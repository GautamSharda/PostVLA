#!/usr/bin/env python3
from __future__ import annotations

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from urllib.parse import parse_qs, urlparse


POSTVLA_ROOT = Path(__file__).resolve().parents[1]
SITE_ROOT = Path(os.environ.get("POSTVLA_SITE_ROOT", Path(__file__).resolve().parent))
ARTIFACT_ROOT = Path(
    os.environ.get("POSTVLA_ARTIFACT_ROOT", POSTVLA_ROOT / "artifacts")
)
RLINF_ROOT = POSTVLA_ROOT / "third_party" / "RLinf"
OPENPI_ROOT = POSTVLA_ROOT / "third_party" / "openpi"
FLASHRT_ROOT = POSTVLA_ROOT / "third_party" / "FlashRT"
DEMO_ROOT = POSTVLA_ROOT / "sim"
EVAL_SCRIPT = POSTVLA_ROOT / "scripts" / "eval" / "eval_noise_control.py"
DEFAULT_TASK_ID = 34

SFT_PATH = os.environ.get(
    "POSTVLA_SFT_CKPT",
    str(ARTIFACT_ROOT / "checkpoints" / "pi05_so100_sim_distilled_sft"),
)
RL_PATH = os.environ.get(
    "POSTVLA_RL_CKPT",
    str(ARTIFACT_ROOT / "checkpoints" / "pi05_so100_sim_rl_stage2_step5"),
)
HYBRID_FRONTEND_CHECKPOINT = os.environ.get(
    "POSTVLA_HYBRID_FRONTEND_CHECKPOINT",
    SFT_PATH,
)

MODELS = {
    "sft": {
        "label": "SFT standard",
        "summary": SITE_ROOT / "sft_summary.json",
        "path": SFT_PATH,
        "backend": "standard",
    },
    "sft_opt": {
        "label": "SFT optimized",
        "summary": SITE_ROOT / "sft_summary.json",
        "path": SFT_PATH,
        "backend": "hybrid",
    },
    "rl": {
        "label": "RL standard",
        "summary": SITE_ROOT / "rl_summary.json",
        "path": RL_PATH,
        "backend": "standard",
    },
    "rl_opt": {
        "label": "RL optimized",
        "summary": SITE_ROOT / "rl_summary.json",
        "path": RL_PATH,
        "backend": "hybrid",
    },
}

JOBS: dict[str, dict] = {}
JOB_LOCK = threading.Lock()


def eval_env(backend: str) -> dict[str, str]:
    env = os.environ.copy()
    if backend == "hybrid":
        pythonpath = [
            str(FLASHRT_ROOT),
            str(POSTVLA_ROOT / "kernels" / "pi05" / "mk_v6"),
            str(POSTVLA_ROOT / "scripts" / "eval"),
            str(RLINF_ROOT),
            str(OPENPI_ROOT / "src"),
            str(OPENPI_ROOT / "packages/openpi-client/src"),
            str(DEMO_ROOT),
            str(POSTVLA_ROOT),
        ]
        hybrid_site_packages = env.get("POSTVLA_HYBRID_SITE_PACKAGES")
        if hybrid_site_packages:
            pythonpath.insert(0, hybrid_site_packages)
        cuda_home = env.get("POSTVLA_CUDA_HOME", "/usr/local/cuda-13.0")
        env["CUDA_HOME"] = cuda_home
        env["PATH"] = f"{cuda_home}/bin:{env.get('PATH', '')}"
        env["LD_LIBRARY_PATH"] = f"{cuda_home}/lib64:{env.get('LD_LIBRARY_PATH', '')}"
        torch_extensions_dir = "/tmp/flashact_torch_ext_cu130"
    else:
        pythonpath = [
            str(POSTVLA_ROOT / "scripts" / "eval"),
            str(POSTVLA_ROOT / "kernels" / "pi05" / "mk_common"),
            str(POSTVLA_ROOT / "kernels" / "pi05" / "mk_v6"),
            str(RLINF_ROOT),
            str(OPENPI_ROOT / "src"),
            str(OPENPI_ROOT / "packages/openpi-client/src"),
            str(DEMO_ROOT),
            str(POSTVLA_ROOT),
        ]
        standard_site_packages = env.get("POSTVLA_STANDARD_SITE_PACKAGES")
        if standard_site_packages:
            pythonpath.insert(0, standard_site_packages)
        torch_extensions_dir = "/tmp/flashact_torch_ext"
    env.update(
        {
            "MUJOCO_GL": "osmesa",
            "PYOPENGL_PLATFORM": "osmesa",
            "EMBODIED_PATH": str(RLINF_ROOT / "examples/embodiment"),
            "REPO_PATH": str(RLINF_ROOT),
            "PYTHONPATH": os.pathsep.join(pythonpath),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TORCH_EXTENSIONS_DIR": torch_extensions_dir,
        }
    )
    return env


def read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def tail_text(path: Path, max_chars: int = 6000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(errors="replace")
    return text[-max_chars:]


def stream_url(job_id: str) -> str:
    return f"/api/live-stream?job_id={job_id}"


def job_with_stream_progress(job: dict) -> dict:
    payload = dict(job)
    if payload.get("state") != "running":
        return payload
    state_path = Path(payload.get("output_dir", "")) / "stream" / "state.json"
    if not state_path.exists():
        return payload
    try:
        stream_state = read_json(state_path)
    except Exception:
        return payload
    payload["stream_state"] = stream_state
    payload["message"] = (
        f"Streaming task {payload['task_id']}, frame {stream_state.get('frame', 0)}..."
    )
    return payload


def run_job(job_id: str) -> None:
    job = JOBS[job_id]
    model_key = job["model"]
    task_id = int(job["task_id"])
    out_dir = SITE_ROOT / "live_outputs" / job_id
    log_path = out_dir / "eval.log"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        episode_file = out_dir / "episode_ids.json"
        episode_file.write_text(json.dumps({"episode_ids": [task_id]}) + "\n")

        job.update(
            {
                "state": "running",
                "message": f"Loading {MODELS[model_key]['label']} for task {task_id}...",
                "output_dir": str(out_dir),
                "task_id": task_id,
                "stream_url": stream_url(job_id),
                "backend": MODELS[model_key]["backend"],
                "label": MODELS[model_key]["label"],
            }
        )

        command = [
            "python3",
            str(EVAL_SCRIPT),
            "--model-path",
            MODELS[model_key]["path"],
            "--output-dir",
            str(out_dir),
            "--episode-ids-json",
            str(episode_file),
            "--attempts",
            "1",
            "--batch-size",
            "1",
            "--max-steps",
            "900",
            "--num-steps",
            "10",
            "--seed",
            "0",
            "--initial-noise-mode",
            "zero",
            "--save-videos",
            "--stream-dir",
            str(out_dir / "stream"),
            "--stream-every-steps",
            "1",
            "--stream-realtime-fps",
            "30",
            "--inference-backend",
            MODELS[model_key]["backend"],
        ]
        if MODELS[model_key]["backend"] == "hybrid":
            command.extend(
                ["--hybrid-frontend-checkpoint", HYBRID_FRONTEND_CHECKPOINT]
            )
        with log_path.open("w") as log:
            proc = subprocess.run(
                command,
                cwd=str(RLINF_ROOT),
                env=eval_env(MODELS[model_key]["backend"]),
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=1200,
                check=False,
            )
        if proc.returncode != 0:
            job.update(
                {
                    "state": "failed",
                    "error": f"live eval exited with code {proc.returncode}",
                    "log_tail": tail_text(log_path),
                }
            )
            return

        summary = read_json(out_dir / "summary.json")
        video = out_dir / "videos" / "attempt_000.mp4"
        if not video.exists():
            raise FileNotFoundError(f"Expected video missing: {video}")
        job.update(
            {
                "state": "done",
                "message": f"Live rollout complete for task {task_id}",
                "summary": summary,
                "video_url": f"/live_outputs/{job_id}/videos/attempt_000.mp4?ts={int(time.time())}",
                "stream_url": stream_url(job_id),
                "log_tail": tail_text(log_path),
            }
        )
    except Exception as exc:
        job.update({"state": "failed", "error": repr(exc), "log_tail": tail_text(log_path)})


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(SITE_ROOT), **kwargs)

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_live_stream(self, job_id: str) -> None:
        job = JOBS.get(job_id)
        if not job:
            self._send_json(404, {"error": "unknown job"})
            return

        latest = Path(job.get("output_dir", "")) / "stream" / "latest.jpg"
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Pragma", "no-cache")
        self.end_headers()

        last_sig = None
        idle_after_done = 0
        while True:
            current_job = JOBS.get(job_id, job)
            if latest.exists():
                stat = latest.stat()
                sig = (stat.st_mtime_ns, stat.st_size)
                if sig != last_sig:
                    data = latest.read_bytes()
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(data)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(data)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                    last_sig = sig

            if current_job.get("state") != "running":
                idle_after_done += 1
                if idle_after_done >= 10:
                    break
            time.sleep(0.1)

    def do_POST(self) -> None:
        if self.path != "/api/live-run":
            self._send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            model_key = str(body.get("model", "")).lower()
            task_id = int(body.get("task_id", DEFAULT_TASK_ID))
            if model_key not in MODELS:
                raise ValueError("model must be one of: " + ", ".join(MODELS))
            if task_id < 0 or task_id >= 40:
                raise ValueError("task_id must be 0..39")
            with JOB_LOCK:
                running = [job for job in JOBS.values() if job.get("state") == "running"]
                if running:
                    self._send_json(409, {"error": "another live inference is already running"})
                    return
                job_id = f"{model_key}_task_{task_id:03d}_{int(time.time())}"
                JOBS[job_id] = {
                    "job_id": job_id,
                    "model": model_key,
                    "task_id": task_id,
                    "state": "running",
                    "message": "Queued",
                    "stream_url": stream_url(job_id),
                    "backend": MODELS[model_key]["backend"],
                    "label": MODELS[model_key]["label"],
                    "created_at": time.time(),
                }
                thread = threading.Thread(target=run_job, args=(job_id,), daemon=True)
                thread.start()
            self._send_json(202, {"job_id": job_id, **JOBS[job_id]})
        except Exception as exc:
            self._send_json(400, {"error": repr(exc)})

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/history", "/history/"}:
            self.path = "/sft_rl_history.html"
            super().do_GET()
            return
        if parsed.path == "/api/live-status":
            job_id = parse_qs(parsed.query).get("job_id", [""])[0]
            job = JOBS.get(job_id)
            if not job:
                self._send_json(404, {"error": "unknown job"})
                return
            self._send_json(200, job_with_stream_progress(job))
            return
        if parsed.path == "/api/live-active":
            running = [job_with_stream_progress(job) for job in JOBS.values() if job.get("state") == "running"]
            self._send_json(200, {"running": running[0] if running else None})
            return
        if parsed.path == "/api/live-stream":
            job_id = parse_qs(parsed.query).get("job_id", [""])[0]
            try:
                self._send_live_stream(job_id)
            except (BrokenPipeError, ConnectionResetError):
                return
            return
        super().do_GET()


def main() -> None:
    SITE_ROOT.mkdir(parents=True, exist_ok=True)
    (SITE_ROOT / "live_outputs").mkdir(exist_ok=True)
    port = int(os.environ.get("POSTVLA_SITE_PORT", "8011"))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"Serving SFT/RL compare site with live runner on :{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
