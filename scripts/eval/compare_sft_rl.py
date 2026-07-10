#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
import threading
import time
from types import SimpleNamespace
from urllib.parse import urlparse

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import numpy as np
from PIL import Image, ImageDraw
import torch

from rlinf.envs.so100_mujoco import So100MujocoEnv
from rlinf.models.embodiment.openpi import get_model
from eval_noise_control import apply_noise_control
from eval_noise_control import make_env_cfg
from eval_noise_control import make_model_cfg
from eval_noise_control import tensor_to_numpy


POSTVLA_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = Path(os.environ.get("POSTVLA_ARTIFACT_ROOT", POSTVLA_ROOT / "artifacts"))
DEFAULT_SFT = str(ARTIFACT_ROOT / "checkpoints" / "pi05_so100_sim_distilled_sft")
DEFAULT_RL = str(ARTIFACT_ROOT / "checkpoints" / "pi05_so100_sim_rl_stage2_step5")
DEFAULT_MANIFEST = str(POSTVLA_ROOT / "configs" / "manifests" / "cubevar_contact_40.json")

GPU_LOCK = threading.Lock()


def make_args(model_path: str, output_dir: str, seed: int, max_steps: int) -> SimpleNamespace:
    return SimpleNamespace(
        model_path=model_path,
        output_dir=output_dir,
        attempts=1,
        batch_size=1,
        max_steps=max_steps,
        num_steps=10,
        seed=seed,
        cube_xy_manifest=DEFAULT_MANIFEST,
        cube_xy_limit=40,
        save_videos=False,
        initial_noise_mode="zero",
        initial_noise_scale=1.0,
        sampling_mode="eval",
        inference_backend="standard",
    )


class PolicyPanel:
    def __init__(self, key: str, label: str, model_path: str, seed: int, max_steps: int) -> None:
        self.key = key
        self.label = label
        self.model_path = model_path
        self.seed = seed
        self.max_steps = max_steps
        self.output_dir = f"/tmp/flashact_sft_rl_compare/{key}"
        self.args = make_args(model_path, self.output_dir, seed, max_steps)
        self.lock = threading.RLock()
        self.ready = False
        self.load_error: str | None = None
        self.load_seconds: float | None = None
        self.episode_counter = 0
        self.episode_id = 0
        self.steps = 0
        self.last_infer_ms: float | None = None
        self.last_loop_ms: float | None = None
        self.success_hold_frames = 0
        self._model = None
        self._env = None
        self._obs = None
        self._actions = np.empty((0, 6), dtype=np.float32)
        self._action_index = 0
        self._load()

    def _load(self) -> None:
        started = time.perf_counter()
        try:
            print(f"[{self.key}] creating model from {self.model_path}", flush=True)
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            self._model = get_model(make_model_cfg(self.model_path, self.args.num_steps))
            print(f"[{self.key}] moving model to cuda", flush=True)
            self._model.eval()
            self._model.cuda()
            apply_noise_control(self._model, "zero", 1.0)
            print(f"[{self.key}] creating SO100 env", flush=True)
            self._env = So100MujocoEnv(
                make_env_cfg(self.args, 1),
                num_envs=1,
                seed_offset=0,
                total_num_processes=1,
                worker_info=None,
            )
            print(f"[{self.key}] resetting env", flush=True)
            self._reset_locked()
            self.ready = True
            self.load_seconds = time.perf_counter() - started
            print(f"[{self.key}] ready in {self.load_seconds:.1f}s", flush=True)
        except Exception as exc:
            self.load_error = repr(exc)
            self.load_seconds = time.perf_counter() - started
            print(f"[{self.key}] load failed after {self.load_seconds:.1f}s: {self.load_error}", flush=True)

    def _reset_locked(self) -> None:
        assert self._env is not None
        self.episode_id = int((self.episode_counter * 7 + self.seed) % 40)
        self.episode_counter += 1
        self._obs, _ = self._env.reset(options={"episode_id": np.asarray([self.episode_id], dtype=np.int64)})
        self.steps = 0
        self.success_hold_frames = 0
        self._actions = np.empty((0, 6), dtype=np.float32)
        self._action_index = 0

    def _predict_actions_locked(self) -> None:
        assert self._model is not None
        started = time.perf_counter()
        with GPU_LOCK, torch.no_grad():
            actions, _ = self._model.predict_action_batch(self._obs, mode="eval")
        self.last_infer_ms = (time.perf_counter() - started) * 1000.0
        actions = tensor_to_numpy(actions)
        if actions.ndim == 3:
            actions = actions[0]
        self._actions = np.asarray(actions, dtype=np.float32)
        self._action_index = 0

    def _step_locked(self) -> None:
        assert self._env is not None
        if self._actions.size == 0 or self._action_index >= self._actions.shape[0]:
            self._predict_actions_locked()
        action = self._actions[self._action_index : self._action_index + 1]
        self._action_index += 1
        self._obs, _, terminated, truncated, _ = self._env.step(action)
        self.steps += 1

        done = bool(tensor_to_numpy(terminated)[0] or tensor_to_numpy(truncated)[0])
        success = bool(tensor_to_numpy(self._env.success_once)[0])
        if success:
            self.success_hold_frames += 1
        if done or self.steps >= self.max_steps or self.success_hold_frames >= 30:
            self._reset_locked()

    def _render_locked(self) -> np.ndarray:
        assert self._env is not None
        self._env._video_renderer.update_scene(self._env.datas[0], camera=self._env.policy_camera)
        return self._env._video_renderer.render().copy()

    def status(self) -> dict:
        with self.lock:
            if not self.ready:
                return {
                    "key": self.key,
                    "label": self.label,
                    "ready": False,
                    "load_error": self.load_error,
                    "load_seconds": self.load_seconds,
                    "model_path": self.model_path,
                }
            assert self._env is not None
            on_pad = bool(tensor_to_numpy(self._env.success_once)[0])
            center = bool(tensor_to_numpy(self._env.pad_center_once)[0])
            max_z = float(np.asarray(self._env._max_cube_z)[0])
            distance = float(self._env._cube_target_xy_distance(self._env.datas[0]))
            return {
                "key": self.key,
                "label": self.label,
                "ready": True,
                "load_seconds": self.load_seconds,
                "model_path": self.model_path,
                "episode_id": self.episode_id,
                "episode_counter": self.episode_counter,
                "steps": self.steps,
                "on_pad_once": on_pad,
                "pad_center_3cm_lift_once": center,
                "max_cube_z": max_z,
                "cube_target_xy_distance": distance,
                "last_infer_ms": self.last_infer_ms,
                "last_loop_ms": self.last_loop_ms,
            }

    def jpeg(self) -> bytes:
        started = time.perf_counter()
        with self.lock:
            if not self.ready:
                image = Image.new("RGB", (640, 480), "#111827")
                draw = ImageDraw.Draw(image)
                draw.text((24, 24), f"{self.label} loading/error", fill="#ffffff")
                draw.text((24, 54), str(self.load_error), fill="#fca5a5")
            else:
                frame = self._render_locked()
                self._step_locked()
                image = Image.fromarray(frame)
                draw = ImageDraw.Draw(image)
                s = self.status()
                overlay = (
                    f"{self.label} | ep {s['episode_id']} | step {s['steps']} | "
                    f"on_pad {int(s['on_pad_once'])} | center {int(s['pad_center_3cm_lift_once'])}"
                )
                draw.rectangle((0, 0, 640, 34), fill=(0, 0, 0))
                draw.text((10, 10), overlay, fill="#ffffff")
            self.last_loop_ms = (time.perf_counter() - started) * 1000.0

        output = io.BytesIO()
        image.save(output, format="JPEG", quality=82, optimize=True)
        return output.getvalue()


INDEX_HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>SO100 SFT vs RL</title>
    <style>
      body { margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #eef2f6; color: #111827; }
      main { padding: 22px; }
      header { display: flex; align-items: end; justify-content: space-between; gap: 20px; margin-bottom: 18px; }
      h1 { font-size: clamp(32px, 5vw, 72px); margin: 0; line-height: 0.95; letter-spacing: 0; }
      p { margin: 0; color: #64748b; font-weight: 700; text-transform: uppercase; letter-spacing: 0; }
      .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }
      article { background: #fff; border: 1px solid #cbd5e1; border-radius: 8px; overflow: hidden; box-shadow: 0 14px 40px rgba(15, 23, 42, 0.08); }
      article header { padding: 14px 16px; margin: 0; align-items: center; border-bottom: 1px solid #d7dee8; }
      h2 { margin: 0; font-size: 30px; }
      img { display: block; width: 100%; aspect-ratio: 4 / 3; object-fit: cover; background: #111827; }
      dl { display: grid; grid-template-columns: repeat(4, 1fr); margin: 0; border-top: 1px solid #d7dee8; }
      div.metric { padding: 13px 14px; border-right: 1px solid #d7dee8; min-width: 0; }
      div.metric:last-child { border-right: 0; }
      dt { color: #64748b; font-size: 12px; font-weight: 800; text-transform: uppercase; }
      dd { margin: 6px 0 0; font-size: 18px; font-weight: 800; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
      .note { margin-top: 14px; font-size: 13px; color: #64748b; text-transform: none; font-weight: 600; }
      @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } dl { grid-template-columns: repeat(2, 1fr); } }
    </style>
  </head>
  <body>
    <main>
      <header>
        <div>
          <p>FlashAct Demo</p>
          <h1>SO100 SFT vs RL</h1>
        </div>
      </header>
      <section class="grid">
        <article>
          <header><h2>SFT</h2><p>calibrated torch policy</p></header>
          <img src="/stream/sft.mjpeg" />
          <dl id="sft"></dl>
        </article>
        <article>
          <header><h2>RL</h2><p>pick/lift repair step 5</p></header>
          <img src="/stream/rl.mjpeg" />
          <dl id="rl"></dl>
        </article>
      </section>
      <p class="note">Both streams use the same SO100 MuJoCo scene, cube-var episode sequence, zero-noise policy eval, and 900-step reset cap.</p>
    </main>
    <script>
      const keys = ["sft", "rl"];
      function fmt(v, digits = 1) { return v === null || v === undefined ? "--" : Number(v).toFixed(digits); }
      function renderMetric(k, s) {
        document.getElementById(k).innerHTML = `
          <div class="metric"><dt>Episode</dt><dd>${s.episode_id ?? "--"}</dd></div>
          <div class="metric"><dt>Step</dt><dd>${s.steps ?? "--"}</dd></div>
          <div class="metric"><dt>On Pad</dt><dd>${s.on_pad_once ? "yes" : "no"}</dd></div>
          <div class="metric"><dt>Infer</dt><dd>${fmt(s.last_infer_ms)} ms</dd></div>
        `;
      }
      async function poll() {
        const res = await fetch("/status");
        const data = await res.json();
        keys.forEach(k => renderMetric(k, data[k]));
      }
      setInterval(poll, 750);
      poll();
    </script>
  </body>
</html>
"""


class CompareHandler(BaseHTTPRequestHandler):
    panels: dict[str, PolicyPanel] = {}

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)

    def _write(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._write(200, "text/html; charset=utf-8", INDEX_HTML.encode("utf-8"))
            return
        if parsed.path == "/status":
            body = json.dumps({key: panel.status() for key, panel in self.panels.items()}).encode("utf-8")
            self._write(200, "application/json", body)
            return
        if parsed.path.startswith("/stream/") and parsed.path.endswith(".mjpeg"):
            key = parsed.path.removeprefix("/stream/").removesuffix(".mjpeg")
            panel = self.panels.get(key)
            if panel is None:
                self._write(404, "text/plain", b"unknown stream")
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while True:
                    frame = panel.jpeg()
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                    self.wfile.flush()
                    time.sleep(1 / 30)
            except (BrokenPipeError, ConnectionResetError):
                return
            return
        self._write(404, "text/plain", b"not found")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", default=DEFAULT_SFT)
    parser.add_argument("--rl", default=DEFAULT_RL)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--max-steps", type=int, default=900)
    args = parser.parse_args()

    CompareHandler.panels = {
        "sft": PolicyPanel("sft", "SFT", args.sft, seed=0, max_steps=args.max_steps),
        "rl": PolicyPanel("rl", "RL", args.rl, seed=0, max_steps=args.max_steps),
    }
    print(json.dumps({"sft": args.sft, "rl": args.rl, "port": args.port}, indent=2), flush=True)
    server = ThreadingHTTPServer((args.host, args.port), CompareHandler)
    print(f"serving on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
