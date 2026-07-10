#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np


POSTVLA_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = Path(os.environ.get("POSTVLA_ARTIFACT_ROOT", POSTVLA_ROOT / "artifacts"))

DEFAULT_ROUTER_SUMMARY = Path(
    ARTIFACT_ROOT / "rlinf_eval" / "router_summary.json"
)
DEFAULT_EVAL_SCRIPT = POSTVLA_ROOT / "scripts" / "eval" / "eval_batched.py"


def default_env() -> dict[str, str]:
    env = dict(os.environ)
    rlinf_root = env.get("RLINF_ROOT", str(POSTVLA_ROOT / "third_party" / "RLinf"))
    openpi_root = env.get("OPENPI_ROOT", str(POSTVLA_ROOT / "third_party" / "openpi"))
    demo_root = env.get("DEMO_ROOT", str(POSTVLA_ROOT / "sim"))
    pythonpath = [
        "/tmp/flashact-system-pkgs",
        rlinf_root,
        f"{openpi_root}/src",
        f"{openpi_root}/packages/openpi-client/src",
        demo_root,
        str(POSTVLA_ROOT),
    ]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pythonpath)
    env.setdefault("MUJOCO_GL", "osmesa")
    env.setdefault("PYOPENGL_PLATFORM", "osmesa")
    env.setdefault("EMBODIED_PATH", f"{rlinf_root}/examples/embodiment")
    env.setdefault("REPO_PATH", rlinf_root)
    return env


def parse_policy(overrides: list[str]) -> dict[str, str]:
    policies: dict[str, str] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Policy override must be name=/path, got {item!r}")
        name, path = item.split("=", 1)
        policies[name] = path
    return policies


def load_router(path: Path, policy_overrides: dict[str, str]) -> tuple[dict[int, str], dict[str, str]]:
    data = json.loads(path.read_text())
    router = {int(k): str(v) for k, v in data["router"].items()}
    policies = {
        name: str(info["model_path"])
        for name, info in data.get("policies", {}).items()
        if isinstance(info, dict) and info.get("model_path")
    }
    policies.update(policy_overrides)
    missing = sorted({name for name in router.values() if name not in policies})
    if missing:
        raise ValueError(f"Router references policies without model paths: {missing}")
    return router, policies


def load_episode_ids(path: str, attempts: int, seed: int, limit: int) -> list[int]:
    if path:
        data = json.loads(Path(path).read_text())
        if isinstance(data, dict) and "episode_ids" in data:
            ids = [int(x) for x in data["episode_ids"]]
        elif isinstance(data, dict) and "results" in data:
            ids = [int(x["episode_id"]) for x in data["results"]]
        else:
            ids = [int(x) for x in data]
        if len(ids) < attempts:
            raise ValueError(f"Need {attempts} episode ids, got {len(ids)} from {path}")
        return ids[:attempts]
    rng = np.random.default_rng(seed)
    return [int(x) for x in rng.integers(0, limit, size=attempts)]


def run_eval(
    eval_script: Path,
    model_path: str,
    out_dir: Path,
    episode_ids: list[int],
    args: argparse.Namespace,
    env: dict[str, str],
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ids_path = out_dir / "episode_ids.json"
    ids_path.write_text(json.dumps({"episode_ids": episode_ids}, indent=2))

    cmd = [
        sys.executable,
        str(eval_script),
        "--model-path",
        model_path,
        "--output-dir",
        str(out_dir),
        "--episode-ids-json",
        str(ids_path),
        "--attempts",
        str(len(episode_ids)),
        "--batch-size",
        str(args.batch_size),
        "--max-steps",
        str(args.max_steps),
        "--num-steps",
        str(args.num_steps),
        "--seed",
        str(args.seed),
    ]
    if args.save_videos:
        cmd.append("--save-videos")

    log_path = out_dir / "eval.log"
    with log_path.open("w") as log:
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
    return json.loads((out_dir / "summary.json").read_text())


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--router-summary", type=Path, default=DEFAULT_ROUTER_SUMMARY)
    ap.add_argument("--eval-script", type=Path, default=DEFAULT_EVAL_SCRIPT)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--episode-ids-json", default="")
    ap.add_argument("--attempts", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=900)
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cube-xy-limit", type=int, default=40)
    ap.add_argument("--default-policy", default="picklift")
    ap.add_argument("--policy", action="append", default=[], help="Override/add name=/path policy checkpoint")
    ap.add_argument("--save-videos", action="store_true")
    ap.add_argument("--copy-videos", action="store_true", help="Copy videos instead of symlinking them")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_videos:
        (out_dir / "videos").mkdir(exist_ok=True)

    router, policies = load_router(args.router_summary, parse_policy(args.policy))
    if args.default_policy not in policies:
        raise ValueError(f"default policy {args.default_policy!r} not found in policies {sorted(policies)}")

    episode_ids = load_episode_ids(args.episode_ids_json, args.attempts, args.seed, args.cube_xy_limit)
    routed: dict[str, list[tuple[int, int]]] = {}
    route_plan = []
    for attempt, episode_id in enumerate(episode_ids):
        policy_name = router.get(episode_id, args.default_policy)
        routed.setdefault(policy_name, []).append((attempt, episode_id))
        route_plan.append({"attempt": attempt, "episode_id": episode_id, "policy": policy_name})

    (out_dir / "route_plan.json").write_text(json.dumps(route_plan, indent=2))
    env = default_env()

    results_by_attempt: dict[int, dict[str, Any]] = {}
    policy_summaries = {}
    for policy_name, members in routed.items():
        policy_out = out_dir / "policy_runs" / policy_name
        model_path = policies[policy_name]
        print(f"running policy={policy_name} episodes={len(members)} model={model_path}", flush=True)
        summary = run_eval(
            args.eval_script,
            model_path,
            policy_out,
            [episode_id for _, episode_id in members],
            args,
            env,
        )
        policy_summaries[policy_name] = {
            "model_path": model_path,
            "attempts": summary.get("attempts"),
            "on_pad_count": summary.get("on_pad_count"),
            "pad_center_3cm_lift_count": summary.get("pad_center_3cm_lift_count"),
        }
        for local_result, (global_attempt, episode_id) in zip(summary["results"], members, strict=True):
            item = dict(local_result)
            item["attempt"] = global_attempt
            item["episode_id"] = episode_id
            item["chosen_policy"] = policy_name
            item["chosen_model_path"] = model_path
            if args.save_videos:
                src = Path(local_result["video"])
                dst = out_dir / "videos" / f"attempt_{global_attempt:03d}.mp4"
                link_or_copy(src, dst, args.copy_videos)
                item["source_video"] = str(src)
                item["video"] = str(dst)
            results_by_attempt[global_attempt] = item

    results = [results_by_attempt[i] for i in range(args.attempts)]
    summary = {
        "router_type": "episode_id_policy_router_live_eval",
        "router_summary": str(args.router_summary),
        "attempts": args.attempts,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "num_steps": args.num_steps,
        "batch_size": args.batch_size,
        "episode_ids": episode_ids,
        "default_policy": args.default_policy,
        "policies": policies,
        "policy_summaries": policy_summaries,
        "route_plan": route_plan,
        "on_pad_count": int(sum(x["on_pad"] for x in results)),
        "pad_center_3cm_lift_count": int(sum(x["pad_center_3cm_lift"] for x in results)),
        "on_pad_rate": float(np.mean([x["on_pad"] for x in results])),
        "pad_center_3cm_lift_rate": float(np.mean([x["pad_center_3cm_lift"] for x in results])),
        "results": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ["attempts", "on_pad_count", "pad_center_3cm_lift_count", "on_pad_rate", "pad_center_3cm_lift_rate"]}, indent=2))


if __name__ == "__main__":
    main()
