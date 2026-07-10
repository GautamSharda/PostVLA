#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf


POSTVLA_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SIM_ROOT = POSTVLA_ROOT / "sim"
DEFAULT_CUBE_MANIFEST = POSTVLA_ROOT / "configs" / "manifests" / "cubevar_contact_40.json"


def make_model_cfg(model_path: str, num_steps: int):
    return OmegaConf.create({
        "model_path": model_path,
        "precision": None,
        "model_type": "openpi",
        "num_action_chunks": 16,
        "action_dim": 6,
        "is_lora": False,
        "lora_rank": 32,
        "use_proprio": True,
        "num_steps": num_steps,
        "add_value_head": True,
        "openpi": {
            "config_name": "pi05_so100_sim",
            "num_images_in_input": 2,
            "noise_level": 0.5,
            "action_chunk": 16,
            "num_steps": num_steps,
            "train_expert_only": True,
            "action_env_dim": 6,
            "noise_method": "flow_noise",
            "add_value_head": True,
            "value_after_vlm": True,
            "value_vlm_mode": "mean_token",
            "detach_critic_input": None,
            "use_dsrl": False,
            "dsrl_state_dim": 8,
            "dsrl_action_noise_dim": 32,
            "dsrl_num_q_heads": 10,
            "dsrl_agg_q": "mean",
            "dsrl_image_latent_dim": 64,
            "dsrl_state_latent_dim": 64,
            "dsrl_hidden_dims": [128, 128, 128],
            "action_horizon": 16,
            "noise_params": [0.16, 0.12, 200],
            "joint_logprob": True,
        },
        "policy_setup": "so100_mujoco",
    })


def make_env_cfg(args, num_envs: int):
    return OmegaConf.create({
        "env_type": "so100_mujoco",
        "total_num_envs": num_envs,
        "auto_reset": False,
        "ignore_terminations": True,
        "use_rel_reward": False,
        "seed": args.seed,
        "group_size": 1,
        "use_fixed_reset_state_ids": False,
        "max_steps_per_rollout_epoch": args.max_steps,
        "max_episode_steps": args.max_steps,
        "is_eval": True,
        "video_cfg": {
            "save_video": False,
            "info_on_video": True,
            "fps": 30,
            "extra_info_on_video": ["on_pad", "pad_center_3cm_lift", "cube_target_xy_distance", "max_cube_z"],
            "video_base_dir": str(Path(args.output_dir) / "video"),
        },
        "enable_offload": False,
        "flashact_demo_root": str(DEFAULT_SIM_ROOT),
        "prompt": "pick up the cube and place it on the pink pad",
        "cube_xy_manifest": args.cube_xy_manifest,
        "cube_xy_limit": args.cube_xy_limit,
        "obs_width": 224,
        "obs_height": 224,
        "render_width": 640,
        "render_height": 480,
        "policy_camera": "exterior_left",
        "wrist_camera": "wrist_left",
        "control_fps": 30,
        "render_fps": 30,
        "steps_per_control": None,
        "reset_warmup_frames": 0,
        "reset_warmup_steps": 0,
        "on_pad_half_extent": 0.139,
        "pad_center_threshold": 0.03,
        "lift_z_threshold": 0.13,
        "cube_rest_z_threshold": 0.065,
        "grasp_reward": 0.03,
        "lift_reward": 0.1,
        "progress_reward_scale": 0.5,
        "on_pad_reward": 1.0,
        "pad_center_reward": 0.5,
        "rollout_epoch": 1,
    })


def tensor_to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def render_env_frames(env):
    frames = []
    for data in env.datas:
        env._video_renderer.update_scene(data, camera=env.policy_camera)
        frames.append(env._video_renderer.render().copy())
    return frames


def load_episode_ids(path: str, attempts: int, seed: int, limit: int):
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


def run_batch(model, env_cls, args, attempt_offset: int, episode_ids: list[int], out_dir: Path):
    num_envs = len(episode_ids)
    env = env_cls(make_env_cfg(args, num_envs), num_envs=num_envs, seed_offset=attempt_offset, total_num_processes=1, worker_info=None)
    obs, _ = env.reset(options={"episode_id": np.asarray(episode_ids, dtype=np.int64)})

    finished = np.zeros(num_envs, dtype=bool)
    frames = [[] for _ in range(num_envs)]
    steps = np.zeros(num_envs, dtype=np.int32)

    while not finished.all() and int(steps.max()) < args.max_steps:
        if args.save_videos:
            render_frames = render_env_frames(env)
            for env_i, frame in enumerate(render_frames):
                if not finished[env_i]:
                    frames[env_i].append(frame)
        with torch.no_grad():
            actions, _ = model.predict_action_batch(obs, mode="eval")
        actions = tensor_to_numpy(actions)
        for chunk_i in range(actions.shape[1]):
            obs, _, terminated, truncated, _ = env.step(actions[:, chunk_i, :])
            steps += (~finished).astype(np.int32)
            done = tensor_to_numpy(terminated).astype(bool) | tensor_to_numpy(truncated).astype(bool)
            finished |= done
            if finished.all() or int(steps.max()) >= args.max_steps:
                break

    if args.save_videos:
        render_frames = render_env_frames(env)
        for env_i, frame in enumerate(render_frames):
            frames[env_i].append(frame)

    on_pad_once = tensor_to_numpy(env.success_once).astype(bool)
    pad_center_once = tensor_to_numpy(env.pad_center_once).astype(bool)
    max_cube_z = np.asarray(env._max_cube_z, dtype=np.float32)
    xy_distance = np.asarray([env._cube_target_xy_distance(data) for data in env.datas], dtype=np.float32)

    summaries = []
    for env_i, episode_id in enumerate(episode_ids):
        attempt = attempt_offset + env_i
        item = {
            "attempt": attempt,
            "episode_id": int(episode_id),
            "steps": int(steps[env_i]),
            "on_pad": bool(on_pad_once[env_i]),
            "pad_center_3cm_lift": bool(pad_center_once[env_i]),
            "max_cube_z": float(max_cube_z[env_i]),
            "cube_target_xy_distance": float(xy_distance[env_i]),
        }
        if args.save_videos:
            video = out_dir / "videos" / f"attempt_{attempt:03d}.mp4"
            imageio.mimsave(video, frames[env_i], fps=30)
            item["video"] = str(video)
        summaries.append(item)
    return summaries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--episode-ids-json", default="")
    ap.add_argument("--attempts", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=900)
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cube-xy-manifest", default=str(DEFAULT_CUBE_MANIFEST))
    ap.add_argument("--cube-xy-limit", type=int, default=40)
    ap.add_argument("--save-videos", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "osmesa")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_videos:
        (out_dir / "videos").mkdir(exist_ok=True)

    from rlinf.envs.so100_mujoco import So100MujocoEnv
    from rlinf.models.embodiment.openpi import get_model

    episode_ids = load_episode_ids(args.episode_ids_json, args.attempts, args.seed, args.cube_xy_limit)

    print(f"loading model {args.model_path}", flush=True)
    t0 = time.time()
    model = get_model(make_model_cfg(args.model_path, args.num_steps))
    model.eval()
    model.cuda()
    print(f"loaded model in {time.time() - t0:.1f}s", flush=True)

    results = []
    for offset in range(0, args.attempts, args.batch_size):
        batch_ids = episode_ids[offset : offset + args.batch_size]
        print(f"running attempts {offset}..{offset + len(batch_ids) - 1}", flush=True)
        results.extend(run_batch(model, So100MujocoEnv, args, offset, batch_ids, out_dir))
        partial = {
            "model_path": args.model_path,
            "output_dir": str(out_dir),
            "attempts": len(results),
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "episode_ids": episode_ids[: len(results)],
            "on_pad_count": int(sum(x["on_pad"] for x in results)),
            "pad_center_3cm_lift_count": int(sum(x["pad_center_3cm_lift"] for x in results)),
            "on_pad_rate": float(np.mean([x["on_pad"] for x in results])),
            "pad_center_3cm_lift_rate": float(np.mean([x["pad_center_3cm_lift"] for x in results])),
            "results": results,
        }
        with open(out_dir / "summary.partial.json", "w") as f:
            json.dump(partial, f, indent=2)

    summary = dict(partial)
    summary["attempts"] = args.attempts
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
