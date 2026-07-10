#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf


POSTVLA_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SIM_ROOT = POSTVLA_ROOT / "sim"
DEFAULT_CUBE_MANIFEST = POSTVLA_ROOT / "configs" / "manifests" / "cubevar_contact_40.json"
MEGAKERNEL_ROOT = Path(
    os.environ.get("POSTVLA_MEGAKERNEL_ROOT", POSTVLA_ROOT / "kernels" / "pi05")
)


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


def apply_noise_control(model, mode: str, scale: float):
    original_sample_noise = model.sample_noise
    fixed_noise: dict[tuple[tuple[int, ...], str], torch.Tensor] = {}

    def sample_noise(shape, device):
        if mode == "zero":
            return torch.zeros(shape, dtype=torch.float32, device=device)
        if mode == "scaled_gaussian":
            return original_sample_noise(shape, device) * scale
        if mode == "gaussian":
            return original_sample_noise(shape, device)
        if mode == "fixed_gaussian":
            key = (tuple(shape), str(device))
            if key not in fixed_noise:
                fixed_noise[key] = original_sample_noise(shape, device) * scale
            return fixed_noise[key].clone()
        if mode == "first6_gaussian":
            noise = torch.zeros(shape, dtype=torch.float32, device=device)
            active_dims = min(6, shape[-1])
            noise[..., :active_dims] = original_sample_noise(shape, device)[..., :active_dims] * scale
            return noise
        raise ValueError(f"Unknown noise control mode: {mode}")

    model.sample_noise = sample_noise


class MegakernelOpenPiPolicy:
    """RLinf/OpenPI frontend with the SO100 pi0.5 mk_v6 CUDA denoise loop."""

    D, H, HD, F, NL, T, AD, QD = 1024, 8, 256, 4096, 18, 16, 32, 2048
    NB, NT = 170, 256

    def __init__(self, base_model, num_steps: int):
        if int(base_model.config.action_horizon) != self.T:
            raise ValueError(
                f"mk_v6 expects action_horizon={self.T}, got {base_model.config.action_horizon}"
            )
        if int(base_model.config.action_dim) != self.AD:
            raise ValueError(
                f"mk_v6 expects action_dim={self.AD}, got {base_model.config.action_dim}"
            )
        self.base = base_model
        self.config = base_model.config
        self.num_steps = int(num_steps)
        self.last_chunk_ms: float | None = None
        self.chunk_times_ms: list[float] = []
        self._buf_lmax: int | None = None
        self._load_extension()
        self._pack_weights()

    def __getattr__(self, name):
        return getattr(self.base, name)

    def _load_extension(self):
        from torch.utils.cpp_extension import load

        here = MEGAKERNEL_ROOT / "mk_v6"
        os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/flashact_torch_ext")
        self.ext = load(
            name="mk_v6_live",
            sources=[str(here / "binding.cpp"), str(here / "mk6.cu")],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-gencode=arch=compute_120,code=sm_120",
            ],
            extra_cflags=["-O3"],
            verbose=False,
        )

    @staticmethod
    def _quant_fp8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = w.float().abs().amax(dim=-1).clamp(min=1e-8) / 448.0
        q = (w.float() / scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
        return q.view(torch.uint8).contiguous(), scale.float().contiguous()

    def _pack_weights(self):
        sys.path.insert(0, str(MEGAKERNEL_ROOT / "mk_common"))
        import pack_weights as pw

        base = pw.pack(self.base, num_steps=self.num_steps, device="cuda")
        out = {
            k: base[k]
            for k in ("mods", "mod_final", "w_ain", "b_ain", "w_aout", "b_aout")
        }

        p = torch.arange(self.D, device="cuda")
        h, j = p // 128, p % 128
        qsrc = torch.stack([h * 256 + j, h * 256 + j + 128], 1).reshape(-1)
        jk = torch.arange(128, device="cuda")
        ksrc = torch.stack([jk, jk + 128], 1).reshape(-1)

        w1 = torch.cat([base["wq"][:, qsrc], base["wk"][:, ksrc], base["wv"]], dim=1)
        w2 = torch.stack([base["wgate"], base["wup"]], dim=2).reshape(
            self.NL, 2 * self.F, self.D
        )

        for name, tensor in (
            ("w1", w1),
            ("wo", base["wo"]),
            ("w2", w2),
            ("w3", base["wdown"]),
        ):
            q, scale = self._quant_fp8(tensor.to(torch.float16).contiguous())
            out[name] = q
            out[f"s_{name}"] = scale

        self.packed = {k: v.contiguous() for k, v in out.items()}

    def _ensure_buffers(self, lmax: int):
        if self._buf_lmax == lmax:
            return
        device = torch.device("cuda")
        self.kcache = torch.zeros(self.NL, lmax, self.HD, dtype=torch.float16, device=device)
        self.vtcache = torch.zeros(self.NL, self.HD, lmax, dtype=torch.float16, device=device)
        self.x_t = torch.zeros(self.T, self.AD, dtype=torch.float32, device=device)
        self.xb = torch.zeros(2 * self.T, self.D, dtype=torch.float32, device=device)
        self.xp = torch.zeros(4, self.T, self.D, dtype=torch.float32, device=device)
        self.xn = torch.zeros(self.T, self.D, dtype=torch.float16, device=device)
        self.qb = torch.zeros(self.T, self.QD, dtype=torch.float16, device=device)
        self.attnb = torch.zeros(self.T, self.QD, dtype=torch.float16, device=device)
        self.hmlp = torch.zeros(self.T, self.F, dtype=torch.float16, device=device)
        self.scores = torch.zeros(self.H, self.T, lmax, dtype=torch.float32, device=device)
        self.probs = torch.zeros(self.H, self.T, lmax, dtype=torch.float16, device=device)
        self.stage_cycles = torch.zeros(16, dtype=torch.int64, device=device)
        self._buf_lmax = lmax

    @torch.no_grad()
    def sample_actions(self, observation, noise=None, mode="eval", compute_values=False):
        if observation.state.shape[0] != 1:
            raise ValueError("megakernel backend currently supports batch_size=1")
        device = observation.state.device
        if noise is None:
            shape = (1, self.config.action_horizon, self.config.action_dim)
            noise = self.base.sample_noise(shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self.base._preprocess_observation(
            observation, train=False
        )
        _, prefix_pad_masks, past_key_values = self.base._build_prefix_cache(
            images, img_masks, lang_tokens, lang_masks
        )

        sys.path.insert(0, str(MEGAKERNEL_ROOT / "mk_common"))
        import pack_weights as pw

        lp = int(prefix_pad_masks[0].sum().item())
        lmax = (lp + self.T + 15) // 16 * 16
        self._ensure_buffers(lmax)
        kv_prefix = pw.compact_kv(past_key_values, prefix_pad_masks, device="cuda")
        ksrc = kv_prefix[:, 0].contiguous()
        vsrc = kv_prefix[:, 1].contiguous()
        cos, sin = pw.rope_table(lp, device="cuda")

        self.x_t.zero_()
        self.x_t.copy_(noise[0].float())
        self.stage_cycles.zero_()
        self.ext.launch(
            self.packed["w1"],
            self.packed["wo"],
            self.packed["w2"],
            self.packed["w3"],
            self.packed["s_w1"],
            self.packed["s_wo"],
            self.packed["s_w2"],
            self.packed["s_w3"],
            self.packed["mods"],
            self.packed["mod_final"],
            cos,
            sin,
            self.packed["w_ain"],
            self.packed["b_ain"],
            self.packed["w_aout"],
            self.packed["b_aout"],
            ksrc,
            vsrc,
            lp,
            self.kcache,
            self.vtcache,
            lp,
            lmax,
            self.num_steps,
            self.T,
            self.x_t,
            self.xb,
            self.xp,
            self.stage_cycles,
            self.xn,
            self.qb,
            self.attnb,
            self.hmlp,
            self.scores,
            self.probs,
            self.NB,
            self.NT,
        )
        actions = self.x_t[None].clone()
        zeros_logprob = torch.zeros(
            1,
            self.config.action_chunk,
            self.config.action_env_dim,
            dtype=actions.dtype,
            device=device,
        )
        zeros_value = torch.zeros(1, 1, dtype=actions.dtype, device=device)
        return {
            "actions": actions,
            "chains": torch.stack([noise, actions], dim=1),
            "prev_logprobs": zeros_logprob,
            "prev_values": zeros_value,
            "denoise_inds": torch.full((1, self.num_steps), -1, dtype=torch.long, device=device),
        }

    @torch.no_grad()
    def predict_action_batch(self, env_obs, mode="eval", compute_values=False, **kwargs):
        from openpi.models import model as _model
        from rlinf.utils.nested_dict_process import copy_dict_tensor

        to_process_obs = self.base.obs_processor(env_obs)
        processed_obs = self.base.input_transform(to_process_obs, transpose=False)
        processed_obs = self.base.precision_processor(processed_obs)
        observation = _model.Observation.from_dict(processed_obs)
        outputs = self.sample_actions(
            observation, mode=mode, compute_values=compute_values
        )
        actions = self.base.output_transform(
            {"actions": outputs["actions"], "state": observation.state}
        )["actions"]
        forward_inputs = {
            "chains": outputs["chains"],
            "denoise_inds": outputs["denoise_inds"],
            "tokenized_prompt": processed_obs["tokenized_prompt"],
            "tokenized_prompt_mask": processed_obs["tokenized_prompt_mask"],
            "action": actions.reshape(actions.shape[0], -1).contiguous(),
            "model_action": outputs["actions"].reshape(outputs["actions"].shape[0], -1).contiguous(),
        }
        forward_inputs.update(
            copy_dict_tensor({k: v for k, v in to_process_obs.items() if k != "prompt"})
        )
        return actions, {
            "prev_logprobs": outputs["prev_logprobs"],
            "prev_values": outputs["prev_values"],
            "forward_inputs": forward_inputs,
        }


def render_env_frames(env):
    frames = []
    for data in env.datas:
        env._video_renderer.update_scene(data, camera=env.policy_camera)
        frames.append(env._video_renderer.render().copy())
    return frames


def stream_dir_for_attempt(args, attempt: int) -> Path | None:
    if not args.stream_dir:
        return None
    stream_root = Path(args.stream_dir)
    if args.attempts == 1 and args.batch_size == 1:
        return stream_root
    return stream_root / f"attempt_{attempt:03d}"


def publish_stream_frame(
    stream_dir: Path | None,
    frame,
    frame_index: int,
    done: bool = False,
    metrics: dict | None = None,
) -> None:
    if stream_dir is None:
        return
    stream_dir.mkdir(parents=True, exist_ok=True)
    tmp_image = stream_dir / "latest.tmp.jpg"
    latest_image = stream_dir / "latest.jpg"
    tmp_state = stream_dir / "state.tmp.json"
    latest_state = stream_dir / "state.json"
    imageio.imwrite(tmp_image, frame, quality=85)
    tmp_image.replace(latest_image)
    state = {"frame": frame_index, "done": done, "time": time.time()}
    if metrics:
        state["metrics"] = metrics
    tmp_state.write_text(json.dumps(state) + "\n")
    tmp_state.replace(latest_state)


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
    stream_frame_counts = np.zeros(num_envs, dtype=np.int32)
    steps = np.zeros(num_envs, dtype=np.int32)
    chunk_times_ms = [[] for _ in range(num_envs)]

    def metrics_for(env_i: int) -> dict:
        times = chunk_times_ms[env_i]
        return {
            "backend": args.inference_backend,
            "last_chunk_ms": float(times[-1]) if times else None,
            "avg_chunk_ms": float(np.mean(times)) if times else None,
            "chunk_count": len(times),
        }

    if args.stream_dir:
        render_frames = render_env_frames(env)
        for env_i, frame in enumerate(render_frames):
            attempt = attempt_offset + env_i
            publish_stream_frame(
                stream_dir_for_attempt(args, attempt),
                frame,
                int(stream_frame_counts[env_i]),
                metrics=metrics_for(env_i),
            )
            stream_frame_counts[env_i] += 1

    while not finished.all() and int(steps.max()) < args.max_steps:
        if args.save_videos:
            render_frames = render_env_frames(env)
            for env_i, frame in enumerate(render_frames):
                if not finished[env_i]:
                    frames[env_i].append(frame)
        chunk_t0 = time.perf_counter()
        with torch.no_grad():
            actions, _ = model.predict_action_batch(obs, mode=args.sampling_mode)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        chunk_ms = (time.perf_counter() - chunk_t0) * 1000.0
        for env_i in range(num_envs):
            if not finished[env_i]:
                chunk_times_ms[env_i].append(chunk_ms)
        actions = tensor_to_numpy(actions)
        for chunk_i in range(actions.shape[1]):
            obs, _, terminated, truncated, _ = env.step(actions[:, chunk_i, :])
            steps += (~finished).astype(np.int32)
            done = tensor_to_numpy(terminated).astype(bool) | tensor_to_numpy(truncated).astype(bool)
            finished |= done
            if args.stream_dir and (int(steps.max()) % args.stream_every_steps == 0):
                render_frames = render_env_frames(env)
                for env_i, frame in enumerate(render_frames):
                    if not finished[env_i]:
                        attempt = attempt_offset + env_i
                        publish_stream_frame(
                            stream_dir_for_attempt(args, attempt),
                            frame,
                            int(stream_frame_counts[env_i]),
                            metrics=metrics_for(env_i),
                        )
                        stream_frame_counts[env_i] += 1
                if args.stream_realtime_fps > 0:
                    time.sleep(1.0 / args.stream_realtime_fps)
            if finished.all() or int(steps.max()) >= args.max_steps:
                break

    if args.save_videos:
        render_frames = render_env_frames(env)
        for env_i, frame in enumerate(render_frames):
            frames[env_i].append(frame)
    if args.stream_dir:
        render_frames = render_env_frames(env)
        for env_i, frame in enumerate(render_frames):
            attempt = attempt_offset + env_i
            publish_stream_frame(
                stream_dir_for_attempt(args, attempt),
                frame,
                int(stream_frame_counts[env_i]),
                done=True,
                metrics=metrics_for(env_i),
            )

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
            "chunk_latency_ms": metrics_for(env_i),
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
    ap.add_argument("--stream-dir", default="")
    ap.add_argument("--stream-every-steps", type=int, default=1)
    ap.add_argument("--stream-realtime-fps", type=float, default=0.0)
    ap.add_argument(
        "--initial-noise-mode",
        choices=["gaussian", "zero", "scaled_gaussian", "fixed_gaussian", "first6_gaussian"],
        default="gaussian",
    )
    ap.add_argument("--initial-noise-scale", type=float, default=1.0)
    ap.add_argument(
        "--sampling-mode",
        choices=["eval", "train"],
        default="eval",
        help="RLinf policy path: eval is deterministic ODE integration; train enables flow-noise sampling.",
    )
    ap.add_argument("--inference-backend", choices=["standard", "megakernel"], default="standard")
    args = ap.parse_args()

    if args.inference_backend == "megakernel" and args.batch_size != 1:
        raise ValueError("megakernel inference backend currently requires --batch-size 1")

    os.environ.setdefault("MUJOCO_GL", "osmesa")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_videos:
        (out_dir / "videos").mkdir(exist_ok=True)
    if args.stream_dir:
        args.stream_every_steps = max(1, args.stream_every_steps)
        Path(args.stream_dir).mkdir(parents=True, exist_ok=True)

    from rlinf.envs.so100_mujoco import So100MujocoEnv
    from rlinf.models.embodiment.openpi import get_model

    episode_ids = load_episode_ids(args.episode_ids_json, args.attempts, args.seed, args.cube_xy_limit)

    print(f"loading model {args.model_path}", flush=True)
    t0 = time.time()
    model = get_model(make_model_cfg(args.model_path, args.num_steps))
    model.eval()
    model.cuda()
    apply_noise_control(model, args.initial_noise_mode, args.initial_noise_scale)
    if args.inference_backend == "megakernel":
        model = MegakernelOpenPiPolicy(model, args.num_steps)
    print(f"loaded model in {time.time() - t0:.1f}s", flush=True)
    print(
        f"noise_control mode={args.initial_noise_mode} scale={args.initial_noise_scale} backend={args.inference_backend}",
        flush=True,
    )

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
            "inference_backend": args.inference_backend,
            "sampling_mode": args.sampling_mode,
            "initial_noise_mode": args.initial_noise_mode,
            "initial_noise_scale": args.initial_noise_scale,
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
