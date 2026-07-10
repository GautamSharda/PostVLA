from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys
from typing import Optional, Union

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import gymnasium as gym
import mujoco
import numpy as np
import torch

__all__ = ["So100MujocoEnv"]

PICK_CUBE_HOME = np.asarray([-0.08, -0.25, 0.078], dtype=np.float64)
ROBOT_HOME_QPOS = np.asarray([0.0, -1.57005, 1.3525, 1.57079, -1.57079, 0.0], dtype=np.float64)


def _cfg_get(cfg, name: str, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _clone_nested(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _clone_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_nested(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_nested(item) for item in value)
    return value


def _tile_images(images: list[np.ndarray]) -> np.ndarray:
    if len(images) == 1:
        return images[0]
    height, width = images[0].shape[:2]
    cols = int(np.ceil(np.sqrt(len(images))))
    rows = int(np.ceil(len(images) / cols))
    canvas = np.zeros((rows * height, cols * width, 3), dtype=np.uint8)
    for index, image in enumerate(images):
        row = index // cols
        col = index % cols
        canvas[row * height : (row + 1) * height, col * width : (col + 1) * width] = image
    return canvas


class So100MujocoEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
        record_metrics=True,
    ):
        self.cfg = cfg
        self.seed = int(_cfg_get(cfg, "seed", 0)) + int(seed_offset)
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.auto_reset = bool(_cfg_get(cfg, "auto_reset", False))
        self.ignore_terminations = bool(_cfg_get(cfg, "ignore_terminations", False))
        self.terminate_on = str(_cfg_get(cfg, "terminate_on", "on_pad"))
        if self.terminate_on not in {"on_pad", "pad_center", "pad_center_3cm_lift", "none"}:
            raise ValueError(f"Unsupported terminate_on={self.terminate_on!r}")
        self.use_fixed_reset_state_ids = bool(_cfg_get(cfg, "use_fixed_reset_state_ids", False))
        self.group_size = int(_cfg_get(cfg, "group_size", 1))
        self.num_group = max(1, int(num_envs) // max(1, self.group_size))
        self.record_metrics = record_metrics
        self._num_envs = int(num_envs)
        self._device = torch.device("cpu")
        self._is_start = True
        self._rng = np.random.default_rng(self.seed)

        self.prompt = str(_cfg_get(cfg, "prompt", "pick up the cube and place it on the pink pad"))
        self.obs_width = int(_cfg_get(cfg, "obs_width", 224))
        self.obs_height = int(_cfg_get(cfg, "obs_height", 224))
        self.render_width = int(_cfg_get(cfg, "render_width", 640))
        self.render_height = int(_cfg_get(cfg, "render_height", 480))
        self.policy_camera = str(_cfg_get(cfg, "policy_camera", "exterior_left"))
        self.wrist_camera = str(_cfg_get(cfg, "wrist_camera", "wrist_left"))
        self.max_episode_steps = int(_cfg_get(cfg, "max_episode_steps", 480))
        self.on_pad_half_extent = float(_cfg_get(cfg, "on_pad_half_extent", 0.139))
        self.pad_center_threshold = float(_cfg_get(cfg, "pad_center_threshold", 0.03))
        self.lift_z_threshold = float(_cfg_get(cfg, "lift_z_threshold", 0.13))
        self.cube_rest_z_threshold = float(_cfg_get(cfg, "cube_rest_z_threshold", 0.065))
        self.grasp_reward = float(_cfg_get(cfg, "grasp_reward", 0.03))
        self.lift_reward = float(_cfg_get(cfg, "lift_reward", 0.10))
        self.progress_reward_scale = float(_cfg_get(cfg, "progress_reward_scale", 0.50))
        self.on_pad_reward = float(_cfg_get(cfg, "on_pad_reward", 1.0))
        self.pad_center_reward = float(_cfg_get(cfg, "pad_center_reward", 0.5))
        self.once_rewards = bool(_cfg_get(cfg, "once_rewards", False))

        demo_root = Path(
            str(
                _cfg_get(
                    cfg,
                    "flashact_demo_root",
                    os.environ.get("POSTVLA_SIM_ROOT", "sim"),
                )
            )
        )
        sys.path.insert(0, str(demo_root))
        from app import mujoco_stream

        self._mujoco_stream = mujoco_stream
        self._scene_path = self._prepare_scene_xml(Path(mujoco_stream.scene_path()))
        self.model = mujoco.MjModel.from_xml_path(str(self._scene_path))
        self.datas = [mujoco.MjData(self.model) for _ in range(self._num_envs)]
        self._obs_renderer = mujoco.Renderer(self.model, height=self.obs_height, width=self.obs_width)
        self._video_renderer = mujoco.Renderer(self.model, height=self.render_height, width=self.render_width)

        self._cube_body_id = self._body_id("pick_cube")
        self._target_body_id = self._body_id("place_target")
        self._cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pick_cube_free")
        self._gripper_body_ids = self._find_gripper_body_ids()
        self._arm_dofs = min(self.model.nu, ROBOT_HOME_QPOS.shape[0], self.model.nq)
        self._ctrl_low = np.full(self._arm_dofs, -np.inf, dtype=np.float64)
        self._ctrl_high = np.full(self._arm_dofs, np.inf, dtype=np.float64)
        limited = np.asarray(self.model.actuator_ctrllimited[: self._arm_dofs], dtype=bool)
        ranges = np.asarray(self.model.actuator_ctrlrange[: self._arm_dofs], dtype=np.float64)
        self._ctrl_low[limited] = ranges[limited, 0]
        self._ctrl_high[limited] = ranges[limited, 1]
        self.action_space = gym.spaces.Box(
            low=self._ctrl_low.astype(np.float32),
            high=self._ctrl_high.astype(np.float32),
            dtype=np.float32,
        )

        configured_steps = _cfg_get(cfg, "steps_per_control", None)
        if configured_steps is None:
            control_fps = float(_cfg_get(cfg, "control_fps", 30))
            self.steps_per_control = max(1, int(round((1.0 / control_fps) / self.model.opt.timestep)))
        else:
            self.steps_per_control = int(configured_steps)
        configured_warmup_steps = _cfg_get(cfg, "reset_warmup_steps", None)
        if configured_warmup_steps is None:
            reset_warmup_frames = int(_cfg_get(cfg, "reset_warmup_frames", 0))
            self.reset_warmup_steps = max(0, reset_warmup_frames * self.steps_per_control)
        else:
            self.reset_warmup_steps = max(0, int(configured_warmup_steps))

        self._cube_xy_positions = self._load_cube_xy_positions()
        self._elapsed_steps = torch.zeros(self._num_envs, dtype=torch.long, device=self.device)
        self.prev_step_reward = torch.zeros(self._num_envs, dtype=torch.float32, device=self.device)
        self.returns = torch.zeros(self._num_envs, dtype=torch.float32, device=self.device)
        self.success_once = torch.zeros(self._num_envs, dtype=torch.bool, device=self.device)
        self.pad_center_once = torch.zeros(self._num_envs, dtype=torch.bool, device=self.device)
        self._prev_cube_target_xy_distance = np.zeros(self._num_envs, dtype=np.float64)
        self._max_cube_z = np.zeros(self._num_envs, dtype=np.float64)
        self.update_reset_state_ids()

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def device(self):
        return self._device

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    @property
    def task_descriptions(self):
        return [self.prompt for _ in range(self.num_envs)]

    @property
    def total_num_group_envs(self):
        return len(self._cube_xy_positions)

    def update_reset_state_ids(self):
        state_ids = self._rng.integers(0, self.total_num_group_envs, size=(self.num_group,))
        state_ids = np.repeat(state_ids, self.group_size)[: self.num_envs]
        if state_ids.shape[0] < self.num_envs:
            state_ids = np.resize(state_ids, self.num_envs)
        self.reset_state_ids = torch.as_tensor(state_ids, dtype=torch.long, device=self.device)

    def reset(
        self,
        *,
        seed: Optional[Union[int, list[int]]] = None,
        options: Optional[dict] = None,
    ):
        del seed
        env_indices = self._option_env_indices(options)
        episode_ids = self._option_episode_ids(options, env_indices)
        for env_id, episode_id in zip(env_indices, episode_ids, strict=True):
            self._reset_one(env_id, int(episode_id))
        self._reset_metrics(env_indices)
        obs = self._get_obs()
        infos = self._get_info()
        return obs, infos

    def step(self, actions=None, auto_reset=True):
        actions = self._normalize_actions(actions)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        on_pad = np.zeros(self.num_envs, dtype=bool)
        pad_center = np.zeros(self.num_envs, dtype=bool)
        lifted = np.zeros(self.num_envs, dtype=bool)
        grasp_contact = np.zeros(self.num_envs, dtype=bool)
        xy_distance = np.zeros(self.num_envs, dtype=np.float32)

        for env_id, (data, action) in enumerate(zip(self.datas, actions, strict=True)):
            ctrl = np.clip(action[: self._arm_dofs], self._ctrl_low, self._ctrl_high)
            data.ctrl[: self._arm_dofs] = ctrl
            for _ in range(self.steps_per_control):
                mujoco.mj_step(self.model, data)
                self._max_cube_z[env_id] = max(self._max_cube_z[env_id], self._cube_pos(data)[2])

            self._elapsed_steps[env_id] += 1
            cube_pos = self._cube_pos(data)
            target_pos = self._target_pos(data)
            delta_xy = cube_pos[:2] - target_pos[:2]
            xy_distance[env_id] = float(np.linalg.norm(delta_xy))
            lifted[env_id] = bool(self._max_cube_z[env_id] > self.lift_z_threshold)
            on_pad[env_id] = bool(
                abs(delta_xy[0]) <= self.on_pad_half_extent
                and abs(delta_xy[1]) <= self.on_pad_half_extent
                and cube_pos[2] >= self.cube_rest_z_threshold
            )
            pad_center[env_id] = bool(
                xy_distance[env_id] < self.pad_center_threshold
                and lifted[env_id]
                and cube_pos[2] >= self.cube_rest_z_threshold
            )
            grasp_contact[env_id] = self._cube_contacting_gripper(data)
            progress = max(0.0, self._prev_cube_target_xy_distance[env_id] - xy_distance[env_id])
            self._prev_cube_target_xy_distance[env_id] = xy_distance[env_id]
            on_pad_rewarded = bool(self.success_once[env_id].item())
            pad_center_rewarded = bool(self.pad_center_once[env_id].item())
            rewards[env_id] += self.grasp_reward if grasp_contact[env_id] else 0.0
            rewards[env_id] += self.lift_reward if lifted[env_id] else 0.0
            rewards[env_id] += self.progress_reward_scale * progress if lifted[env_id] else 0.0
            if self.once_rewards:
                rewards[env_id] += self.on_pad_reward if on_pad[env_id] and not on_pad_rewarded else 0.0
                rewards[env_id] += self.pad_center_reward if pad_center[env_id] and not pad_center_rewarded else 0.0
            else:
                rewards[env_id] += self.on_pad_reward if on_pad[env_id] else 0.0
                rewards[env_id] += self.pad_center_reward if pad_center[env_id] else 0.0

        on_pad_t = torch.as_tensor(on_pad, dtype=torch.bool, device=self.device)
        pad_center_t = torch.as_tensor(pad_center, dtype=torch.bool, device=self.device)
        reward_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        if self.terminate_on in {"pad_center", "pad_center_3cm_lift"}:
            terminations = pad_center_t.clone()
        elif self.terminate_on == "none":
            terminations = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            terminations = on_pad_t.clone()
        truncations = self._elapsed_steps >= self.max_episode_steps
        infos = self._get_info(
            reward_t=reward_t,
            on_pad=on_pad,
            pad_center=pad_center,
            lifted=lifted,
            grasp_contact=grasp_contact,
            xy_distance=xy_distance,
        )

        self.returns += reward_t
        self.success_once |= on_pad_t
        self.pad_center_once |= pad_center_t
        infos["episode"] = self._episode_info(reward_t)
        infos["episode"]["success_at_end"] = on_pad_t.clone()
        infos["episode"]["on_pad_at_end"] = on_pad_t.clone()
        infos["episode"]["pad_center_3cm_lift_at_end"] = pad_center_t.clone()

        if self.ignore_terminations:
            terminations[:] = False

        obs = self._get_obs()
        dones = torch.logical_or(terminations, truncations)
        if dones.any() and auto_reset and self.auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)
        return obs, reward_t, terminations, truncations, infos

    def chunk_step(self, chunk_actions):
        chunk_size = chunk_actions.shape[1]
        obs_list = []
        infos_list = []
        chunk_rewards = []
        raw_chunk_terminations = []
        raw_chunk_truncations = []
        for i in range(chunk_size):
            obs, reward, terminations, truncations, infos = self.step(chunk_actions[:, i], auto_reset=False)
            obs_list.append(obs)
            infos_list.append(infos)
            chunk_rewards.append(reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        chunk_rewards = torch.stack(chunk_rewards, dim=1)
        raw_chunk_terminations = torch.stack(raw_chunk_terminations, dim=1)
        raw_chunk_truncations = torch.stack(raw_chunk_truncations, dim=1)
        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(past_dones, obs_list[-1], infos_list[-1])

        chunk_terminations = torch.zeros_like(raw_chunk_terminations)
        chunk_terminations[:, -1] = past_terminations
        chunk_truncations = torch.zeros_like(raw_chunk_truncations)
        chunk_truncations[:, -1] = past_truncations
        return obs_list, chunk_rewards, chunk_terminations, chunk_truncations, infos_list

    def capture_image(self, infos=None):
        del infos
        images = []
        for data in self.datas:
            self._video_renderer.update_scene(data, camera=self.policy_camera)
            images.append(self._video_renderer.render().copy())
        return _tile_images(images)

    def render(self, info=None, rew=None):
        del info, rew
        return self.capture_image()

    def sample_action_space(self):
        return np.stack([self.action_space.sample() for _ in range(self.num_envs)], axis=0)

    def close(self):
        if hasattr(self._obs_renderer, "close"):
            self._obs_renderer.close()
        if hasattr(self._video_renderer, "close"):
            self._video_renderer.close()

    def _prepare_scene_xml(self, path: Path) -> Path:
        text = path.read_text()
        generated_dir = str(path.parent)
        text = text.replace(f'{generated_dir}/so_arm100_flashact.xml', "so_arm100_flashact.xml")
        path.write_text(text)
        robot_path = path.parent / "so_arm100_flashact.xml"
        robot_text = robot_path.read_text()
        if 'autolimits="true"' not in robot_text:
            robot_text = robot_text.replace("<compiler ", '<compiler autolimits="true" ', 1)
        robot_text = re.sub(r"\n\s*<keyframe>.*?</keyframe>", "", robot_text, flags=re.S)
        robot_path.write_text(robot_text)
        return path

    def _body_id(self, name: str) -> int:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise RuntimeError(f"Missing MuJoCo body: {name}")
        return body_id

    def _find_gripper_body_ids(self) -> set[int]:
        ids = set()
        for body_id in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            if "Jaw" in name:
                ids.add(body_id)
        return ids

    def _load_cube_xy_positions(self) -> list[np.ndarray]:
        manifest = _cfg_get(self.cfg, "cube_xy_manifest", None)
        if not manifest:
            return [PICK_CUBE_HOME[:2].copy()]
        path = Path(str(manifest))
        if not path.exists():
            return [PICK_CUBE_HOME[:2].copy()]
        data = json.loads(path.read_text())
        records = data.get("records", data) if isinstance(data, dict) else data
        limit = _cfg_get(self.cfg, "cube_xy_limit", None)
        positions = []
        for record in records:
            if "cube_xy" in record:
                positions.append(np.asarray(record["cube_xy"], dtype=np.float64))
            if limit is not None and len(positions) >= int(limit):
                break
        return positions or [PICK_CUBE_HOME[:2].copy()]

    def _option_env_indices(self, options: Optional[dict]) -> list[int]:
        if options is not None and "env_idx" in options:
            indices = _to_numpy(options["env_idx"]).astype(np.int64).reshape(-1)
            return [int(index) for index in indices]
        return list(range(self.num_envs))

    def _option_episode_ids(self, options: Optional[dict], env_indices: list[int]) -> np.ndarray:
        if options is not None and "episode_id" in options:
            episode_ids = _to_numpy(options["episode_id"]).astype(np.int64).reshape(-1)
            if episode_ids.shape[0] == self.num_envs:
                return episode_ids[np.asarray(env_indices, dtype=np.int64)]
            if episode_ids.shape[0] == len(env_indices):
                return episode_ids
        if self.use_fixed_reset_state_ids:
            return self.reset_state_ids[env_indices].detach().cpu().numpy()
        return self._rng.integers(0, self.total_num_group_envs, size=(len(env_indices),))

    def _reset_one(self, env_id: int, episode_id: int) -> None:
        data = self.datas[env_id]
        key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key >= 0:
            mujoco.mj_resetDataKeyframe(self.model, data, key)
        else:
            mujoco.mj_resetData(self.model, data)
        data.qpos[: self._arm_dofs] = ROBOT_HOME_QPOS[: self._arm_dofs]
        data.qvel[: self._arm_dofs] = 0.0
        data.ctrl[: self._arm_dofs] = ROBOT_HOME_QPOS[: self._arm_dofs]
        cube_position = PICK_CUBE_HOME.copy()
        cube_position[:2] = self._cube_xy_positions[episode_id % len(self._cube_xy_positions)]
        if self._cube_joint_id >= 0:
            qpos_adr = int(self.model.jnt_qposadr[self._cube_joint_id])
            qvel_adr = int(self.model.jnt_dofadr[self._cube_joint_id])
            data.qpos[qpos_adr : qpos_adr + 3] = cube_position
            data.qpos[qpos_adr + 3 : qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
            data.qvel[qvel_adr : qvel_adr + 6] = 0.0
        mujoco.mj_forward(self.model, data)
        if self.reset_warmup_steps:
            data.ctrl[: self._arm_dofs] = ROBOT_HOME_QPOS[: self._arm_dofs]
            for _ in range(self.reset_warmup_steps):
                mujoco.mj_step(self.model, data)

    def _reset_metrics(self, env_indices: list[int]) -> None:
        for env_id in env_indices:
            self.prev_step_reward[env_id] = 0.0
            self.returns[env_id] = 0.0
            self.success_once[env_id] = False
            self.pad_center_once[env_id] = False
            self._elapsed_steps[env_id] = 0
            cube_pos = self._cube_pos(self.datas[env_id])
            self._max_cube_z[env_id] = cube_pos[2]
            self._prev_cube_target_xy_distance[env_id] = self._cube_target_xy_distance(self.datas[env_id])

    def _normalize_actions(self, actions) -> np.ndarray:
        if actions is None:
            actions = np.stack([data.ctrl[: self._arm_dofs] for data in self.datas], axis=0)
        actions = _to_numpy(actions).astype(np.float64)
        if actions.ndim == 1:
            actions = np.broadcast_to(actions[None, :], (self.num_envs, actions.shape[0]))
        if actions.shape[0] != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} actions, got {actions.shape[0]}")
        return actions

    def _get_obs(self) -> dict:
        main_images = []
        wrist_images = []
        states = []
        for data in self.datas:
            self._obs_renderer.update_scene(data, camera=self.policy_camera)
            main_images.append(self._obs_renderer.render().copy())
            self._obs_renderer.update_scene(data, camera=self.wrist_camera)
            wrist_images.append(self._obs_renderer.render().copy())
            states.append(np.asarray(data.qpos[: self._arm_dofs], dtype=np.float32).copy())
        return {
            "main_images": torch.as_tensor(np.stack(main_images), dtype=torch.uint8, device=self.device),
            "wrist_images": torch.as_tensor(np.stack(wrist_images), dtype=torch.uint8, device=self.device),
            "extra_view_images": None,
            "states": torch.as_tensor(np.stack(states), dtype=torch.float32, device=self.device),
            "task_descriptions": self.task_descriptions,
        }

    def _get_info(
        self,
        reward_t: torch.Tensor | None = None,
        on_pad: np.ndarray | None = None,
        pad_center: np.ndarray | None = None,
        lifted: np.ndarray | None = None,
        grasp_contact: np.ndarray | None = None,
        xy_distance: np.ndarray | None = None,
    ) -> dict:
        if on_pad is None:
            on_pad = np.zeros(self.num_envs, dtype=bool)
            pad_center = np.zeros(self.num_envs, dtype=bool)
            lifted = np.zeros(self.num_envs, dtype=bool)
            grasp_contact = np.zeros(self.num_envs, dtype=bool)
            xy_distance = np.asarray([self._cube_target_xy_distance(data) for data in self.datas], dtype=np.float32)
        assert pad_center is not None and lifted is not None and grasp_contact is not None and xy_distance is not None
        return {
            "success": torch.as_tensor(on_pad, dtype=torch.bool, device=self.device),
            "on_pad": torch.as_tensor(on_pad, dtype=torch.bool, device=self.device),
            "pad_center_3cm_lift": torch.as_tensor(pad_center, dtype=torch.bool, device=self.device),
            "is_lifted": torch.as_tensor(lifted, dtype=torch.bool, device=self.device),
            "is_grasp_contact": torch.as_tensor(grasp_contact, dtype=torch.bool, device=self.device),
            "cube_target_xy_distance": torch.as_tensor(xy_distance, dtype=torch.float32, device=self.device),
            "max_cube_z": torch.as_tensor(self._max_cube_z, dtype=torch.float32, device=self.device),
            "reward": reward_t if reward_t is not None else torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
        }

    def _episode_info(self, step_reward: torch.Tensor) -> dict:
        episode_len = torch.clamp(self._elapsed_steps, min=1).to(torch.float32)
        return {
            "return": self.returns.clone(),
            "episode_len": self._elapsed_steps.clone(),
            "reward": self.returns.clone() / episode_len,
            "step_reward": step_reward.clone(),
            "success_once": self.success_once.clone(),
            "on_pad_once": self.success_once.clone(),
            "pad_center_3cm_lift_once": self.pad_center_once.clone(),
        }

    def _handle_auto_reset(self, dones: torch.Tensor, obs: dict, infos: dict):
        final_obs = _clone_nested(obs)
        final_info = _clone_nested(infos)
        env_idx = torch.arange(0, self.num_envs, device=self.device)[dones]
        options = {"env_idx": env_idx}
        if self.use_fixed_reset_state_ids:
            options["episode_id"] = self.reset_state_ids[env_idx]
        obs, reset_infos = self.reset(options=options)
        reset_infos["final_observation"] = final_obs
        reset_infos["final_info"] = final_info
        reset_infos["_final_info"] = dones.clone()
        reset_infos["_final_observation"] = dones.clone()
        reset_infos["_elapsed_steps"] = dones.clone()
        return obs, reset_infos

    def _cube_pos(self, data: mujoco.MjData) -> np.ndarray:
        return np.asarray(data.xpos[self._cube_body_id], dtype=np.float64).copy()

    def _target_pos(self, data: mujoco.MjData) -> np.ndarray:
        return np.asarray(data.xpos[self._target_body_id], dtype=np.float64).copy()

    def _cube_target_xy_distance(self, data: mujoco.MjData) -> float:
        return float(np.linalg.norm(self._cube_pos(data)[:2] - self._target_pos(data)[:2]))

    def _cube_contacting_gripper(self, data: mujoco.MjData) -> bool:
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            body1 = int(self.model.geom_bodyid[contact.geom1])
            body2 = int(self.model.geom_bodyid[contact.geom2])
            if body1 == self._cube_body_id and body2 in self._gripper_body_ids:
                return True
            if body2 == self._cube_body_id and body1 in self._gripper_body_ids:
                return True
        return False
