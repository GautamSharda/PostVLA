from __future__ import annotations

import asyncio
from dataclasses import dataclass
import io
import math
import os
from pathlib import Path
import time
from typing import AsyncIterator
from typing import Any


os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco
import numpy as np
from PIL import Image


POSTVLA_ROOT = Path(__file__).resolve().parents[2]
MODEL_CACHE = Path(
    os.environ.get(
        "POSTVLA_MUJOCO_MENAGERIE",
        POSTVLA_ROOT / "third_party" / "mujoco_menagerie",
    )
)
LEGACY_CACHE = Path("/tmp/mujoco_menagerie_so/trs_so_arm100")
GENERATED_SCENE_CACHE = Path("/tmp/flashact-demo-assets/flashact_so_arm100_scene")
DASHBOARD_CAMERA = "exterior_left"
PICK_CUBE_HOME = np.asarray([-0.08, -0.25, 0.078], dtype=np.float64)
PLACE_TARGET_CENTER_XY = np.asarray([0.08, -0.25], dtype=np.float64)
PLACE_TARGET_HALF_EXTENT_XY = np.asarray([0.115, 0.115], dtype=np.float64)
PLACE_TARGET_CUBE_Z_RANGE = (0.065, 0.10)
SFT_SUCCESS_RESET_HOLD_S = 1.0
ROBOT_HOME_QPOS = np.asarray([0.0, -1.57079, 1.35, 1.57079, -1.57079, 0.0], dtype=np.float64)


@dataclass(frozen=True)
class SimProfile:
    action_interval_s: float
    phase_rate: float


@dataclass
class OpenPIControlState:
    enabled: bool = False
    prompt: str = "pick up the cube and place it"
    seq: int = 0
    worker_pid: int | None = None
    last_request_wall_ms: float | None = None
    last_policy_infer_ms: float | None = None
    last_action_shape: list[int] | None = None
    last_action: dict[str, float | None] | None = None
    last_error: str | None = None
    updated_at: float | None = None


SIM_PROFILES = {
    "standard": SimProfile(action_interval_s=0.0868, phase_rate=0.72),
    "flashact": SimProfile(action_interval_s=0.0196, phase_rate=0.72),
    "sft": SimProfile(action_interval_s=0.0868, phase_rate=0.72),
}


OPENPI_CAMERAS = {
    "observation/exterior_image_1_left": "exterior_left",
    "observation/wrist_image_left": "wrist_left",
}

SO_ARM100_TO_DROID_ACTION = {
    "Rotation": "actions[:, 0]",
    "Pitch": "actions[:, 1]",
    "Elbow": "actions[:, 2]",
    "Wrist_Pitch": "actions[:, 3]",
    "Wrist_Roll": "actions[:, 4]",
    "Jaw": "actions[:, 5]",
}

OPENPI_ACTION_INDEX_BY_ACTUATOR = {
    "Rotation": 0,
    "Pitch": 1,
    "Elbow": 2,
    "Wrist_Pitch": 3,
    "Wrist_Roll": 4,
    "Jaw": 5,
}
OPENPI_ACTION_DIM = len(OPENPI_ACTION_INDEX_BY_ACTUATOR)
OPENPI_POLICY_INTERVAL_S = 0.25
SFT_POLICY_CONTROL = os.environ.get("FLASHACT_SFT_POLICY_CONTROL", "0") == "1"
_POLICY_CONTROL = OpenPIControlState()


def map_openpi_action(action: list[float] | np.ndarray) -> dict[str, float | None]:
    values = list(action)
    mapped: dict[str, float | None] = {}
    for actuator_name, action_index in OPENPI_ACTION_INDEX_BY_ACTUATOR.items():
        mapped[actuator_name] = float(values[action_index]) if len(values) > action_index else None
    return mapped


def set_openpi_control(enabled: bool, prompt: str | None = None) -> dict[str, Any]:
    _POLICY_CONTROL.enabled = enabled
    _POLICY_CONTROL.seq += 1
    _POLICY_CONTROL.last_error = None
    if prompt is not None:
        normalized_prompt = prompt.strip()
        if normalized_prompt:
            _POLICY_CONTROL.prompt = normalized_prompt
    _POLICY_CONTROL.updated_at = time.perf_counter()
    return openpi_control_status()


def record_openpi_result(result: dict[str, Any], request_wall_ms: float) -> dict[str, Any]:
    actions = result["actions"]["values"]
    _POLICY_CONTROL.worker_pid = int(result["worker"]["pid"])
    _POLICY_CONTROL.last_request_wall_ms = float(request_wall_ms)
    _POLICY_CONTROL.last_policy_infer_ms = float(result.get("policy_timing", {}).get("infer_ms", 0.0))
    _POLICY_CONTROL.last_action_shape = list(result["actions"]["shape"])
    _POLICY_CONTROL.last_action = map_openpi_action(actions[0] if actions else [])
    _POLICY_CONTROL.last_error = None
    _POLICY_CONTROL.updated_at = time.perf_counter()
    return openpi_control_status()


def record_openpi_error(error: str) -> dict[str, Any]:
    _POLICY_CONTROL.last_error = error
    _POLICY_CONTROL.updated_at = time.perf_counter()
    return openpi_control_status()


def openpi_control_status() -> dict[str, Any]:
    updated_ago_ms = None
    if _POLICY_CONTROL.updated_at is not None:
        updated_ago_ms = (time.perf_counter() - _POLICY_CONTROL.updated_at) * 1000.0
    return {
        "enabled": _POLICY_CONTROL.enabled,
        "prompt": _POLICY_CONTROL.prompt,
        "worker_pid": _POLICY_CONTROL.worker_pid,
        "last_request_wall_ms": _POLICY_CONTROL.last_request_wall_ms,
        "last_policy_infer_ms": _POLICY_CONTROL.last_policy_infer_ms,
        "last_action_shape": _POLICY_CONTROL.last_action_shape,
        "last_action": _POLICY_CONTROL.last_action,
        "last_error": _POLICY_CONTROL.last_error,
        "updated_ago_ms": updated_ago_ms,
    }


def model_dir() -> Path:
    configured = os.environ.get("FLASHACT_SO_ARM100_DIR")
    if configured:
        path = Path(configured)
        if (path / "scene.xml").exists():
            return path

    if (LEGACY_CACHE / "scene.xml").exists():
        return LEGACY_CACHE

    target = MODEL_CACHE / "trs_so_arm100"
    if (target / "scene.xml").exists():
        return target

    if not (target / "scene.xml").exists():
        raise FileNotFoundError(
            f"SO-ARM100 MuJoCo model not found at {target}. "
            "Run `git submodule update --init --recursive` from the PostVLA root."
        )
    return target


def _write_if_changed(path: Path, content: str) -> None:
    if path.exists() and path.read_text() == content:
        return
    path.write_text(content)


def _robot_xml_with_wrist_camera(source_dir: Path) -> str:
    robot_path = source_dir / "so_arm100.xml"
    if not robot_path.exists():
        raise FileNotFoundError(f"SO-ARM100 robot XML not found at {robot_path}")

    robot_xml = robot_path.read_text()
    robot_xml = robot_xml.replace("meshdir=\"assets/\"", f"meshdir=\"{source_dir / 'assets'}\"")
    robot_xml = robot_xml.replace(
        '<material name="orange" rgba="1.0 0.331 0.0 1.0" specular="0.1" shininess="0.1"/>',
        '<material name="body_beige" rgba="0.74 0.63 0.48 1.0" specular="0.15" shininess="0.18"/>\n'
        '    <material name="gripper_white" rgba="0.96 0.95 0.91 1.0" specular="0.2" shininess="0.25"/>',
    )
    robot_xml = robot_xml.replace('material="orange"', 'material="body_beige"')
    robot_xml = robot_xml.replace(
        '<geom type="mesh" mesh="Fixed_Jaw" class="visual"/>',
        '<geom type="mesh" mesh="Fixed_Jaw" class="visual" material="gripper_white"/>',
    )
    robot_xml = robot_xml.replace(
        '<geom type="mesh" mesh="Fixed_Jaw_Motor" class="motor_visual"/>',
        '<geom type="mesh" mesh="Fixed_Jaw_Motor" class="motor_visual" material="black"/>',
    )
    robot_xml = robot_xml.replace(
        '<geom type="mesh" mesh="Moving_Jaw" class="visual"/>',
        '<geom type="mesh" mesh="Moving_Jaw" class="visual" material="gripper_white"/>',
    )
    fixed_jaw_body = '<body name="Fixed_Jaw" pos="0 -0.0601 0" euler="0 1.57079 0">'
    wrist_camera = (
        '<camera name="wrist_left" mode="targetbody" target="pick_cube" '
        'pos="0 -0.06 0.12" fovy="75"/>'
    )
    if wrist_camera in robot_xml:
        return robot_xml
    if fixed_jaw_body not in robot_xml:
        raise ValueError("Could not find Fixed_Jaw body to mount wrist camera")
    return robot_xml.replace(fixed_jaw_body, f"{fixed_jaw_body}\n                {wrist_camera}", 1)


def _scene_xml(robot_path: Path) -> str:
    cube_x, cube_y, cube_z = PICK_CUBE_HOME
    return f"""<mujoco model="so_arm100 flashact scene">
  <include file="{robot_path}"/>

  <statistic center="0 -0.2 0.12" extent="0.55"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="140" elevation="-30"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.015 0.013 0.011" rgb2="0.025 0.022 0.019"
      markrgb="0.035 0.032 0.028" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    <material name="table_mat" rgba="0.02 0.02 0.018 1"/>
    <material name="cube_mat" rgba="0.0 0.2 1.0 1"/>
    <material name="target_mat" rgba="1.0 0.45 0.68 1.0"/>
    <material name="target_fold_mat" rgba="0.75 0.25 0.45 0.35"/>
  </asset>

  <worldbody>
    <light pos="0 -0.25 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
    <body name="work_table" pos="0 -0.24 0.025">
      <geom name="table_top" type="box" size="0.55 0.40 0.025" material="table_mat"/>
    </body>
    <body name="pick_cube" pos="{cube_x:.6f} {cube_y:.6f} {cube_z:.6f}">
      <joint name="pick_cube_free" type="free"/>
      <geom name="pick_cube_geom" type="box" size="0.024 0.024 0.024" material="cube_mat" mass="0.04"
        friction="1 0.005 0.0001"/>
    </body>
    <body name="place_target" pos="0.08 -0.25 0.054">
      <geom name="target_pad" type="box" size="0.115 0.115 0.002" material="target_mat"
        friction="1 0.005 0.0001" solref="0.002 1" solimp="0.95 0.99 0.001"/>
      <geom name="target_fold_horizontal" type="box" size="0.11 0.002 0.0004" pos="0 0 0.0026"
        material="target_fold_mat" contype="0" conaffinity="0"/>
      <geom name="target_fold_vertical" type="box" size="0.002 0.105 0.0004" pos="-0.025 0 0.0028"
        material="target_fold_mat" contype="0" conaffinity="0"/>
    </body>
    <camera name="exterior_left" mode="fixed" pos="-0.369 -0.246 0.724"
      xyaxes="-0.009901 -0.999951 0 0.847057 -0.008387 0.531436" fovy="54"/>
  </worldbody>

  <contact>
    <exclude body1="work_table" body2="Rotation_Pitch"/>
  </contact>
</mujoco>
"""


def scene_path() -> Path:
    source_dir = model_dir()
    GENERATED_SCENE_CACHE.mkdir(parents=True, exist_ok=True)
    robot_path = GENERATED_SCENE_CACHE / "so_arm100_flashact.xml"
    generated_scene_path = GENERATED_SCENE_CACHE / "scene.xml"
    _write_if_changed(robot_path, _robot_xml_with_wrist_camera(source_dir))
    _write_if_changed(generated_scene_path, _scene_xml(robot_path))
    return generated_scene_path


class MujocoArmStream:
    def __init__(self, engine_key: str, width: int = 640, height: int = 480) -> None:
        self.engine_key = engine_key
        self.profile = SIM_PROFILES[engine_key]
        self.model_path = scene_path()
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.started = time.perf_counter()
        self.next_action_time = 0.0
        self.current_ctrl = [0.0] * self.model.nu
        self._reset()
        self.home_ctrl = np.asarray(self.data.ctrl[: self.model.nu], dtype=np.float64).copy()
        self.policy_actions = np.empty((0, OPENPI_ACTION_DIM), dtype=np.float64)
        self.policy_action_index = 0
        self.last_policy_request_time = 0.0
        self.control_seq_seen = _POLICY_CONTROL.seq
        self.place_detected_at: float | None = None

    def _reset(self) -> None:
        key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key)
            self.current_ctrl = list(self.data.ctrl)
        else:
            mujoco.mj_resetData(self.model, self.data)
        self._reset_arm_pose()
        self._reset_scene_objects()
        mujoco.mj_forward(self.model, self.data)

    def _reset_arm_pose(self) -> None:
        arm_dofs = min(self.model.nu, self.data.qpos.shape[0], ROBOT_HOME_QPOS.shape[0])
        self.data.qpos[:arm_dofs] = ROBOT_HOME_QPOS[:arm_dofs]
        self.data.qvel[:arm_dofs] = 0.0
        self.data.ctrl[:arm_dofs] = ROBOT_HOME_QPOS[:arm_dofs]
        self.current_ctrl = ROBOT_HOME_QPOS[:arm_dofs].tolist()

    def _reset_scene_objects(self) -> None:
        cube_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pick_cube_free")
        if cube_joint < 0:
            return

        qpos_adr = int(self.model.jnt_qposadr[cube_joint])
        qvel_adr = int(self.model.jnt_dofadr[cube_joint])
        self.data.qpos[qpos_adr : qpos_adr + 3] = PICK_CUBE_HOME
        self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[qvel_adr : qvel_adr + 6] = 0.0

    def _cube_position(self) -> np.ndarray | None:
        cube_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pick_cube")
        if cube_body < 0:
            return None
        return np.asarray(self.data.xpos[cube_body], dtype=np.float64)

    def _cube_is_on_pad(self) -> bool:
        cube_position = self._cube_position()
        if cube_position is None:
            return False
        xy_on_pad = np.all(np.abs(cube_position[:2] - PLACE_TARGET_CENTER_XY) <= PLACE_TARGET_HALF_EXTENT_XY)
        z_min, z_max = PLACE_TARGET_CUBE_Z_RANGE
        return bool(xy_on_pad and z_min <= cube_position[2] <= z_max)

    def _reset_rollout_after_success(self, wall_now: float) -> None:
        self.started = wall_now
        self.next_action_time = 0.0
        self.policy_actions = np.empty((0, OPENPI_ACTION_DIM), dtype=np.float64)
        self.policy_action_index = 0
        self.last_policy_request_time = 0.0
        self.place_detected_at = None
        self._reset()

    def _maybe_reset_after_success(self) -> None:
        if not (self.engine_key == "sft" and SFT_POLICY_CONTROL):
            return
        if not self._cube_is_on_pad():
            self.place_detected_at = None
            return

        wall_now = time.perf_counter()
        if self.place_detected_at is None:
            self.place_detected_at = wall_now
            return
        if wall_now - self.place_detected_at >= SFT_SUCCESS_RESET_HOLD_S:
            self._reset_rollout_after_success(wall_now)

    def _sync_policy_control_epoch(self) -> None:
        if self.control_seq_seen == _POLICY_CONTROL.seq:
            return
        self.control_seq_seen = _POLICY_CONTROL.seq
        self.policy_actions = np.empty((0, OPENPI_ACTION_DIM), dtype=np.float64)
        self.policy_action_index = 0

    def _openpi_action_to_ctrl(self, action: np.ndarray) -> list[float]:
        target = self.home_ctrl.copy()
        raw_values = []
        for actuator_name in OPENPI_ACTION_INDEX_BY_ACTUATOR:
            action_index = OPENPI_ACTION_INDEX_BY_ACTUATOR[actuator_name]
            raw_values.append(float(action[action_index]) if action.shape[0] > action_index else 0.0)

        mapped = np.asarray(raw_values[: self.model.nu], dtype=np.float64)
        target[: mapped.shape[0]] = mapped

        limited = np.asarray(self.model.actuator_ctrllimited[: target.shape[0]], dtype=bool)
        if np.any(limited):
            ctrlrange = self.model.actuator_ctrlrange[: target.shape[0]]
            target[limited] = np.clip(target[limited], ctrlrange[limited, 0], ctrlrange[limited, 1])
        return target.tolist()

    def _uses_policy_control(self) -> bool:
        return (self.engine_key == "sft" and SFT_POLICY_CONTROL) or (
            self.engine_key == "standard" and _POLICY_CONTROL.enabled
        )

    def _next_policy_ctrl(self) -> list[float] | None:
        self._sync_policy_control_epoch()
        if not self._uses_policy_control():
            return None
        if self.policy_actions.size == 0:
            return self.home_ctrl.tolist()

        index = min(self.policy_action_index, self.policy_actions.shape[0] - 1)
        self.policy_action_index += 1
        return self._openpi_action_to_ctrl(self.policy_actions[index])

    def _target_ctrl(self, now: float) -> list[float]:
        policy_ctrl = self._next_policy_ctrl()
        if policy_ctrl is not None:
            return policy_ctrl

        phase = now * self.profile.phase_rate
        pick = 0.5 + 0.5 * math.sin(phase * 1.15)
        lift = 0.5 + 0.5 * math.sin(phase * 0.75 + 0.8)
        return [
            0.55 * math.sin(phase * 0.8),
            -1.45 + 0.22 * math.sin(phase * 0.55),
            1.22 + 0.48 * lift,
            1.25 + 0.35 * math.sin(phase * 0.9 + 1.1),
            -1.57 + 0.7 * math.sin(phase * 1.2),
            0.18 + 0.72 * pick,
        ][: self.model.nu]

    def openpi_observation(self, prompt: str) -> dict[str, Any]:
        return _build_openpi_observation_from_state(self.model, self.data, prompt)

    async def maybe_update_openpi_control(self) -> None:
        self._sync_policy_control_epoch()
        if not self._uses_policy_control():
            return

        now = time.perf_counter()
        if now - self.last_policy_request_time < OPENPI_POLICY_INTERVAL_S:
            return
        if self.policy_actions.size and self.policy_action_index < self.policy_actions.shape[0]:
            return

        self.last_policy_request_time = now
        try:
            from app.openpi_worker import infer_observation

            observation = self.openpi_observation(_POLICY_CONTROL.prompt)
            started = time.perf_counter()
            result = await infer_observation(observation)
            wall_ms = (time.perf_counter() - started) * 1000.0
            actions = np.asarray(result["actions"]["values"], dtype=np.float64)
            if actions.ndim == 1:
                actions = actions.reshape(1, -1)
            self.policy_actions = actions
            self.policy_action_index = 0
            record_openpi_result(result, wall_ms)
        except Exception as exc:
            record_openpi_error(repr(exc))

    def _advance(self) -> None:
        now = time.perf_counter() - self.started
        while self.data.time < now:
            if self.data.time >= self.next_action_time:
                self.current_ctrl = self._target_ctrl(self.data.time)
                self.next_action_time += self.profile.action_interval_s
            self.data.ctrl[: len(self.current_ctrl)] = self.current_ctrl
            mujoco.mj_step(self.model, self.data)
        self._maybe_reset_after_success()

    def jpeg(self) -> bytes:
        self._advance()
        self.renderer.update_scene(self.data, camera=DASHBOARD_CAMERA)
        rgb = self.renderer.render()
        output = io.BytesIO()
        Image.fromarray(rgb).save(output, format="JPEG", quality=82, optimize=True)
        return output.getvalue()


def _render_rgb(model: mujoco.MjModel, data: mujoco.MjData, camera_name: str, width: int = 224, height: int = 224) -> np.ndarray:
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        renderer.update_scene(data, camera=camera_name)
        return renderer.render()
    finally:
        renderer.close()


def _build_openpi_observation_from_state(model: mujoco.MjModel, data: mujoco.MjData, prompt: str) -> dict[str, Any]:
    joint_position, gripper_position = _so_arm100_state(data)
    return {
        "observation/exterior_image_1_left": _render_rgb(
            model,
            data,
            OPENPI_CAMERAS["observation/exterior_image_1_left"],
        ),
        "observation/wrist_image_left": _render_rgb(
            model,
            data,
            OPENPI_CAMERAS["observation/wrist_image_left"],
        ),
        "observation/joint_position": joint_position,
        "observation/gripper_position": gripper_position,
        "prompt": prompt,
    }


def _so_arm100_state(data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    qpos = np.asarray(data.qpos, dtype=np.float32)
    joint_position = np.zeros(5, dtype=np.float32)
    joint_position[:5] = qpos[:5]
    gripper_position = np.asarray([qpos[5] if qpos.shape[0] > 5 else 0.0], dtype=np.float32)
    return joint_position, gripper_position


def build_openpi_observation(engine_key: str, prompt: str) -> dict[str, Any]:
    stream = MujocoArmStream(engine_key)
    try:
        stream._advance()
        return stream.openpi_observation(prompt)
    finally:
        stream.renderer.close()


def _array_summary(value: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "min": float(value.min()),
        "max": float(value.max()),
        "mean": float(value.mean()),
    }


def openpi_observation_summary(engine_key: str, prompt: str) -> dict[str, Any]:
    observation = build_openpi_observation(engine_key, prompt)
    joint_position = observation["observation/joint_position"]
    gripper_position = observation["observation/gripper_position"]
    return {
        "engine_key": engine_key,
        "policy_config": "pi05_so100_sim",
        "prompt": observation["prompt"],
        "observation": {
            "observation/exterior_image_1_left": _array_summary(observation["observation/exterior_image_1_left"]),
            "observation/wrist_image_left": _array_summary(observation["observation/wrist_image_left"]),
            "observation/joint_position": {
                **_array_summary(joint_position),
                "values": [float(value) for value in joint_position],
            },
            "observation/gripper_position": {
                **_array_summary(gripper_position),
                "values": [float(value) for value in gripper_position],
            },
        },
        "action_mapping": SO_ARM100_TO_DROID_ACTION,
        "ignored_action_dims": [],
    }


async def mjpeg_stream(engine_key: str) -> AsyncIterator[bytes]:
    stream = MujocoArmStream(engine_key)
    try:
        while True:
            # OSMesa rendering is process-global and can crash when called from worker threads.
            frame = stream.jpeg()
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            await stream.maybe_update_openpi_control()
            await asyncio.sleep(1 / 24)
    finally:
        stream.renderer.close()
