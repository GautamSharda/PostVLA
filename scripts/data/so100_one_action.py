from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np


# Edit these actions manually. Each row is one target pose:
# [Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw]
ACTIONS = [
    [0.0, -1.47079, 1.57079, -0.1, 0.0, 0.9], # go to neutral
    [-0.5, -1.47079, 1.57079, -0.1, 0.0, 0.9], # go above cube
    [-0.5, -1.47079, 1.57079, 1.265, 2, 1.5], # lower + get ready to grasp
    [-0.5, -1.47079, 1.57079, 1.265, 2, 0.25], # grasp / close gripper
    [-0.5, -1.47079, 1.57079, 1.265, 2, 0.25], # wait a bit
    [-0.5, -1.47079, 0.5, 1.265, 2, 0.25], # lift
    [0.35, -1.475, 0.5, 1.265, 2, 0.25], # go above the pad
    [0.35, -1.475, 1.57079, 1.3275, 2.75, 0.25], # lower onto the pad
    [0.35, -1.475, 1.57079, 1.3275, 2.75, 1.5], # release / open gripper
    [0.35, -1.475, 0.5, 1.265, 2.75, 1.5], # retreat upward
    [0.0, -1.47079, 1.57079, -0.1, 0.0, 0.9], # back to neutral
]

POSTVLA_ROOT = Path(__file__).resolve().parents[2]
DEMO_ROOT = POSTVLA_ROOT / "sim"
OUT = POSTVLA_ROOT / "artifacts" / "manual_actions" / "actions.mp4"
WIDTH, HEIGHT, FPS = 640, 480, 30
MOVE_SECONDS, HOLD_SECONDS = 2.0, 0.5

# "kinematic" sets qpos directly, so your typed values happen exactly.
# "actuator" uses the MuJoCo position actuators.
CONTROL_MODE = "actuator"


def main() -> None:
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    sys.path.insert(0, str(DEMO_ROOT))

    from app.mujoco_stream import DASHBOARD_CAMERA, MujocoArmStream, mujoco

    OUT.parent.mkdir(parents=True, exist_ok=True)
    stream = MujocoArmStream("standard", width=WIDTH, height=HEIGHT)

    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-pix_fmt", "yuv420p", str(OUT),
    ]
    ffmpeg = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write_frame() -> None:
        stream.renderer.update_scene(stream.data, camera=DASHBOARD_CAMERA)
        frame = np.ascontiguousarray(stream.renderer.render())
        ffmpeg.stdin.write(frame.tobytes())

    def step_with_ctrl(ctrl: np.ndarray) -> None:
        stream.data.ctrl[:6] = ctrl
        if CONTROL_MODE == "kinematic":
            stream.data.qpos[:6] = ctrl
            stream.data.qvel[:6] = 0.0
            mujoco.mj_forward(stream.model, stream.data)
        elif CONTROL_MODE == "actuator":
            mujoco.mj_step(stream.model, stream.data)
        else:
            raise ValueError(f"Unknown CONTROL_MODE: {CONTROL_MODE}")

    steps_per_frame = max(1, int((1 / FPS) / stream.model.opt.timestep))

    try:
        for action in ACTIONS:
            action = np.asarray(action, dtype=float)
            if action.shape != (6,):
                raise ValueError(f"Each action must have 6 values, got {action}")

            start = np.asarray(stream.data.ctrl[:6], dtype=float).copy()
            for frame_idx in range(int(MOVE_SECONDS * FPS)):
                alpha = (frame_idx + 1) / int(MOVE_SECONDS * FPS)
                alpha = alpha * alpha * (3 - 2 * alpha)
                ctrl = (1 - alpha) * start + alpha * action
                for _ in range(steps_per_frame):
                    step_with_ctrl(ctrl)
                write_frame()

            for _ in range(int(HOLD_SECONDS * FPS)):
                for _ in range(steps_per_frame):
                    step_with_ctrl(action)
                write_frame()
    finally:
        stream.renderer.close()
        ffmpeg.stdin.close()
        ffmpeg.wait()

    cube_id = mujoco.mj_name2id(stream.model, mujoco.mjtObj.mjOBJ_BODY, "pick_cube")
    target_id = mujoco.mj_name2id(stream.model, mujoco.mjtObj.mjOBJ_BODY, "place_target")
    cube_pos = np.asarray(stream.data.xpos[cube_id], dtype=float)
    target_pos = np.asarray(stream.data.xpos[target_id], dtype=float)

    print(f"saved {OUT}")
    print("final qpos", np.round(stream.data.qpos[:6], 4).tolist())
    print("cube pos", np.round(cube_pos, 4).tolist())
    print("target pos", np.round(target_pos, 4).tolist())
    print("cube-target xy dist", round(float(np.linalg.norm(cube_pos[:2] - target_pos[:2])), 4))


if __name__ == "__main__":
    main()
