from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np


POSTVLA_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SIM_ROOT = POSTVLA_ROOT / "sim"
DEFAULT_OUTPUT_ROOT = POSTVLA_ROOT / "artifacts" / "oracle_pick_place"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scripted SO100 MuJoCo pick-and-place oracle for the FlashAct demo scene."
    )
    parser.add_argument("--demo-root", default=str(DEFAULT_SIM_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--video-name", default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--cube-x", type=float, default=-0.08)
    parser.add_argument("--cube-y", type=float, default=-0.25)
    parser.add_argument("--target-x", type=float, default=0.08)
    parser.add_argument("--target-y", type=float, default=-0.25)
    parser.add_argument(
        "--control-mode",
        choices=["kinematic", "actuator"],
        default="kinematic",
        help="kinematic directly interpolates solved joint states; actuator sends position controls to the MuJoCo actuators.",
    )
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def configure_imports(demo_root: str) -> None:
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    root = Path(demo_root).resolve()
    if not (root / "app" / "mujoco_stream.py").exists():
        raise FileNotFoundError(f"PostVLA simulator package not found at {root}")
    sys.path.insert(0, str(root))


class SO100Oracle:
    def __init__(self, stream: Any, cube_xy: tuple[float, float], target_xy: tuple[float, float]) -> None:
        self.stream = stream
        self.model = stream.model
        self.data = stream.data
        self.mujoco = sys.modules["app.mujoco_stream"].mujoco
        self.cube_xy = cube_xy
        self.target_xy = target_xy
        self.cube_rest_z = 0.081
        self.closed_jaw = -0.2
        self.open_jaw = 0.85
        self.geom_ids = [
            self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, "fixed_jaw_pad_2"),
            self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, "moving_jaw_pad_2"),
        ]
        if any(geom_id < 0 for geom_id in self.geom_ids):
            raise RuntimeError("Could not find jaw pad geoms for pinch-point IK")
        self.cube_body_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, "pick_cube")
        self.target_body_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, "place_target")
        self.cube_joint_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, "pick_cube_free")
        if self.cube_body_id < 0 or self.target_body_id < 0 or self.cube_joint_id < 0:
            raise RuntimeError("Could not find cube/target bodies or cube free joint")
        self.cube_qpos_adr = int(self.model.jnt_qposadr[self.cube_joint_id])
        self.cube_qvel_adr = int(self.model.jnt_dofadr[self.cube_joint_id])
        self.joint_limits = np.asarray(self.model.jnt_range[:6], dtype=np.float64)

    def reset(self) -> None:
        key = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_KEY, "home")
        if key >= 0:
            self.mujoco.mj_resetDataKeyframe(self.model, self.data, key)
        else:
            self.mujoco.mj_resetData(self.model, self.data)
        self.set_cube_position(np.asarray([self.cube_xy[0], self.cube_xy[1], self.cube_rest_z], dtype=np.float64))
        self.mujoco.mj_forward(self.model, self.data)

    def set_cube_position(self, xyz: np.ndarray) -> None:
        self.data.qpos[self.cube_qpos_adr : self.cube_qpos_adr + 3] = xyz
        self.data.qpos[self.cube_qpos_adr + 3 : self.cube_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[self.cube_qvel_adr : self.cube_qvel_adr + 6] = 0.0

    def cube_position(self) -> np.ndarray:
        return np.asarray(self.data.xpos[self.cube_body_id], dtype=np.float64).copy()

    def target_position(self) -> np.ndarray:
        return np.asarray(self.data.xpos[self.target_body_id], dtype=np.float64).copy()

    def pinch_position(self) -> np.ndarray:
        return np.mean([self.data.geom_xpos[geom_id] for geom_id in self.geom_ids], axis=0)

    def pinch_jacobian(self) -> np.ndarray:
        jac = np.zeros((3, self.model.nv), dtype=np.float64)
        for geom_id in self.geom_ids:
            jac_pos = np.zeros((3, self.model.nv), dtype=np.float64)
            jac_rot = np.zeros((3, self.model.nv), dtype=np.float64)
            self.mujoco.mj_jacGeom(self.model, self.data, jac_pos, jac_rot, geom_id)
            jac += jac_pos
        return jac / len(self.geom_ids)

    def solve_pinch_ik(
        self,
        target_xyz: np.ndarray,
        start_q: np.ndarray,
        jaw: float,
        max_iters: int = 240,
        tolerance: float = 0.0015,
    ) -> tuple[np.ndarray, float]:
        self.data.qpos[:6] = start_q
        self.data.qpos[5] = jaw
        self.mujoco.mj_forward(self.model, self.data)

        target_xyz = np.asarray(target_xyz, dtype=np.float64)
        for _ in range(max_iters):
            error = target_xyz - self.pinch_position()
            if np.linalg.norm(error) < tolerance:
                break
            jac = self.pinch_jacobian()[:, :5]
            damping = 0.01
            step = jac.T @ np.linalg.solve(jac @ jac.T + damping * np.eye(3), error)
            self.data.qpos[:5] += np.clip(step, -0.06, 0.06)
            for joint in range(5):
                self.data.qpos[joint] = np.clip(
                    self.data.qpos[joint],
                    self.joint_limits[joint, 0],
                    self.joint_limits[joint, 1],
                )
            self.data.qpos[5] = jaw
            self.mujoco.mj_forward(self.model, self.data)

        final_error = float(np.linalg.norm(target_xyz - self.pinch_position()))
        return self.data.qpos[:6].copy(), final_error

    def plan(self) -> list[dict[str, Any]]:
        home_q = self.data.qpos[:6].copy()
        cube_x, cube_y = self.cube_xy
        target_x, target_y = self.target_xy
        q = home_q
        specs = [
            ("pre_grasp", np.asarray([cube_x, cube_y, 0.150]), self.open_jaw, 1.1, False, False),
            ("grasp", np.asarray([cube_x, cube_y, 0.089]), self.open_jaw, 1.0, False, False),
            ("close_and_attach", np.asarray([cube_x, cube_y, 0.089]), self.closed_jaw, 0.8, True, False),
            ("lift", np.asarray([cube_x, cube_y, 0.170]), self.closed_jaw, 1.0, True, False),
            ("transfer", np.asarray([target_x, target_y, 0.170]), self.closed_jaw, 1.5, True, False),
            ("place", np.asarray([target_x, target_y, 0.092]), self.closed_jaw, 1.0, True, False),
            ("release", np.asarray([target_x, target_y, 0.092]), self.open_jaw, 0.8, False, True),
            ("retreat", np.asarray([target_x, target_y, 0.170]), self.open_jaw, 1.0, False, False),
        ]
        phases: list[dict[str, Any]] = []
        for name, target, jaw, duration_s, attached, release in specs:
            q, error = self.solve_pinch_ik(target, q, jaw)
            phases.append(
                {
                    "name": name,
                    "target": target.tolist(),
                    "q": q.tolist(),
                    "duration_s": duration_s,
                    "attached": attached,
                    "release": release,
                    "ik_error_m": error,
                }
            )
        return phases

    def run(
        self,
        phases: list[dict[str, Any]],
        fps: int,
        control_mode: str,
        write_frame: Any | None = None,
    ) -> dict[str, Any]:
        self.reset()
        max_cube_z = float(self.cube_position()[2])
        attached = False
        attach_offset = np.zeros(3, dtype=np.float64)
        phase_metrics: list[dict[str, Any]] = []
        current_arm_q = np.asarray(self.data.qpos[:6], dtype=np.float64).copy()

        def step_once(arm_q: np.ndarray | None = None) -> None:
            nonlocal current_arm_q
            nonlocal max_cube_z
            if arm_q is not None:
                current_arm_q = np.asarray(arm_q, dtype=np.float64).copy()

            if control_mode == "kinematic":
                self.data.qpos[:6] = current_arm_q
                self.data.qvel[:6] = 0.0
                self.data.ctrl[:6] = current_arm_q
                self.mujoco.mj_forward(self.model, self.data)
            else:
                self.data.ctrl[:6] = current_arm_q
                self.mujoco.mj_step(self.model, self.data)

            if attached:
                self.set_cube_position(self.pinch_position() + attach_offset)
                self.mujoco.mj_forward(self.model, self.data)
            elif control_mode == "kinematic":
                self.mujoco.mj_step(self.model, self.data)
                self.data.qpos[:6] = current_arm_q
                self.data.qvel[:6] = 0.0
                self.data.ctrl[:6] = current_arm_q
                self.mujoco.mj_forward(self.model, self.data)
            max_cube_z = max(max_cube_z, float(self.cube_position()[2]))

        def hold(seconds: float) -> None:
            frame_count = max(1, int(seconds * fps))
            steps_per_frame = max(1, int((1.0 / fps) / self.model.opt.timestep))
            for _ in range(frame_count):
                for _ in range(steps_per_frame):
                    step_once()
                if write_frame is not None:
                    write_frame()

        hold(0.4)
        for phase in phases:
            start_ctrl = np.asarray(self.data.ctrl[:6], dtype=np.float64).copy()
            target_ctrl = np.asarray(phase["q"], dtype=np.float64)
            frame_count = max(1, int(float(phase["duration_s"]) * fps))
            steps_per_frame = max(1, int((1.0 / fps) / self.model.opt.timestep))

            if phase["name"] == "close_and_attach":
                attach_offset = self.cube_position() - self.pinch_position()
                attached = True

            for frame_index in range(frame_count):
                alpha = (frame_index + 1) / frame_count
                alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                ctrl = (1.0 - alpha) * start_ctrl + alpha * target_ctrl
                for _ in range(steps_per_frame):
                    step_once(ctrl)
                if write_frame is not None:
                    write_frame()

            if phase["release"]:
                release_xyz = np.asarray([self.target_xy[0], self.target_xy[1], self.cube_rest_z], dtype=np.float64)
                self.set_cube_position(release_xyz)
                attached = False
                self.mujoco.mj_forward(self.model, self.data)

            hold(0.2)
            phase_metrics.append(
                {
                    "phase": phase["name"],
                    "cube_position": self.cube_position().tolist(),
                    "pinch_position": self.pinch_position().tolist(),
                    "attached": attached,
                }
            )

        hold(1.0)
        final_cube = self.cube_position()
        target = np.asarray([self.target_xy[0], self.target_xy[1], self.cube_rest_z], dtype=np.float64)
        xy_distance = float(np.linalg.norm(final_cube[:2] - target[:2]))
        return {
            "final_cube": final_cube.tolist(),
            "target": target.tolist(),
            "max_cube_z": max_cube_z,
            "final_cube_target_xy_distance": xy_distance,
            "success": bool(xy_distance < 0.03 and max_cube_z > 0.13 and final_cube[2] > 0.07),
            "phase_metrics": phase_metrics,
        }


def main() -> None:
    args = parse_args()
    configure_imports(args.demo_root)

    from app.mujoco_stream import DASHBOARD_CAMERA
    from app.mujoco_stream import MujocoArmStream

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    video_path = output_dir / (args.video_name or f"so100_oracle_pick_place_{stamp}.mp4")
    summary_path = video_path.with_suffix(".json")

    stream = MujocoArmStream("standard", width=args.width, height=args.height)
    oracle = SO100Oracle(stream, cube_xy=(args.cube_x, args.cube_y), target_xy=(args.target_x, args.target_y))
    oracle.reset()
    phases = oracle.plan()

    proc: subprocess.Popen[bytes] | None = None

    def write_frame() -> None:
        if proc is None or proc.stdin is None:
            return
        stream.renderer.update_scene(stream.data, camera=DASHBOARD_CAMERA)
        rgb = np.ascontiguousarray(stream.renderer.render())
        proc.stdin.write(rgb.tobytes())

    try:
        if not args.no_video:
            cmd = [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{args.width}x{args.height}",
                "-r",
                str(args.fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "24",
                "-pix_fmt",
                "yuv420p",
                str(video_path),
            ]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        result = oracle.run(
            phases,
            fps=args.fps,
            control_mode=args.control_mode,
            write_frame=None if args.no_video else write_frame,
        )
    finally:
        if proc is not None:
            if proc.stdin is not None:
                proc.stdin.close()
            stderr = proc.stderr.read().decode("utf-8", "replace") if proc.stderr is not None else ""
            returncode = proc.wait(timeout=30)
            if returncode != 0:
                raise RuntimeError(f"ffmpeg failed with code {returncode}: {stderr[-1000:]}")
        stream.renderer.close()

    probe = None
    if not args.no_video:
        probe = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height,nb_frames,duration",
                    "-of",
                    "json",
                    str(video_path),
                ],
                text=True,
            )
        )

    summary = {
        "mode": f"oracle_attach_{args.control_mode}",
        "note": (
            "The cube is explicitly attached to the IK-controlled gripper during the grasp/carry phases. "
            "The default kinematic mode directly interpolates solved joint states for a clean scripted baseline."
        ),
        "control_mode": args.control_mode,
        "video": None if args.no_video else str(video_path),
        "summary": str(summary_path),
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "cube_xy": [args.cube_x, args.cube_y],
        "target_xy": [args.target_x, args.target_y],
        "phases": phases,
        "result": result,
        "ffprobe": probe,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
