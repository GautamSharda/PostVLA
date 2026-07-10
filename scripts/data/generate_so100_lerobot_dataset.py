from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import mujoco
import numpy as np


POSTVLA_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = Path(os.environ.get("POSTVLA_ARTIFACT_ROOT", POSTVLA_ROOT / "artifacts"))

DEFAULT_CANDIDATE_MANIFEST = (
    str(POSTVLA_ROOT / "configs" / "manifests" / "cubevar_contact_40.json")
)
DEFAULT_DEMO_ROOT = str(POSTVLA_ROOT / "sim")
DEFAULT_ORACLE_PATH = str(POSTVLA_ROOT / "scripts" / "data" / "so100_oracle_pick_place.py")
DEFAULT_ROOT = str(DEFAULT_ARTIFACT_ROOT / "datasets" / "so100_sim_pick_place_ring_40")
DEFAULT_REPO_ID = "flashact/so100_sim_pick_place_ring_40"
DEFAULT_PROMPT = "pick up the cube and place it on the pink pad"
COMPAT_SCENE_CACHE = Path("/tmp/flashact-lerobot-so100-scene")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a LeRobot dataset from SO100 MuJoCo oracle rollouts."
    )
    parser.add_argument("--candidate-manifest", default=DEFAULT_CANDIDATE_MANIFEST)
    parser.add_argument("--demo-root", default=DEFAULT_DEMO_ROOT)
    parser.add_argument("--oracle-path", default=DEFAULT_ORACLE_PATH)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--control-mode", choices=["kinematic", "actuator"], default="kinematic")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_module(path: str, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_placements(path: Path, episodes: int) -> list[dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    placements = manifest.get("accepted", manifest.get("records", []))[:episodes]
    if len(placements) != episodes:
        raise ValueError(f"Expected {episodes} accepted placements, found {len(placements)}")
    seen = set()
    for placement in placements:
        key = tuple(placement["cube_xy"])
        if key in seen:
            raise ValueError(f"Duplicate cube placement: {key}")
        seen.add(key)
    return placements


def create_dataset(args: argparse.Namespace) -> LeRobotDataset:
    root = Path(args.root)
    if root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{root} exists; pass --overwrite to regenerate it")
        shutil.rmtree(root)

    return LeRobotDataset.create(
        repo_id=args.repo_id,
        root=root,
        robot_type="so100",
        fps=args.fps,
        features={
            "exterior_image_1_left": {
                "dtype": "image",
                "shape": (args.image_size, args.image_size, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image_left": {
                "dtype": "image",
                "shape": (args.image_size, args.image_size, 3),
                "names": ["height", "width", "channel"],
            },
            "joint_position": {
                "dtype": "float32",
                "shape": (5,),
                "names": ["joint_position"],
            },
            "gripper_position": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_position"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["actions"],
            },
        },
        image_writer_threads=4,
        image_writer_processes=0,
    )


def patch_demo_scene_path() -> None:
    import app.mujoco_stream as mujoco_stream

    original_scene_path = mujoco_stream.scene_path

    def compatible_scene_path() -> Path:
        source_scene_path = original_scene_path()
        source_dir = source_scene_path.parent
        source_robot_path = source_dir / "so_arm100_flashact.xml"
        COMPAT_SCENE_CACHE.mkdir(parents=True, exist_ok=True)

        robot_path = COMPAT_SCENE_CACHE / source_robot_path.name
        scene_path = COMPAT_SCENE_CACHE / source_scene_path.name
        robot_xml = source_robot_path.read_text(encoding="utf-8").replace(' dampratio="1"', "")
        robot_xml = robot_xml.replace(
            '<key name="home" qpos="0 -1.57079 1.57079 1.57079 -1.57079 0" '
            'ctrl="0 -1.57079 1.57079 1.57079 -1.57079 0"/>',
            '<key name="home" qpos="0 -1.57079 1.57079 1.57079 -1.57079 0 '
            '-0.08 -0.25 0.078 1 0 0 0" ctrl="0 -1.57079 1.57079 1.57079 -1.57079 0"/>',
        )
        scene_xml = source_scene_path.read_text(encoding="utf-8").replace(
            f'file="{source_robot_path}"',
            f'file="{source_robot_path.name}"',
        )
        if "<compiler " not in scene_xml:
            scene_xml = scene_xml.replace(
                '<mujoco model="so_arm100 flashact scene">\n',
                '<mujoco model="so_arm100 flashact scene">\n  <compiler autolimits="true"/>\n',
                1,
            )
        if not robot_path.exists() or robot_path.read_text(encoding="utf-8") != robot_xml:
            robot_path.write_text(robot_xml, encoding="utf-8")
        if not scene_path.exists() or scene_path.read_text(encoding="utf-8") != scene_xml:
            scene_path.write_text(scene_xml, encoding="utf-8")
        return scene_path

    mujoco_stream.scene_path = compatible_scene_path


def render_rgb(renderer: mujoco.Renderer, data: mujoco.MjData, camera: str) -> np.ndarray:
    renderer.update_scene(data, camera=camera)
    return np.ascontiguousarray(renderer.render())


def close_renderer(renderer: Any) -> None:
    close = getattr(renderer, "close", None)
    if close is not None:
        close()


def so100_state(data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    qpos = np.asarray(data.qpos, dtype=np.float32)
    joint_position = np.asarray(qpos[:5], dtype=np.float32)
    gripper_position = np.asarray([qpos[5] if qpos.shape[0] > 5 else 0.0], dtype=np.float32)
    return joint_position, gripper_position


def add_dataset_frame(
    dataset: LeRobotDataset,
    oracle: Any,
    exterior_renderer: mujoco.Renderer,
    wrist_renderer: mujoco.Renderer,
    action: np.ndarray,
    prompt: str,
) -> None:
    joint_position, gripper_position = so100_state(oracle.data)
    dataset.add_frame(
        {
            "exterior_image_1_left": render_rgb(exterior_renderer, oracle.data, "exterior_left"),
            "wrist_image_left": render_rgb(wrist_renderer, oracle.data, "wrist_left"),
            "joint_position": joint_position,
            "gripper_position": gripper_position,
            "actions": np.asarray(action, dtype=np.float32),
            "task": prompt,
        }
    )


def run_recorded_oracle(
    oracle: Any,
    phases: list[dict[str, Any]],
    dataset: LeRobotDataset,
    exterior_renderer: mujoco.Renderer,
    wrist_renderer: mujoco.Renderer,
    fps: int,
    control_mode: str,
    prompt: str,
) -> dict[str, Any]:
    oracle.reset()
    max_cube_z = float(oracle.cube_position()[2])
    attached = False
    attach_offset = np.zeros(3, dtype=np.float64)
    phase_metrics: list[dict[str, Any]] = []
    current_arm_q = np.asarray(oracle.data.qpos[:6], dtype=np.float64).copy()

    def step_once(arm_q: np.ndarray | None = None) -> None:
        nonlocal current_arm_q
        nonlocal max_cube_z

        if arm_q is not None:
            current_arm_q = np.asarray(arm_q, dtype=np.float64).copy()

        if control_mode == "kinematic":
            oracle.data.qpos[:6] = current_arm_q
            oracle.data.qvel[:6] = 0.0
            oracle.data.ctrl[:6] = current_arm_q
            oracle.mujoco.mj_forward(oracle.model, oracle.data)
        else:
            oracle.data.ctrl[:6] = current_arm_q
            oracle.mujoco.mj_step(oracle.model, oracle.data)

        if attached:
            oracle.set_cube_position(oracle.pinch_position() + attach_offset)
            oracle.mujoco.mj_forward(oracle.model, oracle.data)
        elif control_mode == "kinematic":
            oracle.mujoco.mj_step(oracle.model, oracle.data)
            oracle.data.qpos[:6] = current_arm_q
            oracle.data.qvel[:6] = 0.0
            oracle.data.ctrl[:6] = current_arm_q
            oracle.mujoco.mj_forward(oracle.model, oracle.data)

        max_cube_z = max(max_cube_z, float(oracle.cube_position()[2]))

    def hold(seconds: float) -> None:
        frame_count = max(1, int(seconds * fps))
        steps_per_frame = max(1, int((1.0 / fps) / oracle.model.opt.timestep))
        for _ in range(frame_count):
            for _ in range(steps_per_frame):
                step_once()
            add_dataset_frame(
                dataset,
                oracle,
                exterior_renderer,
                wrist_renderer,
                current_arm_q,
                prompt,
            )

    hold(0.4)
    for phase in phases:
        start_ctrl = np.asarray(oracle.data.ctrl[:6], dtype=np.float64).copy()
        target_ctrl = np.asarray(phase["q"], dtype=np.float64)
        frame_count = max(1, int(float(phase["duration_s"]) * fps))
        steps_per_frame = max(1, int((1.0 / fps) / oracle.model.opt.timestep))

        if phase["name"] == "close_and_attach":
            attach_offset = oracle.cube_position() - oracle.pinch_position()
            attached = True

        for frame_index in range(frame_count):
            alpha = (frame_index + 1) / frame_count
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            ctrl = (1.0 - alpha) * start_ctrl + alpha * target_ctrl
            for _ in range(steps_per_frame):
                step_once(ctrl)
            add_dataset_frame(
                dataset,
                oracle,
                exterior_renderer,
                wrist_renderer,
                ctrl,
                prompt,
            )

        if phase["release"]:
            release_xyz = np.asarray([oracle.target_xy[0], oracle.target_xy[1], oracle.cube_rest_z])
            oracle.set_cube_position(release_xyz)
            attached = False
            oracle.mujoco.mj_forward(oracle.model, oracle.data)

        hold(0.2)
        phase_metrics.append(
            {
                "phase": phase["name"],
                "cube_position": oracle.cube_position().tolist(),
                "pinch_position": oracle.pinch_position().tolist(),
                "attached": attached,
            }
        )

    hold(1.0)
    final_cube = oracle.cube_position()
    target = np.asarray([oracle.target_xy[0], oracle.target_xy[1], oracle.cube_rest_z], dtype=np.float64)
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
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    sys.path.insert(0, str(Path(args.demo_root).resolve()))

    patch_demo_scene_path()
    from app.mujoco_stream import MujocoArmStream

    oracle_module = load_module(args.oracle_path, "so100_oracle_pick_place")
    placements = load_placements(Path(args.candidate_manifest), args.episodes)
    dataset = create_dataset(args)
    records = []

    stream = MujocoArmStream("standard", width=args.image_size, height=args.image_size)
    exterior_renderer = mujoco.Renderer(stream.model, height=args.image_size, width=args.image_size)
    wrist_renderer = mujoco.Renderer(stream.model, height=args.image_size, width=args.image_size)
    try:
        for episode_index, placement in enumerate(placements):
            cube_x, cube_y = placement["cube_xy"]
            target_x, target_y = placement["target_xy"]
            oracle = oracle_module.SO100Oracle(
                stream,
                cube_xy=(float(cube_x), float(cube_y)),
                target_xy=(float(target_x), float(target_y)),
            )
            oracle.reset()
            phases = oracle.plan()
            max_ik_error = max(float(phase["ik_error_m"]) for phase in phases)
            result = run_recorded_oracle(
                oracle,
                phases,
                dataset,
                exterior_renderer,
                wrist_renderer,
                fps=args.fps,
                control_mode=args.control_mode,
                prompt=args.prompt,
            )
            dataset.save_episode()

            record = {
                "episode_index": episode_index,
                "cube_xy": [float(cube_x), float(cube_y)],
                "target_xy": [float(target_x), float(target_y)],
                "prompt": args.prompt,
                "source_side": placement.get("side"),
                "max_ik_error_m": max_ik_error,
                "result": result,
            }
            if not result["success"]:
                raise RuntimeError(f"Episode {episode_index} failed: {record}")
            records.append(record)
            print(
                f"saved episode {episode_index + 1:02d}/{len(placements)} "
                f"cube=({cube_x:.3f},{cube_y:.3f}) max_ik={max_ik_error:.4f}",
                flush=True,
            )
    finally:
        close_renderer(exterior_renderer)
        close_renderer(wrist_renderer)
        close_renderer(stream.renderer)

    index = {
        "repo_id": args.repo_id,
        "root": str(Path(args.root)),
        "episodes": len(records),
        "fps": args.fps,
        "image_size": args.image_size,
        "prompt": args.prompt,
        "all_success": all(record["result"]["success"] for record in records),
        "max_ik_error_m": max(record["max_ik_error_m"] for record in records),
        "max_final_cube_target_xy_distance": max(
            record["result"]["final_cube_target_xy_distance"] for record in records
        ),
        "records": records,
    }
    index_path = Path(args.root) / "so100_dataset_index.json"
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(index, indent=2), flush=True)


if __name__ == "__main__":
    main()
