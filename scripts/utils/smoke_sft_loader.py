#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

from rlinf.workers.actor.fsdp_actor_worker import resolve_lerobot_repo_id
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
import openpi.training.data_loader as data_loader


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    artifact_root = Path(os.environ.get("POSTVLA_ARTIFACT_ROOT", root / "artifacts"))
    dataset_path = str(artifact_root / "datasets" / "flashact" / "so100_sim_pick_place_ring_40")
    model_path = str(artifact_root / "checkpoints" / "pi05_so100_sim_distilled_sft")
    repo_id = resolve_lerobot_repo_id(dataset_path)
    print(f"repo={repo_id}")
    cfg = get_openpi_config(
        "pi05_so100_sim",
        model_path=model_path,
        repo_id=repo_id,
        batch_size=2,
    )
    loader = data_loader.create_data_loader(cfg, framework="pytorch", shuffle=True)
    obs, actions = next(iter(loader))
    print(f"obs_keys={sorted(obs.keys())}")
    for key, value in obs.items():
        if hasattr(value, "shape"):
            print(f"obs {key} shape={tuple(value.shape)} dtype={value.dtype}")
    print(f"actions shape={tuple(actions.shape)} dtype={actions.dtype}")


if __name__ == "__main__":
    main()
