#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors


ASSET_ID = "flashact/so100_sim_pick_place_ring_40"


def state_path(ckpt: Path) -> Path:
    candidates = [
        ckpt / "actor" / "model_state_dict" / "full_weights.pt",
        ckpt / "model_state_dict" / "full_weights.pt",
        ckpt / "model.safetensors",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"no supported state file under {ckpt}")


def load_state(ckpt: Path) -> dict[str, torch.Tensor]:
    path = state_path(ckpt)
    if path.suffix == ".safetensors":
        return load_safetensors(str(path), device="cpu")
    return torch.load(path, map_location="cpu")


def parse_component(raw: str) -> tuple[str, float, Path]:
    name, weight, path = raw.split("=", 2)
    return name, float(weight), Path(path)


def copy_assets(out: Path, sources: list[Path]) -> None:
    for src in sources:
        candidates = [
            src / ASSET_ID,
            src / "assets" / ASSET_ID,
        ]
        for asset_src in candidates:
            if (asset_src / "norm_stats.json").exists():
                dest = out / ASSET_ID
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(asset_src, dest)
                return
    raise FileNotFoundError("no norm_stats assets found in soup sources")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--component",
        required=True,
        action="append",
        help="name=weight=/path/to/checkpoint",
    )
    args = parser.parse_args()

    components = [parse_component(raw) for raw in args.component]
    if not components:
        raise SystemExit("no components")

    total = sum(weight for _, weight, _ in components)
    if total <= 0:
        raise SystemExit("component weights must sum positive")
    components = [(name, weight / total, path) for name, weight, path in components]

    base_name, base_weight, base_path = components[0]
    base = load_state(base_path)
    acc: dict[str, torch.Tensor] = {}
    weight_sums: dict[str, float] = {}

    for key, value in base.items():
        if torch.is_tensor(value) and value.is_floating_point():
            acc[key] = value.detach().cpu().to(torch.float32) * base_weight
            weight_sums[key] = base_weight
        else:
            acc[key] = value.detach().cpu() if torch.is_tensor(value) else value

    for name, weight, path in components[1:]:
        state = load_state(path)
        for key, current in list(acc.items()):
            other = state.get(key)
            if (
                torch.is_tensor(current)
                and current.is_floating_point()
                and torch.is_tensor(other)
                and other.shape == current.shape
                and other.is_floating_point()
            ):
                acc[key] = current + other.detach().cpu().to(torch.float32) * weight
                weight_sums[key] = weight_sums.get(key, 0.0) + weight
        del state

    out_state: dict[str, torch.Tensor] = {}
    for key, value in acc.items():
        if torch.is_tensor(value) and value.is_floating_point():
            denom = weight_sums.get(key, 1.0)
            target_dtype = base[key].dtype if torch.is_tensor(base[key]) else value.dtype
            out_state[key] = (value / denom).to(target_dtype)
        else:
            out_state[key] = value

    out = args.out
    weights_dir = out / "actor" / "model_state_dict"
    weights_dir.mkdir(parents=True, exist_ok=True)
    torch.save(out_state, weights_dir / "full_weights.pt")
    copy_assets(out, [path for _, _, path in components])

    meta = {
        "components": [
            {"name": name, "weight": weight, "path": str(path)}
            for name, weight, path in components
        ],
        "state_keys": len(out_state),
    }
    (out / "soup_meta.json").write_text(json.dumps(meta, indent=2))
    print(out)


if __name__ == "__main__":
    main()
