# PostVLA

PostVLA reproduces an SO100 pick-and-place post-training study for OpenPI pi0.5:

1. Build a contact-physics MuJoCo task and scripted IK oracle.
2. Generate exterior-camera, wrist-camera, proprioception, and action trajectories.
3. SFT pi0.5 in OpenPI (JAX).
4. Convert to RLinf's PyTorch OpenPI implementation and calibrate against the JAX teacher.
5. Continue online with two five-iteration PPO stages using RLinf's flow-noise path.
6. Evaluate matched cube-position rollouts under several initial-latent and sampler settings.

Third-party projects are pinned as submodules; our changes are explicit patches and overlays.
Checkpoints, datasets, videos, and run directories are intentionally excluded from Git.

## Results

All 100-episode rows below use the same ordered cube-position IDs.

| Policy | PPO iterations from SFT | Decode | On pad | Center + lift |
| --- | ---: | --- | ---: | ---: |
| Distilled Torch SFT | 0 | unit Gaussian + ODE | 69/100 | 53/100 |
| RL stage 1 | 5 | unit Gaussian + ODE | 64/100 | 55/100 |
| RL stage 2 | 10 total | unit Gaussian + ODE | 70/100 | 52-55/100 |
| RL stage 2 | 10 total | zero latent + ODE | 90/100 | 82/100 |

The `90/100` row is real, but it is not by itself an apples-to-apples estimate of the PPO gain:
the SFT `69/100` baseline used Gaussian initialization, while that row used a zero initial
latent. A full 100-episode SFT zero-latent control is still required to separate policy
improvement from decoder selection. See [results/README.md](results/README.md).

## Quick Start

```bash
git clone --recursive <repository-url> PostVLA
cd PostVLA
./scripts/apply_overlays.sh
export POSTVLA_ARTIFACT_ROOT=/network_volume/PostVLA-artifacts
source scripts/env.sh
```

Generate one oracle rollout:

```bash
python3 scripts/data/so100_oracle_pick_place.py
```

Evaluate a checkpoint with the standard Gaussian-initialized ODE path:

```bash
python3 scripts/eval/eval_noise_control.py \
  --model-path "$POSTVLA_ARTIFACT_ROOT/checkpoints/pi05_so100_sim_rl_stage2_step5" \
  --output-dir "$POSTVLA_ARTIFACT_ROOT/eval/gaussian_n100" \
  --attempts 100 --batch-size 1 --max-steps 900 --save-videos \
  --sampling-mode eval --initial-noise-mode gaussian
```

Use `--initial-noise-mode zero` for the zero-latent control. Full setup, SFT,
conversion, calibration, and two-stage PPO commands are in
[docs/REPRODUCTION.md](docs/REPRODUCTION.md).

## Layout

- `sim/`: standalone SO100 MuJoCo scene and observation renderer.
- `scripts/data/`: oracle and LeRobot dataset generation.
- `scripts/calibration/`: JAX teacher rollout collection and PyTorch calibration.
- `scripts/train/`: OpenPI SFT, conversion, RLinf SFT, and PPO launchers.
- `scripts/eval/`: batched rollout, sampler controls, and comparison tools.
- `overlays/`, `patches/`: auditable changes applied to pinned submodules.
- `configs/`: manifests and resolved historical run configurations.
- `results/`: raw summaries and a compact experiment index.

