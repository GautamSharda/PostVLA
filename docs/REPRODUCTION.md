# Reproduction Guide

## 1. Checkout and overlays

```bash
git clone --recursive <repository-url> PostVLA
cd PostVLA
./scripts/apply_overlays.sh
export POSTVLA_ARTIFACT_ROOT=/network_volume/PostVLA-artifacts
source scripts/env.sh
```

The tested stack used Linux, an NVIDIA RTX 5090, CUDA 12.8, PyTorch 2.8, OSMesa rendering,
and separate OpenPI/JAX and RLinf/PyTorch environments. Follow the pinned submodules' setup
instructions, then install `mujoco`, `gymnasium`, `imageio`, `Pillow`, `omegaconf`, and LeRobot
in the environment used for data generation and evaluation.

## 2. Oracle and LeRobot data

Generate a single contact-physics rollout first:

```bash
python3 scripts/data/so100_oracle_pick_place.py \
  --cube-x -0.08 --cube-y -0.25 --target-x 0.08 --target-y -0.25
```

Generate the cube-position LeRobot dataset:

```bash
python3 scripts/data/generate_so100_lerobot_dataset.py \
  --candidate-manifest configs/manifests/cubevar_contact_40.json \
  --root "$POSTVLA_ARTIFACT_ROOT/datasets/flashact/so100_sim_pick_place_ring_40" \
  --episodes 40 --control-mode actuator --overwrite
```

The manifests preserve the exact accepted cube/pad placements. The `actuator` mode uses MuJoCo
contacts; `kinematic` is useful for debugging but should not be used for the final demonstrations.

## 3. OpenPI SFT

The OpenPI overlay registers `pi05_so100_sim` with a `(16, 32)` action latent and six physical
SO100 action dimensions. Point `HF_LEROBOT_HOME` at the parent of the `flashact/...` dataset and run:

```bash
HF_LEROBOT_HOME="$POSTVLA_ARTIFACT_ROOT/datasets" \
  EXP_NAME=so100_contact_cube_pad_lora_8k \
  ./scripts/train/train_openpi_sft.sh --overwrite
```

The historical run used an 8k checkpoint despite the early config name/defaults changing during
development. Record the resolved OpenPI config with any reproduction.

## 4. JAX-to-PyTorch conversion

```bash
./scripts/train/convert_openpi_checkpoint.sh \
  "$POSTVLA_ARTIFACT_ROOT/checkpoints/openpi_jax_sft/7999" \
  "$POSTVLA_ARTIFACT_ROOT/checkpoints/torch_sft" bfloat16
```

The converter patch copies norm-stat assets into the layout expected by RLinf. The compatibility
shim under `compat/openpi_patch` keeps RoPE inverse frequencies in float32.

## 5. JAX-teacher calibration

Collect 20 JAX-teacher rollouts (57 chunks per 900-step episode in the historical run):

```bash
POSTVLA_JAX_TEACHER_CKPT="$POSTVLA_ARTIFACT_ROOT/checkpoints/openpi_jax_sft/7999" \
POSTVLA_TEACHER_DATASET="$POSTVLA_ARTIFACT_ROOT/teacher_dataset" \
python3 scripts/calibration/collect_jax_teacher.py
```

Calibrate the final two action-expert layers and action/time projections for three epochs:

```bash
POSTVLA_TORCH_SFT_CKPT="$POSTVLA_ARTIFACT_ROOT/checkpoints/torch_sft" \
POSTVLA_TEACHER_DATASET="$POSTVLA_ARTIFACT_ROOT/teacher_dataset" \
POSTVLA_DISTILLED_CKPT="$POSTVLA_ARTIFACT_ROOT/checkpoints/pi05_so100_sim_distilled_sft" \
python3 scripts/calibration/calibrate_teacher.py
```

The recovered historical calibration had 1,140 samples, validation loss `0.05765 -> 0.03937`,
batch size 4, layer LR `1e-5`, and projection LR `5e-5`.

## 6. Two-stage PPO

Stage 1 starts from distilled SFT and targets grasp/lift robustness:

```bash
START_CKPT="$POSTVLA_DISTILLED_CKPT" \
BASE_MANIFEST="$POSTVLA_ROOT/configs/manifests/cubevar_contact_40.json" \
MANIFEST_LIMIT=40 TRAIN_STEPS=5 SAVE_INTERVAL=5 TOTAL_ENVS=8 \
GLOBAL_BATCH_SIZE=80 MICRO_BATCH_SIZE=4 ACTOR_LR=5e-7 KL_BETA=0.02 \
CLIP_RATIO=0.1 UPDATE_EPOCH=1 GRASP_REWARD=0.12 LIFT_REWARD=0.30 \
PROGRESS_REWARD_SCALE=0.50 ON_PAD_REWARD=1.0 PAD_CENTER_REWARD=0.0 \
EXP=rl_stage1_liftboost ./scripts/train/train_rl.sh custom
```

After that process finishes, locate `global_step_5`, copy/fix its norm-stat assets with the
launcher's `fix-assets` mode, and use it as stage 2's start checkpoint:

```bash
START_CKPT=/path/to/rl_stage1/checkpoints/global_step_5 \
BASE_MANIFEST="$POSTVLA_ROOT/configs/manifests/picklift_repair_curriculum.json" \
MANIFEST_LIMIT=118 TRAIN_STEPS=5 SAVE_INTERVAL=5 TOTAL_ENVS=8 \
GLOBAL_BATCH_SIZE=80 MICRO_BATCH_SIZE=4 ACTOR_LR=2.5e-7 KL_BETA=0.02 \
CLIP_RATIO=0.1 UPDATE_EPOCH=1 GRASP_REWARD=0.15 LIFT_REWARD=0.40 \
PROGRESS_REWARD_SCALE=0.50 ON_PAD_REWARD=1.0 PAD_CENTER_REWARD=0.0 \
EXP=rl_stage2_picklift_repair ./scripts/train/train_rl.sh custom
```

The fully resolved historical Hydra configs are under `configs/experiments/`. They retain original
absolute artifact paths for provenance; use the portable commands above for a new checkout.

## 7. Evaluation and videos

Standard Gaussian-initialized deterministic ODE evaluation:

```bash
python3 scripts/eval/eval_noise_control.py \
  --model-path /path/to/checkpoint \
  --output-dir "$POSTVLA_ARTIFACT_ROOT/eval/gaussian_n100" \
  --attempts 100 --batch-size 1 --max-steps 900 --num-steps 10 \
  --sampling-mode eval --initial-noise-mode gaussian --save-videos
```

Change only `--initial-noise-mode` to run controls:

- `zero`: all-zero initial `(16, 32)` latent.
- `scaled_gaussian --initial-noise-scale 0.5`: fresh scaled Gaussian each action chunk.
- `fixed_gaussian --initial-noise-scale 0.5`: one scaled Gaussian tensor reused.
- `first6_gaussian`: Gaussian in physical action dimensions 0-5, zeros in dimensions 6-31.

Use `--sampling-mode train` to exercise RLinf's stochastic flow-noise path. Keep episode IDs,
batch size, step cap, and seeds fixed when making paired comparisons.
