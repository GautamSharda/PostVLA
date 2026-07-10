#!/usr/bin/env bash
set -euo pipefail

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
POSTVLA_ROOT="${POSTVLA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ARTIFACT_ROOT="${POSTVLA_ARTIFACT_ROOT:-${POSTVLA_ROOT}/artifacts}"
RLINF_ROOT="${RLINF_ROOT:-${POSTVLA_ROOT}/third_party/RLinf}"
OPENPI_ROOT="${OPENPI_ROOT:-${POSTVLA_ROOT}/third_party/openpi}"
OPENPI_PATCH_ROOT="${OPENPI_PATCH_ROOT:-${POSTVLA_ROOT}/compat/openpi_patch}"
DEMO_ROOT="${DEMO_ROOT:-${POSTVLA_ROOT}/sim}"
RUN_ROOT="${RUN_ROOT:-${ARTIFACT_ROOT}/rlinf_sft_runs}"
CONFIG_PATH="${CONFIG_PATH:-${POSTVLA_ROOT}/configs/sft}"
CONFIG_NAME="${CONFIG_NAME:-rlinf_openpi_pi05_1gpu}"
DATA_PATH="${DATA_PATH:-${ARTIFACT_ROOT}/datasets/flashact/so100_sim_pick_place_ring_40}"
START_CKPT="${START_CKPT:-${ARTIFACT_ROOT}/checkpoints/pi05_so100_sim_distilled_sft}"
EXP="${EXP:-so100_sft_consolidate_from_picklift_${STAMP}}"
MAX_STEPS="${MAX_STEPS:-20}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
LR="${LR:-5e-7}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-10}"
TRAIN_EXPERT_ONLY="${TRAIN_EXPERT_ONLY:-True}"
ADD_VALUE_HEAD="${ADD_VALUE_HEAD:-False}"

export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export EMBODIED_PATH="${EMBODIED_PATH:-${RLINF_ROOT}/examples/embodiment}"
export PYTHONPATH="/tmp/flashact-system-pkgs:${RLINF_ROOT}:${OPENPI_PATCH_ROOT}:${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src:${DEMO_ROOT}:${POSTVLA_ROOT}:${PYTHONPATH:-}"

rm -rf \
  /tmp/flashact-system-pkgs/torch \
  /tmp/flashact-system-pkgs/torch-*.dist-info \
  /tmp/flashact-system-pkgs/torchvision \
  /tmp/flashact-system-pkgs/torchvision-*.dist-info \
  /tmp/flashact-system-pkgs/functorch \
  /tmp/flashact-system-pkgs/functorch-*.dist-info \
  /tmp/flashact-system-pkgs/torchgen \
  /tmp/flashact-system-pkgs/torchgen-*.dist-info \
  /tmp/flashact-system-pkgs/triton \
  /tmp/flashact-system-pkgs/triton-*.dist-info \
  /tmp/flashact-system-pkgs/nvidia \
  /tmp/flashact-system-pkgs/nvidia_*.dist-info \
  /tmp/flashact-system-pkgs/nvidia-*.dist-info

cd "${RLINF_ROOT}"
out="${RUN_ROOT}/${EXP}"
mkdir -p "${out}"

nohup python3 examples/sft/train_vla_sft.py \
  --config-path "${CONFIG_PATH}" \
  --config-name "${CONFIG_NAME}" \
  runner.logger.log_path="${out}" \
  runner.logger.experiment_name="${EXP}" \
  runner.max_steps="${MAX_STEPS}" \
  runner.save_interval="${SAVE_INTERVAL}" \
  runner.val_check_interval=-1 \
  data.train_data_paths="${DATA_PATH}" \
  actor.model.model_path="${START_CKPT}" \
  actor.model.num_action_chunks=16 \
  actor.model.action_dim=6 \
  actor.model.num_steps=10 \
  actor.model.add_value_head="${ADD_VALUE_HEAD}" \
  actor.model.openpi.config_name=pi05_so100_sim \
  actor.model.openpi.num_images_in_input=2 \
  actor.model.openpi.train_expert_only="${TRAIN_EXPERT_ONLY}" \
  actor.model.openpi.action_env_dim=6 \
  actor.model.openpi.action_chunk=16 \
  actor.model.openpi.num_steps=10 \
  actor.micro_batch_size="${MICRO_BATCH_SIZE}" \
  actor.global_batch_size="${GLOBAL_BATCH_SIZE}" \
  actor.optim.lr="${LR}" \
  actor.optim.weight_decay="${WEIGHT_DECAY}" \
  actor.optim.total_training_steps="${MAX_STEPS}" \
  actor.optim.lr_warmup_steps=0 \
  actor.fsdp_config.sharding_strategy=no_shard \
  actor.fsdp_config.gradient_checkpointing=False \
  > "${out}/sft.log" 2>&1 &

echo "$!" > "${out}/sft.pid"
echo "SFT_PID=$(cat "${out}/sft.pid")"
echo "SFT_DIR=${out}"
echo "EXP=${EXP}"
