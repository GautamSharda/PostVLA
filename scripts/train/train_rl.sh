#!/usr/bin/env bash
set -euo pipefail

# Run this from a clone with the overlays applied. Set POSTVLA_ARTIFACT_ROOT to
# place large datasets, checkpoints, logs, and videos on a network volume.
# Modes:
#   reproduce  Reproduce the promising SFT-start grasp/lift-anchor run.
#   hard       Oversample SFT hard cube IDs, then run the same conservative PPO.
#   continue-hard Continue from START_CKPT on the hard/mixed cube curriculum.
#   eval       Evaluate an existing checkpoint with the standard cubevar benchmark.
#   sweep-eval Evaluate every valid checkpoint under CHECKPOINT_ROOT.
#   fix-assets Copy OpenPI norm stats/assets into RLinf checkpoints after training.
#   latest     Print the latest apparently valid checkpoint under CHECKPOINT_ROOT.
#   quick-eval Fix assets and evaluate latest checkpoint on first 20 starts.
#   cleanup-dcp Remove DCP resume shards when full_weights.pt exists.

MODE="${1:-reproduce}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
POSTVLA_ROOT="${POSTVLA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ARTIFACT_ROOT="${POSTVLA_ARTIFACT_ROOT:-${POSTVLA_ROOT}/artifacts}"

RLINF_ROOT="${RLINF_ROOT:-${POSTVLA_ROOT}/third_party/RLinf}"
OPENPI_ROOT="${OPENPI_ROOT:-${POSTVLA_ROOT}/third_party/openpi}"
OPENPI_PATCH_ROOT="${OPENPI_PATCH_ROOT:-${POSTVLA_ROOT}/compat/openpi_patch}"
DEMO_ROOT="${DEMO_ROOT:-${POSTVLA_ROOT}/sim}"
RUN_ROOT="${RUN_ROOT:-${ARTIFACT_ROOT}/rlinf_runs}"
EVAL_ROOT="${EVAL_ROOT:-${ARTIFACT_ROOT}/rlinf_eval}"
BASE_MANIFEST="${BASE_MANIFEST:-${POSTVLA_ROOT}/configs/manifests/cubevar_contact_40.json}"
START_CKPT="${START_CKPT:-${ARTIFACT_ROOT}/checkpoints/pi05_so100_sim_distilled_sft}"
export POSTVLA_ROOT POSTVLA_ARTIFACT_ROOT="${ARTIFACT_ROOT}" POSTVLA_DISTILLED_CKPT="${START_CKPT}"
MODEL_CKPT="${MODEL_CKPT:-}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-}"
TRAIN_STEPS="${TRAIN_STEPS:-5}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
TOTAL_ENVS="${TOTAL_ENVS:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-80}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
ACTOR_LR="${ACTOR_LR:-1e-7}"
KL_BETA="${KL_BETA:-0.08}"
CLIP_RATIO="${CLIP_RATIO:-0.05}"
UPDATE_EPOCH="${UPDATE_EPOCH:-1}"
VALUE_LR="${VALUE_LR:-1.55e-4}"
CRITIC_WARMUP_STEPS="${CRITIC_WARMUP_STEPS:-0}"
ENTROPY_BONUS="${ENTROPY_BONUS:-0.005}"
NOISE_METHOD="${NOISE_METHOD:-flow_noise}"
JOINT_LOGPROB="${JOINT_LOGPROB:-True}"
ENABLE_SFT_CO_TRAIN="${ENABLE_SFT_CO_TRAIN:-False}"
SFT_DATA_PATH="${SFT_DATA_PATH:-${ARTIFACT_ROOT}/datasets/flashact/so100_sim_pick_place_ring_40}"
SFT_LOSS_WEIGHT="${SFT_LOSS_WEIGHT:-0.05}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-2}"
SFT_CONFIG_NAME="${SFT_CONFIG_NAME:-pi05_so100_sim}"
GRASP_REWARD="${GRASP_REWARD:-0.45}"
LIFT_REWARD="${LIFT_REWARD:-0.85}"
PROGRESS_REWARD_SCALE="${PROGRESS_REWARD_SCALE:-0.25}"
ON_PAD_REWARD="${ON_PAD_REWARD:-1.0}"
PAD_CENTER_REWARD="${PAD_CENTER_REWARD:-0.0}"
HARD_IDS="${HARD_IDS:-0,1,2,3,4,7,9,10,15,16,20,25,28,30,33,35,37}"
HARD_REPEATS="${HARD_REPEATS:-6}"
HARD_BASE_COUNT="${HARD_BASE_COUNT:-40}"
EVAL_ATTEMPTS="${ATTEMPTS:-20}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-900}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-5}"
EVAL_NUM_STEPS="${EVAL_NUM_STEPS:-10}"
EVAL_SEED="${EVAL_SEED:-0}"
MANIFEST_LIMIT="${MANIFEST_LIMIT:-40}"

export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export EMBODIED_PATH="${EMBODIED_PATH:-${RLINF_ROOT}/examples/embodiment}"
export REPO_PATH="${REPO_PATH:-${RLINF_ROOT}}"
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
export PYTHONPATH="/tmp/flashact-system-pkgs:${RLINF_ROOT}:${OPENPI_PATCH_ROOT}:${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src:${DEMO_ROOT}:${POSTVLA_ROOT}:${PYTHONPATH:-}"

cd "${RLINF_ROOT}"

make_hard_manifest() {
  local out="${ARTIFACT_ROOT}/manifests/so100_cubevar_mixed_hard_sft_failures.json"
  mkdir -p "$(dirname "${out}")"
  python3 - "${BASE_MANIFEST}" "${out}" "${HARD_IDS}" "${HARD_REPEATS}" "${HARD_BASE_COUNT}" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
hard_ids = [int(x) for x in sys.argv[3].split(",") if x.strip()]
hard_repeats = int(sys.argv[4])
base_count = int(sys.argv[5])
data = json.loads(src.read_text())
records = data.get("records", data) if isinstance(data, dict) else data

mixed = []
for i, rec in enumerate(records[:base_count]):
    item = dict(rec)
    item["source_episode_id"] = i
    item["curriculum_tag"] = "base"
    mixed.append(item)

for repeat in range(hard_repeats):
    for i in hard_ids:
        item = dict(records[i])
        item["source_episode_id"] = i
        item["curriculum_tag"] = f"hard_repeat_{repeat}"
        mixed.append(item)

dst.write_text(json.dumps({"records": mixed, "hard_ids": hard_ids, "base_count": base_count, "hard_repeats": hard_repeats}, indent=2))
print(dst)
print(f"records={len(mixed)} base={base_count} hard_repeats={len(mixed)-base_count}", file=sys.stderr)
PY
}

manifest_count() {
  python3 - "$1" <<'PY'
import json
import sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text())
records = data.get("records", data) if isinstance(data, dict) else data
print(len(records))
PY
}

train() {
  local exp="$1"
  local manifest="$2"
  local limit="$3"
  local out="${RUN_ROOT}/${exp}"
  mkdir -p "${out}"

  nohup python3 examples/embodiment/train_embodied_agent.py \
    --config-name so100_mujoco_ppo_openpi_pi05 \
    runner.logger.log_path="${out}" \
    runner.logger.experiment_name="${exp}" \
    runner.max_steps="${TRAIN_STEPS}" \
    runner.save_interval="${SAVE_INTERVAL}" \
    runner.val_check_interval=-1 \
    actor.model.model_path="${START_CKPT}" \
    rollout.model.model_path="${START_CKPT}" \
    actor.model.openpi.detach_critic_input=true \
    actor.model.openpi.noise_method="${NOISE_METHOD}" \
    actor.model.openpi.joint_logprob="${JOINT_LOGPROB}" \
    +actor.enable_sft_co_train="${ENABLE_SFT_CO_TRAIN}" \
    +actor.sft_data_path="${SFT_DATA_PATH}" \
    +actor.sft_loss_weight="${SFT_LOSS_WEIGHT}" \
    +actor.sft_batch_size="${SFT_BATCH_SIZE}" \
    +actor.config_name="${SFT_CONFIG_NAME}" \
    actor.optim.lr="${ACTOR_LR}" \
    actor.optim.value_lr="${VALUE_LR}" \
    actor.optim.critic_warmup_steps="${CRITIC_WARMUP_STEPS}" \
    algorithm.kl_beta="${KL_BETA}" \
    algorithm.clip_ratio_high="${CLIP_RATIO}" \
    algorithm.clip_ratio_low="${CLIP_RATIO}" \
    algorithm.update_epoch="${UPDATE_EPOCH}" \
    algorithm.entropy_bonus="${ENTROPY_BONUS}" \
    +weight_syncer.patch.transport_device=cpu \
    actor.enable_offload=true \
    rollout.enable_offload=true \
    actor.micro_batch_size="${MICRO_BATCH_SIZE}" \
    actor.global_batch_size="${GLOBAL_BATCH_SIZE}" \
    env.train.total_num_envs="${TOTAL_ENVS}" \
    env.train.cube_xy_manifest="${manifest}" \
    env.train.cube_xy_limit="${limit}" \
    env.train.grasp_reward="${GRASP_REWARD}" \
    env.train.lift_reward="${LIFT_REWARD}" \
    env.train.progress_reward_scale="${PROGRESS_REWARD_SCALE}" \
    env.train.on_pad_reward="${ON_PAD_REWARD}" \
    env.train.pad_center_reward="${PAD_CENTER_REWARD}" \
    > "${out}/train.log" 2>&1 &

  echo "$!" > "${out}/train.pid"
  echo "TRAIN_PID=$(cat "${out}/train.pid")"
  echo "TRAIN_DIR=${out}"
  echo "EXP=${exp}"
}

eval_ckpt() {
  local ckpt="$1"
  local name="$2"
  local attempts="${3:-20}"
  local videos="${4:-}"
  local out="${EVAL_ROOT}/${name}"
  mkdir -p "${out}"
  local video_arg=()
  if [[ "${videos}" == "videos" ]]; then
    video_arg=(--save-videos)
  fi

  nohup python3 "${POSTVLA_ROOT}/scripts/eval/eval_batched.py" \
    --model-path "${ckpt}" \
    --output-dir "${out}" \
    --attempts "${attempts}" \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --max-steps "${EVAL_MAX_STEPS}" \
    --num-steps "${EVAL_NUM_STEPS}" \
    --seed "${EVAL_SEED}" \
    "${video_arg[@]}" \
    > "${out}/eval.log" 2>&1 &

  echo "$!" > "${out}/eval.pid"
  echo "EVAL_PID=$(cat "${out}/eval.pid")"
  echo "EVAL_DIR=${out}"
}

cleanup_dcp() {
  if [[ -z "${CHECKPOINT_ROOT}" ]]; then
    echo "Set CHECKPOINT_ROOT=/path/to/rlinf/run/checkpoints" >&2
    exit 2
  fi
  local removed=0
  while IFS= read -r -d "" dcp; do
    local full="$(dirname "${dcp}")/model_state_dict/full_weights.pt"
    if [[ -s "${full}" ]]; then
      rm -rf "${dcp}"
      removed=$((removed + 1))
      echo "removed ${dcp}"
    fi
  done < <(find "${CHECKPOINT_ROOT}" -path "*/actor/dcp_checkpoint" -type d -print0)
  echo "removed_dcp_dirs=${removed}"
}

fix_assets() {
  if [[ -z "${CHECKPOINT_ROOT}" ]]; then
    echo "Set CHECKPOINT_ROOT=/path/to/rlinf/run/checkpoints" >&2
    exit 2
  fi
  local src="${START_CKPT}/flashact/so100_sim_pick_place_ring_40"
  if [[ ! -f "${src}/norm_stats.json" ]]; then
    echo "Missing source norm stats at ${src}/norm_stats.json" >&2
    exit 2
  fi
  for ckpt in "${CHECKPOINT_ROOT}"/global_step_*; do
    [[ -d "${ckpt}" ]] || continue
    local dest="${ckpt}/flashact/so100_sim_pick_place_ring_40"
    mkdir -p "$(dirname "${dest}")"
    rm -rf "${dest}"
    cp -a "${src}" "${dest}"
    echo "copied assets into ${ckpt}"
  done
}

latest_ckpt() {
  if [[ -z "${CHECKPOINT_ROOT}" ]]; then
    echo "Set CHECKPOINT_ROOT=/path/to/rlinf/run/checkpoints" >&2
    exit 2
  fi
  python3 - "${CHECKPOINT_ROOT}" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
candidates = []
for ckpt in root.glob("global_step_*"):
    m = re.search(r"global_step_(\d+)$", ckpt.name)
    if not m:
        continue
    full = ckpt / "actor" / "model_state_dict" / "full_weights.pt"
    if not full.exists():
        continue
    # The full weights are roughly 8G; partial files from interrupted saves were smaller.
    if full.stat().st_size < 7_000_000_000:
        continue
    candidates.append((int(m.group(1)), ckpt))
if not candidates:
    raise SystemExit(f"No valid checkpoints found under {root}")
print(max(candidates)[1])
PY
}

sweep_eval() {
  if [[ -z "${CHECKPOINT_ROOT}" ]]; then
    echo "Set CHECKPOINT_ROOT=/path/to/rlinf/run/checkpoints" >&2
    exit 2
  fi
  mapfile -t ckpts < <(python3 - "${CHECKPOINT_ROOT}" <<'PY'
import re
import sys
from pathlib import Path
root = Path(sys.argv[1])
items = []
for ckpt in root.glob("global_step_*"):
    m = re.search(r"global_step_(\d+)$", ckpt.name)
    if not m:
        continue
    full = ckpt / "actor" / "model_state_dict" / "full_weights.pt"
    if full.exists() and full.stat().st_size >= 7_000_000_000:
        items.append((int(m.group(1)), ckpt))
for _, ckpt in sorted(items):
    print(ckpt)
PY
)
  if [[ "${#ckpts[@]}" -eq 0 ]]; then
    echo "No valid checkpoints found under ${CHECKPOINT_ROOT}" >&2
    exit 2
  fi
  fix_assets
  for ckpt in "${ckpts[@]}"; do
    eval_ckpt "${ckpt}" "sweep_$(basename "${ckpt}")_${STAMP}" "${EVAL_ATTEMPTS}" "${VIDEOS:-}"
    local pid_file="${EVAL_ROOT}/sweep_$(basename "${ckpt}")_${STAMP}/eval.pid"
    local pid
    pid="$(cat "${pid_file}")"
    echo "waiting eval pid=${pid} ckpt=${ckpt}"
    while kill -0 "${pid}" 2>/dev/null; do
      sleep 10
    done
    cat "${EVAL_ROOT}/sweep_$(basename "${ckpt}")_${STAMP}/summary.json"
  done
}

case "${MODE}" in
  custom)
    EXP="${EXP:-so100_ppo_custom_${STAMP}}"
    train "${EXP}" "${BASE_MANIFEST}" "${MANIFEST_LIMIT}"
    ;;
  reproduce)
    EXP="so100_ppo_sft_grasplift_anchor_repro_lr1e7_kl008_env8_5_${STAMP}"
    train "${EXP}" "${BASE_MANIFEST}" 40
    ;;
  hard)
    HARD_MANIFEST="$(make_hard_manifest)"
    EXP="so100_ppo_sft_grasplift_hardmix_lr1e7_kl008_env8_5_${STAMP}"
    train "${EXP}" "${HARD_MANIFEST}" "$(manifest_count "${HARD_MANIFEST}")"
    ;;
  continue-hard)
    HARD_MANIFEST="$(make_hard_manifest)"
    EXP="so100_ppo_continue_hardmix_lr1e7_kl008_env8_${TRAIN_STEPS}_${STAMP}"
    train "${EXP}" "${HARD_MANIFEST}" "$(manifest_count "${HARD_MANIFEST}")"
    ;;
  eval)
    if [[ -z "${MODEL_CKPT}" ]]; then
      echo "Set MODEL_CKPT=/path/to/checkpoint for eval mode" >&2
      exit 2
    fi
    eval_ckpt "${MODEL_CKPT}" "eval_$(basename "${MODEL_CKPT}")_${STAMP}" "${EVAL_ATTEMPTS}" "${VIDEOS:-}"
    ;;
  sweep-eval)
    sweep_eval
    ;;
  fix-assets)
    fix_assets
    ;;
  latest)
    latest_ckpt
    ;;
  cleanup-dcp)
    cleanup_dcp
    ;;
  quick-eval)
    ckpt="$(latest_ckpt)"
    CHECKPOINT_ROOT="${CHECKPOINT_ROOT}" START_CKPT="${START_CKPT}" "$0" fix-assets
    eval_ckpt "${ckpt}" "quick20_$(basename "$(dirname "$(dirname "${ckpt}")")")_$(basename "${ckpt}")_${STAMP}" 20 ""
    ;;
  *)
    echo "Usage: $0 [custom|reproduce|hard|continue-hard|eval|sweep-eval|fix-assets|latest|quick-eval|cleanup-dcp]" >&2
    exit 2
    ;;
esac
