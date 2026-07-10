#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/scripts/env.sh"

EXP_NAME="${EXP_NAME:-so100_contact_cube_pad_lora_8k}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${POSTVLA_ARTIFACT_ROOT}/datasets}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${POSTVLA_ARTIFACT_ROOT}/openpi_data}"

cd "${ROOT}/third_party/openpi"
uv run scripts/train.py pi05_so100_sim --exp-name="${EXP_NAME}" "${@}"

