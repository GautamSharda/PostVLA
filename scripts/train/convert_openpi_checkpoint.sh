#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/scripts/env.sh"

if [[ "$#" -lt 2 ]]; then
  echo "usage: $0 JAX_CHECKPOINT OUTPUT_DIRECTORY [PRECISION]" >&2
  exit 2
fi

JAX_CHECKPOINT="$1"
OUTPUT_DIRECTORY="$2"
PRECISION="${3:-bfloat16}"

python3 "${ROOT}/third_party/RLinf/rlinf/utils/ckpt_convertor/convert_openpi_jax_to_python.py" \
  --checkpoint-dir "${JAX_CHECKPOINT}" \
  --config-name pi05_so100_sim \
  --output-path "${OUTPUT_DIRECTORY}" \
  --precision "${PRECISION}"

