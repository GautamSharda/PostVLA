#!/usr/bin/env bash

POSTVLA_ROOT="${POSTVLA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
POSTVLA_ARTIFACT_ROOT="${POSTVLA_ARTIFACT_ROOT:-${POSTVLA_ROOT}/artifacts}"
POSTVLA_DISTILLED_CKPT="${POSTVLA_DISTILLED_CKPT:-${POSTVLA_ARTIFACT_ROOT}/checkpoints/pi05_so100_sim_distilled_sft}"

export POSTVLA_ROOT POSTVLA_ARTIFACT_ROOT POSTVLA_DISTILLED_CKPT
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export EMBODIED_PATH="${EMBODIED_PATH:-${POSTVLA_ROOT}/third_party/RLinf/examples/embodiment}"
export PYTHONPATH="${POSTVLA_ROOT}/third_party/RLinf:${POSTVLA_ROOT}/compat/openpi_patch:${POSTVLA_ROOT}/third_party/openpi/src:${POSTVLA_ROOT}/third_party/openpi/packages/openpi-client/src:${POSTVLA_ROOT}/sim:${POSTVLA_ROOT}:${PYTHONPATH:-}"
