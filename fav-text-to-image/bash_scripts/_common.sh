#!/bin/bash
# Shared environment for fav-text-to-image launch scripts.
# Override PYTHON_BIN / CUDA_VISIBLE_DEVICES / MAIN_PORT via environment.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"

GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC="$(echo "${GPU_LIST}" | awk -F',' '{print NF}')"

MAIN_PORT="${MAIN_PORT:-29520}"

# Sana-Sprint uses vanilla cross-attention; xformers must be disabled.
export DISABLE_XFORMERS="${DISABLE_XFORMERS:-1}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"
