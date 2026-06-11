#!/bin/bash
# Shared environment setup sourced by every per-experiment script.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Drifting (JAX) needs its own env; the rest use the default PyTorch env.
PYTHON_BIN="${PYTHON_BIN:-python}"
JAX_PYTHON_BIN="${JAX_PYTHON_BIN:-python}"

GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC="$(echo "${GPU_LIST}" | awk -F',' '{print NF}')"

MAIN_PORT="${MAIN_PORT:-29500}"

# Reduce CUDA fragmentation OOM on 24 GB GPUs.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"
