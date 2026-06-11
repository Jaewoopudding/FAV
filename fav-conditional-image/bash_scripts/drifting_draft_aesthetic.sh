#!/bin/bash
# drifting_draft_aesthetic (seed=0, JAX backend). Runs on GPU 0; uses the JAX conda env.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"
export TPU_SKIP_MDS_QUERY=1

CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
"${JAX_PYTHON_BIN}" -m scripts.train_drifting \
    +experiment=drifting_draft_aesthetic \
    seed=0 "$@"
