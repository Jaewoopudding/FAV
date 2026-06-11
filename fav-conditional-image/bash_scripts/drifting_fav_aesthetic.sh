#!/bin/bash
# drifting_fav_aesthetic (seed=0, JAX backend). Uses the JAX conda env.
# Requires >=2 GPUs: the FAV/SVGD path shards across devices (hsdp). Defaults to GPU 0,1.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"
export TPU_SKIP_MDS_QUERY=1

CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
"${JAX_PYTHON_BIN}" -m scripts.train_drifting \
    +experiment=drifting_fav_aesthetic \
    seed=0 "$@"
