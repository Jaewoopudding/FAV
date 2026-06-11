#!/bin/bash
# Sana-Sprint × FAV × HPSv2 (multi-GPU data-parallel).
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash bash_scripts/sana_fav_hpsv2.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
"${PYTHON_BIN}" -m accelerate.commands.launch \
    --num_processes "${NPROC}" --main_process_port "${MAIN_PORT}" \
    -m scripts.train \
    +experiment=sana_fav_hpsv2 \
    seed=0 "$@"
