#!/bin/bash
# imm_draft_aesthetic (seed=0). Override GPUs/port via CUDA_VISIBLE_DEVICES / MAIN_PORT.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
"${PYTHON_BIN}" -m accelerate.commands.launch \
    --num_processes "${NPROC}" --main_process_port "${MAIN_PORT}" \
    -m scripts.train \
    +experiment=imm_draft_aesthetic \
    seed=0 "$@"
