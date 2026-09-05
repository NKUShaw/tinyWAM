#!/usr/bin/env bash
set -euo pipefail

unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT \
  GROUP_RANK ROLE_RANK TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT

torchrun --standalone --nproc-per-node="${NPROC_PER_NODE:-8}" \
  -m slim.training.stage1 \
  --config "${1:-configs/libero/stage1_idm0125_fdm1_h8.yaml}" \
  "${@:2}"
