#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 INIT_CHECKPOINT [CONFIG] [OVERRIDES...]" >&2
  exit 2
fi

CHECKPOINT="$1"
CONFIG="${2:-configs/libero/stage2_policy_h8_40ep.yaml}"
shift
[[ $# -gt 0 ]] && shift

unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT \
  GROUP_RANK ROLE_RANK TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT

torchrun --standalone --nproc-per-node="${NPROC_PER_NODE:-8}" \
  -m slim.training.stage2 \
  --config "$CONFIG" \
  --init-checkpoint "$CHECKPOINT" \
  "$@"
