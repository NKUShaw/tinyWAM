#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TRAIN_PYTHON="${SLIM_TRAIN_PYTHON:-$ROOT/.venv/bin/python}"
NPROC="${NPROC_PER_NODE:-8}"
CALVIN_VIDEO_BACKEND="${CALVIN_VIDEO_BACKEND:-}"
CALVIN_ACTION_KEY="${CALVIN_ACTION_KEY:-}"
PIPELINE_ID="${SLIM_PIPELINE_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${SLIM_PIPELINE_LOG_ROOT:-$ROOT/logs/calvin_pipeline/$PIPELINE_ID}"
STAGE1_NAME="calvin_abc_d_stage1_idm0125_fdm1_${PIPELINE_ID}"
STAGE2_NAME="calvin_abc_d_stage2_policy_40ep_${PIPELINE_ID}"
STAGE1_RUN="$ROOT/checkpoints/stage1/$STAGE1_NAME"
STAGE2_RUN="$ROOT/checkpoints/stage2/$STAGE2_NAME"
STAGE1_CKPT="$STAGE1_RUN/checkpoints/epoch_3_pytorch_model.pt"

: "${DINOV2_MODEL_DIR:?Set DINOV2_MODEL_DIR to the local DINOv2 model directory}"
: "${T5_MODEL_DIR:?Set T5_MODEL_DIR to the local T5 model directory}"
: "${CALVIN_LANG_DATASET:?Set CALVIN_LANG_DATASET to the CALVIN LeRobot dataset}"
export DINOV2_MODEL_DIR T5_MODEL_DIR CALVIN_LANG_DATASET
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
export SLIM_CACHE_DIR="${SLIM_CACHE_DIR:-$ROOT/.cache/slim}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

mkdir -p "$LOG_ROOT"
printf '%s\n' "$$" >"$LOG_ROOT/pipeline.pid"

require_path() {
  if [[ ! -e "$1" ]]; then
    printf 'Missing required path: %s\n' "$1" >&2
    exit 1
  fi
}

require_path "$TRAIN_PYTHON"
require_path "$DINOV2_MODEL_DIR/dinov2_vitb14_pretrain.pth"
require_path "$T5_MODEL_DIR/model.safetensors"
require_path "$TORCH_HOME/hub/facebookresearch_dinov2_main/hubconf.py"
require_path "$CALVIN_LANG_DATASET/meta/info.json"
if [[ -z "$CALVIN_ACTION_KEY" ]]; then
  CALVIN_ACTION_KEY="$(
    "$TRAIN_PYTHON" - "$CALVIN_LANG_DATASET/meta/info.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    info = json.load(handle)
features = info.get("features", {})
print("action.rel" if "action.rel" in features else "action")
PY
  )"
fi
export CALVIN_ACTION_KEY
if [[ -z "$CALVIN_VIDEO_BACKEND" || "$CALVIN_VIDEO_BACKEND" == "frames" ]]; then
  require_path "$CALVIN_LANG_DATASET/frames"
fi
if [[ -n "${SLIM_LANGUAGE_CACHE:-}" ]]; then
  export SLIM_LANGUAGE_CACHE
  require_path "$SLIM_LANGUAGE_CACHE/calvin_ABC_D_lerobot_t5small_lang_emb.npy"
fi

BACKEND_ARGS=()
if [[ -n "$CALVIN_VIDEO_BACKEND" ]]; then
  BACKEND_ARGS+=(--data.video_backend="$CALVIN_VIDEO_BACKEND")
fi

{
  printf 'pipeline_id=%s\n' "$PIPELINE_ID"
  printf 'python=%s\n' "$TRAIN_PYTHON"
  printf 'nproc_per_node=%s\n' "$NPROC"
  printf 'cuda_visible_devices=%s\n' "$CUDA_VISIBLE_DEVICES"
  printf 'video_backend_override=%s\n' "${CALVIN_VIDEO_BACKEND:-<config-default>}"
  printf 'action_key=%s\n' "$CALVIN_ACTION_KEY"
  printf 'stage1_run=%s\n' "$STAGE1_RUN"
  printf 'stage2_run=%s\n' "$STAGE2_RUN"
  printf 'stage1_checkpoint=%s\n' "$STAGE1_CKPT"
} >"$LOG_ROOT/run.env"

printf '[pipeline] Stage 1 -> %s\n' "$STAGE1_RUN"
"$TRAIN_PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc-per-node="$NPROC" \
  -m slim.training.stage1 \
  --config configs/calvin/stage1_idm0125_fdm1.yaml \
  "${BACKEND_ARGS[@]}" \
  --run.timestamp=false \
  --run.name="$STAGE1_NAME" \
  2>&1 | tee "$LOG_ROOT/stage1.log"

if [[ ! -f "$STAGE1_CKPT" ]]; then
  printf 'Stage 1 finished without expected checkpoint: %s\n' "$STAGE1_CKPT" >&2
  exit 1
fi

printf '[pipeline] Stage 2 -> %s\n' "$STAGE2_RUN"
"$TRAIN_PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc-per-node="$NPROC" \
  -m slim.training.stage2 \
  --config configs/calvin/stage2_policy_40ep.yaml \
  "${BACKEND_ARGS[@]}" \
  --init-checkpoint "$STAGE1_CKPT" \
  --run.timestamp=false \
  --run.name="$STAGE2_NAME" \
  2>&1 | tee "$LOG_ROOT/stage2.log"

touch "$LOG_ROOT/complete"
printf '[pipeline] complete: %s\n' "$PIPELINE_ID"
