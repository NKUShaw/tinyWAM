#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/disk2/xiaoyang/WorldModels/SLIM"
STAGE1_PID="${STAGE1_PID:-3485661}"
STAGE1_RUN="${STAGE1_RUN:-$ROOT/checkpoints/stage1_dinov3/libero_all90_stage1_dinov3_vits16_d384_h6_4gpu_gb128_0903_010855}"
STAGE1_LOG="${STAGE1_LOG:-$ROOT/logs/dinov3_stage1/train_4gpu_gb128_20260903T010850Z.log}"
STAGE1_CHECKPOINT="$STAGE1_RUN/checkpoints/epoch_3_pytorch_model.pt"

STAGE2_ROOT="$ROOT/checkpoints/stage2_dinov3"
STAGE2_NAME="libero_all_stage2_dinov3_vits16_d384_h6_40ep_4gpu_gb128"
STAGE2_RUN="$STAGE2_ROOT/$STAGE2_NAME"
STAGE2_CHECKPOINT="$STAGE2_RUN/checkpoints/epoch_40_pytorch_model.pt"

QUEUE_ROOT="$ROOT/logs/dinov3_stage2_queue"
OUTPUT_ROOT="$ROOT/outputs/dinov3_vits16_d384_h6_stage2_closed_loop"
STATUS_FILE="$QUEUE_ROOT/status.txt"
mkdir -p "$QUEUE_ROOT" "$STAGE2_ROOT" "$OUTPUT_ROOT"

log() {
  local message="[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
  printf '%s\n' "$message"
  printf '%s\n' "$message" >"$STATUS_FILE"
}

wait_for_stable_file() {
  local path="$1"
  local previous=-1
  while true; do
    if [[ -s "$path" ]]; then
      local current
      current="$(stat -c %s "$path")"
      if [[ "$current" -eq "$previous" ]]; then
        return 0
      fi
      previous="$current"
    fi
    sleep 30
  done
}

log "waiting_stage1 pid=$STAGE1_PID checkpoint=$STAGE1_CHECKPOINT"
while kill -0 "$STAGE1_PID" 2>/dev/null; do
  sleep 30
done

if [[ ! -s "$STAGE1_CHECKPOINT" ]]; then
  log "failed stage1_exited_without_final_checkpoint log=$STAGE1_LOG"
  exit 2
fi
if rg -q 'Traceback|CUDA out of memory|(^|[^a-zA-Z])NaN([^a-zA-Z]|$)|Exception|ERROR' "$STAGE1_LOG"; then
  log "failed stage1_log_contains_error log=$STAGE1_LOG"
  exit 3
fi
wait_for_stable_file "$STAGE1_CHECKPOINT"

if [[ -e "$STAGE2_RUN" ]]; then
  log "failed stage2_run_already_exists path=$STAGE2_RUN"
  exit 4
fi

stage2_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
stage2_log="$QUEUE_ROOT/stage2_4gpu_${stage2_stamp}.log"
log "launching_stage2 log=$stage2_log global_batch=128"

cd "$ROOT"
set +e
env \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  WANDB_MODE=offline \
  LIBERO_DATA_ROOT="$ROOT/datasets/libero" \
  TORCH_HOME="$ROOT/assets/torch-cache" \
  DINOV3_VITS16_MODEL_DIR="/mnt/disk2/xiaoyang/WorldModels/tinyWAM/weights/dinov3_vits16_lvd1689m" \
  T5_MODEL_DIR="$ROOT/assets/t5-small" \
  SLIM_CACHE_DIR="$ROOT/cache/slim" \
  SLIM_LANGUAGE_CACHE="$ROOT/cache/language_embeddings" \
  PYTHONUNBUFFERED=1 \
  "$ROOT/.venv/bin/torchrun" --standalone --nproc-per-node=4 \
    -m slim.training.stage2 \
    --config configs/libero/stage2_dinov3_vits16_d384_h6_40ep.yaml \
    --init-checkpoint "$STAGE1_CHECKPOINT" \
    --run.root="$STAGE2_ROOT" \
    --run.name="$STAGE2_NAME" \
    --run.timestamp=false \
    --data.per_device_batch_size=8 \
    --training.gradient_accumulation_steps=4 \
    >"$stage2_log" 2>&1
stage2_status=$?
set -e
if [[ "$stage2_status" -ne 0 ]]; then
  log "failed stage2_exit_code=$stage2_status log=$stage2_log"
  exit "$stage2_status"
fi

wait_for_stable_file "$STAGE2_CHECKPOINT"
log "stage2_complete checkpoint=$STAGE2_CHECKPOINT"

eval_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
eval_log="$QUEUE_ROOT/libero_standard_4gpu_${eval_stamp}.log"
log "launching_closed_loop_libero log=$eval_log episodes=2000 video=disabled"

set +e
env \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  EVAL_PHASE=standard \
  EVAL_GPU_COUNT=4 \
  SLIM_TRAIN_PYTHON="$ROOT/.venv/bin/python" \
  SLIM_LIBERO_PYTHON="/home/bhui/miniconda3/envs/libero/bin/python" \
  SLIM_PLUS_PYTHON="/home/bhui/miniconda3/envs/libero/bin/python" \
  SLIM_LIBERO_HOME="/mnt/disk2/xiaoyang/WorldModels/LIBERO" \
  SLIM_PLUS_HOME="/mnt/disk2/xiaoyang/WorldModels/LIBERO-plus" \
  CHECK_INTERVAL=30 \
  STANDARD_PROGRESS_INTERVAL_SECONDS=60 \
  bash "$ROOT/scripts/evaluate_all_8gpu.sh" \
    "$STAGE2_CHECKPOINT" "$OUTPUT_ROOT" 12000 \
    >"$eval_log" 2>&1
eval_status=$?
set -e
if [[ "$eval_status" -ne 0 ]]; then
  log "failed closed_loop_exit_code=$eval_status log=$eval_log"
  exit "$eval_status"
fi

log "complete summary=$OUTPUT_ROOT/libero/summary.txt"
