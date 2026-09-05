#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/disk2/xiaoyang/WorldModels/SLIM"
RUN="$ROOT/checkpoints/stage2_dinov3/libero_all_stage2_dinov3_vits16_d384_h6_40ep_4gpu_gb128"
CHECKPOINT="$RUN/checkpoints/epoch_40_pytorch_model.pt"
OUTPUT_ROOT="$ROOT/outputs/dinov3_vits16_d384_h6_stage2_closed_loop"
QUEUE_ROOT="$ROOT/logs/dinov3_final_full_eval_queue"
STATUS_FILE="$QUEUE_ROOT/status.txt"
STANDARD_SUMMARY="$OUTPUT_ROOT/libero/summary.txt"
PLUS_SUMMARY="$OUTPUT_ROOT/libero_plus/summary.txt"

mkdir -p "$QUEUE_ROOT"

log() {
  local message="[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
  printf '%s\n' "$message"
  printf '%s\n' "$message" >"$STATUS_FILE"
}

if [[ ! -s "$CHECKPOINT" ]]; then
  log "failed final_checkpoint_missing checkpoint=$CHECKPOINT"
  exit 2
fi

if [[ -s "$STANDARD_SUMMARY" && -s "$PLUS_SUMMARY" ]]; then
  log "complete evaluations_already_finished standard=$STANDARD_SUMMARY plus=$PLUS_SUMMARY"
  exit 0
fi

eval_log="$QUEUE_ROOT/libero_and_plus_full.log"
log "launching_full_evaluation checkpoint=$CHECKPOINT libero_episodes=2000 libero_plus_cases=10030 log=$eval_log"

cd "$ROOT"
set +e
env \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  EVAL_PHASE=all \
  EVAL_GPU_COUNT=4 \
  STANDARD_PROCESS_COUNT=32 \
  PLUS_PROCESS_COUNT=32 \
  DINOV3_VITS16_MODEL_DIR="/mnt/disk2/xiaoyang/WorldModels/tinyWAM/weights/dinov3_vits16_lvd1689m" \
  T5_MODEL_DIR="$ROOT/assets/t5-small" \
  LIBERO_DATA_ROOT="$ROOT/datasets/libero" \
  SLIM_CACHE_DIR="$ROOT/cache/slim" \
  SLIM_LANGUAGE_CACHE="$ROOT/cache/language_embeddings" \
  SLIM_TRAIN_PYTHON="$ROOT/.venv/bin/python" \
  SLIM_LIBERO_PYTHON="/home/bhui/miniconda3/envs/libero/bin/python" \
  SLIM_PLUS_PYTHON="/home/bhui/miniconda3/envs/libero/bin/python" \
  SLIM_LIBERO_HOME="/mnt/disk2/xiaoyang/WorldModels/LIBERO" \
  SLIM_PLUS_HOME="/mnt/disk2/xiaoyang/WorldModels/LIBERO-plus" \
  CHECK_INTERVAL=30 \
  PROGRESS_INTERVAL_SECONDS=300 \
  bash "$ROOT/scripts/evaluate_all_8gpu.sh" \
    "$CHECKPOINT" "$OUTPUT_ROOT" 12100 \
    >"$eval_log" 2>&1
eval_status=$?
set -e

if [[ "$eval_status" -ne 0 ]]; then
  log "failed full_evaluation_exit_code=$eval_status log=$eval_log"
  exit "$eval_status"
fi
if [[ ! -s "$STANDARD_SUMMARY" || ! -s "$PLUS_SUMMARY" ]]; then
  log "failed summary_missing standard=$STANDARD_SUMMARY plus=$PLUS_SUMMARY"
  exit 4
fi
log "complete standard=$STANDARD_SUMMARY plus=$PLUS_SUMMARY"
