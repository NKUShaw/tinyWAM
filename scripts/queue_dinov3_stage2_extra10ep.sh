#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/disk2/xiaoyang/WorldModels/SLIM"
WAIT_PID="${WAIT_PID:-}"
START_NOW="${START_NOW:-0}"
ALLOW_SHARED_GPUS="${ALLOW_SHARED_GPUS:-0}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-4}"
SOURCE_RUN="$ROOT/checkpoints/stage2_dinov3/libero_all_stage2_dinov3_vits16_d384_h6_40ep_4gpu_gb128"
SOURCE_CHECKPOINT="$SOURCE_RUN/checkpoints/epoch_40_pytorch_model.pt"
RUN_ROOT="$ROOT/checkpoints/stage2_dinov3"
RUN_NAME="libero_all_stage2_dinov3_vits16_d384_h6_extra10ep_from_epoch40_4gpu_gb128"
RUN_DIR="$RUN_ROOT/$RUN_NAME"
QUEUE_ROOT="$ROOT/logs/dinov3_stage2_extra10ep_queue"
STATUS_FILE="$QUEUE_ROOT/status.txt"
mkdir -p "$QUEUE_ROOT"

log() {
  local message="[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
  printf '%s\n' "$message"
  printf '%s\n' "$message" >"$STATUS_FILE"
}

if [[ "$START_NOW" != "1" ]]; then
  if [[ -z "$WAIT_PID" ]]; then
    log "failed WAIT_PID_required_unless_START_NOW_1"
    exit 1
  fi
  log "waiting_full_evaluation pid=$WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 30
  done
fi

if [[ ! -s "$SOURCE_CHECKPOINT" ]]; then
  log "failed source_checkpoint_missing path=$SOURCE_CHECKPOINT"
  exit 2
fi
if [[ -e "$RUN_DIR" ]]; then
  log "failed run_already_exists path=$RUN_DIR"
  exit 3
fi

# Do not collide with any unrelated GPU workload that may start meanwhile.
if [[ "$ALLOW_SHARED_GPUS" != "1" ]]; then
  while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^$/d')" ]]; do
    log "waiting_gpus_to_be_free"
    sleep 30
  done
fi

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
train_log="$QUEUE_ROOT/train_4gpu_${stamp}.log"
log "launching source=$SOURCE_CHECKPOINT epochs=10 global_batch=128 data_workers_per_rank=$TRAIN_NUM_WORKERS shared_gpus=$ALLOW_SHARED_GPUS log=$train_log"

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
    --config configs/libero/stage2_dinov3_vits16_d384_h6_extra10ep.yaml \
    --init-checkpoint "$SOURCE_CHECKPOINT" \
    --run.root="$RUN_ROOT" \
    --run.name="$RUN_NAME" \
    --run.timestamp=false \
    --data.per_device_batch_size=8 \
    --data.num_workers="$TRAIN_NUM_WORKERS" \
    --training.gradient_accumulation_steps=4 \
    >"$train_log" 2>&1
train_status=$?
set -e

if [[ "$train_status" -ne 0 ]]; then
  log "failed training_exit_code=$train_status log=$train_log"
  exit "$train_status"
fi

final_checkpoint="$RUN_DIR/checkpoints/epoch_10_pytorch_model.pt"
if [[ ! -s "$final_checkpoint" ]]; then
  log "failed final_checkpoint_missing path=$final_checkpoint"
  exit 4
fi
log "complete checkpoint=$final_checkpoint log=$train_log"
