#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/disk2/xiaoyang/WorldModels/SLIM"
WAIT_PID_FILE="$ROOT/logs/dinov3_stage2_resume_to50ep/supervisor.pid"
SOURCE_RUN="$ROOT/checkpoints/stage2_dinov3/libero_all_stage2_dinov3_vits16_d384_h6_40ep_4gpu_gb128"
SOURCE_CHECKPOINT="$SOURCE_RUN/checkpoints/epoch_50_pytorch_model.pt"
RUN_ROOT="$ROOT/checkpoints/stage2_dinov3"
RUN_NAME="libero_all_stage2_dinov3_vits16_d384_h6_wrist_hr_adapter_only_40ep_4gpu_gb128"
RUN_DIR="$RUN_ROOT/$RUN_NAME"
QUEUE_ROOT="$ROOT/logs/dinov3_wrist_hr_adapter_40ep_queue"
mkdir -p "$QUEUE_ROOT"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee "$QUEUE_ROOT/status.txt"
}

if [[ -s "$WAIT_PID_FILE" ]]; then
  wait_pid="$(cat "$WAIT_PID_FILE")"
  log "waiting_for_stage2_to50ep pid=$wait_pid"
  while kill -0 "$wait_pid" 2>/dev/null; do sleep 30; done
fi

if [[ ! -s "$SOURCE_CHECKPOINT" ]]; then
  log "failed missing_epoch50_checkpoint path=$SOURCE_CHECKPOINT"
  exit 2
fi
if [[ -e "$RUN_DIR" ]]; then
  log "failed run_already_exists path=$RUN_DIR"
  exit 3
fi

# Do not collide with unrelated jobs that appeared while this queue was waiting.
while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^$/d')" ]]; do
  log "waiting_for_free_gpus"
  sleep 30
done

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
train_log="$QUEUE_ROOT/train_4gpu_${stamp}.log"
printf '%s\n' "$train_log" > "$QUEUE_ROOT/latest_log.txt"
log "launching source=$SOURCE_CHECKPOINT trainable=HR_adapter_only epochs=40 global_batch=128 log=$train_log"

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
    --config configs/libero/stage2_dinov3_vits16_d384_h6_wrist_hr_adapter_40ep.yaml \
    --init-checkpoint "$SOURCE_CHECKPOINT" \
    --run.root="$RUN_ROOT" \
    --run.name="$RUN_NAME" \
    --run.timestamp=false \
    --data.per_device_batch_size=8 \
    --data.num_workers=8 \
    --training.gradient_accumulation_steps=4 \
    >"$train_log" 2>&1
status=$?
set -e

if [[ "$status" -ne 0 ]]; then
  log "failed exit_code=$status log=$train_log"
  exit "$status"
fi
log "complete checkpoint=$RUN_DIR/checkpoints/epoch_40_pytorch_model.pt log=$train_log"
