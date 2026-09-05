#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OFFICIAL_PID_FILE="${OFFICIAL_PID_FILE:-$ROOT/logs/official_stage1/train_4gpu.pid}"
OFFICIAL_FINAL_CHECKPOINT="${OFFICIAL_FINAL_CHECKPOINT:-$ROOT/checkpoints/stage1_official/libero_all90_official_stage1_h8_4gpu_gb128_0902_152225/checkpoints/epoch_3_pytorch_model.pt}"
QUEUE_DIR="${DINOV3_QUEUE_DIR:-$ROOT/logs/dinov3_stage1_queue}"
STATUS_FILE="$QUEUE_DIR/status.txt"
TRAIN_PID_FILE="$QUEUE_DIR/train_4gpu.pid"

mkdir -p "$QUEUE_DIR"

write_status() {
  printf '%s status=%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "${2:-}" > "$STATUS_FILE"
}

if [[ ! -f "$OFFICIAL_PID_FILE" ]]; then
  write_status failed "missing_official_pid_file=$OFFICIAL_PID_FILE"
  exit 1
fi

official_pid="$(tr -d '[:space:]' < "$OFFICIAL_PID_FILE")"
if [[ ! "$official_pid" =~ ^[0-9]+$ ]]; then
  write_status failed "invalid_official_pid=$official_pid"
  exit 1
fi

write_status waiting "official_pid=$official_pid"
while kill -0 "$official_pid" 2>/dev/null; do
  process_state="$(ps -o stat= -p "$official_pid" 2>/dev/null | tr -d '[:space:]' || true)"
  [[ "$process_state" == Z* ]] && break
  sleep 30
done

if [[ ! -s "$OFFICIAL_FINAL_CHECKPOINT" ]]; then
  write_status blocked "official_final_checkpoint_missing=$OFFICIAL_FINAL_CHECKPOINT"
  exit 2
fi

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
train_log="$QUEUE_DIR/train_4gpu_${stamp}.log"
write_status launching "log=$train_log"

cd "$ROOT"
env \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
  WANDB_MODE="${WANDB_MODE:-offline}" \
  LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-$ROOT/datasets/libero}" \
  TORCH_HOME="${TORCH_HOME:-$ROOT/assets/torch-cache}" \
  DINOV3_VITS16_MODEL_DIR="${DINOV3_VITS16_MODEL_DIR:-/mnt/disk2/xiaoyang/WorldModels/tinyWAM/weights/dinov3_vits16_lvd1689m}" \
  T5_MODEL_DIR="${T5_MODEL_DIR:-$ROOT/assets/t5-small}" \
  SLIM_CACHE_DIR="${SLIM_CACHE_DIR:-$ROOT/cache/slim}" \
  SLIM_LANGUAGE_CACHE="${SLIM_LANGUAGE_CACHE:-$ROOT/cache/language_embeddings}" \
  "$ROOT/.venv/bin/torchrun" --standalone --nproc-per-node=4 \
    -m slim.training.stage1 \
    --config configs/libero/stage1_dinov3_vits16_d384_h6.yaml \
    --run.root="$ROOT/checkpoints/stage1_dinov3" \
    --data.per_device_batch_size=8 \
    --training.gradient_accumulation_steps=4 \
    > "$train_log" 2>&1 &
train_pid=$!
printf '%s\n' "$train_pid" > "$TRAIN_PID_FILE"
write_status running "pid=$train_pid log=$train_log global_batch=128"

set +e
wait "$train_pid"
exit_code=$?
set -e
if [[ "$exit_code" -eq 0 ]]; then
  write_status complete "pid=$train_pid log=$train_log"
else
  write_status failed "pid=$train_pid exit_code=$exit_code log=$train_log"
fi
exit "$exit_code"
