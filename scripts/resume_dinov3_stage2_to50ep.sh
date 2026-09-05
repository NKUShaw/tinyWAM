#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/disk2/xiaoyang/WorldModels/SLIM"
RUN_DIR="$ROOT/checkpoints/stage2_dinov3/libero_all_stage2_dinov3_vits16_d384_h6_40ep_4gpu_gb128"
RESUME_STATE="$RUN_DIR/states/step_00081200"
QUEUE_ROOT="$ROOT/logs/dinov3_stage2_resume_to50ep"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-4}"
mkdir -p "$QUEUE_ROOT"

if [[ ! -s "$RESUME_STATE/trainer_state.json" ]]; then
  printf 'missing resume state: %s\n' "$RESUME_STATE" >&2
  exit 2
fi

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
train_log="$QUEUE_ROOT/train_resume_step81200_4gpu_${stamp}.log"
printf '%s\n' "$train_log" > "$QUEUE_ROOT/latest_log.txt"
printf '[%s] launching true resume state=%s target_total_epochs=50 log=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$RESUME_STATE" "$train_log" | tee "$QUEUE_ROOT/status.txt"

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
    --config configs/libero/stage2_dinov3_vits16_d384_h6_50ep.yaml \
    --resume-state "$RESUME_STATE" \
    --data.per_device_batch_size=8 \
    --data.num_workers="$TRAIN_NUM_WORKERS" \
    --training.gradient_accumulation_steps=4 \
    >"$train_log" 2>&1
train_status=$?
set -e

printf '[%s] exit_code=%s log=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$train_status" "$train_log" | tee "$QUEUE_ROOT/status.txt"
exit "$train_status"
