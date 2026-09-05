#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/disk2/xiaoyang/WorldModels/SLIM"
RUN="$ROOT/checkpoints/stage2_dinov3/libero_all_stage2_dinov3_vits16_d384_h6_40ep_4gpu_gb128"
QUEUE_ROOT="$ROOT/logs/dinov3_eval400_queue"
STATUS_FILE="$QUEUE_ROOT/status.txt"
mkdir -p "$QUEUE_ROOT"

log() {
  local message="[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
  printf '%s\n' "$message"
  printf '%s\n' "$message" >"$STATUS_FILE"
}

for step in 10000 20000 30000 40000; do
  checkpoint="$RUN/checkpoints/steps_${step}_pytorch_model.pt"
  output="$ROOT/outputs/dinov3_stage2_step${step}_eval400"
  eval_log="$QUEUE_ROOT/step${step}.log"

  if [[ ! -s "$checkpoint" ]]; then
    log "failed missing_checkpoint=$checkpoint"
    exit 2
  fi
  if [[ -e "$output" ]]; then
    log "failed output_already_exists=$output"
    exit 3
  fi

  log "running step=$step episodes=400 output=$output"
  env \
    BASE_PORT=13200 \
    TASK_COUNT=10 \
    EPISODES_PER_TASK=10 \
    POLICY_SEED=42 \
    bash "$ROOT/scripts/evaluate_dinov3_step35000_64ep_2gpu.sh" \
      "$checkpoint" "$output" >"$eval_log" 2>&1

  completed="$(find "$output" -mindepth 2 -maxdepth 2 -type f -name 'rollout_*.txt' | wc -l | tr -d ' ')"
  if [[ "$completed" -ne 400 ]]; then
    log "failed step=$step completed=$completed expected=400 log=$eval_log"
    exit 4
  fi
  log "finished step=$step summary=$output/summary.txt"
done

log "complete steps=10000,20000,30000,40000 episodes_per_checkpoint=400"
