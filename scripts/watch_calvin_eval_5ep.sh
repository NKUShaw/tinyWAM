#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 1 ]]; then
  printf 'Usage: %s STAGE2_RUN_DIR\n' "$0" >&2
  exit 2
fi
RUN_DIR="$(realpath -m "$1")"
CHECKPOINT_DIR="$RUN_DIR/checkpoints"
RUN_NAME="$(basename "$RUN_DIR")"

TRAIN_PYTHON="${SLIM_TRAIN_PYTHON:-$ROOT/.venv/bin/python}"
CALVIN_PYTHON="${SLIM_CALVIN_PYTHON:?Set SLIM_CALVIN_PYTHON to the CALVIN Python executable}"
CALVIN_ROOT="${CALVIN_ROOT:?Set CALVIN_ROOT to the CALVIN checkout}"
CALVIN_DATASET="${CALVIN_DATASET:-$CALVIN_ROOT/dataset/task_ABC_D}"
ACTION_STATS="${ACTION_STATS:-$RUN_DIR/action_stats_calvin_ABC_D_lerobot.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/evaluation_outputs/calvin_5ep/$RUN_NAME}"
PIPELINE_PID_FILE="${PIPELINE_PID_FILE:-}"

NUM_SEQUENCES="${NUM_SEQUENCES:-1000}"
NUM_WORKERS="${NUM_WORKERS:-24}"
EXEC_STRIDE="${EXEC_STRIDE:-12}"
SERVER_GPU="${SERVER_GPU:-0}"
PORT="${PORT:-}"
POLL_SECONDS="${POLL_SECONDS:-60}"
STABLE_SECONDS="${STABLE_SECONDS:-10}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-900}"
MAX_EVAL_ATTEMPTS="${MAX_EVAL_ATTEMPTS:-3}"
RETRY_SECONDS="${RETRY_SECONDS:-60}"
EPOCHS=( ${EPOCHS:-5 10 15 20 25 30 35 40} )

: "${DINOV2_MODEL_DIR:?Set DINOV2_MODEL_DIR to the local DINOv2 model directory}"
: "${T5_MODEL_DIR:?Set T5_MODEL_DIR to the local T5 model directory}"
: "${CALVIN_LANG_DATASET:?Set CALVIN_LANG_DATASET to the CALVIN LeRobot dataset}"
export DINOV2_MODEL_DIR T5_MODEL_DIR CALVIN_LANG_DATASET
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
export SLIM_CACHE_DIR="${SLIM_CACHE_DIR:-$ROOT/.cache/slim}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUTPUT_ROOT"
exec 9>"$OUTPUT_ROOT/.watcher.lock"
if ! flock -n 9; then
  printf 'Another CALVIN watcher already owns %s\n' "$OUTPUT_ROOT" >&2
  exit 2
fi
printf '%s\n' "$$" >"$OUTPUT_ROOT/watcher.pid"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*"
}

require_path() {
  if [[ ! -e "$1" ]]; then
    log "ERROR: required path does not exist: $1"
    exit 2
  fi
}

port_in_use() {
  (exec 3<>"/dev/tcp/127.0.0.1/$1") >/dev/null 2>&1
}

pick_free_port() {
  local candidate offset
  if [[ -n "$PORT" ]]; then
    if port_in_use "$PORT"; then
      log "ERROR: requested PORT=$PORT is already in use"
      return 1
    fi
    printf '%s\n' "$PORT"
    return
  fi
  for offset in $(seq 0 199); do
    candidate=$((12000 + RANDOM % 40000 + offset))
    if ((candidate > 65000)); then
      continue
    fi
    if ! port_in_use "$candidate"; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  log "ERROR: unable to find a free policy-server port"
  return 1
}

pipeline_is_alive() {
  local pid
  [[ -n "$PIPELINE_PID_FILE" && -f "$PIPELINE_PID_FILE" ]] || return 0
  read -r pid <"$PIPELINE_PID_FILE"
  kill -0 "$pid" 2>/dev/null
}

summary_is_complete() {
  local summary="$1"
  [[ -f "$summary" ]] || return 1
  "$CALVIN_PYTHON" - "$summary" "$NUM_SEQUENCES" <<'PY' >/dev/null 2>&1
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
expected = int(sys.argv[2])
assert payload["num_sequences"] == expected
assert len(payload["results"]) == expected
assert set(payload["success_rates"]) == {"1", "2", "3", "4", "5"}
PY
}

wait_for_checkpoint() {
  local checkpoint="$1"
  local previous_size=-1 size
  while true; do
    if [[ -f "$checkpoint" ]]; then
      size="$(stat -c %s "$checkpoint")"
      if [[ "$size" -gt 0 && "$size" -eq "$previous_size" ]]; then
        log "checkpoint is stable: $checkpoint ($size bytes)"
        return 0
      fi
      previous_size="$size"
      sleep "$STABLE_SECONDS"
      continue
    fi
    if ! pipeline_is_alive; then
      log "ERROR: training pipeline ended before checkpoint appeared: $checkpoint"
      return 1
    fi
    log "waiting for checkpoint: $checkpoint"
    sleep "$POLL_SECONDS"
  done
}

wait_for_server() {
  local server_pid="$1" server_log="$2"
  local waited=0
  while ((waited < SERVER_START_TIMEOUT)); do
    if grep -q "server listening" "$server_log" 2>/dev/null; then
      return 0
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      return 1
    fi
    sleep 2
    waited=$((waited + 2))
  done
  return 1
}

SERVER_PID=""
WORKER_PIDS=()

stop_children() {
  local pid
  for pid in "${WORKER_PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${WORKER_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  WORKER_PIDS=()
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
}

trap stop_children EXIT INT TERM

rebuild_summary() {
  "$CALVIN_PYTHON" - "$OUTPUT_ROOT" "${EPOCHS[@]}" <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
epochs = [int(value) for value in sys.argv[2:]]
lines = ["epoch\tavg_length\tSR1\tSR2\tSR3\tSR4\tSR5"]
for epoch in epochs:
    path = root / f"ep{epoch}" / "summary.json"
    if not path.is_file():
        continue
    payload = json.loads(path.read_text(encoding="utf-8"))
    rates = payload["success_rates"]
    values = [
        epoch,
        payload["avg_length"],
        *(rates[str(length)] for length in range(1, 6)),
    ]
    lines.append("\t".join(str(value) for value in values))
temporary = root / f".summary.tsv.{os.getpid()}.tmp"
temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
temporary.replace(root / "summary.tsv")
PY
}

run_epoch() {
  local epoch="$1"
  local checkpoint="$CHECKPOINT_DIR/epoch_${epoch}_pytorch_model.pt"
  local epoch_dir="$OUTPUT_ROOT/ep${epoch}"
  local server_log="$epoch_dir/server.log"
  local port fail_count worker index

  if summary_is_complete "$epoch_dir/summary.json"; then
    log "ep${epoch}: complete result already exists; skipping"
    return 0
  fi

  mkdir -p "$epoch_dir"
  rm -f "$epoch_dir"/shard_*.json "$epoch_dir"/summary.json \
    "$epoch_dir"/worker_*.log "$server_log"
  port="$(pick_free_port)" || return 1

  log "ep${epoch}: starting policy server on GPU $SERVER_GPU, port $port"
  CUDA_VISIBLE_DEVICES="$SERVER_GPU" \
    "$TRAIN_PYTHON" -m slim.serving.server \
      --checkpoint "$checkpoint" \
      --port "$port" \
      --bf16 \
      --idle-timeout -1 \
      >"$server_log" 2>&1 &
  SERVER_PID=$!
  if ! wait_for_server "$SERVER_PID" "$server_log"; then
    log "ep${epoch}: policy server failed to start; see $server_log"
    tail -n 40 "$server_log" 2>/dev/null || true
    stop_children
    return 1
  fi

  log "ep${epoch}: evaluating $NUM_SEQUENCES sequences with $NUM_WORKERS workers"
  WORKER_PIDS=()
  for ((index = 0; index < NUM_WORKERS; index++)); do
    CUDA_VISIBLE_DEVICES="" \
      "$CALVIN_PYTHON" -m slim.evaluation.calvin.evaluate \
        --dataset-path "$CALVIN_DATASET" \
        --action-stats "$ACTION_STATS" \
        --dataset-name calvin_ABC_D_lerobot \
        --num-sequences "$NUM_SEQUENCES" \
        --num-shards "$NUM_WORKERS" \
        --shard-index "$index" \
        --action-horizon 12 \
        --exec-stride "$EXEC_STRIDE" \
        --host 127.0.0.1 \
        --port "$port" \
        --output-dir "$epoch_dir" \
        >"$epoch_dir/worker_${index}.log" 2>&1 &
    WORKER_PIDS+=("$!")
  done

  fail_count=0
  for worker in "${WORKER_PIDS[@]}"; do
    wait "$worker" || fail_count=$((fail_count + 1))
  done
  WORKER_PIDS=()
  if [[ "$fail_count" -ne 0 ]]; then
    log "ep${epoch}: $fail_count/$NUM_WORKERS workers failed"
    stop_children
    return 1
  fi

  local aggregate_status=0
  if [[ "$NUM_WORKERS" -gt 1 ]]; then
    "$CALVIN_PYTHON" -m slim.evaluation.calvin.evaluate \
      --aggregate \
      --num-sequences "$NUM_SEQUENCES" \
      --num-shards "$NUM_WORKERS" \
      --output-dir "$epoch_dir" \
      >"$epoch_dir/aggregate.log" 2>&1
    aggregate_status=$?
  fi
  stop_children
  if [[ "$aggregate_status" -ne 0 ]] || ! summary_is_complete "$epoch_dir/summary.json"; then
    log "ep${epoch}: shard aggregation failed; see $epoch_dir/aggregate.log"
    return 1
  fi

  rebuild_summary
  log "ep${epoch}: complete; $(tail -n 1 "$OUTPUT_ROOT/summary.tsv")"
}

require_path "$TRAIN_PYTHON"
require_path "$CALVIN_PYTHON"
require_path "$CALVIN_DATASET/validation"
require_path "$ACTION_STATS"
require_path "$RUN_DIR/config.yaml"
require_path "$DINOV2_MODEL_DIR/dinov2_vitb14_pretrain.pth"
require_path "$T5_MODEL_DIR/model.safetensors"
require_path "$SLIM_LANGUAGE_CACHE/calvin_ABC_D_lerobot_t5small_lang_emb.npy"

if [[ "$NUM_WORKERS" -le 0 || "$NUM_SEQUENCES" -le 0 ]]; then
  log "ERROR: NUM_WORKERS and NUM_SEQUENCES must be positive"
  exit 2
fi

NCORES="${EVAL_NCORES:-$(nproc)}"
LP_THREADS=$((NCORES / NUM_WORKERS))
if [[ "$LP_THREADS" -lt 1 ]]; then
  LP_THREADS=1
fi
export LP_NUM_THREADS="$LP_THREADS"

log "run_dir=$RUN_DIR"
log "output=$OUTPUT_ROOT"
log "epochs=${EPOCHS[*]} sequences=$NUM_SEQUENCES workers=$NUM_WORKERS stride=$EXEC_STRIDE"

rebuild_summary
for epoch in "${EPOCHS[@]}"; do
  if summary_is_complete "$OUTPUT_ROOT/ep${epoch}/summary.json"; then
    log "ep${epoch}: complete result already exists; skipping"
    continue
  fi
  wait_for_checkpoint "$CHECKPOINT_DIR/epoch_${epoch}_pytorch_model.pt" || exit 1
  success=0
  for attempt in $(seq 1 "$MAX_EVAL_ATTEMPTS"); do
    log "ep${epoch}: evaluation attempt $attempt/$MAX_EVAL_ATTEMPTS"
    if run_epoch "$epoch"; then
      success=1
      break
    fi
    log "ep${epoch}: attempt $attempt failed"
    if [[ "$attempt" -lt "$MAX_EVAL_ATTEMPTS" ]]; then
      sleep "$RETRY_SECONDS"
    fi
  done
  if [[ "$success" -ne 1 ]]; then
    log "ERROR: ep${epoch} failed after $MAX_EVAL_ATTEMPTS attempts"
    exit 1
  fi
done

rebuild_summary
log "all requested epochs are complete: $OUTPUT_ROOT/summary.tsv"
