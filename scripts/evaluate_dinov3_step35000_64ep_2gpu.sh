#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/disk2/xiaoyang/WorldModels/SLIM"
DEFAULT_CHECKPOINT="$ROOT/checkpoints/stage2_dinov3/libero_all_stage2_dinov3_vits16_d384_h6_40ep_4gpu_gb128/checkpoints/steps_35000_pytorch_model.pt"
DEFAULT_OUTPUT_ROOT="$ROOT/outputs/dinov3_stage2_step35000_eval64"
CHECKPOINT="${1:-$DEFAULT_CHECKPOINT}"
OUTPUT_ROOT="${2:-$DEFAULT_OUTPUT_ROOT}"
LOG_ROOT="$OUTPUT_ROOT/logs"
CONFIG_ROOT="$OUTPUT_ROOT/.libero_config"
LIBERO_HOME="/mnt/disk2/xiaoyang/WorldModels/LIBERO"
TRAIN_PYTHON="$ROOT/.venv/bin/python"
LIBERO_PYTHON="/home/bhui/miniconda3/envs/libero/bin/python"
BASE_PORT="${BASE_PORT:-13100}"
TASK_COUNT="${TASK_COUNT:-8}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-2}"
POLICY_SEED="${POLICY_SEED:-42}"

mkdir -p "$LOG_ROOT" "$CONFIG_ROOT"
if [[ ! -s "$CHECKPOINT" ]]; then
  echo "Missing checkpoint: $CHECKPOINT" >&2
  exit 2
fi

cat >"$CONFIG_ROOT/config.yaml" <<EOF
assets: "$LIBERO_HOME/libero/libero/assets"
bddl_files: "$LIBERO_HOME/libero/libero/bddl_files"
benchmark_root: "$LIBERO_HOME/libero/libero"
datasets: "$LIBERO_HOME/libero/datasets"
init_states: "$LIBERO_HOME/libero/libero/init_files"
EOF

SERVER_PIDS=()
cleanup() {
  if [[ "${#SERVER_PIDS[@]}" -gt 0 ]]; then
    kill "${SERVER_PIDS[@]}" 2>/dev/null || true
    wait "${SERVER_PIDS[@]}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

wait_for_port() {
  local port="$1"
  "$LIBERO_PYTHON" - "$port" <<'PY'
import socket
import sys
import time

port = int(sys.argv[1])
for _ in range(600):
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            raise SystemExit(0)
    time.sleep(1)
raise SystemExit(f"Timed out waiting for policy server on port {port}")
PY
}

for local_rank in 0 1; do
  port=$((BASE_PORT + local_rank))
  if timeout 2 bash -c "exec 3<>/dev/tcp/127.0.0.1/$port" >/dev/null 2>&1; then
    echo "Port already in use: $port" >&2
    exit 3
  fi
  env \
    CUDA_VISIBLE_DEVICES="$local_rank" \
    DINOV3_VITS16_MODEL_DIR="/mnt/disk2/xiaoyang/WorldModels/tinyWAM/weights/dinov3_vits16_lvd1689m" \
    T5_MODEL_DIR="$ROOT/assets/t5-small" \
    LIBERO_DATA_ROOT="$ROOT/datasets/libero" \
    SLIM_CACHE_DIR="$ROOT/cache/slim" \
    SLIM_LANGUAGE_CACHE="$ROOT/cache/language_embeddings" \
    "$TRAIN_PYTHON" -m slim.serving.server \
      --checkpoint "$CHECKPOINT" --port "$port" --bf16 --idle-timeout -1 \
      --seed "$POLICY_SEED" \
      >"$LOG_ROOT/server_gpu${local_rank}.log" 2>&1 &
  SERVER_PIDS+=("$!")
done

wait_for_port "$BASE_PORT"
wait_for_port "$((BASE_PORT + 1))"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LD_PRELOAD="${LD_PRELOAD:-/usr/lib/x86_64-linux-gnu/libstdc++.so.6}"

suites=(libero_10 libero_spatial libero_object libero_goal)
for suite in "${suites[@]}"; do
  suite_root="$OUTPUT_ROOT/$suite"
  mkdir -p "$suite_root"
  client_pids=()
  for shard in 0 1; do
    task_start=$((shard * TASK_COUNT / 2))
    task_end=$(((shard + 1) * TASK_COUNT / 2 - 1))
    port=$((BASE_PORT + shard))
    visible_devices="$shard"
    if [[ "$shard" -eq 1 ]]; then
      visible_devices="1,0"
    fi
    env \
      CUDA_VISIBLE_DEVICES="$visible_devices" \
      MUJOCO_EGL_DEVICE_ID=0 \
      LIBERO_CONFIG_PATH="$CONFIG_ROOT" \
      TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
      PYTHONPATH="$ROOT:$LIBERO_HOME" \
      "$LIBERO_PYTHON" -m slim.evaluation.libero.evaluate \
        --checkpoint "$CHECKPOINT" \
        --host 127.0.0.1 --port "$port" \
        --task_suite_name "$suite" \
        --task_start_idx "$task_start" --task_end_idx "$task_end" \
        --num_trials_per_task "$EPISODES_PER_TASK" \
        --episode_start_idx 0 --episode_end_idx "$((EPISODES_PER_TASK - 1))" \
        --action_chunk_size 8 --send-state --save_video 0 \
        --seed 7 --video_out_path "$suite_root" \
        >"$LOG_ROOT/${suite}_shard${shard}.log" 2>&1 &
    client_pids+=("$!")
  done
  suite_status=0
  for pid in "${client_pids[@]}"; do
    if ! wait "$pid"; then
      suite_status=1
    fi
  done
  "$TRAIN_PYTHON" -m slim.evaluation.summarize \
    "$OUTPUT_ROOT" --output "$OUTPUT_ROOT/summary.txt"
  if [[ "$suite_status" -ne 0 ]]; then
    echo "Evaluation failed for suite: $suite" >&2
    exit 4
  fi
done

"$TRAIN_PYTHON" -m slim.evaluation.summarize \
  "$OUTPUT_ROOT" --output "$OUTPUT_ROOT/summary.txt"
