#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 CHECKPOINT OUTPUT_ROOT [BASE_PORT]" >&2
  exit 2
fi

CHECKPOINT="$(realpath "$1")"
OUTPUT_ROOT="$(realpath -m "$2")"
BASE_PORT="${3:-12000}"
EVAL_GPU_COUNT="${EVAL_GPU_COUNT:-8}"
TRAIN_PYTHON="${SLIM_TRAIN_PYTHON:-python}"
LIBERO_PYTHON="${SLIM_LIBERO_PYTHON:-python}"
PLUS_PYTHON="${SLIM_PLUS_PYTHON:-python}"
CHECK_INTERVAL="${CHECK_INTERVAL:-60}"
GPU_STABLE_CHECKS="${GPU_STABLE_CHECKS:-2}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-900}"
PROGRESS_INTERVAL_SECONDS="${PROGRESS_INTERVAL_SECONDS:-300}"
STANDARD_PROGRESS_INTERVAL_SECONDS="${STANDARD_PROGRESS_INTERVAL_SECONDS:-60}"
PLUS_PROCESS_COUNT="${PLUS_PROCESS_COUNT:-32}"
STANDARD_PROCESS_COUNT="${STANDARD_PROCESS_COUNT:-$EVAL_GPU_COUNT}"
EVAL_PHASE="${EVAL_PHASE:-all}"
POLICY_SEED="${POLICY_SEED:-42}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLIM_LIBERO_HOME="${SLIM_LIBERO_HOME:-}"
SLIM_PLUS_HOME="${SLIM_PLUS_HOME:-}"
STANDARD_CONFIG_DIR="$OUTPUT_ROOT/.libero_standard"
PLUS_CONFIG_DIR="$OUTPUT_ROOT/.libero_plus"

cd "$REPO_ROOT"
ulimit -n 1048576

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*"
}

print_standard_progress() {
  local suite="$1"
  local root="$2"
  local completed success remaining completion_rate success_rate
  completed="$(find "$root" -maxdepth 1 -type f \
    -name "rollout_${suite}_task*_episode*_*.txt" | wc -l | tr -d ' ')"
  success="$(find "$root" -maxdepth 1 -type f \
    -name "rollout_${suite}_task*_episode*_success.txt" | wc -l | tr -d ' ')"
  remaining=$((500 - completed))
  ((remaining < 0)) && remaining=0
  completion_rate="$(awk -v value="$completed" \
    'BEGIN {printf "%.2f", (value * 100 / 500)}')"
  success_rate="$(awk -v value="$success" -v total="$completed" \
    'BEGIN {printf "%.2f", (total > 0 ? value * 100 / total : 0)}')"
  log "[progress][LIBERO][$suite] completed=$completed/500 (${completion_rate}%), remaining=$remaining, success=$success, success_rate=${success_rate}%"
}

monitor_standard_progress() {
  local suite="$1"
  local root="$2"
  while true; do
    sleep "$STANDARD_PROGRESS_INTERVAL_SECONDS"
    print_standard_progress "$suite" "$root"
  done
}

print_plus_progress() {
  local suite="$1"
  local root="$2"
  local total="$3"
  local pending_this_run="$4"
  local completed success remaining completion_rate success_rate
  completed="$(find "$root" -maxdepth 1 -type f \
    -name "rollout_${suite}_task*_episode*_*.txt" | wc -l | tr -d ' ')"
  success="$(find "$root" -maxdepth 1 -type f \
    -name "rollout_${suite}_task*_episode*_success.txt" | wc -l | tr -d ' ')"
  remaining=$((total - completed))
  ((remaining < 0)) && remaining=0
  completion_rate="$(awk -v value="$completed" -v total="$total" \
    'BEGIN {printf "%.2f", (total > 0 ? value * 100 / total : 0)}')"
  success_rate="$(awk -v value="$success" -v total="$completed" \
    'BEGIN {printf "%.2f", (total > 0 ? value * 100 / total : 0)}')"
  log "[progress][$suite] completed=$completed/$total (${completion_rate}%), remaining=$remaining, success=$success, success_rate=${success_rate}%, pending_this_run=$pending_this_run"
}

monitor_plus_progress() {
  local suite="$1"
  local root="$2"
  local total="$3"
  local pending_this_run="$4"
  while true; do
    sleep "$PROGRESS_INTERVAL_SECONDS"
    print_plus_progress "$suite" "$root" "$total" "$pending_this_run"
  done
}

write_plus_suite_summary() {
  local suite="$1"
  local root="$2"
  PYTHONPATH="$REPO_ROOT" "$PLUS_PYTHON" -m slim.evaluation.libero_plus.analyze \
    --results-root "$root" \
    --classification "$PLUS_CLASSIFICATION" \
    --suites "$suite" \
    --model-name "$EVAL_MODEL_NAME" \
    --output "$root/summary_metrics.txt" \
    --csv "$root/rollouts.csv"
}

prepare_libero_config() {
  local config_dir="$1"
  local checkout="$2"
  local benchmark_root="$checkout/libero/libero"
  mkdir -p "$config_dir"
  cat >"$config_dir/config.yaml" <<EOF
assets: "$benchmark_root/assets"
bddl_files: "$benchmark_root/bddl_files"
benchmark_root: "$benchmark_root"
datasets: "$checkout/libero/datasets"
init_states: "$benchmark_root/init_files"
EOF
}

if [[ -z "$SLIM_LIBERO_HOME" || -z "$SLIM_PLUS_HOME" ]]; then
  echo "SLIM_LIBERO_HOME and SLIM_PLUS_HOME must both be set." >&2
  exit 2
fi
prepare_libero_config "$STANDARD_CONFIG_DIR" "$SLIM_LIBERO_HOME"
prepare_libero_config "$PLUS_CONFIG_DIR" "$SLIM_PLUS_HOME"
PLUS_CLASSIFICATION="${LIBERO_PLUS_CLASSIFICATION:-$SLIM_PLUS_HOME/libero/libero/benchmark/task_classification.json}"
EVAL_MODEL_NAME="$(basename "$(dirname "$(dirname "$CHECKPOINT")")")_$(basename "$CHECKPOINT")"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export LD_PRELOAD="${LD_PRELOAD:-/usr/lib/x86_64-linux-gnu/libstdc++.so.6}"

eval_visible_devices() {
  local gpu="$1"
  if [[ "$gpu" -eq 0 ]]; then
    printf "0"
  else
    printf "%s,0" "$gpu"
  fi
}

# On this host EGL selects devices by NVIDIA minor number, which is not
# necessarily the same as the nvidia-smi index (here 0,1,2,3 -> 3,2,1,0).
egl_device_id() {
  local gpu="$1"
  nvidia-smi -q -i "$gpu" \
    | awk -F: '/Minor Number/ && !found {gsub(/[[:space:]]/, "", $2); print $2; found=1}'
}

wait_for_checkpoint() {
  local previous_size=-1
  while true; do
    if [[ -f "$CHECKPOINT" ]]; then
      local size
      size="$(stat -c %s "$CHECKPOINT")"
      if [[ "$size" -gt 0 && "$size" -eq "$previous_size" ]]; then
        return
      fi
      previous_size="$size"
    fi
    sleep "$CHECK_INTERVAL"
  done
}

wait_for_gpus() {
  local stable=0
  while [[ "$stable" -lt "$GPU_STABLE_CHECKS" ]]; do
    local processes
    processes="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^$/d')"
    if [[ -z "$processes" ]]; then
      stable=$((stable + 1))
    else
      stable=0
    fi
    [[ "$stable" -ge "$GPU_STABLE_CHECKS" ]] || sleep "$CHECK_INTERVAL"
  done
}

wait_for_port() {
  local port="$1"
  "$LIBERO_PYTHON" - "$port" "$SERVER_START_TIMEOUT" <<'PY'
import socket
import sys
import time

port = int(sys.argv[1])
timeout = int(sys.argv[2])
for _ in range(timeout):
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            raise SystemExit(0)
    time.sleep(1)
raise SystemExit(f"Timed out waiting for port {port}")
PY
}

assert_ports_free() {
  local port
  for port in $(seq "$BASE_PORT" "$((BASE_PORT + EVAL_GPU_COUNT - 1))"); do
    if timeout 2 bash -c "exec 3<>/dev/tcp/127.0.0.1/$port" \
      >/dev/null 2>&1; then
      echo "Port is already in use: $port" >&2
      return 1
    fi
  done
}

SERVER_PIDS=()
cleanup() {
  if [[ "${#SERVER_PIDS[@]}" -gt 0 ]]; then
    kill "${SERVER_PIDS[@]}" 2>/dev/null || true
    wait "${SERVER_PIDS[@]}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

start_servers() {
  local log_dir="$1"
  mkdir -p "$log_dir"
  assert_ports_free
  SERVER_PIDS=()
  for gpu in $(seq 0 "$((EVAL_GPU_COUNT - 1))"); do
    local port=$((BASE_PORT + gpu))
    CUDA_VISIBLE_DEVICES="$gpu" "$TRAIN_PYTHON" -m slim.serving.server \
      --checkpoint "$CHECKPOINT" --port "$port" --bf16 --idle-timeout -1 \
      --seed "$POLICY_SEED" \
      >"$log_dir/server_gpu${gpu}.log" 2>&1 &
    SERVER_PIDS+=("$!")
  done
  for gpu in $(seq 0 "$((EVAL_GPU_COUNT - 1))"); do
    wait_for_port $((BASE_PORT + gpu))
  done
}

stop_servers() {
  cleanup
  SERVER_PIDS=()
}

run_standard() {
  local root="$OUTPUT_ROOT/libero"
  mkdir -p "$root/logs"
  start_servers "$root/logs"
  local suites=(libero_10 libero_spatial libero_object libero_goal)
  for suite in "${suites[@]}"; do
    mkdir -p "$root/$suite"
    local completed
    completed="$(find "$root/$suite" -maxdepth 1 -type f \
      -name "rollout_${suite}_task*_episode*_*.txt" | wc -l | tr -d ' ')"
    if [[ "$completed" -eq 500 ]]; then
      log "[resume][LIBERO][$suite] already complete; skipping"
      print_standard_progress "$suite" "$root/$suite"
      continue
    fi
    local worker_count="$STANDARD_PROCESS_COUNT"
    if [[ "$worker_count" -gt 50 ]]; then
      worker_count=50
    fi
    local pids=()
    for worker in $(seq 0 "$((worker_count - 1))"); do
      local server_gpu=$((worker % EVAL_GPU_COUNT))
      local start=$((worker * 50 / worker_count))
      local end=$(((worker + 1) * 50 / worker_count - 1))
      local visible_devices
      local egl_id
      egl_id="$(egl_device_id "$server_gpu")"
      # Evaluation uses remote inference, so expose only the physical EGL
      # minor selected for this renderer. This also satisfies robosuite's
      # legacy CUDA_VISIBLE_DEVICES/MUJOCO_EGL_DEVICE_ID assertion.
      visible_devices="$egl_id"
      CUDA_VISIBLE_DEVICES="$visible_devices" MUJOCO_EGL_DEVICE_ID="$egl_id" \
        LIBERO_CONFIG_PATH="$STANDARD_CONFIG_DIR" \
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
        PYTHONPATH="${REPO_ROOT}${SLIM_LIBERO_HOME:+:${SLIM_LIBERO_HOME}}" \
        "$LIBERO_PYTHON" -m slim.evaluation.libero.evaluate \
        --checkpoint "$CHECKPOINT" \
        --host 127.0.0.1 --port $((BASE_PORT + server_gpu)) \
        --task_suite_name "$suite" --num_trials_per_task 50 \
        --episode_start_idx "$start" --episode_end_idx "$end" \
        --action_chunk_size 8 --send-state --save_video 0 \
        --video_out_path "$root/$suite" \
        >"$root/logs/${suite}_worker${worker}_gpu${server_gpu}.log" 2>&1 &
      pids+=("$!")
    done
    print_standard_progress "$suite" "$root/$suite"
    local monitor_pid=""
    if [[ "$STANDARD_PROGRESS_INTERVAL_SECONDS" -gt 0 ]]; then
      monitor_standard_progress "$suite" "$root/$suite" &
      monitor_pid="$!"
    fi
    local wait_status=0
    set +e
    for pid in "${pids[@]}"; do
      wait "$pid"
      local status="$?"
      if [[ "$status" -ne 0 && "$wait_status" -eq 0 ]]; then
        wait_status="$status"
      fi
    done
    set -e
    if [[ -n "$monitor_pid" ]]; then
      kill "$monitor_pid" 2>/dev/null || true
      wait "$monitor_pid" 2>/dev/null || true
    fi
    print_standard_progress "$suite" "$root/$suite"
    if [[ "$wait_status" -ne 0 ]]; then
      echo "$suite evaluation failed with status $wait_status" >&2
      return "$wait_status"
    fi
  done
  stop_servers
  PYTHONPATH="$REPO_ROOT" "$LIBERO_PYTHON" -m slim.evaluation.summarize \
    "$root" --output "$root/summary.txt"
}

run_plus() {
  if [[ ! -f "$PLUS_CLASSIFICATION" ]]; then
    echo "Missing LIBERO-Plus task classification: $PLUS_CLASSIFICATION" >&2
    return 1
  fi
  local root="$OUTPUT_ROOT/libero_plus"
  mkdir -p "$root/logs"
  start_servers "$root/logs"
  local suites=(libero_10 libero_spatial libero_object libero_goal)
  local counts=(2519 2402 2518 2591)
  for index in "${!suites[@]}"; do
    local suite="${suites[$index]}"
    local count="${counts[$index]}"
    mkdir -p "$root/$suite"
    local task_id_dir="$root/.task_ids/$suite"
    local completed_file="$task_id_dir/completed.txt"
    local pending_file="$task_id_dir/pending.txt"
    mkdir -p "$task_id_dir"
    find "$root/$suite" -maxdepth 1 -type f \
      -name "rollout_${suite}_task*_episode*_*.txt" -printf '%f\n' \
      | sed -n 's/.*_task\([0-9][0-9]*\)_episode.*/\1/p' \
      | sort -n -u >"$completed_file"
    awk 'FILENAME == ARGV[1] {completed[$1] = 1; next} !($1 in completed)' \
      "$completed_file" <(seq 0 "$((count - 1))") >"$pending_file"
    local pending_this_run
    pending_this_run="$(wc -l <"$pending_file" | tr -d ' ')"
    if [[ "$pending_this_run" -eq 0 ]]; then
      print_plus_progress "$suite" "$root/$suite" "$count" 0
      write_plus_suite_summary "$suite" "$root/$suite"
      continue
    fi

    local worker_count="$PLUS_PROCESS_COUNT"
    if [[ "$pending_this_run" -lt "$worker_count" ]]; then
      worker_count="$pending_this_run"
    fi
    local pids=()
    for worker in $(seq 0 "$((worker_count - 1))"); do
      local server_gpu=$((worker % EVAL_GPU_COUNT))
      local task_ids_file="$task_id_dir/worker${worker}.txt"
      awk -v workers="$worker_count" -v rank="$worker" \
        '((NR - 1) % workers) == rank' "$pending_file" >"$task_ids_file"
      local visible_devices
      local egl_id
      egl_id="$(egl_device_id "$server_gpu")"
      visible_devices="$egl_id"
      CUDA_VISIBLE_DEVICES="$visible_devices" MUJOCO_EGL_DEVICE_ID="$egl_id" \
        LIBERO_CONFIG_PATH="$PLUS_CONFIG_DIR" \
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
        PYTHONPATH="${REPO_ROOT}${SLIM_PLUS_HOME:+:${SLIM_PLUS_HOME}}" \
        "$PLUS_PYTHON" -m slim.evaluation.libero_plus.evaluate \
        --checkpoint "$CHECKPOINT" \
        --host 127.0.0.1 --port $((BASE_PORT + server_gpu)) \
        --task_suite_name "$suite" \
        --task_ids_file "$task_ids_file" \
        --num_trials_per_task 1 --action_chunk_size 8 \
        --resume 1 --save_video 0 \
        --video_out_path "$root/$suite" \
        >"$root/logs/${suite}_worker${worker}_gpu${server_gpu}.log" 2>&1 &
      pids+=("$!")
    done
    print_plus_progress "$suite" "$root/$suite" "$count" "$pending_this_run"
    local monitor_pid=""
    if [[ "$PROGRESS_INTERVAL_SECONDS" -gt 0 ]]; then
      monitor_plus_progress "$suite" "$root/$suite" "$count" "$pending_this_run" &
      monitor_pid="$!"
    fi
    local wait_status=0
    set +e
    for pid in "${pids[@]}"; do
      wait "$pid"
      local status="$?"
      if [[ "$status" -ne 0 && "$wait_status" -eq 0 ]]; then
        wait_status="$status"
      fi
    done
    set -e
    if [[ -n "$monitor_pid" ]]; then
      kill "$monitor_pid" 2>/dev/null || true
      wait "$monitor_pid" 2>/dev/null || true
    fi
    print_plus_progress "$suite" "$root/$suite" "$count" "$pending_this_run"
    write_plus_suite_summary "$suite" "$root/$suite"
    if [[ "$wait_status" -ne 0 ]]; then
      echo "$suite evaluation failed with status $wait_status" >&2
      return "$wait_status"
    fi
  done
  stop_servers
  PYTHONPATH="$REPO_ROOT" "$PLUS_PYTHON" -m slim.evaluation.libero_plus.analyze \
    --results-root "$root" \
    --classification "$PLUS_CLASSIFICATION" \
    --model-name "$EVAL_MODEL_NAME" \
    --output "$root/summary.txt" \
    --csv "$root/rollouts.csv"
}

wait_for_checkpoint
case "$EVAL_PHASE" in
  all)
    wait_for_gpus
    run_standard
    wait_for_gpus
    run_plus
    ;;
  standard)
    wait_for_gpus
    run_standard
    ;;
  plus)
    wait_for_gpus
    run_plus
    ;;
  *)
    echo "Unknown EVAL_PHASE=$EVAL_PHASE; expected all, standard, or plus." >&2
    exit 2
    ;;
esac
