#!/usr/bin/env bash
set -euo pipefail

ROOT="${MULTIPERSONAL_LAB_ROOT:-${QQCHAT_LAB_ROOT:-/root/autodl-tmp}}"
PROJECT="${KISAKI_PROJECT_ROOT:-$ROOT/qqchat-enhanced}"
PYTHON="${MULTIPERSONAL_LAB_PYTHON:-${QQCHAT_LAB_PYTHON:-$ROOT/envs/qqchat-gpu-qwen3/bin/python}}"
GPU="${1:-0}"
PORT="${KISAKI_CHECKPOINT_PORT:-8001}"
MODEL="$ROOT/runtime/models/Qwen3-8B-Instruct"
ADAPTER_ROOT="$ROOT/runtime/loras/kisaki/r1v4/e1/seed42"
OUTPUT="$ROOT/runtime/experiments/kisaki/r1v4/e1/checkpoint-selection"
DATASET="$OUTPUT/devset30.json"
PROMPT="$PROJECT/backend/data/character_dialogues/kisaki_system_prompt_v3.txt"
LOG="$OUTPUT/vllm.log"
PID_FILE="$OUTPUT/vllm.pid"
COMPLETE_MARKER="$OUTPUT/evaluation.complete"
VLLM_PID=""

declare -a STEPS=(100 150 200 232)

cleanup() {
  if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
}
trap cleanup EXIT INT TERM

[[ -x "$PYTHON" ]] || { echo "python_missing=$PYTHON" >&2; exit 2; }
[[ -f "$MODEL/config.json" ]] || { echo "base_model_missing=$MODEL" >&2; exit 2; }
[[ -f "$DATASET" ]] || { echo "development_dataset_missing=$DATASET" >&2; exit 2; }
[[ -f "$PROMPT" ]] || { echo "system_prompt_missing=$PROMPT" >&2; exit 2; }
for step in "${STEPS[@]}"; do
  [[ -f "$ADAPTER_ROOT/checkpoint-$step/adapter_config.json" ]] || {
    echo "adapter_missing=$ADAPTER_ROOT/checkpoint-$step" >&2
    exit 2
  }
done

mkdir -p "$OUTPUT"
if [[ -f "$COMPLETE_MARKER" ]]; then
  echo "checkpoint_selection_already_complete=$COMPLETE_MARKER"
  exit 0
fi
if curl --silent --fail --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
  echo "port_in_use=$PORT" >&2
  exit 2
fi

used_memory=$(nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
utilization=$(nvidia-smi --id="$GPU" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
if (( used_memory >= 2048 || utilization >= 10 )); then
  echo "gpu_not_idle=index:$GPU,memory_mb:$used_memory,utilization:$utilization" >&2
  exit 2
fi

cd "$PROJECT"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 \
  --port "$PORT" \
  --model "$MODEL" \
  --served-model-name kisaki-e1-base \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --enable-lora \
  --max-lora-rank 32 \
  --max-loras 4 \
  --lora-modules \
    "kisaki-e1-step100=$ADAPTER_ROOT/checkpoint-100" \
    "kisaki-e1-step150=$ADAPTER_ROOT/checkpoint-150" \
    "kisaki-e1-step200=$ADAPTER_ROOT/checkpoint-200" \
    "kisaki-e1-step232=$ADAPTER_ROOT/checkpoint-232" \
  >"$LOG" 2>&1 &
VLLM_PID=$!
printf '%s\n' "$VLLM_PID" >"$PID_FILE"

ready=false
for _ in $(seq 1 240); do
  if curl --silent --fail --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
    ready=true
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    break
  fi
  sleep 2
done
if [[ "$ready" != true ]]; then
  echo "vllm_start_failed=$LOG" >&2
  exit 2
fi

run_benchmark() {
  local label="$1"
  local served_model="$2"
  local adapter_path="${3:-}"
  local result="$OUTPUT/$label.json"
  local -a args
  if [[ -f "$result" ]]; then
    echo "refusing_to_overwrite_existing_result=$result" >&2
    exit 2
  fi
  args=(
    --dataset "$DATASET"
    --output "$result"
    --base-url "http://127.0.0.1:$PORT"
    --model "$served_model"
    --model-path "$MODEL"
    --system-prompt-file "$PROMPT"
    --compose-runtime-policy
    --temperature 0
    --max-tokens 256
    --top-p 0.9
    --repetition-penalty 1.0
    --frequency-penalty 0.0
    --timeout 120
    --gpu "$GPU"
  )
  if [[ -n "$adapter_path" ]]; then
    args+=(--adapter-path "$adapter_path")
  fi
  PYTHONPATH="$PROJECT/backend" "$PYTHON" -m evaluation.character_benchmark_v3 "${args[@]}"
}

run_benchmark prompt_only kisaki-e1-base
for step in "${STEPS[@]}"; do
  run_benchmark "checkpoint-$step" "kisaki-e1-step$step" "$ADAPTER_ROOT/checkpoint-$step"
done

printf 'completed_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$COMPLETE_MARKER"
echo "checkpoint_selection_complete=$OUTPUT"
