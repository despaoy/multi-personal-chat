#!/usr/bin/env bash
set -euo pipefail

ROOT="${MULTIPERSONAL_LAB_ROOT:-${QQCHAT_LAB_ROOT:-/root/autodl-tmp}}"
PROJECT="${KISAKI_PROJECT_ROOT:-$ROOT/qqchat-enhanced}"
PYTHON="${MULTIPERSONAL_LAB_PYTHON:-${QQCHAT_LAB_PYTHON:-$ROOT/envs/qqchat-gpu-qwen3/bin/python}}"
GPU="${1:-0}"
PORT="${KISAKI_CHECKPOINT_PORT:-8001}"
MODEL="$ROOT/runtime/models/Qwen3-8B-Instruct"
ADAPTER_ROOT="${KISAKI_CHECKPOINT_ADAPTER_ROOT:-$ROOT/runtime/loras/kisaki/r1v4/e1/seed42}"
CHECKPOINT_LABELS_CSV="${KISAKI_CHECKPOINT_LABELS:-100,150,200,232}"
SELECTION_ROOT="$ROOT/runtime/experiments/kisaki/r1v4/e1/checkpoint-selection"
OUTPUT="${KISAKI_CHECKPOINT_OUTPUT:-$SELECTION_ROOT}"
DATASET="${KISAKI_CHECKPOINT_DATASET:-$SELECTION_ROOT/devset30.json}"
PROMPT="$PROJECT/backend/data/character_dialogues/kisaki_system_prompt_v3.txt"
COMPOSE_RUNTIME_POLICY="${KISAKI_CHECKPOINT_COMPOSE_POLICY:-true}"
LOG="$OUTPUT/vllm.log"
PID_FILE="$OUTPUT/vllm.pid"
COMPLETE_MARKER="$OUTPUT/evaluation.complete"
VLLM_PID=""

IFS=',' read -r -a CHECKPOINT_LABELS <<<"$CHECKPOINT_LABELS_CSV"
declare -a LORA_MODULES=()

adapter_path() {
  local label="$1"
  if [[ "$label" == final ]]; then
    printf '%s/final\n' "$ADAPTER_ROOT"
  else
    printf '%s/checkpoint-%s\n' "$ADAPTER_ROOT" "$label"
  fi
}

result_label() {
  local label="$1"
  if [[ "$label" == final ]]; then
    printf 'final\n'
  else
    printf 'checkpoint-%s\n' "$label"
  fi
}

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
[[ "$COMPOSE_RUNTIME_POLICY" == true || "$COMPOSE_RUNTIME_POLICY" == false ]] || {
  echo "invalid_compose_runtime_policy=$COMPOSE_RUNTIME_POLICY" >&2
  exit 2
}
(( ${#CHECKPOINT_LABELS[@]} > 0 )) || { echo "checkpoint_labels_empty=true" >&2; exit 2; }
for label in "${CHECKPOINT_LABELS[@]}"; do
  [[ "$label" == final || "$label" =~ ^[0-9]+$ ]] || {
    echo "invalid_checkpoint_label=$label" >&2
    exit 2
  }
  path="$(adapter_path "$label")"
  [[ -f "$path/adapter_config.json" ]] || {
    echo "adapter_missing=$path" >&2
    exit 2
  }
  LORA_MODULES+=("kisaki-e1-$label=$path")
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
  --max-loras 1 \
  --max-cpu-loras "${#LORA_MODULES[@]}" \
  --lora-modules "${LORA_MODULES[@]}" \
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
    --temperature 0
    --max-tokens 256
    --top-p 0.9
    --repetition-penalty 1.0
    --frequency-penalty 0.0
    --timeout 120
    --gpu "$GPU"
  )
  if [[ "$COMPOSE_RUNTIME_POLICY" == true ]]; then
    args+=(--compose-runtime-policy)
  else
    args+=(--no-compose-runtime-policy)
  fi
  if [[ -n "$adapter_path" ]]; then
    args+=(--adapter-path "$adapter_path")
  fi
  PYTHONPATH="$PROJECT/backend" "$PYTHON" -m evaluation.character_benchmark_v3 "${args[@]}"
}

run_benchmark prompt_only kisaki-e1-base
for label in "${CHECKPOINT_LABELS[@]}"; do
  run_benchmark "$(result_label "$label")" "kisaki-e1-$label" "$(adapter_path "$label")"
done

printf 'completed_at=%s\ncompose_runtime_policy=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "$COMPOSE_RUNTIME_POLICY" \
  >"$COMPLETE_MARKER"
echo "checkpoint_selection_complete=$OUTPUT"
