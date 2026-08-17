#!/usr/bin/env bash
set -euo pipefail

ROOT="${MULTIPERSONAL_LAB_ROOT:-${QQCHAT_LAB_ROOT:?set MULTIPERSONAL_LAB_ROOT or QQCHAT_LAB_ROOT to the lab root}}"
PROJECT=$ROOT/multi-personal-chat
PYTHON="${MULTIPERSONAL_LAB_PYTHON:-${QQCHAT_LAB_PYTHON:-$ROOT/envs/qqchat-gpu-qwen3/bin/python}}"
GPU=${1:?usage: lab-run-kisaki-r3.sh GPU BEST_ADAPTER}
BEST_ADAPTER=${2:?usage: lab-run-kisaki-r3.sh GPU BEST_ADAPTER}
BF16=$ROOT/runtime/models/Qwen3-8B-Instruct
AWQ=$ROOT/runtime/models/Qwen3-8B-Instruct-AWQ
PROMPTS=$PROJECT/backend/data/character_dialogues/experiments/research/kisaki_r3_prompts_v1.json
SYSTEM_PROMPT=$PROJECT/backend/data/character_dialogues/kisaki_system_prompt_v3.txt
OUTPUT=$ROOT/runtime/experiments/kisaki/r3
LOG_ROOT=$ROOT/runtime/logs
MERGED=$ROOT/runtime/models/kisaki-r3-best-merged
PORT=${KISAKI_R3_PORT:-8001}
VLLM_PID=''

for path in "$BF16/config.json" "$AWQ/config.json" "$BEST_ADAPTER/adapter_config.json" "$PROMPTS" "$SYSTEM_PROMPT"; do
  [[ -f "$path" ]] || { echo "required_file_missing=$path" >&2; exit 2; }
done
[[ "$BEST_ADAPTER" == "$ROOT/"* ]] || { echo "adapter_outside_allowed_root=$BEST_ADAPTER" >&2; exit 2; }
mkdir -p "$OUTPUT" "$LOG_ROOT"
cleanup(){ if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then kill "$VLLM_PID" 2>/dev/null || true; wait "$VLLM_PID" 2>/dev/null || true; fi; VLLM_PID=''; }
trap cleanup EXIT INT TERM

run_variant(){
  local label=$1 model=$2 quant=$3 adapter=${4:-} optional=${5:-false}
  local result_dir=$OUTPUT/$label log=$LOG_ROOT/kisaki_r3_$label.log
  if [[ -f "$result_dir/complete.marker" ]]; then echo "skip_completed=$label"; return 0; fi
  if curl --silent --fail --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null; then echo "port_in_use=$PORT" >&2; exit 2; fi
  mkdir -p "$result_dir"
  local served="r3-$label" dtype=bfloat16
  local args=(--host 127.0.0.1 --port "$PORT" --model "$model" --served-model-name "$served" --max-model-len 4096 --gpu-memory-utilization 0.90)
  if [[ "$quant" == awq ]]; then args+=(--quantization awq); dtype=auto; else args+=(--dtype bfloat16); fi
  if [[ -n "$adapter" ]]; then args+=(--enable-lora --max-lora-rank 32 --lora-modules "$served=$adapter"); fi
  local started=$(date +%s)
  CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" -m vllm.entrypoints.openai.api_server "${args[@]}" >"$log" 2>&1 &
  VLLM_PID=$!
  local ready=false
  for _ in $(seq 1 240); do
    if curl --silent --fail --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null; then ready=true; break; fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then break; fi
    sleep 2
  done
  if [[ "$ready" != true ]]; then
    cleanup
    if [[ "$optional" == true ]]; then
      printf '{"schema_version":1,"variant":"%s","status":"unsupported","reason":"vllm_start_or_adapter_compatibility_failed","mock":false}\n' "$label" >"$result_dir/unsupported.json"
      echo "unsupported=$label log=$log"
      return 0
    fi
    echo "vllm_start_failed=$label log=$log" >&2
    exit 2
  fi
  local startup=$(($(date +%s)-started))
  local benchmark_args=(
    --vllm-url "http://127.0.0.1:$PORT"
    --model-path "$model"
    --served-model-name "$served"
    --label "$label"
    --quantization "$quant"
    --prompts-file "$PROMPTS"
    --system-prompt-file "$SYSTEM_PROMPT"
    --compose-runtime-policy
    --startup-time-s "$startup"
    --gpu-index "$GPU"
    --output-dir "$result_dir"
  )
  if [[ -n "$adapter" ]]; then benchmark_args+=(--adapter-path "$adapter"); fi
  PYTHONPATH="$PROJECT/backend" "$PYTHON" -m experiments.quantization_benchmark "${benchmark_args[@]}"
  cleanup
  touch "$result_dir/complete.marker"
  sleep 3
}

cd "$PROJECT"
run_variant bf16_base "$BF16" bf16
run_variant bf16_dynamic_lora "$BF16" bf16 "$BEST_ADAPTER"
"$PYTHON" "$PROJECT/scripts/merge_kisaki_adapter_for_eval.py" --base-model "$BF16" --adapter "$BEST_ADAPTER" --output "$MERGED" --allowed-root "$ROOT" --experiment-id R3-BF16-MERGED
run_variant bf16_merged "$MERGED" bf16
run_variant awq_base "$AWQ" awq
run_variant awq_dynamic_lora "$AWQ" awq "$BEST_ADAPTER" true
echo "r3_complete output=$OUTPUT"
