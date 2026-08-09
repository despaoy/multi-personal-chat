#!/usr/bin/env bash
# Historical artifact only. It uses the archived prompt-v1 contract.
set -euo pipefail

ROOT=${QQCHAT_LAB_ROOT:-/home/szw/lhm2}
PROJECT=$ROOT/qqchat-enhanced
PYTHON=${QQCHAT_PYTHON:-$ROOT/envs/qqchat-gpu-qwen3/bin/python}
SEED=${1:?usage: lab-evaluate-kisaki-seed.sh SEED PHYSICAL_GPU}
GPU=${2:?usage: lab-evaluate-kisaki-seed.sh SEED PHYSICAL_GPU}
MODEL=$ROOT/runtime/models/Qwen3-8B-Instruct
E1=$ROOT/runtime/loras/kisaki/canonical/e1/seed$SEED/final
E2=$ROOT/runtime/loras/kisaki/canonical/e2/seed$SEED/final
GOLD=$PROJECT/backend/evaluation/kisaki_gold_set_v2.json
PROMPT=$PROJECT/backend/data/character_dialogues/kisaki_system_prompt.txt
RESULT_ROOT=$ROOT/runtime/experiments/kisaki
VLLM_LOG=$ROOT/runtime/logs/kisaki_eval_seed$SEED.log
PORT=${KISAKI_EVAL_PORT:-8001}

for path in "$MODEL/config.json" "$E1/adapter_config.json" "$E2/adapter_config.json" "$GOLD"; do
  if [[ ! -f "$path" ]]; then
    echo "required_file_missing=$path" >&2
    exit 2
  fi
done

mkdir -p "$ROOT/runtime/logs" "$RESULT_ROOT/e1/seed$SEED" "$RESULT_ROOT/e2/seed$SEED"
if curl --silent --fail --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
  echo "refusing_to_replace_existing_service=127.0.0.1:$PORT" >&2
  exit 2
fi

VLLM_ARGS=(
  --host 127.0.0.1
  --port "$PORT"
  --model "$MODEL"
  --served-model-name qwen3-8b-instruct
  --dtype bfloat16
  --gpu-memory-utilization 0.90
  --max-model-len 4096
  --enable-lora
  --max-lora-rank 32
  --lora-modules "kisaki-e1-seed$SEED=$E1" "kisaki-e2-seed$SEED=$E2"
)
cd "$PROJECT"
CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}" >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!

cleanup() {
  if kill -0 "$VLLM_PID" 2>/dev/null; then
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 180); do
  if curl --silent --fail --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vllm_start_failed log=$VLLM_LOG" >&2
    exit 2
  fi
  sleep 2
done
curl --silent --fail --max-time 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null

COMMON=(
  --dataset "$GOLD"
  --formal
  --base-url "http://127.0.0.1:$PORT"
  --system-prompt-file "$PROMPT"
  --temperature 0
  --max-tokens 256
  --repetition-penalty 1.0
  --frequency-penalty 0.0
  --gpu "$GPU"
)
PYTHONPATH="$PROJECT/backend" "$PYTHON" -m evaluation.character_benchmark_v3 "${COMMON[@]}" --model "kisaki-e1-seed$SEED" --adapter-path "$E1" --output "$RESULT_ROOT/e1/seed$SEED/character_eval.json"
PYTHONPATH="$PROJECT/backend" "$PYTHON" -m evaluation.character_benchmark_v3 "${COMMON[@]}" --model "kisaki-e2-seed$SEED" --adapter-path "$E2" --output "$RESULT_ROOT/e2/seed$SEED/character_eval.json"
PYTHONPATH="$PROJECT/backend" "$PYTHON" -m evaluation.benchmark_gate_v2 --baseline "$RESULT_ROOT/e1/seed$SEED/character_eval.json" --candidate "$RESULT_ROOT/e2/seed$SEED/character_eval.json" --output "$RESULT_ROOT/seed$SEED-quality-gate.json"
"$PYTHON" "$PROJECT/scripts/build_blind_ab_review.py" --a "$RESULT_ROOT/e1/seed$SEED/character_eval.json" --b "$RESULT_ROOT/e2/seed$SEED/character_eval.json" --output-dir "$RESULT_ROOT/blind-review-seed$SEED" --seed "$SEED"
echo "evaluation_complete seed=$SEED gate=$RESULT_ROOT/seed$SEED-quality-gate.json"
