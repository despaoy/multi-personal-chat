#!/usr/bin/env bash
set -euo pipefail

ROOT=${QQCHAT_LAB_ROOT:-/home/szw/lhm2}
PROJECT=$ROOT/qqchat-enhanced
PYTHON=${QQCHAT_PYTHON:-$ROOT/envs/qqchat-gpu-qwen3/bin/python}
SEED=${1:?usage: lab-evaluate-kisaki-r1-seed.sh SEED PHYSICAL_GPU}
GPU=${2:?usage: lab-evaluate-kisaki-r1-seed.sh SEED PHYSICAL_GPU [SCOPE] [VARIANTS_CSV]}
SCOPE=${3:-formal}
VARIANTS_CSV=${4:-e1,e2,e3,e4,e5}
MODEL=$ROOT/runtime/models/Qwen3-8B-Instruct
GOLD=$PROJECT/backend/evaluation/kisaki_gold_set_v2.json
PROMPT=$PROJECT/backend/data/character_dialogues/kisaki_system_prompt_v2.txt
RESULT_ROOT=$ROOT/runtime/experiments/kisaki/r1
[[ "$SCOPE" =~ ^[a-z0-9-]+$ ]] || { echo "invalid_scope=$SCOPE" >&2; exit 2; }
if [[ "$SCOPE" != formal ]]; then RESULT_ROOT=$RESULT_ROOT/evaluation-$SCOPE; fi
MERGED_ROOT=$ROOT/runtime/models/kisaki-r1-merged/seed$SEED
PORT=${KISAKI_R1_EVAL_PORT:-8001}
IFS=',' read -r -a VARIANTS <<< "$VARIANTS_CSV"
for variant in "${VARIANTS[@]}"; do [[ "$variant" =~ ^e[1-5]$ ]] || { echo "invalid_variant=$variant" >&2; exit 2; }; done
VLLM_PID=''

[[ "$PROJECT" == "$ROOT/"* ]] || { echo "project_path_outside_allowed_root=$PROJECT" >&2; exit 2; }
for path in "$MODEL/config.json" "$GOLD" "$PROMPT"; do
  [[ -f "$path" ]] || { echo "required_file_missing=$path" >&2; exit 2; }
done
mkdir -p "$RESULT_ROOT/gates" "$RESULT_ROOT/blind" "$MERGED_ROOT" "$ROOT/runtime/logs"
if curl --silent --fail --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
  echo "refusing_to_replace_existing_service=127.0.0.1:$PORT" >&2
  exit 2
fi

cleanup_vllm(){
  if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
  VLLM_PID=''
}
trap cleanup_vllm EXIT INT TERM
cd "$PROJECT"

for variant in "${VARIANTS[@]}"; do
  adapter="$ROOT/runtime/loras/kisaki/canonical/$variant/seed$SEED/final"
  merged="$MERGED_ROOT/$variant"
  result="$RESULT_ROOT/$variant/seed$SEED/character_eval_prompt_v2.json"
  [[ -f "$adapter/adapter_config.json" ]] || { echo "required_file_missing=$adapter/adapter_config.json" >&2; exit 2; }
  mkdir -p "$(dirname "$result")"

  "$PYTHON" "$PROJECT/scripts/merge_kisaki_adapter_for_eval.py" \
    --base-model "$MODEL" --adapter "$adapter" --output "$merged" \
    --allowed-root "$ROOT" --experiment-id "R1-${variant^^}-seed$SEED"

  if [[ -f "$result" ]]; then
    echo "skip_immutable_result=$result"
    continue
  fi

  log="$ROOT/runtime/logs/kisaki_r1_${variant}_eval_seed$SEED.log"
  started=$(date +%s)
  CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" -m vllm.entrypoints.openai.api_server \
    --host 127.0.0.1 --port "$PORT" --model "$merged" \
    --served-model-name "r1-$variant-seed$SEED" --dtype bfloat16 \
    --gpu-memory-utilization 0.90 --max-model-len 4096 >"$log" 2>&1 &
  VLLM_PID=$!
  ready=false
  for _ in $(seq 1 180); do
    if curl --silent --fail --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null; then ready=true; break; fi
    kill -0 "$VLLM_PID" 2>/dev/null || { echo "vllm_start_failed log=$log" >&2; exit 2; }
    sleep 2
  done
  [[ "$ready" == true ]] || { echo "vllm_ready_timeout log=$log" >&2; exit 2; }

  PYTHONPATH="$PROJECT/backend" "$PYTHON" -m evaluation.character_benchmark_v3 \
    --dataset "$GOLD" --formal --base-url "http://127.0.0.1:$PORT" \
    --system-prompt-file "$PROMPT" --temperature 0 --max-tokens 256 \
    --repetition-penalty 1.0 --frequency-penalty 0.0 --gpu "$GPU" \
    --model "r1-$variant-seed$SEED" --model-path "$merged" --adapter-path "$adapter" --output "$result"
  cleanup_vllm
  echo "variant_evaluation_complete variant=$variant elapsed_seconds=$(($(date +%s)-started))"
done

if [[ "$SCOPE" == formal && "$VARIANTS_CSV" == "e1,e2,e3,e4,e5" ]]; then
for candidate in e2 e3 e4 e5; do
  gate="$RESULT_ROOT/gates/e1-vs-$candidate-seed$SEED.json"
  blind="$RESULT_ROOT/blind/e1-vs-$candidate-seed$SEED"
  if [[ ! -f "$gate" ]]; then
    set +e
    PYTHONPATH="$PROJECT/backend" "$PYTHON" -m evaluation.benchmark_gate_v2 \
      --baseline "$RESULT_ROOT/e1/seed$SEED/character_eval_prompt_v2.json" \
      --candidate "$RESULT_ROOT/$candidate/seed$SEED/character_eval_prompt_v2.json" \
      --output "$gate"
    gate_code=$?
    set -e
    (( gate_code == 0 || gate_code == 2 )) || exit "$gate_code"
  fi
  if [[ ! -d "$blind" ]]; then
    "$PYTHON" "$PROJECT/scripts/build_stratified_blind_review.py" \
      --a "$RESULT_ROOT/e1/seed$SEED/character_eval_prompt_v2.json" \
      --b "$RESULT_ROOT/$candidate/seed$SEED/character_eval_prompt_v2.json" \
      --output-dir "$blind" --seed "$SEED" --per-category 10
  fi
done
fi
echo "r1_evaluation_complete seed=$SEED prompt=v2 strategy=merged_isolated scope=$SCOPE variants=$VARIANTS_CSV"
