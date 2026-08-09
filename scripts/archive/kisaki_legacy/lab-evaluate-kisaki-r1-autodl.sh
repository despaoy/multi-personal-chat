#!/usr/bin/env bash
# Historical artifact only. Use lab-evaluate-kisaki-r1-seed.sh for active work.
# Evaluate R1 adapters with prompt-v2 on autodl server
# Merges one adapter at a time, evaluates, then deletes merged model to save disk
# Usage: bash scripts/lab-evaluate-kisaki-r1-autodl.sh [SEED] [VARIANTS_CSV]
set -euo pipefail

if [[ "${ALLOW_LEGACY_KISAKI_R1:-false}" != "true" ]]; then
  echo "blocked_legacy_r1=true; this evaluator is retained only for historical reproduction" >&2
  exit 2
fi

export QQCHAT_LAB_ROOT=${QQCHAT_LAB_ROOT:-/root/autodl-tmp}
ROOT=$QQCHAT_LAB_ROOT
PROJECT=$ROOT/qqchat-enhanced
PYTHON=${QQCHAT_PYTHON:-$(command -v python)}
SEED=${1:-42}
VARIANTS_CSV=${2:-e1,e2,e3,e4,e5}
MODEL=$ROOT/runtime/models/Qwen3-8B-Instruct
GOLD=$PROJECT/backend/evaluation/kisaki_gold_set_v2.json
PROMPT=$PROJECT/backend/data/character_dialogues/experiments/archive/prompts/kisaki_system_prompt_v2.txt
RESULT_ROOT=$ROOT/runtime/experiments/kisaki/r1
SCOPE=formal
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
  sleep 3
}
trap cleanup_vllm EXIT INT TERM
cd "$PROJECT"

for variant in "${VARIANTS[@]}"; do
  adapter="$ROOT/runtime/loras/kisaki/canonical/$variant/seed$SEED/final"
  merged="$MERGED_ROOT/$variant"
  result="$RESULT_ROOT/$variant/seed$SEED/character_eval_prompt_v2.json"
  [[ -f "$adapter/adapter_config.json" ]] || { echo "required_file_missing=$adapter/adapter_config.json" >&2; exit 2; }
  mkdir -p "$(dirname "$result")"

  echo ""
  echo "=== Evaluating $variant seed$SEED ==="
  echo "Time: $(date --iso-8601=seconds)"

  # Skip if result already exists (immutable)
  if [[ -f "$result" ]]; then
    echo "skip_immutable_result=$result"
    continue
  fi

  # Check disk space before merge
  DISK_AVAIL=$(df --output=avail -B1 "$ROOT" | tail -1)
  DISK_GB=$((DISK_AVAIL / 1024 / 1024 / 1024))
  echo "Disk available before merge: ${DISK_GB}GB"
  if (( DISK_GB < 18 )); then
    echo "WARNING: Low disk space (${DISK_GB}GB). Attempting to clean old merged models..."
    rm -rf "$MERGED_ROOT"/e* 2>/dev/null || true
    DISK_AVAIL=$(df --output=avail -B1 "$ROOT" | tail -1)
    DISK_GB=$((DISK_AVAIL / 1024 / 1024 / 1024))
    echo "Disk available after cleanup: ${DISK_GB}GB"
    if (( DISK_GB < 18 )); then
      echo "ERROR: insufficient disk for merge (need >=18GB, got ${DISK_GB}GB)" >&2
      exit 2
    fi
  fi

  # Merge adapter into base model
  echo "Merging adapter -> $merged"
  "$PYTHON" "$PROJECT/scripts/merge_kisaki_adapter_for_eval.py" \
    --base-model "$MODEL" --adapter "$adapter" --output "$merged" \
    --allowed-root "$ROOT" --experiment-id "R1-${variant^^}-seed$SEED"

  # Start vLLM with merged model
  log="$ROOT/runtime/logs/kisaki_r1_${variant}_eval_seed$SEED.log"
  started=$(date +%s)
  echo "Starting vLLM with merged model..."
  CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m vllm.entrypoints.openai.api_server \
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
  echo "vLLM ready"

  # Run benchmark
  echo "Running character benchmark v3..."
  PYTHONPATH="$PROJECT/backend" "$PYTHON" -m evaluation.character_benchmark_v3 \
    --dataset "$GOLD" --formal --base-url "http://127.0.0.1:$PORT" \
    --system-prompt-file "$PROMPT" --temperature 0 --max-tokens 256 \
    --repetition-penalty 1.0 --frequency-penalty 0.0 --gpu 0 \
    --model "r1-$variant-seed$SEED" --model-path "$merged" --adapter-path "$adapter" --output "$result"

  # Stop vLLM
  cleanup_vllm
  echo "variant_evaluation_complete variant=$variant elapsed_seconds=$(($(date +%s)-started))"

  # Delete merged model to free disk for next variant
  echo "Cleaning merged model to free disk: $merged"
  rm -rf "$merged"
  DISK_AVAIL=$(df --output=avail -B1 "$ROOT" | tail -1)
  DISK_GB=$((DISK_AVAIL / 1024 / 1024 / 1024))
  echo "Disk available after cleanup: ${DISK_GB}GB"
done

# Gate comparisons (only for full formal run)
if [[ "$SCOPE" == formal && "$VARIANTS_CSV" == "e1,e2,e3,e4,e5" ]]; then
echo ""
echo "=== Running gate comparisons ==="
for candidate in e2 e3 e4 e5; do
  gate="$RESULT_ROOT/gates/e1-vs-$candidate-seed$SEED.json"
  blind="$RESULT_ROOT/blind/e1-vs-$candidate-seed$SEED"
  if [[ ! -f "$gate" ]]; then
    echo "Gate: e1 vs $candidate"
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
    echo "Building blind review: e1 vs $candidate"
    "$PYTHON" "$PROJECT/scripts/build_stratified_blind_review.py" \
      --a "$RESULT_ROOT/e1/seed$SEED/character_eval_prompt_v2.json" \
      --b "$RESULT_ROOT/$candidate/seed$SEED/character_eval_prompt_v2.json" \
      --output-dir "$blind" --seed "$SEED" --per-category 10
  fi
done
fi

echo ""
echo "=== R1 Evaluation Complete ==="
echo "seed=$SEED prompt=v2 strategy=merged_isolated_per_variant scope=$SCOPE variants=$VARIANTS_CSV"
echo "Time: $(date --iso-8601=seconds)"
