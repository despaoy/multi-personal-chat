#!/usr/bin/env bash
set -euo pipefail
ROOT="${QQCHAT_LAB_ROOT:?set QQCHAT_LAB_ROOT to the lab root}"
PROJECT=$ROOT/qqchat-enhanced
PYTHON="${QQCHAT_LAB_PYTHON:-$ROOT/envs/qqchat-gpu-qwen3/bin/python}"
GPU=${1:?usage: lab-run-kisaki-r2.sh GPU MODEL_PATH}
MODEL=${2:?usage: lab-run-kisaki-r2.sh GPU MODEL_PATH}
DATASET=$PROJECT/backend/data/character_dialogues/experiments/research/kisaki_rag_eval_v2.json
DOCUMENTS=$PROJECT/backend/data/character_dialogues/experiments/research/character_rag_seed_documents.json
PROMPT=$PROJECT/backend/data/character_dialogues/kisaki_system_prompt_v3.txt
OUTPUT=$ROOT/runtime/experiments/kisaki/r2
PORT=${KISAKI_R2_PORT:-8001}
RERANKER_MODEL=${RERANKER_MODEL_PATH:-$ROOT/runtime/models/bge-reranker-v2-m3}
export VECTOR_DB_PATH=$OUTPUT/vector_db
export EMBEDDING_MODEL_PATH=${EMBEDDING_MODEL_PATH:-$ROOT/runtime/models/bge-m3}
VLLM_PID=''
for path in "$MODEL/config.json" "$DATASET" "$DOCUMENTS" "$PROMPT" "$EMBEDDING_MODEL_PATH/config.json" "$RERANKER_MODEL/config.json"; do
  [[ -f "$path" ]] || { echo "required_file_missing=$path" >&2; exit 2; }
done
[[ "$MODEL" == "$ROOT/"* ]] || { echo "model_outside_allowed_root=$MODEL" >&2; exit 2; }
export RERANKER_ENABLED=true
export RERANKER_MODEL_PATH="$RERANKER_MODEL"
cleanup(){ if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then kill "$VLLM_PID" 2>/dev/null || true; wait "$VLLM_PID" 2>/dev/null || true; fi; }
trap cleanup EXIT INT TERM
mkdir -p "$OUTPUT" "$ROOT/runtime/logs"
cd "$PROJECT"
"$PYTHON" "$PROJECT/scripts/import_kisaki_rag_evidence.py" --documents "$DOCUMENTS" --dataset "$DATASET"
PYTHONPATH="$PROJECT/backend" "$PYTHON" -m experiments.rag_ablation --formal --dataset "$DATASET" --output-dir "$OUTPUT/retrieval"
RETRIEVAL_REPORT=$(find "$OUTPUT/retrieval" -maxdepth 1 -type f -name 'r2_retrieval_ablation_*.json' -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)
[[ -n "$RETRIEVAL_REPORT" ]] || { echo "retrieval_report_missing" >&2; exit 2; }
BEST_STRATEGY=$("$PYTHON" -c 'import json,sys; rows=json.load(open(sys.argv[1],encoding="utf-8"))["results"]; valid=[r for r in rows if not r.get("error")]; assert valid, "no valid retrieval result"; print(max(valid,key=lambda r:(r["mrr"],r["recall_at_5"],r["ndcg_at_5"],-r["p95_latency_ms"]))["variant_name"])' "$RETRIEVAL_REPORT")
echo "selected_retrieval_strategy=$BEST_STRATEGY report=$RETRIEVAL_REPORT"
if curl --silent --fail --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null; then echo "port_in_use=$PORT" >&2; exit 2; fi
CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" -m vllm.entrypoints.openai.api_server --host 127.0.0.1 --port "$PORT" --model "$MODEL" --served-model-name kisaki-r2 --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.90 >"$ROOT/runtime/logs/kisaki_r2_vllm.log" 2>&1 &
VLLM_PID=$!
for _ in $(seq 1 240); do curl --silent --fail --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null && break; kill -0 "$VLLM_PID" 2>/dev/null || { echo "vllm_start_failed" >&2; exit 2; }; sleep 2; done
curl --silent --fail --max-time 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null
PYTHONPATH="$PROJECT/backend" "$PYTHON" -m experiments.rag_answer_ablation --formal --dataset "$DATASET" --system-prompt-file "$PROMPT" --base-url "http://127.0.0.1:$PORT" --model kisaki-r2 --retrieval-strategy "$BEST_STRATEGY" --output "$OUTPUT/answer/r2_answer.json"
echo "r2_complete output=$OUTPUT"
