#!/usr/bin/env bash
# E2'' RAG Citation LoRA v2 训练启动脚本（月社妃，四轮审查后888条数据）
# 使用 GPU 1（GPU 0 被 vLLM 占用）
set -euo pipefail

ROOT=/home/szw/lhm2
PROJECT=$ROOT/qqchat-enhanced
PYTHON=$ROOT/envs/qqchat-gpu-qwen3/bin/python
CONFIG=$PROJECT/backend/data/character_dialogues/experiments/configs/tsukiyashiro_kisaki_e2pp_rag.json
PIDFILE=$ROOT/runtime/kisaki_e2pp_rag.pid
LOGFILE=$ROOT/runtime/logs/kisaki_e2pp_rag.log

mkdir -p "$ROOT/runtime/logs" "$ROOT/runtime/loras/kisaki"

# 1. 检查是否已有训练在跑
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "training_already_running pid=$(cat "$PIDFILE")"
  exit 0
fi

# 2. 检查 GPU 1 是否可用（vLLM 占用 GPU 0）
GPU1_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | head -1)
if [[ "$GPU1_USED" -gt 5000 ]]; then
  echo "refusing_to_start: GPU 1 memory used ${GPU1_USED}MB (>5000MB threshold)" >&2
  exit 2
fi

# 3. 检查训练数据是否存在
TRAIN_DATA=$PROJECT/backend/data/character_dialogues/tsukiyashiro_kisaki_train_e2pp.json
if [[ ! -f "$TRAIN_DATA" ]]; then
  echo "ERROR: Training data not found at $TRAIN_DATA" >&2
  exit 1
fi

# 4. Archive previous training artifacts instead of deleting them.
OUTPUT_DIR=$ROOT/runtime/loras/kisaki/e2pp_rag_r32
if [[ -d "$OUTPUT_DIR" ]]; then
  ARCHIVE_DIR=$ROOT/runtime/archive/training/$(date +%Y%m%d-%H%M%S)-e2pp_rag_r32
  mkdir -p "$(dirname "$ARCHIVE_DIR")"
  echo "Archiving previous output dir: $OUTPUT_DIR -> $ARCHIVE_DIR"
  mv "$OUTPUT_DIR" "$ARCHIVE_DIR"
fi
mkdir -p "$OUTPUT_DIR"

cd "$PROJECT/backend"
nohup env CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
  "$PYTHON" -m training.trainer --config "$CONFIG" \
  >"$LOGFILE" 2>&1 </dev/null &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$PIDFILE"
echo "training_started pid=$TRAIN_PID log=$LOGFILE config=$CONFIG"
echo "monitor: tail -f $LOGFILE"
echo "gpu: 1 (GPU 0 occupied by vLLM)"
