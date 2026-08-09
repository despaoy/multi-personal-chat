#!/usr/bin/env bash
set -euo pipefail

# E2 NEFTune LoRA 训练启动脚本（月社妃，Qwen3-8B-Instruct 基座）
# 配置：r=32, alpha=64, target_modules=7 linear, neftune_alpha=5.0, no DoRA/RSLoRA
# 数据：854 train + eval（E1 的 829 + 元叙事补充15 + 安全拒绝补充10）
# GPU：使用 GPU 0（GPU 1 被 vLLM 占用）

ROOT=/home/szw/lhm2
PROJECT=$ROOT/qqchat-enhanced
PYTHON=$ROOT/envs/qqchat-gpu-qwen3/bin/python
CONFIG=$PROJECT/backend/data/character_dialogues/experiments/configs/tsukiyashiro_kisaki_e2_neftune.json
PIDFILE=$ROOT/runtime/kisaki_e2_neftune.pid
LOGFILE=$ROOT/runtime/logs/kisaki_e2_neftune.log

mkdir -p "$ROOT/runtime/logs" "$ROOT/runtime/loras/kisaki"

# 1. 检查是否已有训练在跑
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "training_already_running pid=$(cat "$PIDFILE")"
  exit 0
fi

# 2. 检查 GPU 0 是否可用（vLLM 用 GPU 1，训练用 GPU 0）
GPU0_FREE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | head -1)
if [[ "$GPU0_FREE" -gt 5000 ]]; then
  echo "refusing_to_start: GPU 0 memory used ${GPU0_FREE}MB (>5000MB threshold)" >&2
  exit 2
fi

cd "$PROJECT/backend"
nohup env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  "$PYTHON" -m training.trainer --config "$CONFIG" \
  >"$LOGFILE" 2>&1 </dev/null &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$PIDFILE"
echo "training_started pid=$TRAIN_PID log=$LOGFILE config=$CONFIG"
echo "monitor: tail -f $LOGFILE"
