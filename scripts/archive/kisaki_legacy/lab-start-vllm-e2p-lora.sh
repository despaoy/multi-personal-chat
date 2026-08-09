#!/usr/bin/env bash
# E2' 评估专用 vLLM 启动脚本：加载 Qwen3-8B-Instruct + E2' LoRA
set -euo pipefail

ROOT=/home/szw/lhm2
MODEL=$ROOT/runtime/models/Qwen3-8B-Instruct
ENV_DIR=$ROOT/envs/qqchat-gpu-qwen3

KISAKI_E2P=$ROOT/runtime/loras/kisaki/e2p_neftune_r32/final

# Triton 编译所需的环境变量
export C_INCLUDE_PATH=$ENV_DIR/include/python3.11
export CPLUS_INCLUDE_PATH=$ENV_DIR/include/python3.11
export TRITON_CACHE_DIR=$ROOT/runtime/cache/triton
mkdir -p "$TRITON_CACHE_DIR"
export CUDA_HOME=/usr/local/cuda

if [[ ! -f "$KISAKI_E2P/adapter_config.json" ]]; then
  echo "ERROR: E2' LoRA adapter not found at $KISAKI_E2P" >&2
  exit 1
fi

VLLM_BIN="$ENV_DIR/bin/vllm"

exec env CUDA_VISIBLE_DEVICES=0 "$VLLM_BIN" serve "$MODEL" \
  --served-model-name kisaki-e2p-neftune \
  --host 127.0.0.1 --port 8002 \
  --gpu-memory-utilization 0.90 --max-model-len 4096 \
  --enable-lora \
  --max-loras 1 \
  --max-lora-rank 32 \
  --lora-modules kisaki-e2p-neftune="$KISAKI_E2P" \
  --enforce-eager
