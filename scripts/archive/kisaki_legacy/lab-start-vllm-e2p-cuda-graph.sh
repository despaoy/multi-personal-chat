#!/usr/bin/env bash
# E2' vLLM 启动脚本（CUDA graph 加速版，去掉 --enforce-eager）
# 修复 Triton 编译：设置 C_INCLUDE_PATH 指向 Python.h 所在目录
set -euo pipefail

ROOT=/home/szw/lhm2
MODEL=$ROOT/runtime/models/Qwen3-8B-Instruct
ENV_DIR=$ROOT/envs/qqchat-gpu-qwen3

KISAKI_E2P=$ROOT/runtime/loras/kisaki/e2p_neftune_r32/final

# Triton 编译所需的环境变量（根因：Python.h 缺失，不是 CUDA 版本不匹配）
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

# 不带 --enforce-eager，启用 CUDA graph 加速
exec env CUDA_VISIBLE_DEVICES=0 "$VLLM_BIN" serve "$MODEL" \
  --served-model-name kisaki-e2p-neftune \
  --host 127.0.0.1 --port 8002 \
  --gpu-memory-utilization 0.90 --max-model-len 4096 \
  --enable-lora \
  --max-loras 1 \
  --max-lora-rank 32 \
  --lora-modules kisaki-e2p-neftune="$KISAKI_E2P"
