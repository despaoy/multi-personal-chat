#!/usr/bin/env bash
# E1 评估专用 vLLM 启动脚本：加载 Qwen3-8B-Instruct + E1 LoRA
# 解决 Triton 编译 Python.h 缺失问题：借用 streamvggt_py311 的头文件
set -euo pipefail

ROOT=/home/szw/lhm2
# 用非 AWQ 的原版模型，与 LoRA 训练时的 base 一致（避免量化模型 LoRA 注入兼容性问题）
MODEL=$ROOT/runtime/models/Qwen3-8B-Instruct
ENV_DIR=$ROOT/envs/qqchat-gpu-qwen3

# E1 LoRA adapter
KISAKI_E1=$ROOT/runtime/loras/kisaki/e1_baseline_r32/final

# Triton 编译所需的环境变量（Python.h 从 streamvggt_py311 借用）
export C_INCLUDE_PATH=$ENV_DIR/include/python3.11
export CPLUS_INCLUDE_PATH=$ENV_DIR/include/python3.11
export TRITON_CACHE_DIR=$ROOT/runtime/cache/triton
mkdir -p "$TRITON_CACHE_DIR"

# CUDA 工具链
export CUDA_HOME=/usr/local/cuda

if [[ ! -f "$KISAKI_E1/adapter_config.json" ]]; then
  echo "ERROR: E1 LoRA adapter not found at $KISAKI_E1" >&2
  exit 1
fi

VLLM_BIN="$ENV_DIR/bin/vllm"

exec env CUDA_VISIBLE_DEVICES=1 "$VLLM_BIN" serve "$MODEL" \
  --served-model-name kisaki-e1-baseline \
  --host 127.0.0.1 --port 8002 \
  --gpu-memory-utilization 0.90 --max-model-len 4096 \
  --enable-lora \
  --max-loras 1 \
  --max-lora-rank 32 \
  --lora-modules kisaki-e1-baseline="$KISAKI_E1" \
  --enforce-eager
