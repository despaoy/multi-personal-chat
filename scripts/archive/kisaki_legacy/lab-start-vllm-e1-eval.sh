#!/usr/bin/env bash
set -euo pipefail

# E1 baseline 评估专用 vLLM 启动脚本
# 加载 Qwen3-8B-Instruct-AWQ + E1 LoRA adapter，暴露为 model 名 "kisaki-e1-baseline"
# 使用 GPU 1（与基座模型推理一致）

ROOT=/home/szw/lhm2
MODEL=$ROOT/runtime/models/Qwen3-8B-Instruct-AWQ
ENV_DIR=$ROOT/envs/qqchat-gpu-qwen3
LORA_E1=$ROOT/runtime/loras/kisaki/e1_baseline_r32/final

if [[ ! -d "$LORA_E1" ]]; then
  echo "ERROR: E1 LoRA adapter not found at $LORA_E1" >&2
  exit 1
fi

if [[ -x "$ENV_DIR/bin/vllm" ]]; then
    VLLM_BIN="$ENV_DIR/bin/vllm"
else
    VLLM_BIN="$ROOT/envs/qqchat-gpu/bin/vllm"
fi

exec env CUDA_VISIBLE_DEVICES=1 "$VLLM_BIN" serve "$MODEL" \
  --served-model-name qwen3-8b-instruct-awq \
  --host 127.0.0.1 --port 8001 \
  --gpu-memory-utilization 0.85 --max-model-len 8192 \
  --quantization awq_marlin \
  --guided-decoding-backend lm-format-enforcer \
  --enforce-eager \
  --enable-lora --max-loras 2 --max-lora-rank 32 \
  --lora-modules kisaki-e1-baseline="$LORA_E1"
