#!/usr/bin/env bash
set -euo pipefail

ROOT="${QQCHAT_LAB_ROOT:?set QQCHAT_LAB_ROOT to the lab root}"
MODEL=$ROOT/runtime/models/Qwen3-8B-Instruct-AWQ
ENV_DIR="${QQCHAT_LAB_ENV_DIR:-$ROOT/envs/qqchat-gpu-qwen3}"

# Adapters are loaded through the authenticated backend after compatibility checks.
# Old Qwen2.5 adapters may remain on disk; the checker rejects them against Qwen3.

if [[ -x "$ENV_DIR/bin/vllm" ]]; then
    VLLM_BIN="$ENV_DIR/bin/vllm"
else
    VLLM_BIN="$ROOT/envs/qqchat-gpu/bin/vllm"
fi

exec env CUDA_VISIBLE_DEVICES=1 VLLM_ALLOW_RUNTIME_LORA_UPDATING=True "$VLLM_BIN" serve "$MODEL" \
  --served-model-name qwen3-8b-instruct-awq \
  --host 127.0.0.1 --port 8001 \
  --gpu-memory-utilization 0.85 --max-model-len 8192 \
  --quantization awq_marlin \
  --guided-decoding-backend lm-format-enforcer \
  --enable-lora --max-loras 4 --max-lora-rank 64 \
  --enforce-eager
