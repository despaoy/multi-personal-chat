#!/bin/bash
# vLLM startup script for the verified AutoDL RTX 3090 baseline.
# Usage: ./start_vllm.sh <GPU_ID> <PORT> <MODEL_PATH> [LORA_PATH]
# Example: ./start_vllm.sh 0 8001 /models/Qwen3-8B-Instruct-AWQ /loras

set -euo pipefail

GPU_ID="${1:?missing GPU ID}"
PORT="${2:?missing port}"
MODEL_PATH="${3:?missing model path}"
LORA_PATH="${4:-/loras}"

SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-qwen3-8b-instruct-awq}"
QUANTIZATION="${VLLM_QUANTIZATION:-awq_marlin}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.82}"
MAX_LORAS="${VLLM_MAX_LORAS:-4}"
MAX_LORA_RANK="${VLLM_MAX_LORA_RANK:-64}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-24576}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-8}"
DTYPE="${VLLM_DTYPE:-float16}"

# 当 QUANTIZATION 为 none/空时，不传 --quantization 参数，让 vLLM 从 config.json 自动检测。
# vLLM 不接受 "none" 作为量化方法名，会抛 ValidationError。
QUANT_ARG=""
if [[ -n "${QUANTIZATION}" && "${QUANTIZATION}" != "none" ]]; then
    QUANT_ARG="--quantization ${QUANTIZATION}"
fi

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

check_gpu() {
    log_info "Checking GPU ${GPU_ID}..."
    if ! command -v nvidia-smi &>/dev/null; then
        log_error "nvidia-smi not found; install NVIDIA driver first."
        exit 1
    fi

    local gpu_count
    gpu_count=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
    if [[ ${GPU_ID} -ge ${gpu_count} ]]; then
        log_error "GPU ID ${GPU_ID} does not exist; system has ${gpu_count} GPU(s)."
        exit 1
    fi

    local gpu_mem
    gpu_mem=$(nvidia-smi -i "${GPU_ID}" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    if [[ -n "${gpu_mem}" && ${gpu_mem} -lt 12000 ]]; then
        log_warn "GPU ${GPU_ID} free memory is ${gpu_mem}MB; Qwen3-8B-AWQ may fail to load."
    else
        log_info "GPU ${GPU_ID} free memory: ${gpu_mem}MB"
    fi
}

check_model() {
    log_info "Checking model files: ${MODEL_PATH}"
    if [[ ! -d "${MODEL_PATH}" ]]; then
        log_error "Model directory does not exist: ${MODEL_PATH}"
        exit 1
    fi

    for f in config.json tokenizer.json; do
        if [[ ! -f "${MODEL_PATH}/${f}" ]]; then
            log_error "Missing required model file: ${MODEL_PATH}/${f}"
            exit 1
        fi
    done

    if ! ls "${MODEL_PATH}"/*.safetensors &>/dev/null && ! ls "${MODEL_PATH}"/model*.bin &>/dev/null; then
        log_error "No model weight files found in ${MODEL_PATH}"
        exit 1
    fi
}

start_vllm() {
    log_info "Starting vLLM..."
    log_info "  GPU: ${GPU_ID}"
    log_info "  Port: ${PORT}"
    log_info "  Model: ${MODEL_PATH}"
    log_info "  Served model: ${SERVED_MODEL_NAME}"
    log_info "  Quantization: ${QUANTIZATION:-auto (no flag)}"
    log_info "  LoRA directory: ${LORA_PATH}"

    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
    export VLLM_ALLOW_RUNTIME_LORA_UPDATING="${VLLM_ALLOW_RUNTIME_LORA_UPDATING:-True}"

    # QUANT_ARG 为空时不传 --quantization，让 vLLM 自动检测量化方式
    # shellcheck disable=SC2086
    python3 -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_PATH}" \
        --served-model-name "${SERVED_MODEL_NAME}" \
        ${QUANT_ARG} \
        --enable-lora \
        --max-loras "${MAX_LORAS}" \
        --max-lora-rank "${MAX_LORA_RANK}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --max-num-seqs "${MAX_NUM_SEQS}" \
        --dtype "${DTYPE}" \
        --tensor-parallel-size 1 \
        --host 0.0.0.0 \
        --port "${PORT}" \
        --enable-prefix-caching \
        --trust-remote-code &
    VLLM_PID=$!
    log_info "vLLM PID: ${VLLM_PID}"
}

wait_for_healthy() {
    local max_retries=60
    local retry=0
    local health_url="http://localhost:${PORT}/health"
    log_info "Waiting for vLLM health check: ${health_url}"

    while [[ ${retry} -lt ${max_retries} ]]; do
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            log_error "vLLM process exited early."
            exit 1
        fi
        if curl -sf "${health_url}" &>/dev/null; then
            log_info "vLLM is ready."
            return 0
        fi
        retry=$((retry + 1))
        sleep 5
    done

    log_error "vLLM startup timed out."
    kill "${VLLM_PID}" 2>/dev/null || true
    exit 1
}

cleanup() {
    log_info "Stopping vLLM..."
    if [[ -n "${VLLM_PID:-}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        kill -TERM "${VLLM_PID}" 2>/dev/null || true
        local wait_count=0
        while kill -0 "${VLLM_PID}" 2>/dev/null && [[ ${wait_count} -lt 30 ]]; do
            sleep 1
            wait_count=$((wait_count + 1))
        done
        if kill -0 "${VLLM_PID}" 2>/dev/null; then
            kill -KILL "${VLLM_PID}" 2>/dev/null || true
        fi
    fi
}
trap cleanup SIGTERM SIGINT SIGQUIT

check_gpu
check_model
start_vllm
wait_for_healthy
wait "${VLLM_PID}"
