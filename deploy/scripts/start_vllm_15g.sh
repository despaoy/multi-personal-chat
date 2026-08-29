#!/bin/bash
# ============================================
# vLLM 启动脚本（适配 3090 ~15GB 可用显存）
# 用法: ./start_vllm_15g.sh
# 使用 AWQ 4bit 量化模型，单卡运行
# ============================================

set -euo pipefail

# ---- 配置 ----
GPU_ID="${1:-0}"
PORT="${2:-8001}"
MODEL_PATH="${3:-./models/Qwen3-8B-Instruct-AWQ}"
LORA_PATH="${4:-./loras}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ---- GPU 检查 ----
check_gpu() {
    log_info "检查 GPU ${GPU_ID}..."
    if ! command -v nvidia-smi &>/dev/null; then
        log_error "nvidia-smi 未找到"
        exit 1
    fi
    nvidia-smi -i "${GPU_ID}" --query-gpu=name,memory.total,memory.free --format=csv,noheader
    local free_mem
    free_mem=$(nvidia-smi -i "${GPU_ID}" --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if [[ -n "${free_mem}" && ${free_mem} -lt 8000 ]]; then
        log_error "可用显存仅 ${free_mem}MB，至少需要 8GB（AWQ模型 ~4.5G + KV Cache）"
        exit 1
    fi
    log_info "GPU ${GPU_ID} 可用显存: ${free_mem}MB ✓"
}

# ---- 模型检查 ----
check_model() {
    log_info "检查模型: ${MODEL_PATH}"
    if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
        log_error "模型 config.json 未找到，请先下载 AWQ 量化模型:"
        log_error "  huggingface-cli download Qwen/Qwen3-8B-AWQ --local-dir ${MODEL_PATH}"
        exit 1
    fi
    log_info "模型检查通过 ✓"
}

# ---- 启动 vLLM ----
start_vllm() {
    log_info "启动 vLLM（AWQ量化 / GPU ${GPU_ID} / 端口 ${PORT}）"

    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
    export VLLM_ALLOW_RUNTIME_LORA_UPDATING="${VLLM_ALLOW_RUNTIME_LORA_UPDATING:-True}"

    python -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_PATH}" \
        --served-model-name qwen3-8b-instruct-awq \
        --quantization awq_marlin \
        --enable-lora \
        --max-loras 4 \
        --max-lora-rank 64 \
        --gpu-memory-utilization 0.5 \
        --max-model-len "${VLLM_MAX_MODEL_LEN:-24576}" \
        --max-num-seqs "${VLLM_MAX_NUM_SEQS:-4}" \
        --dtype float16 \
        --tensor-parallel-size 1 \
        --host 0.0.0.0 \
        --port "${PORT}" \
        --trust-remote-code \
        &
    VLLM_PID=$!
    log_info "vLLM PID: ${VLLM_PID}"
}

# ---- 等待就绪 ----
wait_for_healthy() {
    local max_retries=60
    local retry=0
    local url="http://localhost:${PORT}/health"
    log_info "等待 vLLM 就绪（最多 5 分钟）..."
    while [[ ${retry} -lt ${max_retries} ]]; do
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            log_error "vLLM 进程已退出，检查日志"
            exit 1
        fi
        if curl -sf "${url}" &>/dev/null; then
            log_info "vLLM 已就绪！API: http://localhost:${PORT}/v1"
            return 0
        fi
        retry=$((retry + 1))
        sleep 5
    done
    log_error "启动超时"
    exit 1
}

cleanup() {
    log_info "正在关闭 vLLM..."
    kill -TERM "${VLLM_PID}" 2>/dev/null || true
    sleep 5
    kill -KILL "${VLLM_PID}" 2>/dev/null || true
    exit 0
}
trap cleanup SIGTERM SIGINT

# ---- 入口 ----
main() {
    log_info "===== vLLM 15GB 优化启动 ====="
    check_gpu
    check_model
    start_vllm
    wait_for_healthy
    log_info "运行中，按 Ctrl+C 停止"
    wait "${VLLM_PID}"
}
main "$@"
