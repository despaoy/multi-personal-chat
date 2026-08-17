#!/bin/bash
# MultiPersonal Chat System - 量化对比启动脚本
# 用法: bash deploy/compare_quantization.sh <model_path> [--mock] [--vllm-port <port>]
#
# 按序启动不同量化配置的 vLLM 进程，运行基准测试，收集结果。
# --mock 模式跳过实际 vLLM 启动，直接用预置结果验证流程。

set -euo pipefail

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[0;33m'; NC='\033[0m'

# 切换到项目根目录（deploy/ 的父目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# 参数解析
MODEL_PATH=""
MOCK=false
VLLM_PORT=8001
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mock) MOCK=true; shift ;;
        --vllm-port) VLLM_PORT="$2"; shift 2 ;;
        --help|-h)
            echo "用法: bash deploy/compare_quantization.sh <model_path> [--mock] [--vllm-port <port>]"
            echo ""
            echo "参数:"
            echo "  <model_path>     模型路径（必填，非 mock 模式）"
            echo "  --mock           Mock 模式，跳过 vLLM 启动"
            echo "  --vllm-port      vLLM 端口（默认 8001）"
            exit 0
            ;;
        *)
            if [[ -z "${MODEL_PATH}" ]]; then
                MODEL_PATH="$1"
            else
                echo "未知参数: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

# 确定 Python 命令
PYTHON="${PYTHON:-python3}"
if ! command -v "${PYTHON}" &>/dev/null; then
    PYTHON="python"
fi

OUTPUT_DIR="deploy/results/quantization_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

echo -e "=== 量化对比基准 ==="
echo -e "输出目录: ${YELLOW}${OUTPUT_DIR}${NC}"
echo -e "Mock 模式: ${YELLOW}${MOCK}${NC}"
echo ""

# ============================================
# Mock 模式：直接调用 benchmark 脚本
# ============================================
if [[ "${MOCK}" == "true" ]]; then
    echo -e "${YELLOW}[Mock]${NC} 跳过 vLLM 启动，使用预置结果..."
    if ${PYTHON} -m backend.experiments.quantization_benchmark --mock --output-dir "${OUTPUT_DIR}" 2>&1; then
        echo -e "${GREEN}[PASS]${NC} 量化基准（mock）完成"
        echo -e "报告: ${YELLOW}${OUTPUT_DIR}/quantization_benchmark_report.md${NC}"
        exit 0
    else
        echo -e "${RED}[FAIL]${NC} 量化基准（mock）失败"
        exit 1
    fi
fi

# ============================================
# 非 Mock 模式：检查参数和环境
# ============================================
if [[ -z "${MODEL_PATH}" ]]; then
    echo -e "${RED}错误: 非 mock 模式需要提供 model_path${NC}"
    echo "用法: bash deploy/compare_quantization.sh <model_path> [--mock]"
    exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo -e "${RED}错误: 模型路径不存在: ${MODEL_PATH}${NC}"
    exit 1
fi

# 检查 vLLM 可用性
echo -e "${YELLOW}[检查]${NC} vLLM 可用性..."
if ! ${PYTHON} -c "import vllm" 2>/dev/null; then
    echo -e "${RED}[FAIL]${NC} vLLM 未安装，请先安装: pip install vllm"
    exit 1
fi
echo -e "${GREEN}[PASS]${NC} vLLM 可用"

# 量化配置定义: 标签 | vLLM 量化参数 | 显存利用率
CONFIGS=(
    "fp16|none|0.90"
    "awq|awq|0.50"
    "int8|bitsandbytes|0.70"
)

COMPARISON_FILE="${OUTPUT_DIR}/comparison.md"
echo "# 量化对比报告" > "${COMPARISON_FILE}"
echo "" >> "${COMPARISON_FILE}"
echo "- **模型**: ${MODEL_PATH}" >> "${COMPARISON_FILE}"
echo "- **运行时间**: ${TIMESTAMP}" >> "${COMPARISON_FILE}"
echo "" >> "${COMPARISON_FILE}"
echo "| 配置 | 加载时间(s) | 显存(MB) | TTFT(ms) | 解码(tokens/s) | P95延迟(ms) | 质量 |" >> "${COMPARISON_FILE}"
echo "|------|-------------|----------|----------|----------------|-------------|------|" >> "${COMPARISON_FILE}"

VLLM_PID=""

cleanup() {
    if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo -e "${YELLOW}[清理]${NC} 关闭 vLLM 进程 (PID: ${VLLM_PID})..."
        kill "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

for entry in "${CONFIGS[@]}"; do
    IFS='|' read -r LABEL QUANT GPU_UTIL <<< "${entry}"

    echo ""
    echo -e "${YELLOW}[运行]${NC} 配置: ${LABEL} (quant=${QUANT}, gpu_util=${GPU_UTIL})..."

    # 启动 vLLM
    VLLM_ARGS="--port ${VLLM_PORT} --gpu-memory-utilization ${GPU_UTIL}"
    if [[ "${QUANT}" != "none" ]]; then
        VLLM_ARGS="${VLLM_ARGS} --quantization ${QUANT}"
    fi

    echo -e "  启动 vLLM: vllm serve ${MODEL_PATH} ${VLLM_ARGS}"
    ${PYTHON} -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_PATH}" \
        ${VLLM_ARGS} \
        --host 0.0.0.0 > "${OUTPUT_DIR}/vllm_${LABEL}.log" 2>&1 &
    VLLM_PID=$!

    # 等待 vLLM 就绪
    echo -e "  等待 vLLM 就绪（最多 180s）..."
    READY=false
    for i in $(seq 1 36); do
        if curl -sf "http://localhost:${VLLM_PORT}/health" &>/dev/null; then
            READY=true
            break
        fi
        sleep 5
    done

    if [[ "${READY}" != "true" ]]; then
        echo -e "${RED}[FAIL]${NC} vLLM 启动超时 (${LABEL})，查看日志: ${OUTPUT_DIR}/vllm_${LABEL}.log"
        echo "| ${LABEL} | ❌ 失败 | - | - | - | - | - |" >> "${COMPARISON_FILE}"
        cleanup
        VLLM_PID=""
        continue
    fi

    echo -e "  ${GREEN}vLLM 就绪${NC}，运行基准测试..."

    # 运行基准测试
    if ${PYTHON} -m backend.experiments.quantization_benchmark \
        --vllm-url "http://localhost:${VLLM_PORT}" \
        --model-path "${MODEL_PATH}" \
        --output-dir "${OUTPUT_DIR}" 2>&1; then
        echo -e "  ${GREEN}[PASS]${NC} ${LABEL} 基准完成"
    else
        echo -e "  ${RED}[FAIL]${NC} ${LABEL} 基准失败"
        echo "| ${LABEL} | ❌ 基准失败 | - | - | - | - | - |" >> "${COMPARISON_FILE}"
    fi

    # 关闭 vLLM
    cleanup
    VLLM_PID=""

    # 等待 GPU 显存释放
    echo -e "  等待 GPU 显存释放..."
    sleep 10
done

echo ""
echo -e "=== 量化对比完成 ==="
echo -e "对比报告: ${YELLOW}${COMPARISON_FILE}${NC}"
echo ""
cat "${COMPARISON_FILE}"
