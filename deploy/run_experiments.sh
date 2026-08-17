#!/bin/bash
# MultiPersonal Chat System - 一键运行完整实验套件
# 用法: bash deploy/run_experiments.sh [--mock] [--output-dir <dir>]
#
# 顺序执行 4 个实验脚本：
#   1. LoRA 消融 (ablation_runner)
#   2. RAG 消融 (rag_ablation)
#   3. 量化基准 (quantization_benchmark)
#   4. 偏好训练 (preference_trainer)
#
# 每步输出到 <output-dir>/<experiment>.json，汇总到 <output-dir>/summary.md

set -euo pipefail

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[0;33m'; NC='\033[0m'
PASS=0; FAIL=0; SKIP=0

# 切换到项目根目录（deploy/ 的父目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# 参数解析
MOCK=""
OUTPUT_DIR=""
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mock) MOCK="--mock"; shift ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

OUTPUT_DIR="${OUTPUT_DIR:-deploy/results/${TIMESTAMP}}"
mkdir -p "${OUTPUT_DIR}"

# 确定 Python 命令
PYTHON="${PYTHON:-python3}"
if ! command -v "${PYTHON}" &>/dev/null; then
    PYTHON="python"
fi

echo -e "=== MultiPersonal Chat System 实验套件 ==="
echo -e "输出目录: ${YELLOW}${OUTPUT_DIR}${NC}"
echo -e "Mock 模式: ${YELLOW}${MOCK:-否}${NC}"
echo -e "Python: ${YELLOW}${PYTHON}${NC}"
echo ""

# 实验定义: 名称 | 脚本模块 | 输出文件名
EXPERIMENTS=(
    "LoRA消融|backend.experiments.ablation_runner|lora_ablation.json"
    "RAG消融|backend.experiments.rag_ablation|rag_ablation.json"
    "量化基准|backend.experiments.quantization_benchmark|quantization_benchmark.json"
    "偏好训练|backend.training.preference_trainer|preference_training.json"
)

SUMMARY_FILE="${OUTPUT_DIR}/summary.md"
echo "# 实验运行汇总" > "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"
echo "- **运行时间**: ${TIMESTAMP}" >> "${SUMMARY_FILE}"
echo "- **Mock 模式**: ${MOCK:-否}" >> "${SUMMARY_FILE}"
echo "- **Python**: ${PYTHON}" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"
echo "## 实验结果" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"
echo "| 实验 | 状态 | 耗时(s) | 报告路径 |" >> "${SUMMARY_FILE}"
echo "|------|------|---------|----------|" >> "${SUMMARY_FILE}"

TOTAL_START=$(date +%s)

for entry in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r NAME MODULE OUTPUT_FILE <<< "${entry}"
    echo -e "${YELLOW}[运行]${NC} ${NAME}..."

    EXP_OUTPUT="${OUTPUT_DIR}/${OUTPUT_FILE}"
    START=$(date +%s)

    if ${PYTHON} -m ${MODULE} ${MOCK} --output-dir "${OUTPUT_DIR}" > "${EXP_OUTPUT%.json}_log.txt" 2>&1; then
        END=$(date +%s)
        DURATION=$((END - START))
        echo -e "${GREEN}[PASS]${NC} ${NAME} (${DURATION}s)"
        echo "| ${NAME} | ✅ 通过 | ${DURATION} | ${EXP_OUTPUT} |" >> "${SUMMARY_FILE}"
        ((PASS++))
    else
        END=$(date +%s)
        DURATION=$((END - START))
        echo -e "${RED}[FAIL]${NC} ${NAME} (${DURATION}s) - 查看日志: ${EXP_OUTPUT%.json}_log.txt"
        echo "| ${NAME} | ❌ 失败 | ${DURATION} | ${EXP_OUTPUT%.json}_log.txt |" >> "${SUMMARY_FILE}"
        ((FAIL++))
    fi
done

TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))

echo "" >> "${SUMMARY_FILE}"
echo "## 总耗时" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"
echo "总耗时: ${TOTAL_DURATION}s" >> "${SUMMARY_FILE}"
echo "通过: ${PASS}  失败: ${FAIL}" >> "${SUMMARY_FILE}"

echo ""
echo -e "=== 实验套件完成 ==="
echo -e "通过: ${GREEN}${PASS}${NC}  失败: ${RED}${FAIL}${NC}  总耗时: ${YELLOW}${TOTAL_DURATION}s${NC}"
echo -e "汇总报告: ${YELLOW}${SUMMARY_FILE}${NC}"

if [[ ${FAIL} -gt 0 ]]; then
    echo -e "${RED}部分实验失败，请检查日志文件${NC}"
    exit 1
fi

echo -e "${GREEN}所有实验通过！${NC}"
