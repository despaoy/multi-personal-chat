#!/bin/bash
# QQ智能助手 - 服务器实验编排脚本
# 用法: bash deploy/run_server_experiments.sh [--phase N] [--skip-mock] [--all]
#
# 阶段:
#   0: Mock 验证（4个实验框架）
#   1: 数据生成（SFT数据 + KB导入 + 偏好对）
#   2: Phase C - RAG 消融实验
#   3: Phase D - 量化基准（当前 AWQ）
#   4: Phase E - DPO 偏好训练
#   5: Phase B - LoRA 消融实验

set -euo pipefail

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[0;33m'; NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# 环境变量
export PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
export BASE_MODEL_PATH="${BASE_MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-7B-Instruct}"
export AWQ_MODEL_PATH="${AWQ_MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-7B-Instruct-AWQ}"
export EMBEDDING_MODEL_PATH="${EMBEDDING_MODEL_PATH:-/root/autodl-tmp/models/paraphrase-multilingual-MiniLM-L12-v2}"
export VLLM_URL="${VLLM_URL:-http://localhost:8001}"
export VECTOR_DB_PATH="${VECTOR_DB_PATH:-/root/autodl-tmp/qqchat-data/vector_db}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="deploy/results/real"
mkdir -p "${OUTPUT_DIR}"

# 参数解析
PHASE=""
SKIP_MOCK=false
RUN_ALL=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase) PHASE="$2"; shift 2 ;;
        --skip-mock) SKIP_MOCK=true; shift ;;
        --all) RUN_ALL=true; shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

run_phase() {
    local num=$1
    local name=$2
    echo -e "\n${YELLOW}=== Phase ${num}: ${name} ===${NC}"
    local log="${OUTPUT_DIR}/phase${num}_log.txt"
    echo "[$(date)] Phase ${num} started" > "${log}"
}

phase_done() {
    local num=$1
    local status=$2
    echo "[$(date)] Phase ${num} ${status}" >> "${OUTPUT_DIR}/phase${num}_log.txt"
    if [[ "${status}" == "completed" ]]; then
        echo -e "${GREEN}[PASS]${NC} Phase ${num} completed"
    else
        echo -e "${RED}[FAIL]${NC} Phase ${num} failed"
    fi
}

# ============================================
# Phase 0: Mock 验证
# ============================================
phase_0() {
    run_phase 0 "Mock 验证"
    local pass=0; local fail=0

    echo "Running mock ablation..."
    if ${PYTHON} -m backend.experiments.ablation_runner --mock --output-dir "${OUTPUT_DIR}/mock" >> "${OUTPUT_DIR}/phase0_log.txt" 2>&1; then
        pass=$((pass + 1)); else fail=$((fail + 1)); fi

    echo "Running mock RAG ablation..."
    if ${PYTHON} -m backend.experiments.rag_ablation --mock --output-dir "${OUTPUT_DIR}/mock" >> "${OUTPUT_DIR}/phase0_log.txt" 2>&1; then
        pass=$((pass + 1)); else fail=$((fail + 1)); fi

    echo "Running mock quantization benchmark..."
    if ${PYTHON} -m backend.experiments.quantization_benchmark --mock --output-dir "${OUTPUT_DIR}/mock" >> "${OUTPUT_DIR}/phase0_log.txt" 2>&1; then
        pass=$((pass + 1)); else fail=$((fail + 1)); fi

    echo "Running mock preference trainer..."
    if ${PYTHON} -m backend.training.preference_trainer --mock --output-dir "${OUTPUT_DIR}/mock" >> "${OUTPUT_DIR}/phase0_log.txt" 2>&1; then
        pass=$((pass + 1)); else fail=$((fail + 1)); fi

    echo -e "Mock 验证: ${GREEN}${pass} passed${NC}, ${RED}${fail} failed${NC}"
    if [[ ${fail} -gt 0 ]]; then
        phase_done 0 "failed"
        return 1
    fi
    phase_done 0 "completed"
}

# ============================================
# Phase 1: 数据生成
# ============================================
phase_1() {
    run_phase 1 "数据生成"

    echo "1a: 生成 SFT 训练数据..."
    if ${PYTHON} -m backend.data.gen_sft_training_data \
        --vllm-url "${VLLM_URL}" \
        --output backend/hutao_dialogues.json \
        >> "${OUTPUT_DIR}/phase1_log.txt" 2>&1; then
        echo -e "  ${GREEN}SFT 数据生成完成${NC}"
    else
        echo -e "  ${RED}SFT 数据生成失败${NC}"
        phase_done 1 "failed"
        return 1
    fi

    echo "1b: 导入 KB 种子文档..."
    if ${PYTHON} -m backend.knowledge.seed_kb_importer --verify \
        >> "${OUTPUT_DIR}/phase1_log.txt" 2>&1; then
        echo -e "  ${GREEN}KB 文档导入完成${NC}"
    else
        echo -e "  ${RED}KB 文档导入失败${NC}"
        phase_done 1 "failed"
        return 1
    fi

    echo "1c: 生成偏好对..."
    if ${PYTHON} -m backend.data.gen_preference_pairs \
        --sft-data backend/hutao_dialogues.json \
        --output backend/data/preference_pairs.jsonl \
        >> "${OUTPUT_DIR}/phase1_log.txt" 2>&1; then
        echo -e "  ${GREEN}偏好对生成完成${NC}"
    else
        echo -e "  ${RED}偏好对生成失败${NC}"
        phase_done 1 "failed"
        return 1
    fi

    phase_done 1 "completed"
}

# ============================================
# Phase 2: RAG 消融实验
# ============================================
phase_2() {
    run_phase 2 "Phase C - RAG 消融实验"
    if ${PYTHON} -m backend.experiments.rag_ablation \
        --output-dir "${OUTPUT_DIR}" \
        >> "${OUTPUT_DIR}/phase2_log.txt" 2>&1; then
        echo -e "  ${GREEN}RAG 消融实验完成${NC}"
        phase_done 2 "completed"
    else
        echo -e "  ${RED}RAG 消融实验失败${NC}"
        phase_done 2 "failed"
        return 1
    fi
}

# ============================================
# Phase 3: 量化基准（当前 AWQ）
# ============================================
phase_3() {
    run_phase 3 "Phase D - 量化基准（AWQ）"
    if ${PYTHON} -m backend.experiments.quantization_benchmark \
        --vllm-url "${VLLM_URL}" \
        --model-path "${AWQ_MODEL_PATH}" \
        --served-model-name "qwen2.5-7b-awq" \
        --labels awq \
        --output-dir "${OUTPUT_DIR}" \
        >> "${OUTPUT_DIR}/phase3_log.txt" 2>&1; then
        echo -e "  ${GREEN}量化基准完成${NC}"
        phase_done 3 "completed"
    else
        echo -e "  ${RED}量化基准失败${NC}"
        phase_done 3 "failed"
        return 1
    fi
}

# ============================================
# Phase 4: DPO 偏好训练
# ============================================
phase_4() {
    run_phase 4 "Phase E - DPO 偏好训练"
    echo "  注意: DPO 训练需要 GPU 显存，可能需要先停止 vLLM"
    echo "  停止 vLLM: screen -X -S vllm quit"
    echo "  训练完成后重启: screen -dmS vllm bash -c '...'"

    if ${PYTHON} -m backend.training.preference_trainer \
        --data backend/data/preference_pairs.jsonl \
        --base-model "${BASE_MODEL_PATH}" \
        --method dpo \
        --epochs 1 \
        --beta 0.1 \
        --output-dir loras/preference_dpo \
        >> "${OUTPUT_DIR}/phase4_log.txt" 2>&1; then
        echo -e "  ${GREEN}DPO 训练完成${NC}"
        phase_done 4 "completed"
    else
        echo -e "  ${RED}DPO 训练失败${NC}"
        phase_done 4 "failed"
        return 1
    fi
}

# ============================================
# Phase 5: LoRA 消融实验
# ============================================
phase_5() {
    run_phase 5 "Phase B - LoRA 消融实验"
    echo "  注意: LoRA 训练耗时较长（每变体 2-4 小时）"
    echo "  优先运行 lora_baseline + dora 两个变体"

    if ${PYTHON} -m backend.experiments.ablation_runner \
        --base-model "${BASE_MODEL_PATH}" \
        --train-data backend/hutao_dialogues.json \
        --variants lora_baseline,dora \
        --output-dir reports/ablation \
        >> "${OUTPUT_DIR}/phase5_log.txt" 2>&1; then
        echo -e "  ${GREEN}LoRA 消融实验完成${NC}"
        phase_done 5 "completed"
    else
        echo -e "  ${RED}LoRA 消融实验失败${NC}"
        phase_done 5 "failed"
        return 1
    fi
}

# ============================================
# 主逻辑
# ============================================
echo -e "=== QQ智能助手 服务器实验编排 ==="
echo -e "Python: ${YELLOW}${PYTHON}${NC}"
echo -e "输出目录: ${YELLOW}${OUTPUT_DIR}${NC}"
echo -e "时间戳: ${YELLOW}${TIMESTAMP}${NC}"
echo ""

if [[ "${RUN_ALL}" == "true" ]]; then
    if [[ "${SKIP_MOCK}" != "true" ]]; then
        phase_0 || { echo -e "${RED}Mock 验证失败，终止${NC}"; exit 1; }
    fi
    phase_1 || exit 1
    phase_2 || exit 1
    phase_3 || exit 1
    phase_4 || exit 1
    phase_5 || exit 1
elif [[ -n "${PHASE}" ]]; then
    case "${PHASE}" in
        0) phase_0 ;;
        1) phase_1 ;;
        2) phase_2 ;;
        3) phase_3 ;;
        4) phase_4 ;;
        5) phase_5 ;;
        *) echo "未知阶段: ${PHASE}"; exit 1 ;;
    esac
else
    echo "请指定 --phase N 或 --all"
    echo "  0: Mock 验证"
    echo "  1: 数据生成"
    echo "  2: RAG 消融"
    echo "  3: 量化基准"
    echo "  4: DPO 训练"
    echo "  5: LoRA 消融"
    exit 1
fi

echo ""
echo -e "=== 实验编排完成 ==="
echo -e "结果目录: ${YELLOW}${OUTPUT_DIR}${NC}"
