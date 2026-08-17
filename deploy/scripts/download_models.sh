#!/bin/bash
# ============================================
# MultiPersonal Chat System - 国内镜像模型下载脚本
# 用法: bash download_models.sh [all|base|embed|rerank]
# ============================================
set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

# 国内 HuggingFace 镜像（自动检测）
if [ -z "${HF_ENDPOINT:-}" ]; then
    export HF_ENDPOINT=https://hf-mirror.com
    echo -e "${YELLOW}使用 HF 镜像: ${HF_ENDPOINT}${NC}"
fi

BACKEND_DIR="$(cd "$(dirname "$0")/../../backend" && pwd)"
MODELS_DIR="${BACKEND_DIR}/models"

download_base() {
    echo -e "${GREEN}[1/3] 下载基础模型 Qwen3-8B-Instruct-AWQ (约5GB)...${NC}"
    mkdir -p "${MODELS_DIR}/Qwen3-8B-Instruct-AWQ"
    pip install huggingface_hub -q
    huggingface-cli download Qwen/Qwen3-8B-AWQ \
        --local-dir "${MODELS_DIR}/Qwen3-8B-Instruct-AWQ" \
        --resume-download
    echo -e "${GREEN}基础模型下载完成${NC}"
}

download_embed() {
    echo -e "${GREEN}[2/3] 下载嵌入模型 paraphrase-multilingual-MiniLM-L12-v2 (约420MB)...${NC}"
    mkdir -p "${MODELS_DIR}/paraphrase-multilingual-MiniLM-L12-v2"
    huggingface-cli download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
        --local-dir "${MODELS_DIR}/paraphrase-multilingual-MiniLM-L12-v2" \
        --resume-download
    # 设置环境变量路径，后续启动自动使用
    export EMBEDDING_MODEL_PATH="${MODELS_DIR}/paraphrase-multilingual-MiniLM-L12-v2"
    echo -e "${GREEN}嵌入模型下载完成 → ${EMBEDDING_MODEL_PATH}${NC}"
}

download_rerank() {
    echo -e "${GREEN}[3/3] 下载重排序模型 BAAI/bge-reranker-base (约1.1GB)...${NC}"
    mkdir -p "${BACKEND_DIR}/bge-reranker-base"
    huggingface-cli download BAAI/bge-reranker-base \
        --local-dir "${BACKEND_DIR}/bge-reranker-base" \
        --resume-download
    echo -e "${GREEN}重排序模型下载完成${NC}"
}

# 也可用 ModelScope 作为备选源
download_embed_modelscope() {
    echo -e "${YELLOW}尝试 ModelScope 源下载嵌入模型...${NC}"
    pip install modelscope -q
    python3 -c "
from modelscope import snapshot_download
snapshot_download('iic/nlp_corom_sentence-embedding_chinese-base',
    cache_dir='${HOME}/.cache/modelscope')
# 创建软链接方便代码查找
import os
src = os.path.expanduser('${HOME}/.cache/modelscope/hub/iic/nlp_corom_sentence-embedding_chinese-base')
dst = '${MODELS_DIR}/paraphrase-multilingual-MiniLM-L12-v2'
if not os.path.exists(dst) and os.path.exists(src):
    os.symlink(src, dst)
    print(f'已创建软链接: {dst} → {src}')
" 2>&1
}

case "${1:-all}" in
    base)   download_base ;;
    embed)  download_embed ;;
    rerank) download_rerank ;;
    all)
        download_base
        download_embed
        download_rerank
        ;;
    modelscope)
        download_embed_modelscope
        ;;
    *)
        echo "用法: bash download_models.sh [all|base|embed|rerank|modelscope]"
        echo "  all      - 下载全部模型（推荐）"
        echo "  base     - 仅下载基础模型 Qwen3-8B-AWQ (5GB)"
        echo "  embed    - 仅下载嵌入模型 (420MB)"
        echo "  rerank   - 仅下载重排序模型 (1.1GB)"
        echo "  modelscope - 使用 ModelScope 源下载嵌入模型（备选）"
        ;;
esac

echo -e "${GREEN}=================================${NC}"
echo -e "${GREEN}模型下载完成！${NC}"
echo -e "${GREEN}在 backend/.env 中配置以下变量（如未自动设置）：${NC}"
echo "  EMBEDDING_MODEL_PATH=${MODELS_DIR}/paraphrase-multilingual-MiniLM-L12-v2"
echo "  export HF_ENDPOINT=https://hf-mirror.com"
