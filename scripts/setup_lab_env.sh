#!/usr/bin/env bash
# 快速配置 KISAKI 实验环境
# 用法: bash scripts/setup_lab_env.sh [LAB_ROOT]
# 默认 LAB_ROOT 通过 QQCHAT_LAB_ROOT 环境变量或自动检测
set -euo pipefail

# 自动检测 LAB_ROOT
detect_lab_root() {
    if [[ -n "${QQCHAT_LAB_ROOT:-}" ]]; then
        echo "$QQCHAT_LAB_ROOT"
    elif [[ -d /root/autodl-tmp ]]; then
        echo "/root/autodl-tmp"
    elif [[ -d /home/szw/lhm2 ]]; then
        echo "/home/szw/lhm2"
    else
        echo "$HOME/lab"
    fi
}

LAB_ROOT=${1:-$(detect_lab_root)}
PROJECT=$LAB_ROOT/qqchat-enhanced

echo "=== KISAKI 实验环境配置 ==="
echo "LAB_ROOT: $LAB_ROOT"
echo "PROJECT:  $PROJECT"

# 1. 环境检测
echo ""
echo "=== 1. 环境检测 ==="
echo "OS:       $(uname -s) $(uname -r)"
echo "Python:   $(python3 --version 2>&1)"
if command -v nvidia-smi &>/dev/null; then
    echo "GPU:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader 2>&1 | sed 's/^/  /'
else
    echo "GPU:      (未检测到)"
fi
echo "磁盘:"
df -h "$LAB_ROOT" 2>/dev/null | tail -1 | awk '{print "  总计:"$2, "已用:"$3, "可用:"$4, "使用率:"$5}'

# 2. 创建目录结构
echo ""
echo "=== 2. 创建目录结构 ==="
mkdir -p "$LAB_ROOT/runtime/models"
mkdir -p "$LAB_ROOT/runtime/loras/kisaki"
mkdir -p "$LAB_ROOT/runtime/experiments/kisaki"
mkdir -p "$LAB_ROOT/runtime/logs"
mkdir -p "$LAB_ROOT/runtime/locks"
echo "已创建: $LAB_ROOT/runtime/{models,loras,experiments,logs,locks}"

# 3. 依赖检查
echo ""
echo "=== 3. 依赖检查 ==="
MISSING=()
for pkg in torch transformers peft trl vllm accelerate datasets tensorboard bitsandbytes; do
    ver=$(python3 -c "import importlib.metadata; print(importlib.metadata.version('$pkg'))" 2>/dev/null || echo "MISSING")
    printf "  %-20s %s\n" "$pkg" "$ver"
    [[ "$ver" == "MISSING" ]] && MISSING+=("$pkg")
done
if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "缺失依赖: ${MISSING[*]}"
    echo "安装: pip install ${MISSING[*]}"
fi

# 4. 代码检查
echo ""
echo "=== 4. 代码检查 ==="
if [[ -d "$PROJECT/.git" ]]; then
    cd "$PROJECT"
    echo "  git HEAD:   $(git rev-parse --short HEAD)"
    echo "  git branch: $(git branch --show-current 2>/dev/null || echo '?')"
    if git diff --quiet 2>/dev/null && git diff --cached --quiet 2>/dev/null && [[ -z "$(git ls-files --others --exclude-standard 2>/dev/null)" ]]; then
        echo "  git status: clean"
    else
        echo "  git status: DIRTY (lab-queue 脚本要求 clean)"
        echo "  未跟踪文件:"
        git ls-files --others --exclude-standard 2>/dev/null | head -5 | sed 's/^/    /'
    fi
else
    echo "  代码未找到: $PROJECT"
    echo "  请先 clone: git clone https://github.com/despaoy/qqchat-enhanced.git $PROJECT"
fi

# 5. 模型检查
echo ""
echo "=== 5. 模型检查 ==="
for model in Qwen3-8B-Instruct Qwen3-8B-Instruct-AWQ bge-m3; do
    path="$LAB_ROOT/runtime/models/$model"
    if [[ -f "$path/config.json" ]]; then
        size=$(du -sh "$path" 2>/dev/null | cut -f1)
        echo "  $model: FOUND ($size)"
    else
        echo "  $model: not found"
    fi
done

# 6. Python 路径配置
echo ""
echo "=== 6. Python 路径配置 ==="
# 检查 venv 符号链接是否存在
VENV_PYTHON="$LAB_ROOT/envs/qqchat-gpu-qwen3/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
    ACTUAL_PYTHON=$(which python3)
    echo "  创建 venv 符号链接: $VENV_PYTHON -> $ACTUAL_PYTHON"
    mkdir -p "$(dirname "$VENV_PYTHON")"
    ln -sfn "$ACTUAL_PYTHON" "$VENV_PYTHON"
else
    echo "  venv python 已存在: $VENV_PYTHON"
fi

# 7. 环境变量提示
echo ""
echo "=== 7. 环境变量配置 ==="
echo "请在 ~/.bashrc 中添加以下内容（或运行脚本前 export）:"
echo ""
echo "  export QQCHAT_LAB_ROOT=$LAB_ROOT"
echo "  export QQCHAT_PYTHON=$VENV_PYTHON"
echo ""
echo "或者一行搞定:"
echo "  echo 'export QQCHAT_LAB_ROOT=$LAB_ROOT' >> ~/.bashrc"
echo "  echo 'export QQCHAT_PYTHON=$VENV_PYTHON' >> ~/.bashrc"
echo ""

# 8. 下载缺失模型
echo "=== 8. 下载缺失模型 ==="
NEED_DOWNLOAD=()
for model in Qwen3-8B-Instruct Qwen3-8B-Instruct-AWQ bge-m3; do
    path="$LAB_ROOT/runtime/models/$model"
    [[ -f "$path/config.json" ]] || NEED_DOWNLOAD+=("$model")
done
if [[ ${#NEED_DOWNLOAD[@]} -gt 0 ]]; then
    echo "以下模型缺失，可一键下载:"
    for model in "${NEED_DOWNLOAD[@]}"; do
        echo "  python $PROJECT/scripts/download_model.py --model $model"
    done
    echo ""
    echo "或下载全部:"
    echo "  for m in ${NEED_DOWNLOAD[*]}; do python $PROJECT/scripts/download_model.py --model \$m; done"
else
    echo "所有必需模型已就绪"
fi

echo ""
echo "=== 配置完成 ==="
