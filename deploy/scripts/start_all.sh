#!/bin/bash
# ============================================
# MultiPersonal Chat System - 一键启动脚本
# 支持两种模式：docker-compose / bare-metal
# 用法: ./start_all.sh [docker|bare]
# ============================================

set -euo pipefail

# ------------------------------------------
# 配置
# ------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(dirname "${SCRIPT_DIR}")"
PROJECT_DIR="$(dirname "${DEPLOY_DIR}")"
ENV_FILE="${DEPLOY_DIR}/.env"

# 默认配置
DEFAULT_VLLM_PORT1=8001
DEFAULT_VLLM_PORT2=8002
DEFAULT_BACKEND_PORT=8000
DEFAULT_FRONTEND_PORT=5000
DEFAULT_NGINX_PORT=80
DEFAULT_MODEL_PATH="${PROJECT_DIR}/backend/models/Qwen3-8B-Instruct-AWQ"
DEFAULT_LORA_PATH="${PROJECT_DIR}/backend/loras"
DEFAULT_DATA_DIR="${PROJECT_DIR}/backend/data"

# ------------------------------------------
# 颜色输出
# ------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_step()  { echo -e "${BLUE}[STEP]${NC} $*"; }
log_title() { echo -e "${CYAN}========================================${NC}"; echo -e "${CYAN} $* ${NC}"; echo -e "${CYAN}========================================${NC}"; }

# ------------------------------------------
# 重复进程保护
# ------------------------------------------
check_existing_processes() {
    log_step "检查端口占用（重复启动保护）..."

    local ports=("8001" "8000" "5000")
    local names=("vLLM" "FastAPI" "Next.js")
    local conflict=0

    for i in "${!ports[@]}"; do
        local port="${ports[$i]}"
        local name="${names[$i]}"
        # 用 ss 检查端口是否被监听
        if ss -tlnp 2>/dev/null | grep -q ":${port} "; then
            local pid
            pid=$(ss -tlnp 2>/dev/null | grep ":${port} " | grep -oP 'pid=\K[0-9]+' | head -1)
            log_error "${name} 端口 ${port} 已被 PID=${pid:-unknown} 占用"
            conflict=1
        fi
    done

    if [[ ${conflict} -eq 1 ]]; then
        log_error "检测到端口冲突，请先停止现有服务再启动"
        log_error "停止命令: pkill -f 'vllm.entrypoints'; pkill -f 'uvicorn app.main'; pkill -f 'next/dist/bin/next'"
        exit 1
    fi
    log_info "端口检查通过，无冲突"
}

# ------------------------------------------
# 环境检查
# ------------------------------------------
check_environment() {
    log_step "检查运行环境..."

    # 检查操作系统
    if [[ "$(uname -s)" != "Linux" ]]; then
        log_warn "当前系统非 Linux，部分功能可能受限"
    fi

    # 检查 Docker
    if command -v docker &>/dev/null; then
        local docker_version
        docker_version=$(docker --version 2>/dev/null)
        log_info "Docker: ${docker_version}"
    else
        log_warn "Docker 未安装，将无法使用 docker-compose 模式"
        DOCKER_AVAILABLE=false
    fi

    # 检查 nvidia-docker
    if [[ "${DOCKER_AVAILABLE:-true}" == true ]]; then
        if docker info 2>/dev/null | grep -q "NVIDIA" || docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi &>/dev/null; then
            log_info "NVIDIA Container Toolkit: 已安装"
        else
            log_warn "NVIDIA Container Toolkit 未安装，Docker 模式无法使用 GPU"
            NVIDIA_DOCKER_AVAILABLE=false
        fi
    fi

    # 检查 NVIDIA 驱动
    if command -v nvidia-smi &>/dev/null; then
        local gpu_count
        gpu_count=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
        log_info "GPU 数量: ${gpu_count}"

        if [[ ${gpu_count} -lt 2 ]]; then
            log_warn "检测到 ${gpu_count} 块 GPU，建议至少 2 块 GPU 以运行双 vLLM 实例"
        fi

        nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader 2>/dev/null | while IFS= read -r line; do
            log_info "  GPU ${line}"
        done
    else
        log_error "nvidia-smi 未找到，请确认 NVIDIA 驱动已安装"
        exit 1
    fi

    # NOTE: check_model_files 已移至 load_env 之后执行。
    # 此前在 check_environment 中调用时，deploy/.env 尚未加载，
    # MODEL_PATH 会走默认值 backend/models/Qwen3-8B-Instruct-AWQ 导致误报。

    # 检查 Python
    if command -v python3 &>/dev/null; then
        log_info "Python: $(python3 --version 2>/dev/null)"
    elif command -v python &>/dev/null; then
        log_info "Python: $(python --version 2>/dev/null)"
    else
        log_warn "Python 未找到，bare-metal 模式可能无法运行"
    fi

    # 检查 Node.js
    if command -v node &>/dev/null; then
        log_info "Node.js: $(node --version 2>/dev/null)"
    else
        log_warn "Node.js 未找到，bare-metal 模式无法启动前端"
    fi

    # 检查 pnpm
    if command -v pnpm &>/dev/null; then
        log_info "pnpm: $(pnpm --version 2>/dev/null)"
    else
        log_warn "pnpm 未找到，bare-metal 模式无法启动前端"
    fi
}

# ------------------------------------------
# 检查模型文件
# ------------------------------------------
check_model_files() {
    log_step "检查模型文件..."

    local model_path="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"

    if [[ ! -d "${model_path}" ]]; then
        log_error "模型目录不存在: ${model_path}"
        log_error "请先下载模型到该目录，或修改 .env 中的 MODEL_PATH"
        exit 1
    fi

    if [[ ! -f "${model_path}/config.json" ]]; then
        log_error "模型配置文件不存在: ${model_path}/config.json"
        exit 1
    fi

    local model_size
    model_size=$(du -sh "${model_path}" 2>/dev/null | cut -f1)
    log_info "模型路径: ${model_path} (大小: ${model_size})"
}

# ------------------------------------------
# 生成默认 .env 配置
# ------------------------------------------
generate_env_file() {
    if [[ -f "${ENV_FILE}" ]]; then
        log_info "配置文件已存在: ${ENV_FILE}"
        return 0
    fi

    log_step "生成默认配置文件: ${ENV_FILE}"

    cat > "${ENV_FILE}" << EOF
# MultiPersonal Chat System - 部署配置
# 由 start_all.sh 自动生成，请根据实际情况修改

# ---- 模型配置 ----
MODEL_PATH=${DEFAULT_MODEL_PATH}
LORA_PATH=${DEFAULT_LORA_PATH}
VLLM_LORA_ROOT=${DEFAULT_LORA_PATH}

# ---- 端口配置 ----
VLLM_PORT1=${DEFAULT_VLLM_PORT1}
VLLM_PORT2=${DEFAULT_VLLM_PORT2}
BACKEND_PORT=${DEFAULT_BACKEND_PORT}
FRONTEND_PORT=${DEFAULT_FRONTEND_PORT}
NGINX_PORT=${DEFAULT_NGINX_PORT}

# ---- vLLM 配置 ----
VLLM_GPU_MEMORY_UTILIZATION=0.82
VLLM_MAX_MODEL_LEN=24576
VLLM_MAX_NUM_SEQS=8
VLLM_MAX_LORAS=4
VLLM_MAX_LORA_RANK=64
VLLM_DTYPE=float16

# ---- 后端配置 ----
DATABASE_PATH=${DEFAULT_DATA_DIR}/qq_assistant.db
VECTOR_DB_PATH=${DEFAULT_DATA_DIR}/vector_db
LOG_LEVEL=INFO

# ---- 前端配置 ----
NEXT_PUBLIC_API_URL=/api

# ---- Docker Compose 必需变量 (M3 fix) ----
# docker-compose.yml 使用 \${VAR:?set VAR} 强制校验，缺失会导致启动失败
# 请将 CHANGE_ME_ 替换为实际的强随机值
PG_PASSWORD=CHANGE_ME_TO_A_SECURE_PASSWORD
JWT_SECRET=CHANGE_ME_TO_A_SECURE_JWT_SECRET_AT_LEAST_32_CHARS
ENCRYPTION_KEY=CHANGE_ME_TO_BASE64_ENCODED_32_BYTE_KEY
ALLOWED_ORIGINS=http://localhost:${DEFAULT_FRONTEND_PORT},http://localhost:${DEFAULT_NGINX_PORT}
ASTRBOT_INTEGRATION_TOKEN=CHANGE_ME_TO_A_RANDOM_TOKEN
EOF

    log_info "默认配置已生成，请检查: ${ENV_FILE}"
}

# ------------------------------------------
# 加载 .env 配置
# ------------------------------------------
load_env() {
    if [[ -f "${ENV_FILE}" ]]; then
        set -a
        # shellcheck disable=SC1090
        source "${ENV_FILE}"
        set +a
        log_info "已加载配置: ${ENV_FILE}"
    fi

    # 设置默认值
    MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
    LORA_PATH="${LORA_PATH:-${DEFAULT_LORA_PATH}}"
    VLLM_LORA_ROOT="${VLLM_LORA_ROOT:-${LORA_PATH}}"
    VLLM_PORT1="${VLLM_PORT1:-${DEFAULT_VLLM_PORT1}}"
    VLLM_PORT2="${VLLM_PORT2:-${DEFAULT_VLLM_PORT2}}"
    BACKEND_PORT="${BACKEND_PORT:-${DEFAULT_BACKEND_PORT}}"
    FRONTEND_PORT="${FRONTEND_PORT:-${DEFAULT_FRONTEND_PORT}}"
    NGINX_PORT="${NGINX_PORT:-${DEFAULT_NGINX_PORT}}"
}

# ------------------------------------------
# Docker Compose 模式启动
# ------------------------------------------
start_docker() {
    log_title "Docker Compose 模式启动"

    if [[ "${DOCKER_AVAILABLE:-true}" == false ]]; then
        log_error "Docker 未安装，无法使用 docker-compose 模式"
        exit 1
    fi

    if [[ "${NVIDIA_DOCKER_AVAILABLE:-true}" == false ]]; then
        log_error "NVIDIA Container Toolkit 未安装，Docker 模式无法使用 GPU"
        log_error "请安装: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        exit 1
    fi

    # 创建数据目录
    mkdir -p "${DEPLOY_DIR}/data/models" "${DEPLOY_DIR}/data/loras" "${DEPLOY_DIR}/data/app"

    # 如果模型目录为空，提示用户
    if [[ -z "$(ls -A "${DEPLOY_DIR}/data/models/" 2>/dev/null)" ]]; then
        log_warn "模型目录为空，请将模型文件放入: ${DEPLOY_DIR}/data/models/"
        log_warn "或创建软链接: ln -s ${MODEL_PATH} ${DEPLOY_DIR}/data/models/Qwen3-8B-Instruct-AWQ"
    fi

    log_step "启动 Docker Compose..."
    cd "${DEPLOY_DIR}"
    docker compose up -d

    log_step "等待服务就绪..."
    wait_for_services_docker

    print_status_docker
}

# ------------------------------------------
# Bare-metal 模式启动
# ------------------------------------------
start_bare_metal() {
    log_title "Bare-metal 模式启动"

    # 存储子进程 PID
    local pids=()

    # 优雅关闭处理
    # f4 fix: 此前 cleanup_bare 末尾硬编码 exit 0，导致任何显式调用 cleanup_bare
    # 的场景（如 standalone 构建缺失）都以成功码退出，后续 exit 1 成为死代码。
    # 现接受可选退出码参数：信号触发时默认 0，显式调用时可传入 1 表示错误退出。
    cleanup_bare() {
        local exit_code=${1:-0}
        log_info "正在停止所有服务..."
        for pid in "${pids[@]}"; do
            if kill -0 "${pid}" 2>/dev/null; then
                kill -TERM "${pid}" 2>/dev/null || true
            fi
        done
        # 等待进程退出
        sleep 5
        for pid in "${pids[@]}"; do
            if kill -0 "${pid}" 2>/dev/null; then
                kill -KILL "${pid}" 2>/dev/null || true
            fi
        done
        log_info "所有服务已停止"
        exit "${exit_code}"
    }

    trap 'cleanup_bare 0' SIGTERM SIGINT SIGQUIT

    # 启动 vLLM 实例 1
    log_step "启动 vLLM 实例 1 (GPU 0, 端口 ${VLLM_PORT1})..."
    bash "${SCRIPT_DIR}/start_vllm.sh" 0 "${VLLM_PORT1}" "${MODEL_PATH}" "${LORA_PATH}" &
    pids+=($!)

    # 启动 vLLM 实例 2
    log_step "启动 vLLM 实例 2 (GPU 1, 端口 ${VLLM_PORT2})..."
    local gpu_count
    gpu_count=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
    if [[ "${VLLM_SECONDARY_ENABLED:-false}" == "true" && ${gpu_count} -gt 1 ]]; then
        bash "${SCRIPT_DIR}/start_vllm.sh" 1 "${VLLM_PORT2}" "${MODEL_PATH}" "${LORA_PATH}" &
        pids+=($!)
    else
        log_info "Single-GPU mode: secondary vLLM instance disabled"

    fi
    # 等待 vLLM 就绪
    log_step "等待 vLLM 服务就绪..."
    wait_for_url "http://localhost:${VLLM_PORT1}/health" 120 "vLLM-1"
    if [[ "${VLLM_SECONDARY_ENABLED:-false}" == "true" && ${gpu_count} -gt 1 ]]; then
        wait_for_url "http://localhost:${VLLM_PORT2}/health" 120 "vLLM-2"
    fi

    # 启动 FastAPI 后端
    log_step "启动 FastAPI 后端 (端口 ${BACKEND_PORT})..."
    cd "${PROJECT_DIR}/backend"
    python run.py --host 0.0.0.0 --port "${BACKEND_PORT}" &
    pids+=($!)

    # 等待后端就绪
    wait_for_url "http://localhost:${BACKEND_PORT}/health" 30 "FastAPI"

    # 启动 Next.js 前端
    log_step "启动 Next.js 前端 (端口 ${FRONTEND_PORT})..."
    cd "${PROJECT_DIR}"
    if [[ -f ".next/standalone/server.js" ]]; then
        PORT="${FRONTEND_PORT}" HOSTNAME="0.0.0.0" node .next/standalone/server.js &
    else
        # M4 fix: 此前缺失 standalone 构建时会静默回退到 `pnpm dev`（开发服务器），
        # 在生产部署中存在多个问题：
        #   1. 开发服务器未做性能优化，吞吐远低于 standalone；
        #   2. 暴露源码映射与详细错误，增加信息泄露面；
        #   3. 不会启用 middleware 的生产路径匹配等行为，与线上表现不一致。
        # 现改为硬失败并给出构建指引，避免“看似启动成功实则是 dev 模式”的隐患。
        log_error "未找到 standalone 前端构建产物: .next/standalone/server.js"
        log_error "请先执行生产构建: cd \"${PROJECT_DIR}\" && pnpm build"
        log_error "（next.config.ts 需配置 output: 'standalone'）"
        # f4 fix: 触发 cleanup 停止已启动的 vLLM/后端进程，并传入退出码 1。
        # 此前 cleanup_bare 硬编码 exit 0，导致此致命错误被误报为成功。
        cleanup_bare 1
    fi
    pids+=($!)

    # 等待前端就绪
    wait_for_url "http://localhost:${FRONTEND_PORT}" 60 "Next.js"

    print_status_bare "${pids[@]}"

    # 保持脚本运行
    log_info "所有服务已启动，按 Ctrl+C 停止"
    wait
}

# ------------------------------------------
# 等待 URL 可访问
# ------------------------------------------
wait_for_url() {
    local url="$1"
    local max_wait="${2:-60}"
    local name="${3:-service}"
    local retry=0
    local interval=5
    local max_retries=$((max_wait / interval))

    while [[ ${retry} -lt ${max_retries} ]]; do
        if curl -sf "${url}" &>/dev/null; then
            log_info "${name} 已就绪 (${url})"
            return 0
        fi
        retry=$((retry + 1))
        log_info "等待 ${name}... (${retry}/${max_retries})"
        sleep "${interval}"
    done

    log_error "${name} 启动超时 (${url})"
    return 1
}

# ------------------------------------------
# 等待 Docker 服务就绪
# ------------------------------------------
wait_for_services_docker() {
    local services=("vllm1" "vllm2" "backend" "frontend" "nginx")
    local max_wait=300
    local elapsed=0

    while [[ ${elapsed} -lt ${max_wait} ]]; do
        local all_healthy=true

        for service in "${services[@]}"; do
            local status
            status=$(docker inspect --format='{{.State.Health.Status}}' "multipersonal-${service}" 2>/dev/null || echo "unknown")

            if [[ "${status}" != "healthy" ]]; then
                all_healthy=false
                log_info "等待 ${service}... 状态: ${status}"
            fi
        done

        if [[ "${all_healthy}" == true ]]; then
            log_info "所有服务已就绪！"
            return 0
        fi

        sleep 10
        elapsed=$((elapsed + 10))
    done

    log_warn "部分服务未在 ${max_wait}s 内就绪，请手动检查"
}

# ------------------------------------------
# 打印 Docker 模式服务状态
# ------------------------------------------
print_status_docker() {
    log_title "服务状态"

    echo ""
    docker compose -f "${DEPLOY_DIR}/docker-compose.yml" ps
    echo ""

    log_info "访问地址:"
    log_info "  前端:      http://localhost:${NGINX_PORT}"
    log_info "  后端 API:  http://localhost:${NGINX_PORT}/api"
    log_info "  vLLM API:  http://localhost:${NGINX_PORT}/v1"
    log_info "  健康检查:  http://localhost:${NGINX_PORT}/health"
    echo ""
    log_info "管理命令:"
    log_info "  查看日志:  docker compose -f ${DEPLOY_DIR}/docker-compose.yml logs -f"
    log_info "  停止服务:  docker compose -f ${DEPLOY_DIR}/docker-compose.yml down"
    log_info "  重启服务:  docker compose -f ${DEPLOY_DIR}/docker-compose.yml restart"
}

# ------------------------------------------
# 打印 Bare-metal 模式服务状态
# ------------------------------------------
print_status_bare() {
    log_title "服务状态"

    log_info "访问地址:"
    log_info "  前端:      http://localhost:${FRONTEND_PORT}"
    log_info "  后端 API:  http://localhost:${BACKEND_PORT}"
    log_info "  vLLM API:  http://localhost:${VLLM_PORT1}/v1 (实例1)"
    log_info "  vLLM API:  http://localhost:${VLLM_PORT2}/v1 (实例2)"
    log_info "  健康检查:  http://localhost:${BACKEND_PORT}/health"
    echo ""
    log_info "进程 PID:"
    log_info "  vLLM-1 / vLLM-2 / FastAPI / Next.js"
    for pid in "$@"; do
        log_info "  PID: ${pid}"
    done
    echo ""
    log_info "按 Ctrl+C 停止所有服务"
}

# ------------------------------------------
# 选择启动模式
# ------------------------------------------
select_mode() {
    local mode="${1:-}"

    if [[ -n "${mode}" ]]; then
        case "${mode}" in
            docker|d) echo "docker" ;;
            bare|b|bare-metal) echo "bare-metal" ;;
            *) log_error "未知模式: ${mode}（可选: docker, bare）"; exit 1 ;;
        esac
        return 0
    fi

    # 交互式选择
    echo ""
    echo "请选择启动模式:"
    echo "  1) Docker Compose（推荐，需要 Docker + NVIDIA Container Toolkit）"
    echo "  2) Bare-metal（直接在宿主机运行各服务）"
    echo ""
    read -r -p "请输入选项 [1/2]: " choice

    case "${choice}" in
        1) echo "docker" ;;
        2) echo "bare-metal" ;;
        *) log_error "无效选项"; exit 1 ;;
    esac
}

# ------------------------------------------
# 主流程
# ------------------------------------------
main() {
    log_title "MultiPersonal Chat System - 一键启动"

    # 初始化
    DOCKER_AVAILABLE=true
    NVIDIA_DOCKER_AVAILABLE=true

    # 重复进程保护（在任何启动操作之前检查端口冲突）
    check_existing_processes

    # 环境检查
    check_environment

    # 生成/加载配置
    generate_env_file
    load_env

    # 检查模型文件（必须在 load_env 之后，确保 MODEL_PATH 已从 deploy/.env 加载）
    check_model_files

    # 选择启动模式
    local mode
    mode=$(select_mode "$*")

    log_info "启动模式: ${mode}"

    # 启动
    case "${mode}" in
        docker)
            start_docker
            ;;
        bare-metal)
            start_bare_metal
            ;;
    esac
}

main "$@"
