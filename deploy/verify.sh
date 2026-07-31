#!/usr/bin/env bash
# QQ智能助手 - Docker Compose 部署验证
#
# 默认只访问 Compose 发布的 Nginx 入口（http://127.0.0.1）。
# 可选：
#   VERIFY_BASE_URL=https://chat.example.com
#   VERIFY_USERNAME=<existing-user> VERIFY_PASSWORD=<password>
#   VERIFY_CREATE_KB=true  # 仅管理员；创建后立即删除唯一命名的测试知识库

set -uo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'
PASS=0
FAIL=0
SKIP=0

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${COMPOSE_FILE:-${SCRIPT_DIR}/docker-compose.yml}"
VERIFY_BASE_URL="${VERIFY_BASE_URL:-http://127.0.0.1}"
VERIFY_BASE_URL="${VERIFY_BASE_URL%/}"
VERIFY_TIMEOUT_SECONDS="${VERIFY_TIMEOUT_SECONDS:-300}"
VERIFY_CREATE_KB="${VERIFY_CREATE_KB:-false}"
COMPOSE=(docker compose --project-directory "${SCRIPT_DIR}" -f "${COMPOSE_FILE}")

COOKIE_JAR=""
CREATED_KB_ID=""

pass() {
    printf "%b[PASS]%b %s\n" "${GREEN}" "${NC}" "$1"
    PASS=$((PASS + 1))
}

fail() {
    printf "%b[FAIL]%b %s\n" "${RED}" "${NC}" "$1"
    FAIL=$((FAIL + 1))
}

skip() {
    printf "%b[SKIP]%b %s\n" "${YELLOW}" "${NC}" "$1"
    SKIP=$((SKIP + 1))
}

is_true() {
    case "${1,,}" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

json_escape() {
    local value="$1"
    value="${value//\\/\\\\}"
    value="${value//\"/\\\"}"
    value="${value//$'\n'/\\n}"
    value="${value//$'\r'/\\r}"
    value="${value//$'\t'/\\t}"
    printf '%s' "${value}"
}

json_succeeded() {
    grep -Eq '"success"[[:space:]]*:[[:space:]]*true' <<<"$1"
}

wait_for_url() {
    local url="$1"
    local deadline=$((SECONDS + VERIFY_TIMEOUT_SECONDS))

    while ((SECONDS < deadline)); do
        if curl -fsS --connect-timeout 5 --max-time 15 -o /dev/null "${url}"; then
            return 0
        fi
        sleep 5
    done
    return 1
}

check_url() {
    local name="$1"
    local url="$2"
    if curl -fsS --connect-timeout 5 --max-time 30 -o /dev/null "${url}"; then
        pass "${name}"
    else
        fail "${name}"
    fi
}

cleanup() {
    local exit_status=$?

    if [[ -n "${CREATED_KB_ID}" && -n "${COOKIE_JAR}" ]] && command -v curl >/dev/null 2>&1; then
        if curl -fsS --connect-timeout 5 --max-time 30 \
            -b "${COOKIE_JAR}" -X DELETE \
            "${VERIFY_BASE_URL}/api/knowledge/bases/${CREATED_KB_ID}" \
            >/dev/null 2>&1; then
            printf "%b[CLEANUP]%b 已删除测试知识库 %s\n" "${GREEN}" "${NC}" "${CREATED_KB_ID}"
        else
            printf "%b[WARN]%b 无法删除测试知识库 %s，请手动检查\n" \
                "${YELLOW}" "${NC}" "${CREATED_KB_ID}" >&2
        fi
    fi

    if [[ -n "${COOKIE_JAR}" ]]; then
        rm -f -- "${COOKIE_JAR}"
    fi
    return "${exit_status}"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

print_summary() {
    echo
    echo "=== 验证完成 ==="
    printf "通过: %b%s%b  失败: %b%s%b  跳过: %b%s%b\n" \
        "${GREEN}" "${PASS}" "${NC}" \
        "${RED}" "${FAIL}" "${NC}" \
        "${YELLOW}" "${SKIP}" "${NC}"
}

echo "=== QQ智能助手 Docker Compose 部署验证 ==="
echo "公开入口: ${VERIFY_BASE_URL}"

echo "[1/5] 检查本机依赖..."
if ! command -v docker >/dev/null 2>&1; then
    fail "Docker CLI"
    print_summary
    exit 1
fi
if ! docker compose version >/dev/null 2>&1; then
    fail "Docker Compose 插件"
    print_summary
    exit 1
fi
pass "Docker Compose 插件"

if ! command -v curl >/dev/null 2>&1; then
    fail "curl"
    print_summary
    exit 1
fi
pass "curl"

if ! [[ "${VERIFY_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    fail "VERIFY_TIMEOUT_SECONDS 必须是正整数"
    print_summary
    exit 1
fi

echo "[2/5] 启动 Docker Compose..."
if "${COMPOSE[@]}" up -d; then
    pass "Docker Compose 启动"
else
    fail "Docker Compose 启动"
    print_summary
    exit 1
fi

echo "[3/5] 等待 Nginx 公开入口就绪（最长 ${VERIFY_TIMEOUT_SECONDS}s）..."
if wait_for_url "${VERIFY_BASE_URL}/ready"; then
    pass "Backend /ready（经 Nginx）"
else
    fail "Backend /ready（经 Nginx）"
fi
check_url "Backend /health（经 Nginx）" "${VERIFY_BASE_URL}/health"
check_url "Frontend /api/health（经 Nginx）" "${VERIFY_BASE_URL}/api/health"
check_url "Frontend 首页（经 Nginx）" "${VERIFY_BASE_URL}/"

echo "[4/5] 可选认证业务链路..."
VERIFY_USERNAME="${VERIFY_USERNAME:-}"
VERIFY_PASSWORD="${VERIFY_PASSWORD:-}"

if [[ -z "${VERIFY_USERNAME}" && -z "${VERIFY_PASSWORD}" ]]; then
    skip "未提供 VERIFY_USERNAME/VERIFY_PASSWORD；跳过认证与知识库检查"
elif [[ -z "${VERIFY_USERNAME}" || -z "${VERIFY_PASSWORD}" ]]; then
    fail "VERIFY_USERNAME 与 VERIFY_PASSWORD 必须同时提供"
else
    COOKIE_JAR="$(mktemp "${TMPDIR:-/tmp}/qqchat-verify-cookie.XXXXXX")"
    LOGIN_PAYLOAD="$(printf '{"username":"%s","password":"%s"}' \
        "$(json_escape "${VERIFY_USERNAME}")" "$(json_escape "${VERIFY_PASSWORD}")")"
    LOGIN_BODY=""

    if LOGIN_BODY="$(curl -fsS --connect-timeout 5 --max-time 30 \
        -c "${COOKIE_JAR}" -X POST "${VERIFY_BASE_URL}/api/auth/login" \
        -H "Content-Type: application/json" --data "${LOGIN_PAYLOAD}")" \
        && json_succeeded "${LOGIN_BODY}"; then
        pass "登录"

        ME_BODY=""
        if ME_BODY="$(curl -fsS --connect-timeout 5 --max-time 30 \
            -b "${COOKIE_JAR}" "${VERIFY_BASE_URL}/api/auth/me")" \
            && json_succeeded "${ME_BODY}"; then
            pass "读取当前用户"
        else
            fail "读取当前用户"
        fi

        SEARCH_BODY=""
        if SEARCH_BODY="$(curl -fsS --connect-timeout 5 --max-time 60 \
            -b "${COOKIE_JAR}" -X POST "${VERIFY_BASE_URL}/api/knowledge/search" \
            -H "Content-Type: application/json" \
            --data '{"query":"deployment verification","topK":3}')" \
            && json_succeeded "${SEARCH_BODY}"; then
            pass "知识库搜索"
        else
            fail "知识库搜索"
        fi

        if is_true "${VERIFY_CREATE_KB}"; then
            if grep -Eq '"role"[[:space:]]*:[[:space:]]*"admin"' <<<"${LOGIN_BODY}"; then
                KB_NAME="deployment-verify-$(date +%Y%m%d%H%M%S)-$$"
                KB_PAYLOAD="$(printf '{"name":"%s","description":"temporary deployment verification"}' \
                    "$(json_escape "${KB_NAME}")")"
                KB_BODY=""
                if KB_BODY="$(curl -fsS --connect-timeout 5 --max-time 30 \
                    -b "${COOKIE_JAR}" -X POST "${VERIFY_BASE_URL}/api/knowledge/bases" \
                    -H "Content-Type: application/json" --data "${KB_PAYLOAD}")" \
                    && json_succeeded "${KB_BODY}"; then
                    CREATED_KB_ID="$(sed -nE \
                        's/.*"base"[[:space:]]*:[[:space:]]*\{[^}]*"id"[[:space:]]*:[[:space:]]*([0-9]+).*/\1/p' \
                        <<<"${KB_BODY}")"
                    if [[ -n "${CREATED_KB_ID}" ]]; then
                        pass "创建临时知识库"
                        if DELETE_BODY="$(curl -fsS --connect-timeout 5 --max-time 30 \
                            -b "${COOKIE_JAR}" -X DELETE \
                            "${VERIFY_BASE_URL}/api/knowledge/bases/${CREATED_KB_ID}")" \
                            && json_succeeded "${DELETE_BODY}"; then
                            pass "删除临时知识库"
                            CREATED_KB_ID=""
                        else
                            fail "删除临时知识库（退出时将重试）"
                        fi
                    else
                        fail "解析临时知识库 ID（请按名称 ${KB_NAME} 手动检查）"
                    fi
                else
                    fail "创建临时知识库"
                fi
            else
                skip "VERIFY_CREATE_KB=true 需要管理员账号"
            fi
        fi

        LOGOUT_BODY=""
        if LOGOUT_BODY="$(curl -fsS --connect-timeout 5 --max-time 30 \
            -b "${COOKIE_JAR}" -X POST "${VERIFY_BASE_URL}/api/auth/logout")" \
            && json_succeeded "${LOGOUT_BODY}"; then
            pass "注销"
        else
            fail "注销"
        fi
    else
        fail "登录"
    fi
fi

echo "[5/5] 汇总..."
print_summary
if ((FAIL > 0)); then
    echo "请检查失败项及日志：${COMPOSE[*]} logs"
    exit 1
fi

echo "部署验证通过。"
