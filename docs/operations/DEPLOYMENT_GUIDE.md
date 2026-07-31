# 部署与验收指南

本指南面向个人研究展示和单机 RTX 3090 部署。Kubernetes、服务网格和多机高可用不属于当前必要范围。

## 1. 前置条件

- Linux x86_64，NVIDIA 驱动能够运行 CUDA 12.x 应用。
- RTX 3090 24GB；启动前至少预留 20GB 显存。
- Python 3.12、Node.js 22、pnpm 10。
- 代码和全部运行数据位于当前用户有权限的目录。

## 2. 获取源码

```bash
cd /home/szw/lhm2
git clone https://github.com/despaoy/qqchat-enhanced.git
cd qqchat-enhanced
git status -sb
```

服务器工作区非干净时，不要直接执行 `git pull`。先保存实验文件，再同步。

## 3. 激活环境

```bash
source /home/szw/lhm2/activate_qqchat.sh
python --version
python -m pip check
node --version
pnpm --version
```

预期至少满足 Python 3.12、Node.js 22，且 `pip check` 不报告冲突。

## 4. 配置

配置文件按运行模式分开，仓库根目录 `.env` 不作为约定入口。

### 4.1 裸机模式

```bash
cd /home/szw/lhm2/qqchat-enhanced
cp .env.example backend/.env
chmod 600 backend/.env
```

后端只从 `backend/.env` 加载配置。必须修改：

- `JWT_SECRET`
- `ENCRYPTION_KEY`
- `ASTRBOT_INTEGRATION_TOKEN`
- `ALLOWED_ORIGINS`
- `MODEL_PROVIDER=vllm`
- `VLLM_BASE_URL=http://127.0.0.1:8001`
- `BASE_MODEL_PATH=/home/szw/lhm2/runtime/models/Qwen3-8B-Instruct`
- `LORA_PATH=/home/szw/lhm2/runtime/loras`
- `VLLM_LORA_ROOT=/home/szw/lhm2/runtime/loras`
- `EMBEDDING_MODEL_PATH=/home/szw/lhm2/runtime/models/bge-m3`
- `RERANKER_MODEL_PATH=/home/szw/lhm2/runtime/models/bge-reranker-v2-m3`

Next.js 本地启动如需覆盖 `BACKEND_URL`，写入仓库根目录 `.env.local`，不要写入根目录 `.env`。

### 4.2 Docker Compose 模式

```bash
cd /home/szw/lhm2/qqchat-enhanced
cp .env.example deploy/.env
chmod 600 deploy/.env
```

Compose 只从 `deploy/.env` 做变量插值。复制后至少设置：

- `ENVIRONMENT=production`
- `PG_PASSWORD=<强随机密码>`
- `JWT_SECRET=<至少 32 字符的强随机值>`
- `ENCRYPTION_KEY=<URL-safe Base64 编码的 32 字节随机密钥>`
- `ASTRBOT_INTEGRATION_TOKEN=<至少 32 字符的强随机值>`
- `ALLOWED_ORIGINS=https://<实际管理台域名>`

标准 Compose 通过 `PG_HOST/PG_USER/PG_PASSWORD/PG_DATABASE` 传递数据库配置，后端会使用结构化 URL 构造器安全编码密码；不要再把原始 `PG_PASSWORD` 手工拼接进 `DATABASE_URL`。裸机如使用完整 `DATABASE_URL`，其中的保留字符必须先做 URL 编码。
标准 Compose 把同一个 `deploy/data/loras` 分别挂载为后端的 `/app/loras` 和 vLLM 的 `/loras`。这两个容器内路径有意不同，不应改成字面相同。

模型文件放在 `deploy/data/models/Qwen3-8B-Instruct-AWQ`；不要把宿主机模型路径误写成容器内路径。

生产或公开网络禁止空 token、默认 JWT 密钥和宽泛 CORS。

## 5. 验证源码

```bash
cd /home/szw/lhm2/qqchat-enhanced
python -m pytest backend/tests -q
pnpm install --frozen-lockfile
pnpm ts-check
pnpm build
```

## 6. 启动顺序

1. PostgreSQL/Redis（如启用）。
2. vLLM。
3. FastAPI。
4. Next.js。
5. AstrBot。

vLLM 示例：

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve \
  /home/szw/lhm2/runtime/models/Qwen3-8B-Instruct-AWQ \
  --served-model-name qwen3-8b-instruct-awq \
  --host 127.0.0.1 \
  --port 8001 \
  --quantization awq_marlin \
  --gpu-memory-utilization 0.88 \
  --max-model-len 8192 \
  --enable-lora \
  --max-lora-rank 64
```

后端：

```bash
cd /home/szw/lhm2/qqchat-enhanced
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

SQLite 只使用一个 worker。切换 PostgreSQL并确认共享限流、队列和缓存后，才考虑增加 worker。

前端：

```bash
cd /home/szw/lhm2/qqchat-enhanced
pnpm build
pnpm start
```

## 7. 健康检查

裸机模式直接检查各进程：

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/ready
curl -fsS http://127.0.0.1:8001/v1/models
curl -fsS http://127.0.0.1:5000/api/health
```

Docker Compose 模式运行：

```bash
bash deploy/verify.sh
```

标准 Compose 只向宿主机发布 Nginx 的 80 端口。验证脚本因此只访问 Nginx 公开入口，不要求发布前端、后端或 vLLM 的内部端口。外部 TLS 或非默认地址可通过 `VERIFY_BASE_URL` 指定。 若 TLS 在外层代理终止，外层代理必须传递 `X-Forwarded-Proto: https`；内层 Nginx 会保留该值，确保 Next.js 的同源/CSRF 判断使用浏览器可见协议。

已有账号时可追加认证链路；管理员还可选择创建并自动删除一个唯一命名的临时知识库：

```bash
VERIFY_BASE_URL=https://chat.example.com VERIFY_USERNAME=operator VERIFY_PASSWORD='<password>' bash deploy/verify.sh
VERIFY_BASE_URL=https://chat.example.com VERIFY_USERNAME=admin VERIFY_PASSWORD='<password>' VERIFY_CREATE_KB=true bash deploy/verify.sh
```

未提供账号时脚本只做无写入的部署就绪检查，不会自动注册或残留测试用户。

真实验收还应覆盖：登录、生成、历史入库、LoRA 扫描/切换、知识库导入/检索、AstrBot 鉴权与幂等、监控指标。

## 8. SSH 映射

在个人电脑执行：

```bash
ssh -L 5000:127.0.0.1:5000 \
    -L 8000:127.0.0.1:8000 \
    -L 8001:127.0.0.1:8001 \
    -L 6185:127.0.0.1:6185 \
    <lab-user>@<lab-host>
```

浏览器访问 `http://127.0.0.1:5000`，AstrBot 面板访问 `http://127.0.0.1:6185`。
## 9. SQLite 备份与恢复

SQLite 模式下，`BACKUP_DIR` 的相对路径以 `DATABASE_PATH` 所在目录为基准；未配置时默认使用数据库旁的 `backups/`。管理 API 支持列出和创建备份，但运行中的线程可能仍持有旧数据库连接，因此在线恢复会明确返回 `409`，不会伪装成成功。

离线恢复步骤：

```bash
# 1. 停止 FastAPI、NoneBot 和其他所有 SQLite 写入进程。
# 2. 执行带完整性检查和旧库安全副本的原子恢复。
python scripts/restore_sqlite_backup.py \
  --backup /path/to/qq_assistant_YYYYMMDD_HHMMSS_full.bak.gz \
  --database /path/to/qq_assistant.db \
  --confirm-backend-stopped
# 3. 重新启动后端并检查 /health、/ready 和历史记录。
```

脚本先解压到数据库同目录的临时文件，执行 `PRAGMA integrity_check`，再保留旧数据库的 `*.safety-<UTC时间>.db` 副本并原子替换。恢复成功并完成业务核验前不要删除安全副本。