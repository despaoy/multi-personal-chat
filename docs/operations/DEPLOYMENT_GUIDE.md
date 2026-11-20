# 部署与验收指南

本指南面向个人研究展示和单机 RTX 3090 部署。Kubernetes、服务网格和多机高可用不属于当前必要范围。

## 1. 前置条件

- Linux x86_64，NVIDIA 驱动能够运行 CUDA 12.x 应用。
- RTX 3090 24GB；启动前至少预留 20GB 显存。
- Python 3.12、Node.js 22、pnpm 10。
- 代码和全部运行数据位于当前用户有权限的目录。

## 2. 获取源码

```bash
cd "$MULTIPERSONAL_LAB_ROOT"
git clone https://github.com/despaoy/multi-personal-chat.git
cd multi-personal-chat
git status -sb
```

服务器工作区非干净时，不要直接执行 `git pull`。先保存实验文件，再同步。

## 3. 激活环境

```bash
source "$MULTIPERSONAL_LAB_ROOT"/activate_qqchat.sh
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
cd "$MULTIPERSONAL_LAB_ROOT/multi-personal-chat"
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
- `BASE_MODEL_PATH="$MULTIPERSONAL_LAB_ROOT"/runtime/models/Qwen3-8B-Instruct`
- `LORA_PATH="$MULTIPERSONAL_LAB_ROOT"/runtime/loras`
- `VLLM_LORA_ROOT="$MULTIPERSONAL_LAB_ROOT"/runtime/loras`
- `EMBEDDING_MODEL_PATH="$MULTIPERSONAL_LAB_ROOT"/runtime/models/paraphrase-multilingual-MiniLM-L12-v2`
- `RERANKER_MODEL_PATH="$MULTIPERSONAL_LAB_ROOT"/runtime/models/bge-reranker-v2-m3`
- `RERANKER_DEVICE=auto` 默认按可用设备选择。需要把重排留在 CPU 时设为 `cpu`，也可显式指定 `cuda:0` 等设备。
- 原作资料启用模型重排时，每个 API 进程最多同时执行两次。`CHARACTER_SOURCE_RERANK_TIMEOUT_SECONDS` 默认 25 秒，限制这一轮等待模型重排的时间。名额已满或超时时，按已核实原文的字面覆盖率排序并继续回复。
- `CHARACTER_RAG_INDEX_ROOT="$MULTIPERSONAL_LAB_ROOT"/runtime/rag/character_knowledge_index_v3`
- `CHARACTER_RAG_ABSTAIN_THRESHOLD=0.25`

Next.js 本地启动如需覆盖 `BACKEND_URL`，写入仓库根目录 `.env.local`，不要写入根目录 `.env`。

### 4.2 Docker Compose 模式

`deploy/scripts/start_all.sh docker` 使用 Compose 的 `up --wait --wait-timeout 300` 等待实际服务的健康检查，要求支持该选项的 Docker Compose v2。服务失败或超时会以非零状态退出，不再依赖手写容器名清单，也不会将未就绪报告为启动成功。

生产 PostgreSQL 首次启动或版本升级时，先按[启动顺序](#6-启动顺序)完成数据库迁移，再执行一键启动。

```bash
cd "$MULTIPERSONAL_LAB_ROOT/multi-personal-chat"
cp .env.example deploy/.env
chmod 600 deploy/.env
```

从 `deploy/` 运行 Compose 时，`deploy/.env` 提供插值默认值，shell 中已导出的同名变量优先。复制后至少设置：[Compose 环境变量优先级](https://docs.docker.com/compose/how-tos/environment-variables/variable-interpolation/#ways-to-set-variables-with-interpolation)

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

#### Compose 使用云端模型

从 `deploy/` 运行时，将 `OPENAI_COMPAT_BASE_URL`、`OPENAI_COMPAT_MODEL`、`OPENAI_COMPAT_API_KEY` 配置在受保护的 Compose 环境中。先完成 PostgreSQL 迁移，再叠加云端配置启动。云端配置需要 Compose v2.24.4 或更新版本。[Compose 文件合并规则](https://docs.docker.com/reference/compose-file/merge/#replace-value)

```bash
docker compose -f docker-compose.yml -f docker-compose.cloud.yml up -d --wait
```

此配置默认启动 PostgreSQL、Redis、API、网页和 Nginx，不调度 vLLM。需要聊天 worker 时，以同一组配置叠加 `--profile character-chat`，按[启动顺序](#6-启动顺序)设置 Temporal。API 与 worker 使用同一云端接入配置。该环境没有本地生成模型时，本地限定会话会提示缺少本地模型，保留其网络限制。已运行环境切换模型应另行安排停机和验证。

### 4.3 角色知识索引

多粒度角色知识索引是生成产物，不随源码发布。部署前使用与运行时相同的
本地 `paraphrase-multilingual-MiniLM-L12-v2` 模型构建（384 维，不可替换为 BGE-M3）：

```bash
cd "$MULTIPERSONAL_LAB_ROOT/multi-personal-chat"
python backend/scripts/build_character_rag_index.py \
  --embedding-model-path "$MULTIPERSONAL_LAB_ROOT/runtime/models/paraphrase-multilingual-MiniLM-L12-v2"
```

默认输出到
`backend/data/knowledge/tsukiyashiro_kisaki/character_knowledge_index_v3/`。
标准 Compose 和 15GB 显存 Compose 都会将该目录只读挂载为
`/app/character_knowledge_index`，并设置 `CHARACTER_RAG_INDEX_ROOT`。

裸机部署如将索引复制到仓库外，必须在 `backend/.env` 中更新
`CHARACTER_RAG_INDEX_ROOT`。每个索引分区都应同时包含 `documents.jsonl`、
`vectors.npy` 和 `manifest.json`。索引缺失时后端仍可启动，但需要该索引的
角色作品检索会明确失败；聊天检索接口返回 503，不会自动改查通用知识库或改用其他提供方。

### 4.4 文件来源

管理端扫描和导入使用同一个只读文件来源接口，默认读取 `backend/knowledge_bases/`。接口只接收相对键，提供条目元数据、直接子项和文件内容，后续可接入 S3 兼容对象存储。部署当前版本只需准备本地目录。导入后的正文、分块和版本继续由数据库管理，备份文件继续走独立的备份流程。

## 5. 验证源码

```bash
cd "$MULTIPERSONAL_LAB_ROOT/multi-personal-chat"
python -m pytest backend/tests -q
pnpm install --frozen-lockfile
pnpm ts-check
pnpm build
```

## 6. 启动顺序

1. PostgreSQL 和 Redis（按部署选择）。使用角色聊天的后台功能时，准备可达的 Temporal 服务。使用原作资料检索时，准备可达的 OpenSearch 服务。
2. 启动 vLLM，或配置已授权的云端模型。
3. 升级 PostgreSQL，启动 FastAPI。
4. 按需启动独立的角色聊天 worker。
5. 启动 Next.js，按需接入 AstrBot。

vLLM 示例：

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve \
  "$MULTIPERSONAL_LAB_ROOT"/runtime/models/Qwen3-8B-Instruct-AWQ \
  --served-model-name qwen3-8b-instruct-awq \
  --host 127.0.0.1 \
  --port 8001 \
  --quantization awq_marlin \
  --gpu-memory-utilization 0.88 \
  --max-model-len 24576 \
  --max-num-seqs 8 \
  --enable-lora \
  --max-lora-rank 64
```

后端。PostgreSQL 部署先备份数据库，将与运行时一致的 `DATABASE_URL` 或 `PG_*` 注入迁移进程，并为 API 和 worker 设置 `USE_POSTGRESQL=true`。目标 PostgreSQL 需安装 pgvector，由数据库管理员预先启用 `vector` 扩展，或让迁移账号启用扩展。

```bash
cd "$MULTIPERSONAL_LAB_ROOT/multi-personal-chat/backend"
python -m alembic upgrade head
python -m alembic current
python run.py --host 127.0.0.1 --port 8000 --workers 1
```

角色聊天分支的迁移 head 为 `037_character_chat_history`。该迁移将继承历史的正文与出处转为账号内共享节点，保留每条路径各自的关系和记忆检查点。SQLite 本地运行由后端启动时完成同样的迁移。升级前先备份，并停止旧版 API 和聊天 worker。两种数据库都采用 `BACKEND_WORKERS=1`，因为部分会话锁和 nonce 使用进程内状态。旧 AstrBot 部署仍需让插件和后端使用兼容版本。

生产 PostgreSQL 的 API 和聊天 worker 启动前会检查数据库的 Alembic 版本。未迁移、落后或超前的数据库会拒绝启动。标准 Docker Compose 在首次启动或版本升级时，可在备份后从 `deploy/` 使用当前版本的后端镜像运行迁移：

```bash
docker compose up -d --wait postgres
docker compose build backend
docker compose run --rm --no-deps backend python -m alembic upgrade head
docker compose run --rm --no-deps backend python -m alembic current
```

迁移成功后再按正常流程启动应用。运行迁移的容器使用与 API 相同的 PostgreSQL 配置。

独立角色聊天 worker 在同一代码版本和后端目录运行，与 API 指向同一个数据库：

```bash
python -m workflows.run_chat_worker
```

- Worker 通过 `TEMPORAL_ADDRESS`、`TEMPORAL_NAMESPACE` 和 `TEMPORAL_CHAT_TASK_QUEUE` 连接 Temporal，可按部署配置 TLS 和 API key。
- API 开启 `CHARACTER_CHAT_BACKGROUND_ENABLED` 后，正式回复会生成记忆、关系及摘要的后台处理任务。准备好 worker 后再开启。私人记忆向量、摘要和模型分别配置。
- 原作资料索引由 worker 的 `CHARACTER_SOURCE_INDEXING_ENABLED` 开启。API 和 worker 使用相同的 `CHARACTER_SEARCH_URL`、`CHARACTER_SEARCH_INDEX` 和本地嵌入模型配置。OpenSearch 认证配置可以在两端分别注入。
- 应用内主动联系由 worker 的 `CHARACTER_CHAT_PROACTIVE_ENABLED` 开启，实际发送仍需要会话授权。浏览器提醒还需向 API 和 worker 注入同一套 VAPID 配置，并通过浏览器的安全上下文访问网页。配置格式见 [浏览器提醒交付记录](../development/MODULE_DELIVERY.md#浏览器提醒)。
- 虚构日常由聊天 worker 使用后端模型配置生成。独立启动时为 worker 配置所选模型和后端运行环境，沿用会话的本地模型限制。生成超时或模型不可用时，应用内记录采用带虚构标识的模板。

标准 Docker Compose 可使用 `character-chat` profile 启动同版本的聊天 worker，也可使用上面的独立进程。两种方式均连接到已就绪的 Temporal。资料功能还需运行可从 API 和 worker 访问的 OpenSearch。

- 使用 Compose 时，将配置注入 `deploy/` 下的 Compose 环境。设置容器可到达的 `TEMPORAL_ADDRESS` 与 `CHARACTER_SEARCH_URL`，并按需提供认证。只有服务与容器在同一网络时才使用对应服务名。容器的 `127.0.0.1` 指向容器自身。
- 资料查询与索引分别由 `CHARACTER_CHAT_SOURCES_ENABLED` 和 `CHARACTER_SOURCE_INDEXING_ENABLED` 启用。模型目录位于 Compose 的 `deploy/data/models/`，容器内以只读方式挂载到 `/app/models`。默认使用 CPU，也可以在容器可用的设备范围内显式配置。
- 本地限定会话默认可访问 Compose 内的 vLLM。自托管 OpenSearch 或其他本地模型端点需要将其 origin 显式加入 `LOCAL_MODEL_ALLOWED_ORIGINS`，例如 `http://search.lan:9200`。
- 在 LHM 使用 Compose 前，应确认容器运行时的镜像、卷和日志目录都在 `/home/boot/lhm/` 下。预先核对模型目录与符号链接的实际目标。

迁移成功后，启用可选 worker：

```bash
docker compose --profile character-chat up -d --build --wait
docker compose --profile character-chat logs --tail 80 chat-worker
```

在 Temporal 核对对应任务队列已有聊天 worker，再用测试会话验证后台处理或索引更新。旧版 15G Compose 可继续配合独立启动的 worker 使用。

前端：

```bash
cd "$MULTIPERSONAL_LAB_ROOT/multi-personal-chat"
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

真实验收还应覆盖：登录、生成、历史入库、LoRA 扫描/切换、角色知识检索、通用知识库导入/检索、AstrBot 鉴权与幂等、监控指标。

AstrBot 发送成功后调用 `/api/integrations/astrbot/delivery`；首次确认后才提交角色历史、记忆与关系更新。验证发送失败重试、已生成回复复用和重复确认。平台发送成功不代表用户已读，跨平台发送和数据库确认不构成原子事务，不能承诺严格恰好一次。

角色知识检索低置信时，聊天入口会让模型按人物口吻表达不知道，仍标记 `abstained=true`、`answerMode=abstention` 且引用为空。固定弃答是表达生成失败时的兜底；独立证据问答接口有自己的弃答策略。

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
# 1. 停止 FastAPI 和其他所有 SQLite 写入进程，并暂停 AstrBot 转发。
# 2. 执行带完整性检查和旧库安全副本的原子恢复。
python scripts/restore_sqlite_backup.py \
  --backup /path/to/qq_assistant_YYYYMMDD_HHMMSS_full.bak.gz \
  --database /path/to/qq_assistant.db \
  --confirm-backend-stopped
# 3. 重新启动后端并检查 /health、/ready 和历史记录。
```

脚本先解压到数据库同目录的临时文件，执行 `PRAGMA integrity_check`，再保留旧数据库的 `*.safety-<UTC时间>.db` 副本并原子替换。恢复成功并完成业务核验前不要删除安全副本。

恢复旧备份时，脚本会合并目标数据库已保存的会话删除回执，先从待恢复文件清除对应会话及附属记录，再替换目标文件。回执只保存会话 ID。目标库丢失时，可用 `--deletion-records-from /path/to/newer.sqlite` 提供一份较新的数据库快照。应保留含最新回执的快照；缺少回执的历史删除无法从旧备份中推断。未被勾选删除的独立分支与副本仍会保留。

离线恢复会关闭备份中的主动联系授权，并取消尚未发送的安排。已送达的应用内消息继续作为正式记录，用户可以在恢复后重新授权。

离线恢复也会关闭备份中的角色分享链接，使恢复前的发布预览失效。角色设定、已发布快照和接收方已经创建的独立副本随备份恢复。作者可以重新预览并发布新链接。

SQLite 和 PostgreSQL 的离线恢复都会清除跨角色记忆共享授权。来源会话保存的记忆和接收会话已导入的副本随备份恢复。需要让新会话带入该记忆时，用户须在来源会话重新共享。

如检测到目标 SQLite 文件存在未检查点的 WAL，脚本会中止。需先确保所有写入方停止，再让 SQLite 完成检查点后重试。

## 10. PostgreSQL 备份与离线恢复

定期使用 `pg_dump --format=custom` 生成备份。恢复时保留原数据库与备份，停止 API、聊天 worker 和其他写入方。在独立且名称不同的数据库中用 `pg_restore --exit-on-error` 导入备份，再运行项目的 Alembic 迁移到当前版本。这些操作沿用上面的 PostgreSQL 连接配置，迁移进程连接恢复副本。

通过受保护的环境配置设置 `RESTORE_STAGED_DATABASE_URL` 和 `RESTORE_RECEIPT_DATABASE_URL`。前者指向迁移后的恢复副本，后者指向原数据库或含最新删除回执的独立数据库。回执连接只需读取删除回执表。两个数据库需来自同一原服务，回执源应包含直到所有写入方停止时的删除。缺失的回执无法从旧备份推断。

```bash
python scripts/prepare_postgres_restore.py --confirm-writers-stopped
```

脚本在同一事务内合并回执、删除恢复出来的已擦除会话及附属记录，并清理无人引用的历史节点。仍有路径引用的节点、未选中的分支和已送达的应用内消息保留。恢复副本上的主动联系授权关闭，未发送任务取消，浏览器需要重新授权。角色分享链接关闭，旧发布预览失效，作者可以重新预览并发布新链接。若遇到表锁或数据约束错误，事务整体回滚。先核对恢复副本中的删除和保留样本、启动检查及登录，再切换 API 和 worker 的数据库连接。保留原库供回退。
