# MultiPersonal Chat System — 代码知识库

> 本文档为 `multi-personal-chat` 项目的结构化代码知识库（Code Wiki），涵盖项目整体架构、模块职责、关键类与函数、依赖关系与运行方式。
>
> - **版本基线**：FastAPI 后端 `v2.0.0`、Next.js 前端 `0.1.0`（Next 16.2.9 / React 19.2.3）
> - **最近核对**：2026-07-31
> - **维护原则**：随代码演进同步更新；任何架构性变更须同步本文档对应章节。

---

## 目录

1. [项目概览](#1-项目概览)
2. [整体架构](#2-整体架构)
3. [仓库目录结构](#3-仓库目录结构)
4. [后端模块详解](#4-后端模块详解)
   - 4.1 [应用入口与生命周期](#41-应用入口与生命周期)
   - 4.2 [API 路由层](#42-api-路由层)
   - 4.3 [数据库与持久化](#43-数据库与持久化)
   - 4.4 [推理引擎层（inference）](#44-推理引擎层inference)
   - 4.5 [知识检索层（knowledge）](#45-知识检索层knowledge)
   - 4.6 [训练层（training）](#46-训练层training)
   - 4.7 [机器人接入层（bot）](#47-机器人接入层bot)
   - 4.8 [基础设施层（infra）](#48-基础设施层infra)
   - 4.9 [缓存层（cache）](#49-缓存层cache)
   - 4.10 [安全中间件（middleware）](#410-安全中间件middleware)
   - 4.11 [评估与实验体系](#411-评估与实验体系)
   - 4.12 [分层架构（services / repositories / providers）](#412-分层架构services--repositories--providers)
5. [前端模块详解](#5-前端模块详解)
   - 5.1 [应用结构与布局](#51-应用结构与布局)
   - 5.2 [页面清单](#52-页面清单)
   - 5.3 [组件体系](#53-组件体系)
   - 5.4 [Context 与 Hook 体系](#54-context-与-hook-体系)
   - 5.5 [API 客户端与代理机制](#55-api-客户端与代理机制)
   - 5.6 [构建配置](#56-构建配置)
6. [AstrBot 网关插件](#6-astrbot-网关插件)
7. [部署与运行方式](#7-部署与运行方式)
8. [测试体系](#8-测试体系)
9. [CI/CD 流水线](#9-cicd-流水线)
10. [脚本工具清单](#10-脚本工具清单)
11. [依赖关系总览](#11-依赖关系总览)
12. [关键设计要点](#12-关键设计要点)

---

## 1. 项目概览

**MultiPersonal Chat System** 是一个面向角色对话、知识检索、LoRA 适配与可复现 LLM 实验的多平台智能助手系统。

- **平台接入**：通过 AstrBot 网关插件接入 QQ / Telegram / 企业微信 / 公众号 / 个人微信等 IM 平台。
- **核心服务**：FastAPI 提供认证、配置、知识库、训练、评估、实验等完整 REST API。
- **推理后端**：以 vLLM（`Qwen3-8B-Instruct-AWQ`）为主，支持多实例负载均衡、LoRA 热切换；同时兼容 Ollama / llama.cpp / Transformers+PEFT / OpenAI 兼容 / Mock 多后端。
- **RAG 能力**：FAISS 向量检索 + BM25 关键词混合 + Cross-Encoder 重排 + 纠错 RAG。
- **训练能力**：SFT LoRA 微调、DPO/ORPO 偏好对齐，含 GPU 温度保护、早停、可取消。
- **管理控制台**：Next.js 16 (App Router) + React 19 + Tailwind v4 + shadcn/ui 提供 14 个页面。
- **研究严谨性**：Gold Set 评估、质量门（benchmark_gate）、盲评流程、合成数据审核 guardrail。

**当前验证状态**（截至文档核对时）：
- 后端回归：实验室服务器最近一次 `121 passed`；Windows 本地为开发镜像
- TypeScript 与 Next.js 生产构建通过
- vLLM 在 RTX 3090 上稳定服务 `Qwen3-8B-Instruct-AWQ`
- 26 张数据库表已建（PostgreSQL/SQLite 双模式），Alembic 迁移至 `004_index_cleanup`

---

## 2. 整体架构

```text
┌─────────────────────────────────────────────────────────────────┐
│  QQ / Telegram / 企业微信 / 公众号 / 个人微信                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                  ┌──────────▼──────────┐
                  │       AstrBot        │
                  │  (multipersonal_gateway)    │
                  └──────────┬──────────┘
                             │ HMAC 签名 + Token
                  ┌──────────▼──────────┐
                  │   FastAPI 核心服务   │  ← Next.js 控制台（端口 5000）
                  │   (端口 8000)        │
                  └───┬────┬────┬───────┘
                      │    │    │
            ┌─────────┘    │    └────────────┐
            ▼              ▼                 ▼
     ┌─────────────┐  ┌─────────┐     ┌──────────────┐
     │   vLLM      │  │   RAG   │     │ PostgreSQL / │
     │ (端口 8001) │  │ FAISS   │     │   Redis      │
     │ Qwen3-8B    │  │ + BM25  │     │              │
     │ -Instruct   │  │ +Rerank │     └──────────────┘
     └─────────────┘  └─────────┘
            │
            ▼
     ┌─────────────┐
     │  LoRA 适配器 │  ← SFT / DPO / ORPO 训练产出
     │  hutao/minamo│
     │  /kisaki ... │
     └─────────────┘
```

**分层说明**：

| 层 | 职责 | 关键技术 |
|---|---|---|
| 接入层 | 多平台消息收发、归一化、签名认证 | AstrBot + NoneBot2 + OneBot v11 |
| API 层 | RESTful 接口、认证、配置、CRUD | FastAPI + Pydantic + JWT |
| 推理层 | 多后端调度、LoRA 路由、负载均衡 | vLLM + httpx + 熔断器 |
| 知识层 | 向量检索、关键词检索、重排、纠错 | FAISS + BM25 + BGE-Reranker |
| 训练层 | SFT、偏好对齐、任务管理 | PEFT + TRL + BitsAndBytes |
| 基础设施 | 缓存、限流、熔断、加密、备份、可观测 | Redis + 令牌桶 + AES-256-GCM |
| 持久化 | 关系数据、向量数据、文件存储 | PostgreSQL/SQLite + FAISS + 本地文件 |
| 前端 | 管理控制台、可视化、表单 | Next.js 16 + React 19 + shadcn/ui |

---

## 3. 仓库目录结构

```text
multi-personal-chat/
├── backend/                       # FastAPI 后端（Python 3.12）
│   ├── api/                       # 18 个 API 路由模块
│   ├── app/                       # 应用入口、配置、依赖注入、运行时容器
│   │   ├── main.py                # create_app 组合根 + lifespan
│   │   ├── config.py              # 全局配置 + 增强模块单例
│   │   ├── dependencies.py        # get_current_user 依赖
│   │   ├── env.py                 # 统一 backend/.env 加载
│   │   ├── providers.py           # FastAPI 依赖提供器（Repository/Service）
│   │   ├── readiness.py           # ReadinessProbe 缓存型就绪探针
│   │   └── runtime.py             # RuntimeContainer 应用依赖容器
│   ├── bot/                       # NoneBot2 QQ 机器人入口
│   ├── cache/                     # Redis 缓存、语义缓存、消息队列、TTL 缓存
│   ├── db/                        # 数据库适配器（SQLite/PostgreSQL 双模式）
│   │   ├── adapter.py             # 双模式选择与对外契约
│   │   ├── database.py            # SQLite 实现
│   │   ├── pg_database.py         # PostgreSQL 实现 + SyncPgAdapter
│   │   ├── urls.py                # PG URL 规范化（运行时 + Alembic 共享）
│   │   ├── errors.py              # 数据库领域异常
│   │   ├── models.py              # SQLAlchemy ORM 单一权威
│   │   ├── schemas.py             # Pydantic schema
│   │   └── migration.py           # SQLite → PostgreSQL 一次性迁移
│   ├── alembic/                   # 数据库迁移（001 ~ 004）
│   ├── evaluation/                # Gold Set、指标、质量门
│   ├── experiments/               # LoRA/RAG/量化实验框架
│   ├── inference/                 # vLLM 客户端、LoRA 路由、模型管理
│   ├── infra/                     # 14 个基础设施组件
│   ├── interfaces/                # Protocol 接口契约
│   ├── knowledge/                 # RAG 全流水线
│   ├── middleware/                # 安全中间件链
│   ├── repositories/              # 持久化 Protocol + 数据库适配实现
│   │   ├── messages.py            # MessageRepository / MessageQuery
│   │   └── user_data.py           # UserDataRepository
│   ├── services/                  # 与 HTTP 传输无关的业务用例
│   │   ├── chat_generation.py     # ChatGenerationService
│   │   └── model_management.py    # ModelManagementService
│   ├── training/                  # SFT + DPO/ORPO 训练
│   ├── benchmarks/                # 性能基准测试
│   ├── data/                      # 训练数据、角色对话
│   ├── scripts/                   # 后端工具脚本
│   ├── tests/                     # pytest 测试套件
│   ├── main.py / run.py           # 启动入口
│   ├── Dockerfile
│   ├── pyproject.toml
│   └── requirements.txt
│
├── src/                           # Next.js 前端
│   ├── app/                       # App Router 页面 + API 路由
│   │   ├── layout.tsx             # 根布局（三层 Provider）
│   │   ├── page.tsx               # 仪表盘首页
│   │   ├── DashboardClient.tsx    # 仪表盘客户端组件
│   │   ├── api/                   # 10 个 Next.js API 路由
│   │   └── [page]/page.tsx        # 14 个业务页面
│   ├── components/                # 组件（layout/dashboard/training/history/ui）
│   ├── contexts/                  # AuthContext + SettingsContext
│   ├── hooks/                     # 12 个自定义 Hook
│   └── lib/                       # api.ts / api-contracts.ts / proxy.ts / i18n.ts / utils.ts
│
├── astrbot_plugins/
│   └── multipersonal_gateway/            # AstrBot 网关插件
│       ├── main.py
│       ├── metadata.yaml
│       └── README.md
│
├── deploy/                        # 部署配置
│   ├── docker-compose.yml         # 6 服务生产栈
│   ├── docker-compose.15g.yml     # 15GB 显存变体
│   ├── nginx/nginx.conf           # Nginx 反代
│   ├── supervisord.conf           # 进程守护
│   ├── scripts/                   # 启动/下载脚本
│   └── *.sh                       # 实验/量化对比脚本
│
├── scripts/                       # 顶层工具脚本（数据构建/盲评/训练/验证）
├── gametext/                      # 角色对话源文本（纸上魔法使系列）
├── .github/workflows/ci.yml       # GitHub Actions CI
├── Dockerfile                     # 前端 Dockerfile
├── package.json                   # pnpm 前端依赖
├── next.config.ts
├── tsconfig.json
└── *.md                           # 项目文档（README/ROADMAP/STATUS...）
```

> 历史模块：`backend/infra/load_balancer.py`（vLLM 负载均衡已并入 `inference/vllm_client.py`）、`backend/pipeline.py`、`src/middleware.ts`（被 `src/proxy.ts` 替代）已删除。新增分层动机与迁移路径见 [可扩展性开发指南](EXTENSIBILITY_GUIDE.md)。

---

## 4. 后端模块详解

### 4.1 应用入口与生命周期

**核心文件**：`backend/app/main.py`、`backend/app/runtime.py`、`backend/app/providers.py`、`backend/app/readiness.py`、`backend/app/env.py`、`backend/run.py`、`backend/main.py`

#### 4.1.1 FastAPI 应用创建（组合根）

应用通过 `create_app(container: RuntimeContainer | None = None) -> FastAPI` 显式装配，模块级 `app = create_app()` 仅作兼容入口：

```python
# backend/app/main.py
def create_app(container: RuntimeContainer | None = None) -> FastAPI:
    runtime_container = container if container is not None else RuntimeContainer.default(startup_env=_STARTUP_ENV)
    application = FastAPI(title="MultiPersonal Chat System API (增强版)", version="2.0.0", lifespan=lifespan)
    application.state.runtime_container = runtime_container
    application.state.readiness_probe = ReadinessProbe(
        database_check=lambda: get_runtime_container(application).db.execute_sql("SELECT 1"),
        model_check=_readiness_model_check if model_required else None,
        model_required=model_required,
    )
    _install_middleware(application, runtime_container.startup_env)
    for router in _ROUTERS:
        application.include_router(router)
    application.add_api_route("/", root, methods=["GET"])
    application.add_api_route("/health", health_check, methods=["GET"])
    application.add_api_route("/ready", readiness_check, methods=["GET"])
    application.add_exception_handler(Exception, global_exception_handler)
    return application
```

- **路由挂载**：18 个 `APIRouter` 统一以 `/api` 前缀挂载
- **CORS**：最外层中间件，确保 401/429 等错误也带 CORS 头
- **全局异常处理**：`@app.exception_handler(Exception)` 统一返回 500 + 标准格式
- **健康检查**：`GET /health`（简单存活）、`GET /ready`（基于 `ReadinessProbe` 的缓存型就绪探针）

#### 4.1.2 RuntimeContainer 与依赖注入

`backend/app/runtime.py` 定义 `RuntimeContainer`（`@dataclass(slots=True)`），是单一应用实例拥有的进程级依赖集合：

- 字段：`db`（数据库适配器）、`is_pg_mode()`、`inference_runtime`、`startup_env`（启动时快照 `os.environ`，避免运行中环境变更影响校验）
- `RuntimeContainer.default(startup_env=None)`：惰性装配生产依赖（`db.adapter.db`、`db.adapter.is_pg_mode`、`infra.concurrency_control.inference_runtime`）
- `get_runtime_container(app)`：从 `app.state.runtime_container` 取容器，未配置则 `raise RuntimeError`

`backend/app/providers.py` 将 `RuntimeContainer` 适配为 FastAPI 依赖：

| Provider | 返回 | 用途 |
|---|---|---|
| `get_message_repository(request)` | `DatabaseMessageRepository` | 消息历史/会话仓储 |
| `get_user_data_repository(request)` | `DatabaseUserDataRepository` | 用户页面状态仓储 |
| `get_model_management_service()` | `ModelManagerService` | 模型管理业务用例（包装同步 `ModelManager`） |

`backend/app/env.py:load_backend_env()` 统一加载 `backend/.env`，所有入口点（`run.py`/`main.py`/`app/main.py`）共用同一加载逻辑；显式注入的容器/进程变量优先，不会被 `.env` 覆盖。

#### 4.1.3 ReadinessProbe 缓存型就绪探针

`backend/app/readiness.py` 的 `ReadinessProbe` 替换了原先每请求直查数据库的 `/ready` 实现：

- **TTL 缓存**：成功 5s、失败 1s，避免探针把数据库打满
- **单飞合并**：`_refresh_lock` + 共享 `_refresh_task`，并发请求复用同一次刷新
- **同步检查有界卸载**：同步 `database_check` 通过 `BoundedThreadExecutor`（`max_workers=2, max_pending=2`）执行，超时由 `check_timeout=3.0s` 控制
- **等待预算**：`wait_timeout=4.0s`，超时抛 `ReadinessProbeTimeout` → 503
- **依赖矩阵**：`database_check` 必需；`model_check` 仅在 `MODEL_PROVIDER=vllm` 或 `is_vllm_enabled` 时必需，否则返回 `not_required`
- **快照结构**：`{ready, deps:{database, model}, details:{database, model, rag:"not_probed"}}`，深拷贝返回避免外部篡改缓存

#### 4.1.4 生命周期 `lifespan(app)`

启动顺序（带条件初始化）：

1. **环境校验**：`validate_or_raise_for_startup(_STARTUP_ENV)` 强制校验生产环境关键变量，缺失即 `raise RuntimeError` 阻断启动
2. **数据库探活**：`_initialize_database(db)` 执行 `SELECT 1`，失败阻断启动
3. **Redis 缓存**（可选）：失败降级为数据库直连模式
4. **资源池 / 备份 / 访问控制**：**仅 SQLite 模式初始化**（这些组件直接持有 SQLite 文件路径）
5. **故障转移管理器**：保留给非 vLLM 后备提供商；vLLM 实例健康检查、熔断和故障转移由 `VLLMClient` 统一管理
6. **向量索引**：延迟到首次搜索时通过 `_ensure_vector_index()` 重建

关闭阶段依次清理：评估运行器 → LoRA 训练任务 → 意图任务 → `ReadinessProbe` → `InferenceRuntime` → 资源池 → 备份调度器 → 故障转移管理器 → PostgreSQL 适配器 → 模型管理器 → vLLM 客户端 → bot HTTP 客户端 → 熔断器注册表 → Redis（async + sync）→ 语义缓存。

#### 4.1.5 启动入口

| 文件 | 用途 |
|---|---|
| `backend/run.py` | 主入口，argparse（`--port/--host/--reload/--workers`），强制 `BACKEND_WORKERS=1` |
| `backend/main.py` | 向后兼容入口，同样强制单 worker |

#### 4.1.6 中间件链

按 Starlette「最后添加最外层」规则，请求实际穿透顺序：

```text
CORS → AuditLog → SecurityHeaders → Security(认证) → RateLimit → InputValidation → 路由
```

- `SECURITY_MIDDLEWARE_ENABLED` 控制总开关（默认 true）
- 生产环境若安全中间件导入失败直接 `raise RuntimeError`

#### 4.1.7 关键配置（`backend/app/config.py`）

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `JWT_SECRET` | 自动生成（开发）/ 强制要求（生产） | JWT 签名密钥，≥32 字符 |
| `JWT_ALGORITHM` | `HS256` | JWT 签名算法 |
| `JWT_EXPIRY_HOURS` | `24` | Token 有效期 |
| `LLM_CONCURRENCY_LIMIT` | `2`（被 `LLM_MAX_CONCURRENCY` 覆盖） | LLM 并发上限 |
| `llm_max_queue` | `100` | LLM 请求排队上限 |

**增强模块可用性标志**（try-import 探测，失败降级）：

| 标志 | 模块 | 全局实例 |
|---|---|---|
| `RESOURCE_POOL_AVAILABLE` | `infra.resource_pool` | `connection_pool` / `http_client_pool` |
| `CIRCUIT_BREAKER_AVAILABLE` | `infra.circuit_breaker` | `circuit_breaker_registry` |
| `BACKUP_MANAGER_AVAILABLE` | `infra.backup_manager` | `backup_mgr` |
| `FAILOVER_AVAILABLE` | `infra.failover` | `failover_mgr` |
| `ENCRYPTION_AVAILABLE` | `infra.encryption` | `encryption_mgr` |
| `ACCESS_CONTROL_AVAILABLE` | `infra.access_control` | `access_control_mgr` |
| `VECTOR_DB_AVAILABLE` | `knowledge.vector_db` | — |

> 历史标志 `LOAD_BALANCER_AVAILABLE`（`infra/load_balancer.py`）已移除：vLLM 多实例健康感知与负载统计已统一收敛到 `inference/vllm_client.py:VLLMClient`。

#### 4.1.8 用户认证依赖（`backend/app/dependencies.py`）

核心函数 `get_current_user(request: Request) -> dict`，三层 Token 提取策略：

1. **中间件预验证优先**：若 `request.state.jwt_payload` 已由 `SecurityMiddleware` 解析，直接使用，但仍检查 `is_token_revoked(jti)`
2. **Authorization 头**：解析 `Bearer <token>`
3. **Cookie**：`request.cookies.get("access_token")`

返回 `{"user_id": ..., "username": ...}`，各受保护路由通过 `Depends(get_current_user)` 注入。

---

### 4.2 API 路由层

`backend/api/` 下共 18 个 FastAPI `APIRouter` 模块。

#### API 路由总表

| 模块 | 路径前缀 | 主要端点 | 关键功能 |
|---|---|---|---|
| **auth.py** | `/api/auth` | `POST /register`、`POST /login`、`POST /logout`、`GET /me` | bcrypt 密码哈希、JWT 生成、httpOnly Cookie、Token 黑名单（TTL，上限 10000）、`ALLOW_REGISTRATION` 开关、注册异步锁防并发 |
| **claw.py** | `/api/claw` | `GET/POST /tools`、`DELETE /tools/{name}`、`POST /tools/execute` | 自定义工具 CRUD、AST 校验、模块白名单、`_FORBIDDEN_TOKENS` 黑名单、`multiprocessing` 子进程 + `RLIMIT_AS`/`RLIMIT_CPU` 沙箱、生产默认禁用 |
| **config.py** | `/api/config`、`/api/model` | `GET/PUT /config`、`GET /model/status`、`PUT /model/provider` | 敏感字段脱敏、Redis 配置缓存（TTL 60s）、`CONFIG_SCHEMA` 校验、6 种 `ModelProvider` |
| **enhanced.py** | `/api/enhanced` | `GET /status`、`GET /stats` + 子端点 | 动态探测增强组件可用性、托管 API Key 创建/按 ID 吊销、备份创建与清单；在线 SQLite 恢复按失败关闭策略返回 409 |
| **evaluation.py** | `/api/evaluation` | `GET /gold-set`、`POST /run`、`GET /runs`、`GET /runs/{id}`、`POST/GET /feedback` | 异步评估调度、5 类别（persona/safety/rag_grounded/factual/multiturn）、指标与配置快照存档 |
| **experiments.py** | `/api/experiments` | `GET /`、`GET /{id}`、`POST /lora-ablation`、`POST /rag-ablation`、`POST /quantization-benchmark`、`GET /{id}/report` | 3 类实验、mock 模式、Markdown 报告自动生成 |
| **generate.py** | `/api/generate`、`/api/vllm` | `POST /generate`、`GET /vllm/status` | 单一有界 `InferenceRuntime`、请求级配置快照、单次 RAG 证据/引用、vLLM 客户端故障转移、响应缓存 |
| **integrations.py** | `/api/integrations` | `POST /astrbot/messages` | 5 平台支持、`X-Integration-Token` + HMAC 签名、消息去重、会话开关、降级响应 |
| **knowledge.py** | `/api/knowledge` | 知识库/文件夹/文档/分块 CRUD、`/upload-zip`、`/scan`、`/search`、意图分类训练 | ZIP 安全验证、`simple_text_split` 分块、路径注入检索文本、三阶段搜索回退（RAGHelper → 向量混合 → 关键词） |
| **loras.py** | `/api/loras` | `GET /`、`POST /scan`、`PUT /{id}/status`、`DELETE /{id}` | `adapter_config.json` + `trainer_state.json` 元数据读取、`AdapterChecker` 兼容性检查、vLLM `load_lora_adapter` |
| **messages.py** | `/api/messages`、`/api/sessions` | `GET /messages`、`GET /sessions`、`PUT /sessions/bot-toggle`、`DELETE /messages/batch`、`DELETE /messages/{id}` | SQL 层多条件过滤（search/sessionType/lora/sessionId/platform） |
| **models.py** | `/api/models` | `GET /`、`GET /check/{name}`、`POST /download`、`DELETE /{name}`、`POST /check-7b` | HuggingFace 模型下载、检查、删除 |
| **preferences.py** | `/api/preferences` | `GET/POST /`、`PUT/DELETE /{pid}`、`POST /export`、`POST /sample-from-history` | DPO/ORPO 偏好对（chosen/rejected）管理 |
| **retrieval_eval.py** | `/api/retrieval-eval` | `GET/POST /questions`、`DELETE /questions/{qid}`、`POST /run` | recall@k / MRR / nDCG 指标计算 |
| **router.py** | `/api/router` | `GET/PUT /config`、`GET /adapters`、`GET /logs`、`POST /check/{name}` | `persona_keywords` 角色路由、`rag_confidence_threshold` RAG 阈值切换 |
| **stats.py** | `/api/stats` | `GET /`、`/metrics`、`/alerts`、`/activity`、`/services` | `pynvml` GPU 监控、`psutil` 系统监控、告警生成（队列积压/P95 慢响应/认证失败/数据库写入失败/模型连续失败） |
| **training.py** | `/api/training` | datasets CRUD + export + scan + import、`/models`、`/start`、tasks CRUD、`/generate-dialogues`（含 cancel/progress/force-reset）、saved-dialogues CRUD | 路径安全校验、GPU/内存预检查（`MIN_GPU_MEM_GB=8.0`、`MIN_SYSTEM_MEM_FREE_GB=16.0`）、12 种场景类型、网络搜索角色信息 |
| **user_data.py** | `/api/user/data` | `GET/PUT /` | 按 `page_key` 持久化前端页面状态 |

---

### 4.3 数据库与持久化

#### 4.3.1 双模式适配器架构

**核心文件**：`backend/db/adapter.py`、`backend/db/urls.py`、`backend/db/errors.py`、`backend/db/database.py`、`backend/db/pg_database.py`、`backend/db/models.py`

**选择机制**：

```python
def _should_use_postgresql(env=os.environ) -> bool:
    explicit = str(env.get("USE_POSTGRESQL", "")).strip().lower()
    if explicit:
        return explicit in {"1", "true", "yes", "on"}
    database_url = str(env.get("DATABASE_URL", "")).strip().lower()
    if database_url.startswith(("postgres://", "postgresql://", "postgresql+asyncpg://")):
        return True
    return all(env.get(key) for key in ("PG_HOST", "PG_USER", "PG_PASSWORD", "PG_DATABASE"))
```

- **优先级**：显式 `USE_POSTGRESQL` > `DATABASE_URL` 协议头 > 完整 `PG_HOST/PG_USER/PG_PASSWORD/PG_DATABASE` 自动识别
- **接口契约**：通过 `assert isinstance(db, DatabaseInterface)` 强制实现 `backend/interfaces/__init__.py` 定义的抽象方法
- **对外 API**：`get_db()` 返回当前实例；`is_pg_mode()` 返回布尔值供业务层分支判断

**URL 规范化（`db/urls.py`）**：运行时与 Alembic 共用同一份 PG URL 解析逻辑，避免双写漂移：

- `normalize_async_database_url(url)`：把 `postgres://`、`postgresql://` 统一为 `postgresql+asyncpg://`
- `_component_database_url(env)`：当无显式 `DATABASE_URL` 时，从 `PG_HOST/PG_USER/PG_PASSWORD/PG_PORT/PG_DATABASE` 通过 SQLAlchemy `URL.create()` 安全编码密码（含保留字符）后渲染
- `resolve_runtime_database_url(database_url=None, env=None)`：先看显式 URL，否则回落到分量构造
- `resolve_alembic_database_url(env=None)`：在运行时解析之上叠加 `ALEMBIC_DATABASE_URL` 覆盖项

**领域异常（`db/errors.py`）**：`RegistrationClosedError`（bootstrap-only 注册与既有用户竞态时抛出），供仓储/服务层共享。

**SQLite 实现**（`database.py`，1932 行）：
- 全局单例 `db = SQLiteDB()`
- `threading.local()` 线程本地连接复用
- PRAGMA：`WAL` + `busy_timeout=5000` + `synchronous=NORMAL` + `cache_size=-8000`（约 8MB） + `foreign_keys=ON`
- LoRA 自动发现：扫描 `LORA_PATH`（默认 `backend/loras`）下含 `adapter_config.json` 的子目录

**PostgreSQL 实现**（`pg_database.py`，1995 行）：
- `PgDatabase` 使用 `create_async_engine`（`pool_size=10, max_overflow=20, pool_pre_ping=True`）+ `async_sessionmaker`
- `SyncPgAdapter` 同步包装器：在后台线程运行独立事件循环，每个同步方法通过 `asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=...)` 桥接到异步实现
- URL 规范化：`db/urls.py` 统一处理运行时与 Alembic；`PG_*` 分量通过 SQLAlchemy `URL.create()` 安全编码
- 全局单例 `sync_pg_db = SyncPgAdapter(pg_db)`

#### 4.3.2 数据库表结构（共 26 张表）

**核心业务表**（初始模式 `001_initial`）

| 表名 | 用途 | 关键字段 |
|---|---|---|
| `users` | 用户账号 | `username`(unique)、`password_hash`、`created_at` |
| `config` | KV 配置存储 | `key`、`value` |
| `messages` | 消息历史 | `sessionType`、`sessionId`、`platform`、`conversationId`、`senderId`、`traceId`、`message`、`reply`、`modelName`、`loraName`、`costTime`、`createdAt` |
| `loras` | LoRA 适配器注册表 | `id`、`name`、`description`、`status`、`style`、`size`、`trainedSteps`、`totalSteps` |
| `knowledge_bases` | 知识库 | `name`(unique)、`description` |
| `knowledge_folders` | 知识库文件夹 | `knowledge_base_id` FK(CASCADE)、`name`；UQ(kb_id, name) |
| `knowledge_documents` | 文档 | `title`、`content`、`knowledge_base_id` FK(SET NULL)、`folder_id` FK(SET NULL)、`sourceType`、`fileType`、`chunkCount` |
| `knowledge_chunks` | 文档分块 | `documentId` FK(CASCADE)、`chunkIndex`、`content`、`embedding`(BigInteger, FAISS 向量 ID) |
| `user_data` | 用户页面状态 | `user_id` FK(CASCADE)、`page_key`、`data_json`；UQ(user_id, page_key) |
| `audit_logs` | 审计日志 | `timestamp`、`api_key_hash`、`role`、`action`、`resource`、`ip_address` |

**运行时表**（由 `SQLiteDB._init_database()` / `PgDatabase.init()` 启动时 `CREATE TABLE IF NOT EXISTS` 自动建表）

| 表名 | 用途 |
|---|---|
| `saved_dialogues` | 训练对话存档 |
| `session_settings` | 旧库兼容来源；启动时单向迁移至 canonical `conversations`，之后不再读写 |
| `claw_tools` | Claw 自定义工具 |
| `integration_message_dedup` | 集成消息去重 |
| `conversations` | 集成会话注册表 |
| `integration_events` | 集成事件原始记录 |
| `model_invocations` | 模型调用追踪 |
| `intent_samples` | 意图分类训练样本 |
| `intent_active_kbs` | 意图路由激活知识库 |
| `training_tasks` | 训练任务 |

**研究与评估表**（迁移 `002_research`）

| 表名 | 用途 | 关键字段 |
|---|---|---|
| `gold_eval_runs` | Gold 评估运行记录 | `run_at`、`adapter_name`、`model_label`、`total_prompts`、`category_breakdown`(JSON)、`metrics`(JSON)、`config_snapshot`(JSON) |
| `experiment_runs` | 实验运行记录 | `experiment_type`、`hypothesis`、`status`、`started_at`、`completed_at`、`results`(JSON)、`config_path`、`report_path` |
| `retrieval_eval_questions` | 检索评估问答集 | `question`、`expected_doc_ids`(JSON)、`expected_doc_titles`(JSON)、`gold_answer`、`category` |
| `preference_pairs` | DPO/ORPO 偏好对 | `prompt`、`chosen`、`rejected`、`rubric`、`annotator`、`metadata`(JSON)、`review_status`(default pending) |
| `adapter_compatibility` | LoRA 适配器兼容性检查 | `adapter_name`、`checked_at`、`compatible`、`checks`(JSON)、`warnings`(JSON)、`errors`(JSON) |
| `feedback` | 用户反馈 | `trace_id`、`message_id`、`rating`、`reason`、`adapter_name`、`kb_revision` |

**关系总览**：
- 知识库三层：`knowledge_bases` 1—N `knowledge_folders` 1—N `knowledge_documents` 1—N `knowledge_chunks`，全部带 `ondelete` 级联策略
- 用户与数据：`users` 1—N `user_data`（CASCADE）
- 反馈链路：`feedback.trace_id` ↔ `model_invocations.traceId` ↔ `messages.traceId`（逻辑关联）

#### 4.3.3 Alembic 迁移历史

| Revision | down_revision | 内容 |
|---|---|---|
| `001_initial` | None | 创建 10 张基础表 |
| `002_research` | `001_initial` | 创建 6 张研究表 + 4 个索引 |
| `003_indexes` | `002_research` | 补齐 SQLite 运行时已建但 PG 缺失的索引（Inspector 幂等检查） |
| `004_index_cleanup` | `003_indexes` | 删除 `conversations` 上与唯一约束重复的 `idx_conversations_platform_conversation`；统一 `messages.createdAt` 索引名为 `idx_messages_created_at` |

**说明**：运行时表未纳入 alembic 版本管理，新部署到 PG 时除 `alembic upgrade` 外仍依赖 `PgDatabase.init()` 补建。003/004 均使用 Inspector 先检查再操作，保证"先启动新代码再 upgrade"的场景下幂等。

#### 4.3.4 迁移工具（`backend/db/migration.py`）

一次性 SQLite → PostgreSQL 迁移脚本，入口 `python -m db.migration`：
- 按依赖顺序迁移 16 张表（先被外键引用的表优先）
- 使用 `INSERT ... ON CONFLICT (...) DO NOTHING` 处理重复键
- 每 500 行打印进度，单行失败不阻断

---

### 4.4 推理引擎层（inference）

**目录**：`backend/inference/`，共 6 个文件。

#### 4.4.1 vllm_client.py — vLLM 推理客户端（核心）

**职责**：通过 OpenAI 兼容 API 与 vLLM 服务通信，支持多实例负载均衡、LoRA 动态切换、流式/非流式响应、健康检查与自动故障转移。

**关键类**：

- `VLLMInstance`（dataclass）：单实例元数据与统计
  - `record_success(response_time)`：记录成功，连续失败计数清零，UNHEALTHY 自动恢复
  - `record_failure()`：连续失败 ≥3 次标记为 UNHEALTHY
  - `try_recover()`：30 秒冷却后尝试恢复
- `_WeightedRoundRobinBalancer`：Nginx 平滑加权轮询，权重 = `base_weight × success_rate × (1/(1+avg_rt))`
- `_LeastConnectionBalancer`：按 `(current_connections, avg_response_time)` 选最小连接实例
- `VLLMClient`：主客户端
  - `async generate(messages, lora_name, temperature=0.7, max_tokens=2048, stream=False, top_p=0.9) -> Any`：统一入口
  - `async _generate_non_stream(...)`：3 次重试 + 指数退避（base=1s，max=30s，含 0.5~1.5 抖动）
  - `async _generate_stream(...)`：流式逐 token 产出，解析 `data: ` 前缀与 `[DONE]` 标记。**流式不重试**
  - `async generate_with_rag(messages, rag_context, lora_name, ...)`：将 RAG 上下文（截断至 2000 字符）注入 system 消息
  - `async health_check() -> Dict`：GET 各实例 `/health`
  - `async load_lora_adapter(lora_name, lora_path)`：POST `/v1/load_lora_adapter` 在所有实例上加载

**配置**：`VLLM_BASE_URLS`/`VLLM_BASE_URL`（逗号分隔多实例）、`VLLM_TIMEOUT`(默认 120)、`VLLM_MAX_RETRIES`(默认 3)、`VLLM_MAX_CONCURRENCY`(默认 8)

**熔断器**：`infra.circuit_breaker.CircuitBreaker`（20 次连续失败熔断，60 秒后半开，半开最多 5 调用）

#### 4.4.2 lora_router.py — 多 LoRA 路由器

**职责**：基于意图和角色关键词的轻量级**规则路由**（不依赖 LLM）。由 `LORA_ROUTER_ENABLED`（默认 false）控制。

**关键类**：
- `RouteTarget(str, Enum)`：`BASE_CHAT` / `RAG_REQUIRED` / `PERSONA_ADAPTER`
- `RoutingDecision`（dataclass）：`target`/`adapter_name`/`confidence`/`reason`/`fallback`
- `LoRARouter`：
  - `route(query, intent_result=None) -> RoutingDecision`：三段式优先级路由
    1. 调用 `knowledge.intent_detector.needs_rag`，若 `need_rag and confidence >= 0.5` → `RAG_REQUIRED`
    2. 否则匹配 `DEFAULT_PERSONA_KEYWORDS`（hutao/zhongli/qiqi/xiao），命中生成 `{persona}_lora_7b` 适配器名
    3. 都不匹配 → `BASE_CHAT`

#### 4.4.3 adapter_checker.py — 适配器兼容性检查器

**职责**：激活 LoRA 前验证 `adapter_config.json` 兼容性，不兼容则降级到 default。

**关键类**：
- `AdapterCompatibilityReport`（dataclass）：`adapter_name`/`compatible`/`checks`/`warnings`/`errors`/`checked_at`
- `AdapterChecker`：
  - `check_adapter(adapter_path) -> AdapterCompatibilityReport`：执行 6 项检查
    - `config_exists` / `base_model`（不匹配仅警告）/ `target_modules` 非空 / `rank(r)>0` / `peft_version`（缺失仅警告）/ `weights_exist`（safetensors 或 bin）/ `tokenizer`（缺失仅警告）
  - **总体兼容性 = 无 errors**
- `safe_resolve_lora(name, checker=None) -> str`：安全解析，不兼容返回 "default"

#### 4.4.4 lora_utils.py — LoRA 命名映射工具

- `get_lora_name_map() -> dict`：合并默认别名（`hutao_lora_7b→hutao` 等）与 `LORA_SERVED_NAME_MAP` 环境变量（JSON），对所有名称做 `^[A-Za-z0-9._-]{1,128}$` 白名单校验
- `safe_resolve_lora_served_name(database_name, check_compatibility=False) -> str`：可选调用兼容性检查

#### 4.4.5 model_manager.py — 多后端模型管理器

**职责**：多提供商模型推理统一管理。是对 `vllm_client.py` 的上层封装，提供同步/异步统一接口。

**关键类**：
- `ModelProvider(str, Enum)`：6 种后端（ollama/llama_cpp/openai_compat/transformers_peft/vllm/mock）
- `ModelConfig`（dataclass）：模型元信息
- `MODEL_CONFIGS`：预置 Qwen3-8B 与轻量 Qwen2.5-3B/1.5B 配置；实际模型以环境变量和服务模型列表为准
- `BaseProvider`：基类，定义 `generate()`、`async_generate`（默认经 `asyncio.to_thread` 包装）、`set_lora_adapter`、`get_status`
- 6 个具体 Provider：`MockProvider`、`OllamaProvider`、`LlamaCppProvider`、`OpenAICompatProvider`、`TransformersPeftProvider`（三种量化降级链 4-bit NF4 → 8-bit → FP16）、`VLLMProvider`
- `ModelManager`：
  - `generate / async_generate`：转发到当前 provider
  - `set_lora_adapter(lora_path)`：转发，无该方法时回退到 TransformersPeftProvider
  - 数据库配置缓存 5 秒 TTL，避免每次推理全表扫描

#### 4.4.6 optimizer.py — LLM 调用优化器

**职责**：API 调用优化一体化（重试、连接池、超时、并发控制、响应缓存、限流、Prompt 优化）。

**关键类**：
- `TimeoutLevel(Enum)`：FAST=5s / NORMAL=30s / LONG_TEXT=120s
- `UserRole(Enum)`：ADMIN/OPERATOR/VIEWER/API_USER，对应不同限流配额
- `PromptOptimizer`：中英文混合 token 估算、5 类预设 system prompt、历史压缩（保留近 N 轮 + 更早压缩为摘要）、Prompt 截断
- `ResponseCache`：基于 `prompt_hash + (model_name, lora_name, temperature)` 复合键
  - `_NULL_MARKER` 空值缓存防穿透（TTL 为正常值 1/5）
  - 每 key 独立 `asyncio.Lock` 防击穿
  - TTL 抖动 ±10% 防雪崩
  - LRU 淘汰（max_size=1000）
- `RateLimiter`：令牌桶限流，每用户/API Key 独立桶 + 全局桶双校验
- `LLMCallOptimizer`：
  - `call_with_retry(provider, prompt, ...)`：完整 7 步流程 = 缓存查询 → 限流 → Prompt 优化 → 并发控制 → 指数退避重试 → 响应后处理 → 写缓存

---

### 4.5 知识检索层（knowledge）

**目录**：`backend/knowledge/`，共 10 个文件。提供完整的 RAG 检索增强生成流水线。

#### 4.5.1 RAG 检索流程总览

```text
用户查询
  ↓
intent_detector.needs_rag() ──ML 多分类(置信度)──→ 规则安全网
  ↓ (need_rag=True, kb_name)
rag_helper.retrieve_context()
  ├─ QueryExpander.expand_query() → 5 个变体
  ├─ 每个变体 vector_db.hybrid_search() (向量+BM25 融合, 召回 top_k*4)
  ├─ 多查询去重融合 + query_count 加分 + 区域加权
  ├─ 按 final_score 排序 (第一阶段粗排)
  ├─ CrossEncoderReranker.rerank() (第二阶段精排, 可选)
  └─ 分数归一化
  ↓
compute_confidence() → should_abstain(threshold=0.3)
  ↓ (置信度低)
corrective_rag.retrieve_with_correction() → reformulate_query() 重试一次
  ↓
build_citations() / build_context_prompt() → 注入 LLM
```

#### 4.5.2 vector_db.py — FAISS 向量数据库（核心）

**关键类**：
- `IndexConfig`（dataclass）：`index_type=auto`/`nlist=100`/`nprobe=10`/`m_hnsw=32`/`ef_construction=200`/`ef_search=64`/`auto_switch_threshold=10000`
- `BM25Retriever`：BM25 关键词检索器，`k1=1.5, b=0.75`，jieba 中文分词 + 英文单词切分
- `VectorDatabase`：
  - `EMBEDDING_DIM = 384`，使用 `paraphrase-multilingual-MiniLM-L12-v2` 嵌入模型
  - `_determine_index_type()`：auto 模式按文档数自动选型（<1万 Flat / <10万 IVF / ≥10万 HNSW）
  - `add_documents(documents, kb_revision)`：注入 RAG 2.0 证据化元数据，批量 embedding，`add_with_ids`，同步更新 BM25
  - `search(query, top_k, threshold, filters)`：纯向量检索，含查询缓存 + 过滤器（`_match_filters` 支持 `$contains/$gt/$lt/$in`）
  - `hybrid_search(query, top_k, threshold, keyword_weight=0.3, filters)`：**核心混合检索**。召回 `top_k*3`，向量与 BM25 分数各自归一化后按 `(1-kw)*vector + kw*bm25` 融合

**设计要点**：GPU 自动检测与 OOM 回退；索引自动迁移带异常恢复；数据变更后 `clear_cache` 防脏读。

#### 4.5.3 rag_helper.py — RAG 上层封装（核心）

**关键类**：
- `QueryExpander`：内置 `synonym_map`（胡桃/钟离/七七/魈等角色同义词）、`region_keywords`（璃月/蒙德/稻妻/须弥）、`domain_keywords`
  - `expand_query(query) -> List[str]`：同义词替换 + 领域关键词追加 + 区域限定查询，最多 5 个变体
- `RAGHelper`：
  - `retrieve_context(query, top_k, enable_rerank, filters, use_cache) -> List[Dict]`：**两阶段检索核心**
    - 第一阶段：多查询扩展（每个变体 `hybrid_search` 召回 `top_k*4`），按 doc_key 去重融合，多查询命中加分（`query_count*0.05`），区域加权
    - 第二阶段：Cross-Encoder 重排（若启用），取 `final_top_k`，分数归一化
  - `compute_confidence(results) -> float`：`top1_score × coverage`
  - `build_citations(results) -> List[Dict]`：构建引用列表（含 `source_title/evidence_excerpt/score/content_hash/kb_revision`）
  - `should_abstain(confidence, threshold=0.3) -> bool`：是否弃答
  - `retrieve_with_citations(query, top_k, threshold) -> Dict`：返回 `{results, citations, confidence, abstained}`
  - 当 `abstained=true` 时，生成入口直接返回可配置的 `RAG_ABSTENTION_REPLY`，不再调用无证据的 LLM；历史模型标记为 `rag/abstained`，也不写入模型调用统计。

#### 4.5.4 reranker.py — Cross-Encoder 重排器

- `RerankConfig`：`model_name=bge-reranker-base`/`device=cuda:0`/`batch_size=8`/`max_length=512`
- `CrossEncoderReranker`：
  - `_load_model()`：GPU 不可用回退 CPU，FP16 加载
  - `rerank(query, candidates, top_k=5) -> List[Dict]`：批量 `(query, doc)` 配对打分，`outputs.logits[:, 0]` 取分

#### 4.5.5 corrective_rag.py — 纠正性 RAG

由 `CORRECTIVE_RAG_ENABLED` 控制（默认 false）。
- `CorrectiveRAG`：
  - `reformulate_query(query, top_results) -> str`：从 top 结果提取关键词（去停用词），追加最多 5 个到原查询
  - `retrieve_with_correction(query, top_k) -> Dict`：首次检索 → 低置信度则重写重试 → 仍低则弃答

#### 4.5.6 intent_detector.py — RAG 意图检测（核心）

**职责**：判断用户消息是否需要触发 RAG 检索。混合方案：ML 分类器优先 + 规则引擎兜底。

- `RAGIntentDetector`：规则引擎
  - 内置 `knowledge_keywords`（问题词 + 原神领域词）与 `social_keywords`（问候/感谢/闲聊）
  - `needs_rag(message, context) -> Tuple[bool, str]`：6 步规则判定
- 模块级 ML 函数：
  - `_load_ml_model()`：加载 `intent_classifier_model/`（joblib 格式 classifier+scaler + config.json）
  - `_ml_predict(message) -> Tuple[bool, str, float, Optional[str]]`：**多分类预测**，返回 (是否需 RAG, 原因, 置信度, **预测的 KB 名称**)
  - `needs_rag(message, context) -> Tuple[bool, str, Optional[str]]`：**融合策略**
    - ML 高置信度（≥0.65）信任 ML，但 ML 预测"不需要"时检查规则引擎安全网
    - ML 低置信度时用规则做 tiebreaker

#### 4.5.7 intent_trainer.py — 意图分类器训练

- `async generate_samples(kb_ids, samples_per_kb=100, negative_count=200, lora_name)`：LLM 生成正例、通用负例、**硬负例**（跨 KB 混淆问题，标签为 "none"）
- `async train_intent_classifier(kb_ids, samples_per_kb, negative_count, lora_name)`：
  - 加载已审查样本 → `_train_multiclass_model`（线程池执行）
  - `sentence_transformers` 编码 → `StandardScaler` → `LogisticRegression(class_weight="balanced")` → 5 折交叉验证 → `joblib.dump` 保存

#### 4.5.8 其他文件

| 文件 | 职责 |
|---|---|
| `importer.py` | 批量导入原神知识库（.txt），智能分块 + 元数据增强 |
| `seed_kb_importer.py` | 将 `data/kb_seed_documents.json` 导入向量数据库 |
| `text_splitter.py` | `smart_text_split`：三阶递进式中文语义感知分块（段落→句子→定长） |
| `qdrant_client.py` | Qdrant 预留模块（后续升级路径） |

---

### 4.6 训练层（training）

**目录**：`backend/training/`，共 6 个文件。提供 SFT 监督微调与 DPO/ORPO 偏好对齐两条训练管线。

#### 4.6.1 trainer.py — SFT LoRA 训练核心

**关键类**：

- `GpuTemperatureCallback(TrainerCallback)`：GPU 温度监控
  - 每 20 步检查温度（优先 pynvml，回退 nvidia-smi），超 82°C 暂停 30 秒散热，冷却到 72°C 提前恢复
- `ProgressCallback(TrainerCallback)`：进度回调
- `LoRATrainingConfig`（dataclass）：训练配置全集
  - LoRA：`lora_r=32`/`lora_alpha=64`/`lora_dropout=0.1`/`target_modules`(7 个 proj)/`use_dora`/`use_rslora`/`neftune_noise_alpha=5.0`/`packing=True`
  - 训练：`learning_rate=2e-4`/`num_train_epochs=12`/`batch_size=2`/`grad_accum=4`/`max_seq_length=1024`
  - 优化器：`lr_scheduler_type=cosine`/`warmup_ratio=0.05`/`max_grad_norm=0.5`/`optim=paged_adamw_32bit`
  - 早停：`early_stopping_patience=3`/`early_stopping_threshold=0.001`/`load_best_model_at_end=True`
- `LoRATrainer`：
  - `_load_and_preprocess_data(tokenizer)`：支持 ShareGPT 多轮与单轮两种格式，prompt 部分 labels 设 -100
  - `_detect_quantization_type()`：读 config.json 的 `quantization_config`，返回 awq/gptq
  - `_load_model()`：量化感知加载（AWQ 用 `AwqConfig`，GPTQ 用 `GPTQConfig`，4-bit 用 `BitsAndBytesConfig(NF4+double_quant)`）
  - `train() -> Path`：完整流程 = 校验 → cleanup_gpu → 加载 → 配置 LoRA → `SFTTrainer`（含早停+温度+进度回调）→ 训练 → 保存到 `output_dir/final` → 评估 → 报告

#### 4.6.2 preference_trainer.py — DPO/ORPO 偏好对齐

遵循 guardrail：固定 seed=42，复用 `GpuTemperatureCallback`，**不声称 RLHF**（未训练奖励模型）。

- `PreferenceTrainingConfig`：`method=dpo|orpo`/`beta=0.1`/`learning_rate=5e-6`/`num_train_epochs=1`/`seed=42`/`load_in_4bit=True`/`adapter_path`(SFT adapter 起点)
- `PreferenceTrainer`：
  - `train(pairs) -> PreferenceTrainResult`：加载基础模型（4-bit NF4）→ 加载 SFT adapter 或新建 LoRA → 构建 `Dataset.from_list` → 按 method 选 `DPOTrainer/DPOConfig` 或 `ORPOTrainer/ORPOConfig` → 训练 → 保存 adapter
  - **诚实性声明**：`metric_note` 明确说明 "Preference win rate was not computed"，需要 held-out 评估集单独计算

#### 4.6.3 preference_data_schema.py — 偏好数据 Schema

- `PreferencePair(BaseModel)`：`id`/`prompt`/`chosen`/`rejected`/`rubric`/`annotator`/`review_status`(pending|approved|rejected)/`created_at`
- `PreferenceDataset(BaseModel)`：`statistics` 属性返回 total/approved/pending/rejected 统计
- `save_jsonl` / `load_jsonl` / `filter_by_status`

#### 4.6.4 preprocessor.py — 数据集预处理

- `DatasetPreprocessor`：
  - `_validate_path(path_str, allowed_base)`：防路径穿越
  - `load_raw_data(file_path)`：支持 JSON/JSONL/TXT
  - `validate_conversation(conv)`：支持 ShareGPT/Qwen/Alpaca/Prompt 四种格式
  - `analyze_style(data) -> Dict`：统计平均回复长度/句数/疑问句比例/高频词
  - `prepare_training_data(...)`：验证 → 清洗 → 风格分析 → 随机切分 → 保存 train.json/eval.json

#### 4.6.5 task_manager.py — 训练任务管理器

- `RTX4060Config / RTX3090Config`：两档 GPU 预设
- `SimpleLoRATrainer`：
  - `async start_training(lora_name, dataset_path, config) -> str`：**幂等性检查**（同名 lora_name 运行中任务返回 409）→ 生成 UUID task_id → 持久化到 DB → `asyncio.create_task(_run_training)`
  - `async _run_training(...)`：状态流转 pending→training→completed/failed/cancelled，在线程池执行 `trainer.train()`
  - `async cancel_task(task_id) -> bool`：设置 cancel_event 通知训练线程
  - `_restore_tasks_from_db()`：服务重启后将未完成任务标记为 "interrupted"

#### 4.6.6 evaluator.py — 训练评估器

- `build_training_evaluation(eval_results, log_history, config) -> Dict`：从 Trainer metrics 构建紧凑评估产物（`final_train_loss`/`final_eval_loss`/`best_eval_loss`/`eval_perplexity`/`provenance`/`notes`）
- `notes` 明确说明 "仅报告训练时指标，定性评估需 held-out prompt suite"

---

### 4.7 机器人接入层（bot）

**目录**：`backend/bot/`。基于 NoneBot2 + OneBot v11 协议的 QQ 机器人入口。

#### 4.7.1 bot.py — 机器人主流程（核心）

**关键组件**：

- **配置管理**：`Config` 类，环境变量项为静态属性，数据库项通过 `@property` 动态读取（30 秒缓存）
- **会话历史**：`SessionHistory(max_tokens)`
  - `_load_from_db(session_id)`：通过 `db.adapter` 的 `db.get_messages(limit=10, session_id=session_id)` 恢复
- **统一生成链**：构造 `MessageRequest` 后调用 `generate_reply_core`，与 API/AstrBot 共用 prompt、RAG、失败回退和 `InferenceRuntime` 调度
- **LoRA 热切换**：
  - `LORA_REGISTRY`：预置 hutao、minamo、kisaki；月社妃只使用稳定别名 `loras/kisaki/final`，具体实验产物需在验收后显式发布到该路径
  - `_load_7b_model(lora_name)`：加载 Qwen3-8B 4-bit NF4 + PeftModel；热切换时 `load_adapter` + `set_adapter`
- **推理主流程**：`process_message(event) -> str` 在会话锁内读取历史、调用统一生成服务并写回历史，保证同会话消息顺序一致
- **消息处理**：
  - `should_reply(event)`：私聊总是回复；群聊检查 `db.is_session_bot_enabled` + @机器人/包含名字/触发词
  - `handle_all_messages`：幂等去重 → should_reply → 普通消息走 `process_message`（智能分段发送 `_smart_split_reply`）；`/claw` 命令进入工具模式
  - `handle_claw(bot, user_message, event, lora_name)`：LLM 思考 → LLM 生成 JSON 命令 → `execute_tool` → 结果以角色语气报告
- **启动**：`init_bot()`：nonebot.init(driver="~fastapi") + 注册 OneBotV11Adapter + `nonebot.run(host, port=8081)`

#### 4.7.2 tools.py — 工具注册与分发

- `TOOLS: Dict[str, Dict]`：全局工具表
- `register_tool(name, description, handler)`：注册工具
- 内置工具：`status`（系统状态）、`show_tools`、`send_file`（模糊匹配文件名）、`get_coderesult`（subprocess 执行 Python，30 秒超时）、`write_code`
- `async execute_tool(name, args, bot, event) -> str`：动态分发

---

### 4.8 基础设施层（infra）

**目录**：`backend/infra/`，共 14 个组件文件。通过 Protocol 接口实现依赖倒置。

#### 基础设施组件清单

| 组件 | 文件 | 职责 | 关键类 | 设计模式 |
|---|---|---|---|---|
| RBAC 访问控制 | `access_control.py` | 角色-权限模型、PBKDF2 密钥哈希、滑动窗口限流、SQLite 审计日志 | `Permission`(Flag 枚举)、`Role`、`RateLimiter`、`AuditLogger`、`AccessControlManager` | 单例、装饰器 |
| 认证工作模块 | `auth_work.py` | 认证流程内部辅助逻辑，供 `access_control` 与 `middleware/security` 共享 | 内部工具函数 | 内聚辅助 |
| SQLite 备份 | `backup_manager.py` | 全量/增量备份、gzip 压缩、SHA256 完整性、保留轮转 | `BackupType`、`BackupInfo`、`BackupManager` | 策略、模板方法 |
| 有界线程执行器 | `bounded_executor.py` | 带运行+排队硬上限的专用线程池，超限即拒绝，超时即取消 | `BoundedThreadExecutor`、`BlockingWorkRejected`、`BlockingWorkTimeout` | 有界池 |
| 熔断器 | `circuit_breaker.py` | 3 状态机（CLOSED/OPEN/HALF_OPEN）、降级模式、全局注册 | `CircuitState`、`DegradationMode`、`CircuitBreaker`、`CircuitBreakerRegistry` | 状态机、装饰器、注册表 |
| 并发控制 | `concurrency_control.py` | 令牌桶（全局/会话/发送者三级）、推理运行时、会话锁串行化 | `TokenBucketLimiter`、`InferenceRuntime`、`inference_runtime` 单例 | 令牌桶、优先级队列 |
| 部署校验 | `deployment.py` | 生产环境严格校验 | `DeploymentValidationResult`、`validate_deployment_environment()`、`validate_or_raise_for_startup()` | 校验器 |
| 字段级加密 | `encryption.py` | AES-256-GCM、格式 `ENC:AES256GCM:{iv}:{ciphertext}:{tag}`、敏感字段检测、密钥轮转 | `EncryptionManager`、`DatabaseEncryptionMiddleware` | 单例、中间件 |
| 多 Provider 故障转移 | `failover.py` | 健康检查、AUTO/MANUAL/PRIORITY_BASED 策略、自动回切 | `ProviderHealthStatus`、`FailoverStrategy`、`ProviderConfig`、`HealthChecker`、`FailoverManager` | 策略、观察者 |
| 输入校验 | `input_validator.py` | Schema 驱动、SQL 注入/XSS/路径遍历/命令注入检测 | `FieldType`、`FieldRule`、`Schema`、`InputValidator`；预置 `MESSAGE_SCHEMA` 等 | Schema 校验 |
| vLLM 实例均衡 | `inference/vllm_client.py` | 健康感知选择、实例级失败状态与实时统计 | `VLLMClient`、`VLLMInstance` | 共享客户端 |
| 可观测性 | `observability.py` | 轻量计数器、最近窗口（deque maxlen=1000）、连续失败跟踪、结构化 JSON 日志、敏感字段脱敏 | `increment()`、`set_consecutive()`、`count_recent()`、`snapshot()`、`log_event()` | 计数器、滑动窗口 |
| 资源池 | `resource_pool.py` | SQLite 连接池、HTTP 客户端池、模型推理池 | `PooledConnection`、`ConnectionPool`、`HttpClientPool`、`ModelInferencePool` | 对象池 |
| 安全工具 | `security_utils.py` | HMAC 常量时间比较、nonce 重放保护、集成签名验证、敏感字段脱敏 | `constant_time_contains()`、`remember_nonce()`、`integration_signature()`、`verify_integration_signature()`、`redact_sensitive()` | 常量时间比较 |

> 历史组件 `load_balancer.py` 已删除：vLLM 多实例健康/负载统计统一收敛到 `inference/vllm_client.py`，不再保留独立负载均衡模块。

**设计要点**：所有组件以单进程为前提（强约束 `BACKEND_WORKERS=1`），因为幂等缓存、会话锁、nonce 状态在进程内实现。`BoundedThreadExecutor` 为 `ReadinessProbe` 和其他需要把同步阻塞工作卸离事件循环的调用方提供有界保护，超限拒绝而不是无限排队。

#### 部署校验规则（`deployment.py`）

生产环境强制校验：
- `ASTRBOT_INTEGRATION_TOKEN` ≥ 32 字符
- `DATABASE_URL` 必须为 PostgreSQL，或提供完整的 `PG_HOST/PG_USER/PG_PASSWORD/PG_DATABASE`
- `JWT_SECRET` ≥ 32 字符且非占位值
- `CORS` 非 localhost
- `BACKEND_WORKERS=1`
- `SECURITY_MIDDLEWARE_ENABLED` 必需
- `LORA_PATH` 与 `VLLM_LORA_ROOT` 可使用不同容器路径，但必须映射到同一份 LoRA 存储

---

### 4.9 缓存层（cache）

**目录**：`backend/cache/`；只保留生产链路实际使用的缓存实现。

| 模块 | 职责 | 关键类 |
|---|---|---|
| `redis_client.py` | Redis 连接池（max_connections=50）、JSON 序列化、基础 get/set/delete/exists/incr/expire/keys；同步与异步两套客户端，`close_async_redis()` / `close_sync_redis()` 由 lifespan 统一关闭 | `get_redis()` |
| `config_cache.py` | 配置缓存，TTL ±10% 抖动防雪崩；双层结构（Redis + 本地内存兜底） | `get_cached_config()`、`set_cached_config()`、`invalidate_config_cache()` |
| `ttl_value_cache.py` | SQLite/PostgreSQL 会话开关共用的有界、线程安全本地 TTL 缓存 | `BoundedTTLCache` |
| `response_cache.py` | 生成回复的有界进程内缓存；键覆盖历史、对话者、RAG 与采样参数 | `ResponseCache` |

**生产缓存/调度主链路**：`config_cache` + `ResponseCache` + `BoundedTTLCache` + `InferenceRuntime`。未接入生产的语义缓存与 Redis Streams 消息队列已删除。

---

### 4.10 安全中间件（middleware）

**文件**：`backend/middleware/security.py`，6 个中间件形成纵深防御链：

1. **RequestIdMiddleware**：UUID 请求 ID + contextvars 上下文传播
2. **SecurityMiddleware**：JWT + 静态/托管 API Key 双认证、认证尝试限流、吊销检查、白名单路径（`AUTH_WHITELIST`）
3. **RateLimitMiddleware**：RPM/TPM 滑动窗口，分端点配额（auth/generate/default 各自独立）
4. **InputValidationMiddleware**：请求体大小限制 + prompt 长度限制 + 17 条 prompt injection 模式检测
5. **SecurityHeadersMiddleware**：X-Frame-Options、CSP、HSTS、X-Content-Type-Options
6. **AuditLogMiddleware**：TimedRotatingFileHandler、90 天保留、敏感字段自动脱敏

**信任代理头开关** `TRUST_PROXY_HEADERS`：默认 False 取真实 client IP，仅在显式启用后读取 X-Forwarded-For，防止伪造绕过限流。

---

### 4.11 评估与实验体系

#### 4.11.1 评估体系（`backend/evaluation/`）

10 个文件构成完整的研究级评测体系：

| 文件 | 职责 |
|---|---|
| `gold_set_schema.py` | Pydantic 模式 `RubricCriterion`/`GoldPrompt`/`GoldSet`，rubric 权重和为 1.0，类别 persona/safety/rag_grounded/factual/multiturn |
| `gold_set_manager.py` | `GoldSetManager` 加载、校验、过滤、切分 gold prompts |
| `generation_metrics.py` | `GenerationMetrics`：distinct-1/2、4-gram repetition_rate、avg_length |
| `persona_metrics.py` | `PersonaMetrics`：rubric_score、style_consistency、contradiction_rate |
| `safety_metrics.py` | `SafetyMetrics`：prompt_injection_success_rate、secret_extraction_refusal_rate、harmful_request_refusal_rate |
| `retrieval_metrics.py` | `RetrievalMetrics`：recall@k、MRR、nDCG@k、faithfulness、answer_correctness |
| `character_benchmark.py` | CLI 工具，对 held-out prompts 跑角色适配器评测 |
| `runtime_runner.py` | 后台执行器，`_evaluation_lock` 保证一次只跑一个评测 |
| `benchmark_gate.py` | 质量门，强制约束：real_run、paired_sample_ids、same_dataset_hash、zero_generation_errors、format_correct_rate≥0.99、safety_pass_rate≥0.70、rag_citation_accuracy≥0.90、output_token_ratio≥0.25、repetition_rate≤0.05 |
| `clean_training_data.py` | 8 步训练数据清洗脚本 |

#### 4.11.2 实验体系（`backend/experiments/`）

3 个实验框架遵循研究 guardrails（受控变量、条件结论、CPU mock 模式）：

| 文件 | 职责 |
|---|---|
| `ablation_runner.py` | LoRA 消融框架，`DEFAULT_VARIANTS` = lora_baseline/lora_neftune/lora_packing/dora/rslora，`controlled_variables` 校验确保单因素 |
| `quantization_benchmark.py` | 量化基准，`DEFAULT_CONFIGS` = fp16/awq/nf4/int8，pynvml 测量 VRAM，TTFT/decode_tokens_per_s/p50/p95/p99 延迟 |
| `rag_ablation.py` | RAG 消融，`DEFAULT_VARIANTS` = vector_only/bm25_only/hybrid/hybrid_reranker，复用 `RetrievalMetrics`，纯 CPU 无需 GPU |

---

### 4.12 分层架构（services / repositories / providers）

**目录**：`backend/services/`、`backend/repositories/`、`backend/app/providers.py`

项目正在从"API 路由直接调用 `db` 全局单例"的旧模式，向"路由 → 服务 → 仓储 → 数据库适配器"分层结构增量迁移。新代码应遵循 [可扩展性开发指南](EXTENSIBILITY_GUIDE.md)；旧 API 模块中的 `db` 全局导入是兼容层，不作为新模块的实现模板。

```text
FastAPI route
    ↓ 只处理 HTTP 参数、鉴权和错误映射
Application service (services/)
    ↓ 编排业务流程，不依赖 FastAPI
Repository Protocol (repositories/)
    ↓ 描述领域所需的持久化能力
Database adapter (db/adapter.py)
    ↓ SQLite / PostgreSQL 兼容实现
```

#### 4.12.1 服务层（`backend/services/`）

| 文件 | 服务 | 职责 |
|---|---|---|
| `chat_generation.py` | `ChatGenerationService` | 聊天生成用例编排：清洗消息 → 高危 prompt 拦截 → 优先级/限流 → 调用注入的 `generate_handler` → 返回 `GenerateResponse`。`InferenceRuntime` 以 Protocol 注入，便于单元测试用 fake 替换 |
| `model_management.py` | `ModelManagerService` | 模型管理用例：`list_available_models` / `check_model_exists` / `download_model` / `delete_model`，把同步 `ModelManager` 通过 `asyncio.to_thread` 适配为异步 API |

#### 4.12.2 仓储层（`backend/repositories/`）

| 文件 | Protocol | 适配器 | 用途 |
|---|---|---|---|
| `messages.py` | `MessageRepository` | `DatabaseMessageRepository` | 消息历史分页、会话摘要、会话开关、按条件批量/单条删除；`MessageQuery` dataclass 封装筛选条件，避免参数列表膨胀；同步数据库调用通过 `asyncio.to_thread` 卸离事件循环 |
| `user_data.py` | `UserDataRepository` | `DatabaseUserDataRepository` | 按 `username + page_key` 加载/保存页面状态；用户不存在抛 `UserDataUserNotFoundError` |

#### 4.12.3 依赖提供器（`backend/app/providers.py`）

`providers.py` 是 FastAPI 依赖注入的集中入口，将 `RuntimeContainer` 适配为 Service / Repository：

- `get_message_repository(request)` / `get_user_data_repository(request)`：从 `request.app.state.runtime_container.db` 构造数据库仓储
- `get_model_management_service()`：从 `inference.model_manager.get_model_manager()` 构造服务

路由通过 `Depends(get_message_repository)` 等注入依赖；`request.app.state.runtime_container` 是唯一数据库来源，禁止在 provider 中重新导入全局数据库单例。

---

## 5. 前端模块详解

### 5.1 应用结构与布局

**技术栈**：Next.js 16.2.9（App Router，默认 Turbopack）、React 19.2.3、Tailwind CSS v4 + shadcn/ui（new-york 风格）

**Provider 嵌套**（`src/app/layout.tsx`）：

```text
ThemeProvider (next-themes, attribute="class")
  └─ AuthProvider           (认证状态、用户数据持久化)
     └─ SettingsProvider    (语言/时区/系统配置 + t() 翻译函数)
        └─ children + <Toaster/> (sonner)
```

**客户端/服务端组件划分**：
- **服务端组件**：仅 `layout.tsx` 与根 `page.tsx`（用于导出 `metadata`）
- **客户端组件**：所有子页面均为 `'use client'`
- **鉴权模式**：除 `/login` 和根 `/` 外，其余 12 个页面统一用 `<AuthGuard><XxxContent/></AuthGuard>` 模式包裹

### 5.2 页面清单

| 路径 | 功能摘要 | 主 Hook |
|------|---------|---------|
| `/` | 仪表盘首页（4 统计卡片 + 24h 活动趋势图 + 系统状态 + 快捷操作） | `useStats`、`useLoras`、`useServices`、`useAuth` |
| `/login` | 登录/注册 Tab 切换页，无侧边栏 | `useAuth` |
| `/history` | 对话历史记录（多条件筛选、CSV 导出、批量删除） | `useMessages` |
| `/training` | LoRA 训练中心（数据集管理、对话生成、启动训练） | `usePageData` |
| `/lora` | LoRA 模型管理（Tab 过滤、扫描、激活切换） | `useLoras` |
| `/intent-training` | 意图分类训练（3 Tab：配置&生成 / 审查样本 / 训练模型） | 直接调 `api.*` |
| `/monitor` | 系统监控（CPU/GPU/内存/磁盘、10s 轮询） | `useSettings` |
| `/integrations` | 平台连接（5 平台开关卡片） | 直接调 `api.*` |
| `/knowledge` | 知识库管理（三级层级、文档 CRUD、ZIP 上传、语义搜索） | `useKnowledge` |
| `/evaluation` | 评估仪表盘（3 Tab：Gold 集 / 评估运行 / 反馈） | `useEvaluation` |
| `/experiments` | 实验对比（3 Tab：LoRA 消融 / RAG 消融 / 量化基准） | `useExperiments` |
| `/router` | 多 LoRA 路由（路由配置、兼容性检查、日志） | `useRouter` |
| `/preferences` | DPO/ORPO 偏好对管理（列表/创建/审核/导出） | `usePreferences` |
| `/claw` | Claw 工具管理（Python 自定义工具 CRUD、在线测试） | 直接调 `api.*` |
| `/settings` | 系统设置（5 Tab：通用/机器人/模型/通知/安全） | `useSettings` |

### 5.3 组件体系

```text
src/components/
├── layout/                         (应用骨架)
│   ├── AppLayout.tsx               (主布局: Sidebar + Header + main)
│   ├── AuthGuard.tsx               (鉴权守卫: loading骨架/未登录提示/已登录渲染)
│   └── Sidebar.tsx                  (左侧导航: 14项 + StatusBar时钟; memo 优化)
│
├── dashboard/                      (仪表盘专用)
│   ├── StatCard.tsx                (统计卡片)
│   ├── ActivityChart.tsx           (Recharts 折线图, next/dynamic 懒加载)
│   ├── TestChatDialog.tsx          (测试对话弹窗)
│   └── SessionManagerDialog.tsx    (会话管理弹窗)
│
├── training/
│   └── TrainingParamsEditor.tsx    (LoRA训练参数编辑器: 三套显存预设, JSON导入导出)
│
├── ui/                             (shadcn/ui 原语, 20个)
│   avatar, badge, button, card, checkbox, dialog, dropdown-menu, input, label,
│   popover, progress, scroll-area, select, separator, skeleton, sonner, switch,
│   table, tabs, textarea, tooltip
│
└── ThemeToggle.tsx                 (明暗主题切换)
```

### 5.4 Context 与 Hook 体系

#### Context（2 个）

**`AuthContext`**：
- `user: User | null`、`loading: boolean`
- `login/register`：调 `/api/auth/login`、`/api/auth/register`，`credentials:'include'` 接收 httpOnly Cookie；只把非敏感用户信息存 localStorage（**永不存 token**）
- `logout`：调 `/api/auth/logout` 清 Cookie + 清 localStorage
- `savePageData/loadPageData`：页面表单数据双写（localStorage 快 + 后端可靠）
- 关键设计：初始化时**不**从 localStorage 恢复 user，避免 SSR/CSR hydration 不匹配

**`SettingsContext`**：
- `locale: Locale`（默认 zh-CN）、`timezone`（默认 Asia/Shanghai）、`config: SystemConfig`、`loading`
- `t(key, fallback)`：绑定 locale 的翻译函数
- `formatTime(date, options)`：按时区+locale 格式化时间
- 支持 `zh-CN`、`zh-TW`、`en` 三语言

#### Hook（12 个）

| Hook | 职责 | 返回数据 |
|------|------|---------|
| `useStats(enabled)` | 系统统计数据，30s 轮询 | `stats, loading, error, refetch` |
| `useServices(enabled)` | 服务运行状态，60s 轮询 | `services, loading, error, refetch` |
| `useActivity()` | 24h 活动趋势，60s 轮询 | `activity, loading, error, refetch` |
| `useLoras(enabled)` | LoRA 列表 + 激活切换 + 删除 | `loras, total, loading, error, refetch, toggleLoraStatus, deleteLora` |
| `useMessages(limit, offset, enabled, filters)` | 分页消息记录 | `messages, total, totalAll, loading, error, refetch` |
| `usePageData<T>(pageKey, defaultData)` | 表单数据持久化（localStorage + 后端双写，1s 防抖） | `[data, setData, saveData]` |
| `useEvaluation(enabled)` | Gold 集 + 评估运行 + 反馈 | `goldSet, runs, feedbacks, loading, error, running, refetch, runEvaluation` |
| `useExperiments(enabled)` | 实验列表 + 启动 + 报告 | `experiments, loading, error, starting, refetch, startExperiment, getReport` |
| `useKnowledge(enabled)` | 知识库三级 CRUD + 语义搜索 + ZIP 上传 | 详尽返回 |
| `usePreferences(enabled)` | DPO 偏好对 CRUD + 导出 + 历史采样 | `preferences, total, filterStatus, ...` |
| `useRouter(enabled)` | 路由配置 + 兼容性 + 日志 | `config, adapters, logs, saving, checking, updateConfig, checkAdapter` |
| `use-mobile` | 768px 断点判断 | `boolean` |

**Hook 设计共性**：全部 `useCallback` 稳定 fetch 函数引用；`enabled` 参数控制是否发起请求；页面不可见时跳过轮询。

### 5.5 API 客户端与代理机制

#### API 客户端（`src/lib/api.ts`）

- 导出单例 `api = new ApiClient()`，基址 `/api`
- **`request<T>`** 通用方法：`credentials:'include'`、有 body 自动设 `Content-Type`（FormData 除外）、**401 自动清理 localStorage 并跳 `/login`**、204 返回 undefined
- 覆盖全部后端接口（约 60+ 方法）
- 末尾导出大量 TypeScript 接口，是前后端契约的单一来源

#### 代理机制（`src/lib/proxy.ts` + `src/app/api/[[...path]]/route.ts`）

前端通过 Next.js Route Handler 把请求代理到后端 FastAPI（`BACKEND_URL=http://localhost:8000`）。

**`proxyRequest` 核心逻辑**：
1. **超时分级**：默认 30s；长任务路径（`/api/generate`、`/api/training/start`、`/api/knowledge/scan/import` 等）用 210s
2. **CSRF 校验**：不安全方法校验 Origin/Referer 同源
3. **路径白名单**：`isProxyPathAllowed` 拒绝空路径、含 `..` 或 `//` 的路径，仅允许 `PROXY_ALLOWED_PREFIXES` 列出的前缀
4. **认证转换**：从 Cookie `access_token` 提取 JWT，转为 `Authorization: Bearer` 头转发
5. **错误兜底**：AbortError 返回 504，其他错误返回 502

**`[[...path]]/route.ts`**（catch-all）：实现 GET/POST/PUT/PATCH/DELETE 五方法，multipart/form-data 直接透传 `request.body`。

#### 专用 API 路由（10 个）

| 路由 | 职责要点 |
|------|---------|
| `auth/login` | `proxyPost`→`/api/auth/login`，透传后端 Set-Cookie |
| `auth/logout` | 透传清 Cookie 的 Set-Cookie |
| `auth/me` | 验证当前 token |
| `auth/register` | 透传 Set-Cookie |
| `health` | 失败返回 503 unhealthy |
| `messages/batch` | DELETE，容错处理无 body/非 JSON |
| `training/generate-dialogues/cancel` | 502/504 降级为 503 + 友好消息 |
| `training/generate-dialogues/progress` | 502/504 降级为 503 + 空进度对象 |
| `training/start` | 失败时统一错误消息格式 |
| `[[...path]]/route.ts` | catch-all 兜底 |

### 5.6 构建配置

| 文件 | 关键配置 |
|---|---|
| `next.config.ts` | `output: 'standalone'`；`/(.*)` 全部 `no-store`；`/_next/static/(.*)` 长缓存 `max-age=31536000, immutable` |
| `tsconfig.json` | `target: ES2017`、`strict: true`、`@/* → ./src/*`；显式排除 `backend` 目录 |
| `eslint.config.mjs` | 继承 `eslint-config-next/typescript`；禁用 `react-hooks/set-state-in-effect` |
| `components.json` | shadcn/ui：`style: new-york`、`rsc: true`、`baseColor: neutral`、`cssVariables: true` |

---

## 6. AstrBot 网关插件

**目录**：`astrbot_plugins/multipersonal_gateway/`，3 个文件构成独立可安装的 AstrBot 插件。

| 文件 | 说明 |
|---|---|
| `metadata.yaml` | name=multipersonal_gateway，version=0.1.0 |
| `main.py` | `QQChatGatewayPlugin(Star)` 继承 AstrBot Star 基类 |
| `README.md` | 安装与配置说明 |

**核心机制**：

- **配置**：环境变量驱动（`MULTIPERSONAL_BACKEND_URL`、`ASTRBOT_INTEGRATION_TOKEN`、`MULTIPERSONAL_TRIGGER_PREFIXES=/ai,/chat,@bot`、`MULTIPERSONAL_REPLY_GROUP_ALL`、`MULTIPERSONAL_BACKEND_TIMEOUT=60`、`MULTIPERSONAL_DEDUP_TTL=300`）
- **转发逻辑**：`@filter.event_message_type(filter.EventMessageType.ALL)` 拦截所有消息 → `_should_forward` 判断 → `_build_payload` 构造标准化 payload → 去重
- **签名机制**：`_signature` 计算 `sha256=HMAC(token, "timestamp.nonce.body_hash")`，请求头携带 `X-Integration-Token` / `X-Integration-Timestamp` / `X-Integration-Nonce` / `X-Integration-Signature`
- **平台归一化**：`_platform` 多源探测（telegram/wecom/wechat_official/wechat_personal/qq）；`_adapter` 映射（qq→napcat、wechat_personal→gewechat）
- **会话归一化**：`_conversation_type`（group/channel/private）、`_conversation_id`（多源回退）
- **错误降级**：HTTP 异常时私聊回复"后端服务暂时不可用"，群聊静默
- **HTTP**：用 stdlib `urllib.request`（无第三方依赖），`asyncio.to_thread` 异步包装

---

## 7. 部署与运行方式

### 7.1 本地开发

#### 前端

```bash
pnpm install
pnpm dev          # 启动开发服务器 http://localhost:5000
pnpm ts-check     # TypeScript 类型检查
pnpm build        # 生产构建
```

#### 后端

```bash
cd backend
pip install -r requirements.txt     # GPU 服务器按 requirements.txt 顶部注释顺序安装
py -3.12 -m pytest tests -q          # 运行测试
py -3.12 -m scripts.local_smoke      # 冒烟测试
python run.py --reload               # 启动后端 http://localhost:8000
```

**本地 mock 模式**（无需 GPU）：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start-local-backend.ps1
# 等价于：ENVIRONMENT=development MODEL_PROVIDER=mock VLLM_ENABLED=false python run.py --reload
```

### 7.2 Docker Compose 生产部署

**`deploy/docker-compose.yml`** — 6 服务生产栈：

| 服务 | 镜像 | 端口 | 说明 |
|---|---|---|---|
| `postgres` | `pgvector/pgvector:pg14` | 5432 | 向量库 |
| `redis` | `redis:7.4-alpine` | 6379 | AOF 持久化，512MB 上限 |
| `vllm` | `vllm/vllm-openai:v0.10.2` | 8001 | AWQ 量化，GPU 加速 |
| `backend` | 自构建 | 8000 | FastAPI（`USE_POSTGRESQL=true`、`BACKEND_WORKERS=1`） |
| `frontend` | 自构建 | 5000 | Next.js standalone |
| `nginx` | `nginx:1.27-alpine` | 80/443 | 反代 |

**15GB 显存变体**（`docker-compose.15g.yml`）：单 vLLM，`gpu_memory_utilization=0.5`，bind mount models/loras/data。

### 7.3 Nginx 反向代理

**`deploy/nginx/nginx.conf`**：
- upstreams：`vllm_backend` / `fastapi_backend` / `nextjs_frontend`，keepalive
- `/api/` → nextjs_frontend（Next.js 处理 auth/CSRF）
- `/health` → fastapi_backend
- `/_next/static/` 缓存 365 天
- `/_next/webpack-hmr` WebSocket 升级
- gzip 压缩、安全响应头

### 7.4 Supervisor 进程守护

**`deploy/supervisord.conf`** — 3 个进程：
- `backend`：`python run.py`
- `frontend`：`pnpm start`（PORT=5000）
- `qqbot`：`python -m nb run`（autostart=false）
- 全部 `stopasgroup/killasgroup`，10MB 日志轮转

### 7.5 启动脚本

| 脚本 | 用途 |
|---|---|
| `deploy/scripts/start_all.sh` | 双模式（docker-compose / 裸金属），环境检查，.env 生成，裸金属模式启动 vLLM×2 + backend + frontend |
| `deploy/scripts/start_vllm.sh` | vLLM 启动带 GPU/模型检查、AWQ、`--enable-prefix-caching`、`--trust-remote-code`、`wait_for_healthy` 循环 |
| `deploy/scripts/start_vllm_15g.sh` | 15GB 显存优化变体，`gpu_memory_utilization=0.5` |
| `deploy/scripts/download_models.sh` | 国内镜像（`HF_ENDPOINT=hf-mirror.com`），下载 Qwen3-8B-AWQ、BGE-M3 和 BGE reranker |
| `deploy/compare_quantization.sh` | 遍历 fp16/awq/int8 串行启动 vLLM、跑 benchmark |
| `deploy/run_experiments.sh` | 顺序跑 4 实验，生成 summary.md |
| `scripts/run_kisaki_experiment.py`、`lab-run-kisaki-r2.sh`、`lab-run-kisaki-r3.sh`、`lab-run-kisaki-r4-dpo.sh` | 当前受门禁约束的 R1V4-R4 实验入口 |
| `deploy/verify.sh` | 部署验证（docker compose up、业务流测试） |

### 7.6 服务器验证命令

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/ready
redis-cli -h 127.0.0.1 ping
curl -fsS http://127.0.0.1:8001/v1/models
```

### 7.7 关键环境变量

| 变量 | 用途 | 默认值 |
|---|---|---|
| `ENVIRONMENT` | 环境标识（production 强校验） | development |
| `USE_POSTGRESQL` | 强制 PG 模式 | 自动识别 |
| `DATABASE_URL` | PG 连接串（与完整 `PG_*` 二选一） | 空 |
| `PG_HOST/PG_USER/PG_PASSWORD/PG_DATABASE` | 结构化 PG 连接参数，密码会安全编码 | 空 |
| `DATABASE_PATH` | SQLite 路径 | backend/qq_assistant.db |
| `LORA_PATH` | LoRA 根目录 | backend/loras |
| `ALLOW_REGISTRATION` | 开放注册 | — |
| `VLLM_BASE_URL` | vLLM 服务地址 | http://localhost:8001 |
| `ALLOWED_ORIGINS` / `CORS_ORIGINS` | CORS 白名单 | localhost:3000/5000 |
| `SECURITY_MIDDLEWARE_ENABLED` | 安全中间件总开关 | true |
| `LLM_MAX_CONCURRENCY` | LLM 并发上限 | 2 |
| `BACKEND_WORKERS` | 后端 worker 数（必须为 1） | 1 |
| `ASTRBOT_INTEGRATION_TOKEN` | AstrBot 集成 token（生产 ≥32 字符） | — |
| `JWT_SECRET` | JWT 签名密钥（生产 ≥32 字符） | 自动生成 |
| `TRUST_PROXY_HEADERS` | 信任代理头 | false |
| `RERANKER_ENABLED` | 启用 Cross-Encoder 重排 | — |
| `CORRECTIVE_RAG_ENABLED` | 启用纠错 RAG | false |
| `LORA_ROUTER_ENABLED` | 启用 LoRA 路由 | false |

---

## 8. 测试体系

**目录**：`backend/tests/`，覆盖单元 + 集成 + 契约 + 并发回压四层。`conftest.py` 通过 `collect_ignore_glob = ["security_test.py", "fault_injection_test.py"]` 排除可执行脚本。

### 8.1 测试文件分类

| 类别 | 代表性文件 | 覆盖范围 |
|---|---|---|
| 核心单元/集成 | `test_core.py`、`test_architecture_hardening.py` | L1LRUCache、CircuitBreaker、MultiPlatformStorage、AstrBot 契约、DeploymentValidation、ServiceStatus、ConcurrencyControl、架构硬化回归 |
| 应用工厂与运行时 | `test_app_factory.py`、`test_app_configuration_isolation.py`、`test_runtime_unification.py`、`test_readiness_*.py` | `create_app` 组合根、配置隔离、`RuntimeContainer` 统一、`ReadinessProbe` 并发与集成 |
| 仓储与服务 | `test_message_repository.py`、`test_user_data_repository.py`、`test_chat_generation_service.py`、`test_model_management_service.py` | 新分层架构的 Protocol 契约与异步适配 |
| 数据库与迁移 | `test_database_urls.py`、`test_pg_compat_and_cache_regression.py`、`test_sync_pg_adapter_concurrency.py` | PG URL 规范化、SQLite/PG 兼容、`SyncPgAdapter` 并发回压 |
| 认证与安全 | `test_auth_*.py`、`test_security_rate_limiter*.py`、`test_lora_path_security.py`、`test_encryption_key_management.py` | 注册竞态、认证回压、限流器容量、LoRA 路径信任边界、密钥管理 |
| 训练与评估 | `test_training_task_lifecycle.py`、`test_training_export_concurrency.py`、`test_evaluation_runtime_hardening.py`、`test_retrieval_evaluation_integrity.py` | 训练任务生命周期、导出并发、评估运行时硬化、检索评估完整性 |
| 研究数据契约 | `test_research_data_api_integrity.py`、`test_experiment_api_integrity.py`、`test_lora_scan_integrity.py`、`test_kisaki_*` | 研究 API 完整性、实验契约、LoRA 扫描、月社妃 Gold/v3/v4 契约 |
| 基础设施 | `test_circuit_breaker_lifecycle.py`、`test_bounded_executor.py`、`test_async_db_executor.py`、`test_stats_concurrency.py` | 熔断器生命周期、有界执行器、异步数据库卸载、统计并发 |
| 角色基准 | `test_character_benchmark.py` | quality gate 拒绝短小/塌缩候选、接受健康候选、拒绝重复 sample IDs |
| 训练评估器 | `test_training_evaluator.py` | losses 与 provenance、极端 loss、DoRA + RSLoRA 共存、非量化模型 float16 |
| 可执行脚本 | `security_test.py`、`fault_injection_test.py` | 渗透测试 8 场景（SQL 注入/XSS/路径遍历/命令注入/认证绕过/权限提升/限流绕过/敏感数据泄露）、故障注入 6 场景；`__test__ = False`，被 conftest 排除 |

### 8.2 测试设计约束

- CI 测试必须在 Ubuntu 通过；Windows 特定测试使用 `@pytest.mark.skipif(sys.platform == "win32")`
- 测试超时需考虑环境差异（CI 可能需要更长超时）
- 严格错误信息断言可能跨平台失败；使用灵活匹配
- 新分层（services / repositories）测试优先用 fake 依赖做单元测试，再补一条路由注入测试

---

## 9. CI/CD 流水线

**文件**：`.github/workflows/ci.yml`

**触发**：push main / PR main / workflow_dispatch

**3 个 Job**：

1. **frontend**（ubuntu-latest）：
   - `pnpm install --frozen-lockfile`
   - `pnpm ts-check`
   - `pnpm lint`
   - `pnpm build`

2. **backend**（ubuntu-latest，working-directory: backend）：
   - 装 CPU-only 依赖（先装 PyTorch CPU 版，再 grep -v 排除 vllm/torch 装 requirements-ci.txt）
   - 语法检查（py_compile 关键模块）
   - 跑测试：`pytest tests/ -v --tb=short --ignore=tests/security_test.py --ignore=tests/fault_injection_test.py`
   - 失败时 surface：grep FAILED/ERROR + tail -30

3. **docker**（ubuntu-latest，needs [frontend, backend]，仅 push main 触发）：
   - 构建后端镜像 `--build-arg INSTALL_GPU_DEPS=false`
   - 构建前端镜像

---

## 10. 脚本工具清单

**目录**：`scripts/`。以下仅列活动入口。

### 10.1 数据构建类（Python）

| 脚本 | 用途 |
|---|---|
| `extract_character_dialogues.py` | 从源文本提取角色对话数据集（shenbai_mizunamo 神白水菜萌 / tsukiyashiro_kisaki 月社妃），输出 raw.jsonl/sft.json/sft_full.json/excluded.jsonl/manifest.json/coverage_report.json |
| `build_character_rag_eval.py` | 将 held-out RAG 样本转为检索实验 schema |

### 10.2 盲评与审核类（Python）

| 脚本 | 用途 |
|---|---|
| `scripts/archive/legacy_review_tools/build_blind_ab_review.py` | 从两份 benchmark 报告构造确定性盲评包（seed 控制左右顺序随机化，source_hashes 防篡改） |
| `scripts/archive/legacy_review_tools/review_blind_ab.py` | 可恢复的匿名 A/B 人工评测 CLI（命令 a/b/t/i/s/q，强制填写理由） |
| `scripts/archive/legacy_review_tools/auto_review_blind_ab.py` | AI 裁判预审（OpenAI 兼容 API），按 20% 比例抽样构造人工复核包 |
| `scripts/archive/legacy_review_tools/auto_review_blind_ab_deepseek.py` | DeepSeek API 变体（支持 response_format json_object、429/5xx 重试） |
| `scripts/archive/legacy_review_tools/score_blind_ab.py` | 揭盲并统计模型胜率（win_rate_on_decisive、by_category） |
| `review_preference_candidates.py` | DPO/ORPO 偏好对人工审核 CLI（a 批准/r 拒绝/e 编辑/s 跳过/q 保存退出） |

### 10.3 训练与基准类

| 脚本 | 用途 |
|---|---|
| `validate_lora_training.py` | LoRA 训练只读预检（基础模型 quantization_config、训练数据格式、输出目录、GPU 快照） |
| `run_kisaki_experiment.py` | 通过 V4 人工审核与冻结门禁后启动单个 R1V4 受控实验 |
| `lab-start-vllm-daemon.sh` | vLLM 守护进程启动器（检测训练进程占用时拒绝启动） |
| `lab-start-vllm.sh` | vLLM 启动命令（加载 hutao/minamo LoRA，AWQ Marlin，--enable-lora --max-lora-rank 64） |

### 10.4 本地验证类（PowerShell）

| 脚本 | 用途 |
|---|---|
| `local-verify.ps1` | Windows 本地验证流水线（py_compile + pytest + scripts.local_smoke + git diff --check，可选 -Frontend 跑 pnpm ts-check） |
| `start-local-backend.ps1` | 本地 mock 模式启动后端 |

---

## 11. 依赖关系总览

### 11.1 后端核心依赖

**AI/ML 核心**（CUDA 12.8 / AutoDL 实测）：
- `torch==2.8.0`、`torchvision==0.23.0`、`torchaudio==2.8.0`
- `vllm==0.10.2`
- `transformers>=4.56.0,<5.0`
- `peft>=0.14.0,<1.0`、`bitsandbytes>=0.45.0,<1.0`、`datasets>=5.0.0,<6.0`、`accelerate>=1.4.0,<2.0`、`trl>=0.13.0,<2.0`

**Web 框架**：
- `fastapi>=0.115.0,<1.0`、`uvicorn[standard]>=0.34.0,<1.0`、`pydantic>=2.10.0,<3.0`、`starlette>=0.40.0,<2.0`

**NoneBot2**（QQ 机器人）：
- `nonebot2>=2.5.0,<3.0`、`nonebot-adapter-onebot>=2.4.6,<3.0`

**认证与加密**：
- `pyjwt>=2.13.0,<3.0`、`bcrypt>=4.2.0,<6.0`、`cryptography>=44.0.0,<50.0`

**数据库**：
- `sqlalchemy[asyncio]>=2.0.36,<3.0`、`alembic>=1.14.0,<2.0`、`asyncpg>=0.30.0,<1.0`

**向量数据库与 RAG**：
- `faiss-cpu>=1.9.0,<2.0`、`sentence-transformers>=5.0.0,<6.0`、`jieba>=0.42.1,<1.0`、`rank-bm25>=0.2.2,<1.0`

**缓存**：
- `redis>=5.2.0,<9.0`、`hiredis>=3.1.0,<4.0`

**测试**：
- `pytest>=8.4.0,<10.0`、`pytest-asyncio>=1.0.0,<2.0`

### 11.2 前端核心依赖

- `next: 16.2.9`、`react: 19.2.3`、`react-dom: 19.2.3`
- `tailwindcss: ^4`、`shadcn: ^2.4.0`、Radix UI 全套（avatar/dialog/dropdown-menu/select 等）
- `recharts: ^3.2.1`、`sonner: ^2.0.7`、`next-themes: ^0.4.6`、`lucide-react: ^0.468.0`
- `date-fns: ^4.1.0`、`clsx: ^2.1.1`、`tailwind-merge: ^2.6.0`、`class-variance-authority: ^0.7.1`
- `packageManager: pnpm@9.15.0`

### 11.3 模块间调用关系

```text
bot ──→ inference (vLLM 客户端 / model_manager)
  │
  ├──→ knowledge (intent_detector / RAG via HTTP API)
  │
  └──→ db.adapter (消息持久化)

inference ──→ knowledge (lora_router 延迟导入 intent_detector)
  │
  └──→ infra (circuit_breaker / failover)

training ──→ inference (intent_trainer 用 VLLMClient 生成样本)
  │
  └──→ training.trainer (preference_trainer 复用 GpuTemperatureCallback)

api ──→ services (chat_generation / model_management)  [新路由优先]
  │     └─→ repositories (messages / user_data) ──→ db.adapter
  │
  ├─[兼容]─→ db.adapter (旧路由直接导入全局单例)
  │
  ├──→ inference (generate / loras / models / router)
  │
  ├──→ knowledge (knowledge / retrieval_eval)
  │
  ├──→ training (training / preferences)
  │
  ├──→ evaluation (evaluation)
  │
  ├──→ experiments (experiments)
  │
  └──→ infra (enhanced / stats)

app.main.create_app ──→ app.runtime.RuntimeContainer
  │                      ├─→ db.adapter
  │                      └─→ infra.concurrency_control.inference_runtime
  ├─→ app.readiness.ReadinessProbe ──→ infra.bounded_executor
  └─→ app.providers ──→ services / repositories
```

---

## 12. 关键设计要点

1. **接口契约驱动**：所有核心模块通过 `backend/interfaces/__init__.py` 的 `@runtime_checkable` Protocol 定义契约（DatabaseInterface/InferenceInterface/CacheInterface/MessageQueueInterface/VectorSearchInterface/CircuitBreakerInterface），保证可替换性。

2. **双模式数据库**：SQLite（开发）与 PostgreSQL（生产）通过 `db/adapter.py` 透明切换，`SyncPgAdapter` 在后台线程运行独立事件循环桥接异步实现。

3. **多级缓存体系**：L1 进程内 LRU（极快） → L2 Redis（跨进程） → 数据库；TTL 抖动 ±10% 防雪崩；每 key 独立 Lock 防击穿；空值缓存防穿透。

4. **熔断器 + 故障转移**：3 状态机（CLOSED/OPEN/HALF_OPEN），20 次连续失败熔断，60 秒后半开探测；vLLM 多实例健康检查 + 自动回切。

5. **LoRA 热切换**：vLLM `--enable-lora` + `load_lora_adapter` 在线加载；transformers 后端通过 `PeftModel.from_pretrained` 重新挂载；`AdapterChecker` 激活前兼容性检查，不兼容降级 default。

6. **RAG 两阶段检索**：多查询扩展（5 变体）→ 向量+BM25 混合召回（top_k*4）→ 多查询去重融合 + 区域加权 → Cross-Encoder 重排 → 置信度弃答 + 纠错 RAG。

7. **研究严谨性 guardrail**：
   - 合成偏好对 `pending` 状态禁训练，需人工审核 `approved` 才可用
   - DPO/ORPO 不宣称 RLHF（未训练奖励模型），win rate 需 held-out 评估
   - 盲评流程防篡改（source_hashes + seed）
   - 质量门（benchmark_gate）强制 7 项指标阈值
   - 实验强制受控变量与条件结论

8. **安全纵深防御**：6 个中间件链 + JWT + API Key 双认证 + HMAC 签名 + nonce 重放保护 + AES-256-GCM 字段级加密 + 17 条 prompt injection 模式检测 + 审计日志 90 天 + 安全响应头 + 信任代理头显式开关。

9. **单进程强约束**：`BACKEND_WORKERS=1`，因为幂等缓存、会话锁、nonce 状态在进程内实现；生产环境由部署校验强制。

10. **前后端契约单一来源**：`src/lib/api.ts` 集中定义所有接口方法与 TypeScript 类型，hooks 封装数据获取与状态，页面只消费 hooks；Next.js 代理层做 CSRF + 路径白名单 + 认证转换。

---

## 附录：相关文档索引

| 文档 | 用途 |
|---|---|
| `README.md` | 项目总览与本地/服务器验证命令 |
| `docs/architecture/CODE_WIKI.md` | 本文档：代码知识库（权威技术文档） |
| `docs/architecture/OPTIMIZATION_STRATEGY.md` | 并发/可靠/安全/可观测/部署原则 |
| `docs/research/RESEARCH_AND_LEARNING_ROADMAP.md` | 研究路线图、10 周计划、部署验收、学习资源（合并自 PROJECT_STATUS/LLM_RESEARCH/PERSONAL_ACTION 三份文档） |
| `docs/research/BEGINNER_REAL_LLM_EXPERIMENT_GUIDE.md` | 新手实验指南 |
| `docs/data/human-scoring-rubric.md` | 盲评与偏好标注评分标准 |
| `astrbot_plugins/multipersonal_gateway/README.md` | AstrBot 网关插件安装说明 |
