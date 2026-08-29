# MultiPersonal Chat System

一个面向角色对话的单机多平台 LLM 系统。它在一个仓库中提供可审计的角色语料、LoRA/DoRA/RSLoRA 微调、AWQ 高效推理、多粒度角色知识检索、自动化评测、AstrBot 多平台消息接入和 Web 管理台。

> **定位**：证据驱动的研究原型，不是云原生组件堆叠。每个实验结果都要求保留数据哈希、随机种子、硬件、命令和原始结果。
>
> **发布状态**：工程收尾中。已保留可复现的负实验结果，但当前没有通过门禁的正式 LoRA adapter；发布结论以 [`docs/RELEASE_CHECKLIST.md`](docs/RELEASE_CHECKLIST.md) 的本次验证记录为准。

## 核心能力

### 角色训练与推理

- 以 `Qwen/Qwen3-8B` 为基础模型，支持 LoRA、DoRA、RSLoRA、NEFTune 和 Sequence Packing。
- 固定训练集、验证集和随机种子，进行单变量受控消融实验。
- LoRA adapter 扫描、兼容性检查、激活、回滚和多 LoRA 路由。
- 支持 AWQ 量化模型与 vLLM 推理后端。

### 知识检索（RAG）

- 多粒度角色知识检索将作品知识组织为故事、场景、事实/关系/事件知识卡和原文证据，按问题类型路由到对应粒度。
- BM25、向量和实体混合召回，经 RRF 融合与重排序后回填父场景，并返回引用、置信度和拒答状态。
- 通用用户知识库独立处理上传文档、分块、索引更新和带 `knowledge_base_id` 的检索，不与角色作品索引混用。
- Grounded Answer 对检索证据进行引用校验；证据不足时请求澄清或拒答。

架构、索引构建和配置见[多粒度角色知识检索](docs/architecture/CHARACTER_KNOWLEDGE_RETRIEVAL.md)。

### 多平台接入

- AstrBot 网关插件 `multipersonal_gateway` 统一接入 QQ、Telegram、企业微信、公众号和个人微信。
- `traceId` 全链路追踪，`platform + adapter + messageId` 幂等。
- 平台、会话、发送者和全局限流；群聊与私聊降级策略。

### 工程与评测

- JWT 认证、同源校验、内部集成 token、输入限制和密钥管理。
- 优先级队列、并发控制、会话串行化、熔断和结构化日志。
- 角色一致性、格式、重复率、Distinct-N、安全、RAG 引用和人工盲评。
- Alembic 迁移、SQLite/PostgreSQL 数据层和完整测试套件。

## 架构

```text
QQ / WeChat / Telegram
          |
       AstrBot
          |
  multipersonal_gateway plugin
          |
      FastAPI core
   /       |        \
vLLM  Character RAG  SQLite/PostgreSQL
  |        |              |
LoRA   BGE/FAISS         Redis
          |
     Next.js console
```

关键边界：

- AstrBot 只负责平台事件标准化、鉴权调用和回复发送。
- FastAPI 负责会话策略、幂等、队列、RAG、模型调用、历史和指标。
- vLLM/Transformers 是可替换的推理后端。
- 模型、LoRA、数据库、日志和向量索引不进入 Git，默认放在仓库外。
- 所有外部边界都设置超时、降级和结构化错误。

## 仓库结构

```text
astrbot_plugins/     AstrBot 网关插件
backend/
├── README.md         后端边界、入口与验证
├── api/              FastAPI 路由
├── app/              配置、依赖、应用生命周期与运行时容器
├── data/             角色语料、canonical 数据集与历史数据归档
├── cache/            Redis/内存缓存、队列、语义缓存
├── db/               SQLite/PostgreSQL、模型与迁移
├── evaluation/       Gold Set、角色、安全与检索指标
├── experiments/      LoRA/RAG/量化消融
├── inference/        vLLM 客户端、模型与 LoRA 路由
├── infra/            安全、并发、熔断、观测
├── knowledge/        角色知识检索、通用知识库、共享检索内核与证据回答
├── repositories/     持久化 Protocol 与数据库适配实现
├── services/         与 HTTP 传输无关的业务用例
├── tests/            后端测试
└── training/         SFT、偏好训练和任务管理
deploy/               Compose、Nginx 和部署脚本
docs/                 用户、运维和研究文档
gametext/             可审计的原始角色语料
scripts/              验证、训练、评测和服务器启动工具
src/                  Next.js 管理台
```

详细脚本分类见 [`scripts/README.md`](scripts/README.md)，数据目录说明见 [`backend/data/character_dialogues/README.md`](backend/data/character_dialogues/README.md)。

## 快速开始

### 环境要求

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| Node.js | 22 |
| pnpm | 10 |
| 数据库 | SQLite（本地）或 PostgreSQL（部署推荐） |
| 推理后端 | vLLM 0.10.2 + NVIDIA GPU，或 mock 模式用于无 GPU 验证 |

### 验证源码

```bash
# 后端
python -m pytest backend/tests -q
python -m compileall -q backend scripts astrbot_plugins
python scripts/check_repository_integrity.py

# 前端
pnpm install --frozen-lockfile
pnpm ts-check
pnpm lint
pnpm build
pnpm audit --prod
```

Windows 下也可以一键执行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/local-verify.ps1 -Frontend
```

无 NVIDIA GPU 的环境可以验证前端、API、数据库、鉴权、Schema 和 mock 边界，但不能替代真实 vLLM/LoRA 评测。

### 配置

配置文件按运行模式分开，仓库根目录的 `.env` 不是约定入口：

- 裸机后端：复制 `.env.example` 为 `backend/.env`。
- Docker Compose：复制 `.env.example` 为 `deploy/.env`。
- Next.js 本地变量：写入根目录 `.env.local`，或由部署平台注入。

至少需要配置：

- `JWT_SECRET`、`ENCRYPTION_KEY`
- `ASTRBOT_INTEGRATION_TOKEN`
- `MODEL_PROVIDER=vllm`
- `VLLM_BASE_URL`
- `BASE_MODEL_PATH`、`LORA_PATH`
- `VECTOR_DB_PATH`、`CHARACTER_RAG_INDEX_ROOT`、`EMBEDDING_MODEL_PATH`
- 数据库：PostgreSQL 使用 `DATABASE_URL` 或 `PG_*`；SQLite 使用 `DATABASE_PATH`
- `ALLOWED_ORIGINS`

真实密钥不得进入 Git、前端响应或日志。完整步骤见[部署与验收指南](docs/operations/DEPLOYMENT_GUIDE.md)。

## 当前研究状态

仓库内置示例角色为“月社妃”，当前 canonical 数据为 **926 条训练记录、1961 个有效 assistant 监督目标和 70 条独立验证记录**。R1V4 E1 及 recovery checkpoint 均未通过真实生成与语义门禁，当前无正式 adapter，E2-E5 暂停。数据、审核和训练状态的权威入口见：

- [月社妃实验总览](docs/research/KISAKI_EXPERIMENT_INDEX.md)
- [V4 人工审核与重训练](docs/research/KISAKI_V4_HUMAN_REVIEW_AND_RETRAINING.md)

正式训练目前仍被数据质量门禁阻塞，因此仓库**不提供**任何可以对外引用的训练提升结论。

## 文档

- [文档中心](docs/README.md)
- [部署与验收指南](docs/operations/DEPLOYMENT_GUIDE.md)
- [服务器目录规范](docs/operations/SERVER_LAYOUT.md)
- [代码知识库](docs/architecture/CODE_WIKI.md)
- [多粒度角色知识检索](docs/architecture/CHARACTER_KNOWLEDGE_RETRIEVAL.md)
- [可扩展性开发指南](docs/architecture/EXTENSIBILITY_GUIDE.md)
- [生产准备审查](docs/architecture/PRODUCTION_READINESS_REVIEW_2026-07-18.md)
- [发布前检查清单](docs/RELEASE_CHECKLIST.md)
- [后端目录与入口](backend/README.md)
- [人工评分标准](docs/data/human-scoring-rubric.md)
- [贡献指南](CONTRIBUTING.md)
- [变更记录](CHANGELOG.md)

## 研究诚信

- mock 输出不能作为真实实验结果。
- 旧版本结果仅从 Git 历史追溯，不与当前结果混算。
- 偏好数据必须保留审核状态；DPO/ORPO 不表述为 RLHF。
- 每次实验记录代码提交、数据哈希、模型版本、随机种子、硬件、命令和原始结果。
- Gold Set 不得进入训练集，任何重叠都必须在报告中披露。
- 结论同时报告质量、延迟、显存和失败样本，不只展示最好结果。

## 安全

密钥策略和漏洞报告方式见 [SECURITY.md](SECURITY.md)。

## 许可证

[MIT](LICENSE)
