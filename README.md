# QQChat Enhanced

QQChat Enhanced 是一个面向角色对话研究与保研展示的多平台 LLM 系统。项目覆盖数据治理、LoRA/DoRA/RSLoRA 微调、AWQ 高效推理、混合 RAG、评测体系、AstrBot 消息网关以及可观测的 Web 管理台。

> 当前定位：单机可部署、证据驱动的研究原型。项目强调完整的“数据 -> 训练 -> 推理 -> 检索 -> 评测 -> 多平台交付”链路，不以堆叠云原生组件为目标。
>
> 发布状态：代码可构建、测试可复现；R1V4 正式训练仍被数据质量门禁阻塞，当前仓库不包含任何“正式实验结论”。

## 当前技术基线

| 模块 | 推荐版本或实现 |
| --- | --- |
| 基础模型 | `Qwen/Qwen3-8B`（本地兼容别名 `Qwen3-8B-Instruct`） |
| 量化推理 | `Qwen/Qwen3-8B-AWQ`（本地兼容别名 `Qwen3-8B-Instruct-AWQ`）+ vLLM 0.10.2 |
| 训练 | PyTorch 2.8、Transformers 4.57、PEFT、TRL |
| 后端 | Python 3.12 + FastAPI |
| 前端 | Node.js 22 + Next.js 16 + React 19 |
| RAG | FAISS/BM25 混合检索 + BGE-M3 + 可选 Reranker |
| 平台网关 | AstrBot 插件，统一接入 QQ、微信系、Telegram 等平台 |
| 数据库 | SQLite（本地/单进程）或 PostgreSQL（部署推荐） |
| 缓存 | Redis 可选；不可用时使用受限的进程内退化实现 |

实验室服务器已验证 PyTorch 2.8.0+cu128、RTX 3090 CUDA、BGE-M3 和 vLLM 0.10.2 可用。月社妃 V4 当前包含 1002 条训练数据（576 条原作、150 条既有构造数据、276 条经审核晋升的 V4.1 五轮会话，其中 DeepSeek round06 4 条、Codex 自动化批次 272 条）和 70 条独立验证数据，Gold v2.1 已批准为开发集，Gold v3 的 150 条最终盲测已冻结。正式训练仍由 Game Train 上下文质量复审门禁阻塞；权威计数以 V4 canonical manifest 为准。

## 系统架构

```text
QQ / WeChat / Telegram
          |
       AstrBot
          |
  qqchat_gateway plugin
          |
      FastAPI core
   /       |        \
vLLM     RAG      SQLite/PostgreSQL
  |        |              |
LoRA   BGE/FAISS         Redis
          |
     Next.js console
```

关键设计边界：

- AstrBot 只负责平台事件标准化、鉴权调用和回复发送。
- FastAPI 负责会话策略、幂等、队列、RAG、模型调用、历史和指标。
- vLLM/Transformers 是可替换的推理后端。
- 模型、LoRA、数据库、日志和向量索引存放在仓库之外。
- 所有外部边界都设置超时、降级和结构化错误。

## 主要能力

### LLM 与 LoRA

- Qwen3-8B SFT，支持 LoRA、DoRA、RSLoRA、NEFTune、Sequence Packing。
- 固定训练集、验证集和随机种子的受控消融实验。
- adapter 兼容性检查、扫描、激活、回滚和多 LoRA 路由。
- 角色一致性、格式、重复率、Distinct-N、安全、RAG 引用和人工盲评。

### RAG

- 向量检索、BM25、Hybrid 和可选 Reranker。
- Corrective RAG、引用、置信度和拒答策略。
- 文档导入、分块、索引更新、检索评测和缓存失效。

### AstrBot 多平台

- 统一消息协议和 `traceId`。
- `platform + adapter + messageId` 幂等。
- 平台/会话/发送者/全局限流。
- QQ、Telegram、企业微信、公众号和个人微信适配边界。
- 群聊静默降级、私聊简短降级提示。

### 工程能力

- JWT 认证、CSRF/同源校验、内部集成 token 和输入限制。
- 优先级队列、模型并发控制、会话串行化和熔断。
- 结构化日志、指标、告警、健康检查和服务状态。
- Alembic 数据库迁移、分页查询和多平台数据模型。
- 单元、集成、契约、本地 smoke 和真实模型评测。

## 仓库结构

```text
astrbot_plugins/     AstrBot 网关插件
backend/
├── api/              FastAPI 路由
├── app/              配置、依赖、应用生命周期与运行时容器
├── data/             月社妃语料、canonical 数据集与历史数据归档
├── cache/            Redis/内存缓存、队列、语义缓存
├── db/               SQLite/PostgreSQL、模型与迁移
├── evaluation/       Gold Set、角色、安全与检索指标
├── experiments/      LoRA/RAG/量化消融
├── inference/        vLLM 客户端、模型与 LoRA 路由
├── infra/            安全、并发、熔断、观测
├── knowledge/        导入、分块、检索、重排序
├── repositories/     持久化 Protocol 与数据库适配实现
├── services/         与 HTTP 传输无关的业务用例
├── tests/            后端测试
└── training/         SFT、偏好训练和任务管理
deploy/               Compose、Nginx 和部署脚本
docs/                 架构、运维、研究和数据文档
gametext/             可审计的原始角色语料
scripts/              验证、训练、评测和服务器启动工具（分类索引见 scripts/README.md）
src/                  Next.js 管理台
```

服务器上的模型与训练资产遵循 [服务器目录规范](docs/operations/SERVER_LAYOUT.md)，不放入源码目录。

## 快速验证

### Windows 本地

```powershell
pnpm install --frozen-lockfile
pnpm ts-check
pnpm lint
pnpm build
pnpm audit --prod
py -3.12 -m pytest backend/tests -q
powershell -ExecutionPolicy Bypass -File scripts/local-verify.ps1
```

无 NVIDIA GPU 的本地环境可以验证前端、API、数据库、鉴权、Schema 和 mock 边界，但不能替代真实 vLLM/LoRA 评测。

### 实验室服务器

实验室脚本统一通过 `QQCHAT_LAB_ROOT` 指定服务器根目录，不在代码中硬编码个人路径：

```bash
export QQCHAT_LAB_ROOT=/path/to/lab-root
source "$QQCHAT_LAB_ROOT/activate_qqchat.sh"
cd "$QQCHAT_LAB_ROOT/qqchat-enhanced"
python -m pip check
python -m pytest backend/tests -q
pnpm ts-check
```

服务验收：

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/ready
curl -fsS http://127.0.0.1:8001/v1/models
curl -fsS http://127.0.0.1:5000/api/health
```

默认 vLLM 端口是 8001；独立 LoRA 对比实验可使用 8002，实际地址始终由 `VLLM_BASE_URL` 配置。

## 配置原则

配置文件按运行模式分开，仓库根目录的 `.env` 不是约定入口：

- 裸机后端：复制 `.env.example` 为 `backend/.env`。
- Docker Compose：复制 `.env.example` 为 `deploy/.env`。
- Next.js 本地变量：写入根目录 `.env.local`，或由部署平台注入。

至少配置：

- `JWT_SECRET`
- `ENCRYPTION_KEY`
- `ASTRBOT_INTEGRATION_TOKEN`
- `MODEL_PROVIDER=vllm`
- `VLLM_BASE_URL`
- PostgreSQL：`DATABASE_URL`，或 `USE_POSTGRESQL=true` + 完整 `PG_*`；SQLite：`DATABASE_PATH`
- `BASE_MODEL_PATH`
- `LORA_PATH`
- `VECTOR_DB_PATH`
- `EMBEDDING_MODEL_PATH`
- `ALLOWED_ORIGINS`

真实密钥不得进入 Git、前端响应或日志。部署细节见 [部署与验收指南](docs/operations/DEPLOYMENT_GUIDE.md)。

## 文档

完整索引见 [docs/README.md](docs/README.md)。

- [代码知识库](docs/architecture/CODE_WIKI.md)
- [可扩展性开发指南](docs/architecture/EXTENSIBILITY_GUIDE.md)
- [优化策略](docs/architecture/OPTIMIZATION_STRATEGY.md)
- [生产准备审查](docs/architecture/PRODUCTION_READINESS_REVIEW_2026-07-18.md)
- [发布前检查清单](docs/RELEASE_CHECKLIST.md)
- [部署指南](docs/operations/DEPLOYMENT_GUIDE.md)
- [保研项目答辩与深度问答手册](docs/research/POSTGRADUATE_RECOMMENDATION_DEFENSE_PLAYBOOK.md)
- [研究与学习路线](docs/research/RESEARCH_AND_LEARNING_ROADMAP.md)
- [月社妃 V4 人工审核与重训练](docs/research/KISAKI_V4_HUMAN_REVIEW_AND_RETRAINING.md)
- [月社妃实验总览](docs/research/KISAKI_EXPERIMENT_INDEX.md)
- [人工评分标准](docs/data/human-scoring-rubric.md)

## 研究诚信

- mock 输出不能作为真实实验结果。
- 旧版本结果仅从 Git 历史追溯，不与 Qwen3 当前结果混算。
- 偏好数据必须保留审核状态；DPO/ORPO 不表述为 RLHF。
- 每次实验记录代码提交、数据哈希、模型版本、随机种子、硬件、命令和原始结果。
- Gold Set 不得进入训练集，任何重叠都必须在报告中披露。
- 结论同时报告质量、延迟、显存和失败样本，不只展示最好结果。

## 安全

密钥策略和报告方式见 [SECURITY.md](SECURITY.md)。

## 许可证

详见 [LICENSE](LICENSE)。
