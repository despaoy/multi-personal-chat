# Backend

FastAPI 后端使用 Python 3.12，负责认证、会话与消息、角色记忆、推理编排、RAG、训练、评测、实验和运行观测。业务入口统一由 `app/main.py` 组装；生产与本地启动统一使用 `run.py`。

## 目录边界

| 目录 | 职责 |
| --- | --- |
| `api/` | HTTP 路由与传输层校验 |
| `app/` | 应用工厂、生命周期、配置与依赖装配 |
| `services/` | 与 HTTP 无关的业务用例 |
| `repositories/` | 持久化接口与数据库适配 |
| `db/`、`alembic/` | SQLite/PostgreSQL 与迁移 |
| `inference/`、`training/` | 推理后端、LoRA 路由与训练任务 |
| `knowledge/` | 角色知识检索、通用用户知识库、共享检索内核与证据约束回答 |
| `evaluation/`、`experiments/` | 评测与受控实验 |
| `infra/`、`cache/`、`middleware/` | 并发、熔断、缓存、安全与观测 |
| `character/` | 角色上下文、关系与记忆 |
| `scripts/`、`benchmarks/` | 后端维护工具与显式运行的性能测试 |
| `tests/` | 单元、集成、契约与研究数据回归测试 |

## 入口与验证

```bash
cd backend
python run.py --host 127.0.0.1 --port 8000
python -m pytest tests -q
python -m alembic heads
```

`main.py` 仅保留为旧部署兼容入口；新脚本和文档不得再依赖它。当前架构要求 `BACKEND_WORKERS=1`，因为幂等、会话锁和 nonce 状态仍是进程内状态。

运行时数据库、模型、LoRA、日志、索引和密钥不属于源码目录，边界见 [`../docs/operations/SERVER_LAYOUT.md`](../docs/operations/SERVER_LAYOUT.md)。

## 知识检索边界

- `knowledge/multiscale_rag/` 是经过审核的作品知识生产入口。
- `knowledge/rag_helper.py` 与 `knowledge/vector_db.py` 服务用户管理的通用知识库。
- `knowledge/retrieval_core/` 是两类检索可复用的底层组件，不是独立服务。
- `knowledge/grounded_answer/` 负责证据约束生成、引用绑定和拒答。

聊天生成链路在检索置信度不足时采用角色化弃答：保留人物设定、动态上下文和所选 LoRA，
由模型自然表达不知道或无法确定。低置信度候选及其引用不会进入生成请求，系统指令禁止
猜测剧情、关系、原文和出处。返回仍标记 `abstained=true`、`answerMode=abstention`，
`citations` 为空；`modelInvoked=true` 表示尝试了表达生成，不代表成功回答了事实问题。
生成异常、非文本或空回复时明确失败，不返回固定弃答。
这一行为适用于聊天链路，不改变独立证据问答接口的弃答策略；提示词约束仍需通过真实模型评测验证。

检索或证据审核异常返回 HTTP 503，并保留异常原因供内部定位，不再进入回答生成。
模型管理器与 vLLM 共用这套策略；生成失败不自动切换模型提供商。

## 平台发送与重试

升级此版本时需同时更新 AstrBot 插件和后端，并执行 `python -m alembic upgrade head`
（在 backend 目录）。新增迁移 `008_integration_receipts` 保存消息处理及发送回执；
SQLite 本地初始化也支持创建该表。已有历史数据不会删除，旧去重表保留。

消息按平台、适配器、会话类型、会话、发送者和消息 ID 进行幂等处理。失败允许重试，
正在处理的重复请求不会再次生成；已生成但尚未确认发送的请求重用已保存回复。
插件等待平台 `send` 成功后调用 `/api/integrations/astrbot/delivery`，该接口使用与
消息入口相同的令牌和签名验证。未确认发送的回复不进入角色历史，人物记忆和关系更新
推迟到首次成功确认；重复确认不会重复更新。

这里的“已送达”表示平台适配器发送调用成功，不代表用户已读。确认失败时插件重试三次，
随后可在重复事件到达时再次确认；平台已接收但确认丢失、进程重启等情况下仍存在重复发送的
可能性，不承诺跨平台严格恰好一次。发送确认与人物状态更新不是跨系统事务，进程在确认后
立即退出时可能缺少一次记忆更新。无原生消息 ID 的事件使用事件级 ID，跨进程重新构造的事件
无法可靠识别为同一次消息。

构建角色知识索引：

```bash
python scripts/build_character_rag_index.py
```

完整的数据层级、配置和降级行为见
[`../docs/architecture/CHARACTER_KNOWLEDGE_RETRIEVAL.md`](../docs/architecture/CHARACTER_KNOWLEDGE_RETRIEVAL.md)。

模型提供商选择：非空 `MODEL_PROVIDER` 优先于数据库 `modelProvider`；缺失或非法值明确失败，不切换为 mock。mock 仅可在 development/test 环境显式启用，日志会标明模拟推理。模型配置数据库读取失败不会使用空配置继续运行。

## 就绪检查

`/health` 只表示进程响应；`/ready` 检查数据库和模型管理器当前选中的提供商，模型检查不能因关闭 vLLM 而跳过。
OpenAI 兼容服务使用与生成相同的配置和鉴权查询模型列表；Ollama 查询所选模型详情，llama.cpp 查询健康状态，vLLM 使用聊天链路的共享客户端。
鉴权、连接、解析或配置失败返回 503；模型不在列表、mock、尚未加载的 Transformers 模型也不报告就绪。Transformers 必须先完成模型预热，探测本身不会加载权重。
成功结果缓存 5 秒、失败 1 秒，并发请求共享一次探测；运行时模型切换最迟在当前缓存到期后反映。
检查不发送聊天请求，不消耗生成 token，不代表余额、回答质量、全部 LoRA 或可选 RAG 已通过验证。`details.rag=not_probed` 明确保留该边界。

动态上下文：显式关闭语义审核、清晰轮次无需审核和安全规则保护属于正常路径；需要执行的状态/策略审核失败、输入超出完整证据预算、递归审核或规则分析失败会停止本轮准备。聊天入口返回 503 并保留内部异常原因，不丢弃角色继续普通聊天。语义审核超时配置非法时明确报错。

历史、关系备忘录和已启用的记忆原文召回属于上下文准备的必要读取。正式数据库缺少接口或读取失败时异常传播到聊天入口，不以空记录继续；合法空查询、已删除来源和未启用原文召回仍保留正常语义。假想分支继续按明确的独立来源策略处理。
