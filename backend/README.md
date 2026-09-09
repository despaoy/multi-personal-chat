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
生成异常或空回复时使用固定弃答兜底，并返回 `character_abstention_fallback` 警告。
这一行为适用于聊天链路，不改变独立证据问答接口的弃答策略；提示词约束仍需通过真实模型评测验证。

检索超时或异常同样进入受限回答，提示依据暂不可核实。模型管理器回退与 vLLM 共用这套
检索和提示词策略。弃答表达生成失败仍记录为模型调用失败，固定回复只是服务兜底。

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
