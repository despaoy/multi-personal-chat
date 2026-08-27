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
| `knowledge/`、`evaluation/`、`experiments/` | RAG、评测与受控实验 |
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
