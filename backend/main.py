#!/usr/bin/env python3
"""MultiPersonal Chat System - 后端主服务入口（向后兼容）

此文件保留用于向后兼容。新入口请使用 run.py。
实际应用逻辑已拆分到 app/main.py。
"""
import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

if __name__ == "__main__":
    import uvicorn
    from app.env import load_backend_env
    from run import _positive_int_env
    load_backend_env()
    import os
    worker_count = int(os.getenv("BACKEND_WORKERS", "1"))
    if worker_count != 1:
        raise RuntimeError("当前架构要求 BACKEND_WORKERS=1，以保证幂等、会话锁和 nonce 状态一致")
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=worker_count,
        limit_concurrency=_positive_int_env("BACKEND_LIMIT_CONCURRENCY", 256),
        backlog=_positive_int_env("BACKEND_BACKLOG", 512),
        timeout_keep_alive=_positive_int_env("BACKEND_KEEPALIVE_TIMEOUT", 10),
        timeout_graceful_shutdown=30,
    )
