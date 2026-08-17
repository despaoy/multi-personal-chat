#!/usr/bin/env python3
"""MultiPersonal Chat System - 后端启动入口

使用方式:
    python run.py               # 启动服务（端口8000）
    python run.py --port 8080   # 指定端口
    python run.py --reload      # 开发模式（热重载）
"""
import sys
import os
from pathlib import Path

# 确保 backend 根目录在 Python 路径中
_BACKEND_ROOT = Path(__file__).parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

# 使用与 app.main 相同的 dotenv 加载器；外部注入变量保持最高优先级。
from app.env import load_backend_env

load_backend_env()

import uvicorn


def _positive_int_env(name: str, default: int) -> int:
    """Read a positive integer runtime limit with a startup-friendly error."""
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be a positive integer") from exc
    if value < 1:
        raise RuntimeError(f"{name} must be a positive integer")
    return value

def main():
    import argparse
    parser = argparse.ArgumentParser(description="MultiPersonal Chat System后端服务")
    parser.add_argument("--port", type=int, default=8000, help="监听端口（默认8000）")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--reload", action="store_true", help="开发模式热重载")
    parser.add_argument("--workers", type=int, default=None, help="Worker进程数（当前架构必须为1）")
    args = parser.parse_args()

    worker_count = args.workers if args.workers is not None else int(os.getenv("BACKEND_WORKERS", "1"))
    if worker_count != 1:
        parser.error(
            "当前幂等缓存、会话锁和集成 nonce 状态为进程内实现，BACKEND_WORKERS/--workers 必须为 1"
        )

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=worker_count,
        limit_concurrency=_positive_int_env("BACKEND_LIMIT_CONCURRENCY", 256),
        backlog=_positive_int_env("BACKEND_BACKLOG", 512),
        timeout_keep_alive=_positive_int_env("BACKEND_KEEPALIVE_TIMEOUT", 10),
        timeout_graceful_shutdown=30,
    )


if __name__ == "__main__":
    main()
