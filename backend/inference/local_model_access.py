"""Shared exclusion for all entry points using a stateful local adapter."""

import asyncio

_lock: asyncio.Lock | None = None
_loop: asyncio.AbstractEventLoop | None = None


def get_local_model_lock() -> asyncio.Lock:
    global _lock, _loop
    loop = asyncio.get_running_loop()
    if _lock is None or _loop is not loop:
        _lock = asyncio.Lock()
        _loop = loop
    return _lock
