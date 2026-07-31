"""Lightweight observability helpers for structured logs and alert counters."""

from __future__ import annotations

import json
import logging
import time
import threading
from collections import defaultdict, deque
from typing import Any

logger = logging.getLogger("observability")

_COUNTERS: dict[str, int] = defaultdict(int)
_STATE_LOCK = threading.RLock()

_SENSITIVE_FIELD_PARTS = {"token", "secret", "password", "authorization", "cookie", "openid", "unionid", "phone", "mobile"}


def _redact_fields(value: Any) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if any(part in str(key).lower() for part in _SENSITIVE_FIELD_PARTS):
                result[str(key)] = "***" if item else item
            else:
                result[str(key)] = _redact_fields(item)
        return result
    if isinstance(value, list):
        return [_redact_fields(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_fields(item) for item in value)
    return value

_RECENT: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=1000))
_CONSECUTIVE: dict[str, int] = defaultdict(int)


def _now() -> float:
    return time.time()


def increment(name: str, amount: int = 1) -> int:
    with _STATE_LOCK:
        _COUNTERS[name] += amount
        _RECENT[name].append(_now())
        return _COUNTERS[name]


def set_consecutive(name: str, success: bool) -> int:
    with _STATE_LOCK:
        if success:
            _CONSECUTIVE[name] = 0
        else:
            _CONSECUTIVE[name] += 1
        return _CONSECUTIVE[name]


def get_counter(name: str) -> int:
    with _STATE_LOCK:
        return _COUNTERS.get(name, 0)


def get_consecutive(name: str) -> int:
    with _STATE_LOCK:
        return _CONSECUTIVE.get(name, 0)


def count_recent(name: str, window_seconds: float = 300.0) -> int:
    with _STATE_LOCK:
        cutoff = _now() - window_seconds
        records = _RECENT.get(name)
        if not records:
            return 0
        while records and records[0] < cutoff:
            records.popleft()
        return len(records)


def snapshot() -> dict[str, Any]:
    with _STATE_LOCK:
        return {
            "counters": dict(_COUNTERS),
            "consecutive": dict(_CONSECUTIVE),
            "recent5m": {key: count_recent(key, 300.0) for key in list(_RECENT.keys())},
        }


def log_event(event: str, level: str = "info", **fields: Any) -> None:
    payload = {"event": event, **_redact_fields(fields)}
    message = json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":"))
    log_method = getattr(logger, level.lower(), logger.info)
    log_method(message)
