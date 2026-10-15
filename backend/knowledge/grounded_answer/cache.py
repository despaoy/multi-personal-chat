"""Grounded Answer 有限缓存（沿用进程内设施，不建分布式缓存）。

缓存键绑定：
- domain（或 auto 域集合）
- 原始 query / speaker（不合并可能改变模型输入的差异）
- filters / top_k
- index_version（索引更新后旧缓存不再命中，避免过期证据）
- prompt 契约版本
- 生成模型 ID、实际连接身份摘要与关键生成参数
- answer_mode / persona 指纹
- 输入对话历史指纹（同一句追问在不同上下文下不能复用答案）

缓存值：完整 GroundedAnswerResult.to_api_payload 的内部形态
（服务端直接复用，不再检索与生成）。
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from copy import deepcopy
from math import isfinite
from typing import TYPE_CHECKING, Any

from .prompt import GROUNDED_PROMPT_VERSION

if TYPE_CHECKING:
    from .models import GroundedAnswerResult

_MAX_CACHE_ENTRIES = 128


def answer_cache_key(
    *,
    domain_id: str | None,
    domains: list[str],
    query: str,
    filters: dict[str, Any] | None,
    top_k: int,
    index_version: str,
    model_id: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    persona_prompt: str,
    speaker: str,
    want_citations: bool,
    history: list[dict[str, str]] | None = None,
    cache_identity: str = "",
) -> str:
    payload = {
        "domains": sorted(domains) if domains else [str(domain_id or "auto")],
        "query": query,
        "filters": filters or {},
        "top_k": int(top_k),
        "index_version": index_version,
        "prompt_version": GROUNDED_PROMPT_VERSION,
        "model_id": model_id,
        "cache_identity": cache_identity,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "top_p": float(top_p),
        "persona": hashlib.sha256((persona_prompt or "").encode("utf-8")).hexdigest()[:16],
        "speaker": speaker,
        "want_citations": bool(want_citations),
        "history": hashlib.sha256(
            json.dumps(history or [], ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class AnswerCache:
    """进程内 TTL + 容量受限 LRU（线程安全）。"""

    def __init__(self, ttl_seconds: float = 120.0, max_entries: int = _MAX_CACHE_ENTRIES):
        self.ttl = float(ttl_seconds)
        if isinstance(ttl_seconds, bool) or not isfinite(self.ttl) or self.ttl < 0:
            raise ValueError("AnswerCache ttl_seconds must be a finite non-negative number")
        if type(max_entries) is not int or max_entries <= 0:
            raise ValueError("AnswerCache max_entries must be a positive integer")
        self.max_entries = max_entries
        self._store: OrderedDict[str, tuple[float, GroundedAnswerResult]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> GroundedAnswerResult | None:
        if self.ttl <= 0:
            return None
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self.misses += 1
                return None
            stored_at, result = entry
            if time.monotonic() - stored_at > self.ttl:
                del self._store[key]
                self.misses += 1
                return None
            cached = deepcopy(result)
            self._store.move_to_end(key)
            self.hits += 1
            return cached

    def set(self, key: str, result: GroundedAnswerResult) -> None:
        if self.ttl <= 0:
            return
        with self._lock:
            self._store[key] = (time.monotonic(), deepcopy(result))
            self._store.move_to_end(key)
            if len(self._store) > self.max_entries:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "entries": len(self._store),
                "ttl_seconds": self.ttl,
                "hits": self.hits,
                "misses": self.misses,
            }
