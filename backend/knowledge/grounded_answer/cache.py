"""P7 有限缓存（沿用进程内设施，不建分布式缓存）。

缓存键绑定：
- domain（或 auto 域集合）
- query 归一化（去控制字符/折叠空白/小写 ASCII）
- filters / top_k
- index_version（索引更新后旧缓存不再命中，避免过期证据）
- prompt 契约版本
- 生成模型 ID 与关键生成参数
- answer_mode / persona 指纹

缓存值：完整 GroundedAnswerResult.to_api_payload 的内部形态
（服务端直接复用，不再检索与生成）。
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import unicodedata
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from .prompt import GROUNDED_PROMPT_VERSION

if TYPE_CHECKING:
    from .models import GroundedAnswerResult

_MAX_CACHE_ENTRIES = 128


def normalize_query(query: str) -> str:
    """查询归一化（缓存键组成部分；不改变检索语义，仅归一空白）。"""
    cleaned = "".join(ch for ch in (query or "") if unicodedata.category(ch)[0] != "C")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.casefold()


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
) -> str:
    payload = {
        "domains": sorted(domains) if domains else [str(domain_id or "auto")],
        "query": normalize_query(query),
        "filters": filters or {},
        "top_k": int(top_k),
        "index_version": index_version,
        "prompt_version": GROUNDED_PROMPT_VERSION,
        "model_id": model_id,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "top_p": float(top_p),
        "persona": hashlib.sha256((persona_prompt or "").encode("utf-8")).hexdigest()[:16],
        "speaker": normalize_query(speaker)[:48],
        "want_citations": bool(want_citations),
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class AnswerCache:
    """进程内 TTL + 容量受限 LRU（线程安全）。"""

    def __init__(self, ttl_seconds: float = 120.0, max_entries: int = _MAX_CACHE_ENTRIES):
        self.ttl = max(0.0, float(ttl_seconds))
        self.max_entries = max(1, int(max_entries))
        self._store: dict[str, tuple[float, GroundedAnswerResult]] = {}
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
            self.hits += 1
            return deepcopy(result)

    def set(self, key: str, result: GroundedAnswerResult) -> None:
        if self.ttl <= 0:
            return
        with self._lock:
            if key in self._store:
                del self._store[key]
            elif len(self._store) >= self.max_entries:
                oldest = min(self._store, key=lambda k: self._store[k][0])
                del self._store[oldest]
            self._store[key] = (time.monotonic(), deepcopy(result))

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
