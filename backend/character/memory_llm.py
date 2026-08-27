"""后台 LLM 长期记忆判断。

在线回复不等待本模块：成功生成后只把一个有界任务放入队列。单个后台
worker 按入队顺序处理，避免同一用户连续修正事实时旧任务后完成并覆盖
新事实。LLM 只负责提出候选；用户拒绝、敏感信息、原文证据、类型、长度
和数量仍由本地代码最终决定。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from character.memory_extractor import (
    MAX_EXTRACTED_MEMORIES,
    MAX_MEMORY_CONTENT_CHARS,
    ExtractedMemory,
    memory_evidence_allowed,
    memory_name_allowed,
    memory_write_allowed,
)
from character.models import MemoryItem, UserScope
from repositories.character_memory import CharacterMemoryRepository

logger = logging.getLogger(__name__)

_ALLOWED_KINDS = {
    "name",
    "like",
    "dislike",
    "major",
    "study_stage",
    "location",
    "workplace",
    "goal",
    "promise",
    "shared_event",
    "other_user_fact",
}
_CONFIDENCE_THRESHOLD = 0.85
_MAX_VALUE_CHARS = 48
_MAX_EVIDENCE_CHARS = 120

_SYSTEM_PROMPT = """你是长期记忆写入判断器。你的任务不是回复用户，而是从用户原话中找出以后跨会话仍有帮助的明确事实。

只保留：用户自己的稳定身份、明确好恶、持续目标、共同经历和明确约定。
拒绝：当前临时请求或情绪、疑问、猜测、不确定说法、第三方事实、角色设定、模型推测、密码/验证码/令牌/支付与身份凭据、用户要求不要记录的内容。

rule_hints 只是规则产生的候选，必须独立核验，不能盲目批准。每条 evidence 必须是 source_message 中连续出现的原文；value 必须是 evidence 中连续出现的短语。最多返回 4 条，置信度不足 0.85 时不要返回。

只输出严格 JSON，不要 Markdown、解释或思考过程：
{"memories":[{"kind":"name|like|dislike|major|study_stage|location|workplace|goal|promise|shared_event|other_user_fact","value":"原文中的短语","evidence":"用户原文证据","confidence":0.0}]}
没有合格记忆时输出 {"memories":[]}。"""


class MemoryCompletion(Protocol):
    async def complete(self, messages: list[dict[str, str]]) -> str: ...

    async def close(self) -> None: ...


@dataclass(frozen=True)
class MemoryLlmConfig:
    enabled: bool
    base_url: str
    model: str
    api_key: str = ""
    timeout_seconds: float = 30.0
    queue_size: int = 64
    max_input_chars: int = 2000

    @classmethod
    def from_env(cls) -> MemoryLlmConfig:
        enabled = os.getenv("MEMORY_LLM_ENABLED", "false").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        base_url = os.getenv("MEMORY_LLM_BASE_URL", "").strip()
        if not base_url:
            base_url = (
                os.getenv("VLLM_BASE_URLS", "").split(",", 1)[0].strip() or os.getenv("VLLM_BASE_URL", "").strip()
            )
        model = (
            os.getenv("MEMORY_LLM_MODEL", "").strip()
            or os.getenv("VLLM_SERVED_MODEL_NAME", "").strip()
            or os.getenv("VLLM_MODEL", "").strip()
        )
        return cls(
            enabled=enabled and bool(base_url and model),
            base_url=base_url,
            model=model,
            api_key=os.getenv("MEMORY_LLM_API_KEY", "").strip() or os.getenv("VLLM_API_KEY", "").strip(),
            timeout_seconds=max(1.0, float(os.getenv("MEMORY_LLM_TIMEOUT", "30"))),
            queue_size=max(1, int(os.getenv("MEMORY_LLM_QUEUE_SIZE", "64"))),
            max_input_chars=max(256, int(os.getenv("MEMORY_LLM_MAX_INPUT_CHARS", "2000"))),
        )


class OpenAICompatibleMemoryCompletion:
    """仅用于记忆判断的 OpenAI 兼容客户端。"""

    def __init__(self, config: MemoryLlmConfig) -> None:
        base = config.base_url.rstrip("/")
        self._endpoint = f"{base}/chat/completions" if base.endswith("/v1") else f"{base}/v1/chat/completions"
        self._model = config.model
        self._api_key = config.api_key
        self._timeout = config.timeout_seconds
        self._client: httpx.AsyncClient | None = None

    async def complete(self, messages: list[dict[str, str]]) -> str:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        response = await self._client.post(
            self._endpoint,
            headers=headers,
            json={
                "model": self._model,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 512,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        response.raise_for_status()
        payload = response.json()
        return str(payload["choices"][0]["message"]["content"])

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


@dataclass(frozen=True)
class _MemoryJob:
    repository: CharacterMemoryRepository
    character_id: str
    user_scope: UserScope
    message: str
    rule_hints: tuple[ExtractedMemory, ...]
    source_message_id: str | None


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text or "").strip("，。！？,!?：:；; ")


def _truncate(text: str, limit: int) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1].rstrip() + "…"


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("记忆 LLM 未返回 JSON 对象")
    value, _end = json.JSONDecoder().raw_decode(cleaned[start:])
    if not isinstance(value, dict):
        raise ValueError("记忆 LLM 顶层结果必须是对象")
    return value


def _candidate_to_memory(raw: Any, *, source_message: str) -> ExtractedMemory | None:
    if not isinstance(raw, dict):
        return None
    kind = str(raw.get("kind") or "").strip()
    value = re.sub(r"\s+", " ", str(raw.get("value") or "")).strip()
    evidence = re.sub(r"\s+", " ", str(raw.get("evidence") or "")).strip()
    try:
        confidence = float(raw.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return None
    if (
        kind not in _ALLOWED_KINDS
        or confidence < _CONFIDENCE_THRESHOLD
        or not value
        or not evidence
        or len(value) > _MAX_VALUE_CHARS
        or len(evidence) > _MAX_EVIDENCE_CHARS
        or not memory_evidence_allowed(evidence)
    ):
        return None

    normalized_source = _normalize(source_message)
    normalized_evidence = _normalize(evidence)
    normalized_value = _normalize(value)
    if (
        not normalized_evidence
        or normalized_evidence not in normalized_source
        or not normalized_value
        or normalized_value not in normalized_evidence
    ):
        return None

    if kind == "name":
        if not memory_name_allowed(value):
            return None
        memory_type, key, content, importance = (
            "user_fact",
            "user_name",
            f"用户说自己叫{value}",
            0.9,
        )
    elif kind in {"like", "dislike"}:
        memory_type, key, content, importance = (
            "user_fact",
            f"preference_{value[:20]}",
            f"用户说{'喜欢' if kind == 'like' else '不喜欢'}{value}",
            0.6 if kind == "like" else 0.5,
        )
    elif kind in {"major", "study_stage", "location", "workplace"}:
        field = {
            "major": ("user_major", "用户说自己的专业是", 0.8),
            "study_stage": ("user_study_stage", "用户说自己是", 0.7),
            "location": ("user_location", "用户说自己来自或居住在", 0.6),
            "workplace": ("user_workplace", "用户说自己在", 0.7),
        }[kind]
        suffix = "工作" if kind == "workplace" else ""
        memory_type, key, content, importance = (
            "user_fact",
            field[0],
            f"{field[1]}{value}{suffix}",
            field[2],
        )
    elif kind == "goal":
        memory_type, key, content, importance = (
            "shared_event",
            f"goal_{value[:24]}",
            f"用户正在进行或准备：{value}",
            0.7,
        )
    elif kind == "promise":
        memory_type, key, content, importance = (
            "promise",
            f"promise_{value[:20]}",
            f"用户提到约定：{evidence}",
            0.8,
        )
    elif kind == "shared_event":
        memory_type, key, content, importance = (
            "shared_event",
            f"event_{value[:24]}",
            f"用户提到共同经历：{evidence}",
            0.7,
        )
    else:
        memory_type, key, content, importance = (
            "user_fact",
            f"fact_{value[:24]}",
            f"用户明确提到：{evidence}",
            0.6,
        )

    return ExtractedMemory(
        memory_type=memory_type,
        memory_key=_truncate(key, 60),
        content=_truncate(content, MAX_MEMORY_CONTENT_CHARS),
        importance=importance,
    )


def parse_llm_memories(text: str, *, source_message: str) -> list[ExtractedMemory]:
    """解析并本地复核 LLM 记忆候选。"""
    if not memory_write_allowed(source_message):
        return []
    raw_memories = _extract_json(text).get("memories", [])
    if not isinstance(raw_memories, list):
        raise ValueError("记忆 LLM 的 memories 必须是数组")
    by_key: dict[str, ExtractedMemory] = {}
    for raw in raw_memories[: MAX_EXTRACTED_MEMORIES * 2]:
        item = _candidate_to_memory(raw, source_message=source_message)
        if item is not None:
            by_key[item.memory_key] = item
    return sorted(by_key.values(), key=lambda item: item.importance, reverse=True)[:MAX_EXTRACTED_MEMORIES]


def _build_messages(
    message: str, rule_hints: tuple[ExtractedMemory, ...], max_input_chars: int
) -> list[dict[str, str]]:
    payload = {
        "source_message": message[:max_input_chars],
        "rule_hints": [
            {
                "memory_type": item.memory_type,
                "memory_key": item.memory_key,
                "content": item.content,
            }
            for item in rule_hints
        ],
    }
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


class MemoryEnrichmentScheduler:
    """有界、单 worker、可优雅关闭的后台记忆判断队列。"""

    def __init__(
        self,
        *,
        config: MemoryLlmConfig,
        completion: MemoryCompletion,
    ) -> None:
        self.enabled = config.enabled
        self._completion = completion
        self._max_input_chars = config.max_input_chars
        self._queue: asyncio.Queue[_MemoryJob] = asyncio.Queue(config.queue_size)
        self._worker: asyncio.Task[None] | None = None
        self._closed = False

    def schedule(
        self,
        *,
        repository: CharacterMemoryRepository,
        character_id: str,
        user_scope: UserScope,
        message: str,
        rule_hints: list[ExtractedMemory],
        source_message_id: str | None,
    ) -> bool:
        if self._closed or not self.enabled or not memory_write_allowed(message):
            return False
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run(), name="character-memory-llm-worker")
        try:
            self._queue.put_nowait(
                _MemoryJob(
                    repository=repository,
                    character_id=character_id,
                    user_scope=user_scope,
                    message=message,
                    rule_hints=tuple(rule_hints),
                    source_message_id=source_message_id,
                )
            )
            return True
        except asyncio.QueueFull:
            logger.warning("后台记忆判断队列已满，本轮不写入长期记忆")
            return False

    async def _run(self) -> None:
        while True:
            job = await self._queue.get()
            try:
                response = await self._completion.complete(
                    _build_messages(job.message, job.rule_hints, self._max_input_chars)
                )
                memories = parse_llm_memories(response, source_message=job.message)
                for item in memories:
                    await job.repository.add_or_update_memory(
                        job.character_id,
                        job.user_scope,
                        MemoryItem(
                            memory_id="",
                            memory_type=item.memory_type,  # type: ignore[arg-type]
                            content=item.content,
                            importance=item.importance,
                        ),
                        memory_key=item.memory_key,
                        source_message_id=job.source_message_id,
                    )
                logger.info(
                    "后台记忆判断完成 character=%s accepted=%d",
                    job.character_id,
                    len(memories),
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("后台记忆判断失败，本轮跳过写入", exc_info=True)
            finally:
                self._queue.task_done()

    async def shutdown(self, timeout: float = 10.0) -> None:
        self._closed = True
        if self._worker is not None and not self._worker.done():
            try:
                await asyncio.wait_for(self._queue.join(), timeout=max(timeout, 0.0))
            except TimeoutError:
                pass
            self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)
        self._worker = None
        await self._completion.close()


_default_scheduler: MemoryEnrichmentScheduler | None = None


def get_memory_enrichment_scheduler() -> MemoryEnrichmentScheduler:
    global _default_scheduler
    if _default_scheduler is None:
        config = MemoryLlmConfig.from_env()
        _default_scheduler = MemoryEnrichmentScheduler(
            config=config,
            completion=OpenAICompatibleMemoryCompletion(config),
        )
    return _default_scheduler


async def shutdown_memory_enrichment() -> None:
    global _default_scheduler
    scheduler = _default_scheduler
    _default_scheduler = None
    if scheduler is not None:
        timeout = float(os.getenv("MEMORY_LLM_SHUTDOWN_TIMEOUT", "10"))
        await scheduler.shutdown(timeout=timeout)
