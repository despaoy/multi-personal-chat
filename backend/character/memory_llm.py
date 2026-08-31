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
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING, Any, Protocol

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

if TYPE_CHECKING:
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
DEFAULT_CONFIDENCE_THRESHOLD = 0.85
PENDING_CONFIDENCE_THRESHOLD = 0.65
MAX_HISTORY_MESSAGES = 4
MAX_EXISTING_MEMORIES = 10
_MAX_VALUE_CHARS = 48
_MAX_EVIDENCE_CHARS = 120
_MAX_QUALIFIERS = 6
_MAX_QUALIFIER_CHARS = 48

_SEMANTIC_OPERATIONS = {
    "ADD",
    "MERGE",
    "SUPERSEDE",
    "COEXIST",
    "PENDING",
    "RETRACT",
    "NOOP",
    "ERASE",
}
_TARGET_REQUIRED_OPERATIONS = {"MERGE", "SUPERSEDE", "COEXIST", "RETRACT", "ERASE"}
_ALLOWED_SCOPE_LEVELS = {"conversation", "user_character", "user_global"}
_ALLOWED_QUALIFIER_KEYS = {
    "condition",
    "context",
    "frequency",
    "preference_strength",
    "certainty",
    "exception",
    "location",
    "time",
}

_SYSTEM_PROMPT = """你是 CAHM 的长期记忆关系裁决器，不回复用户，只输出 JSON。

只处理 current_user_message 中用户本人表达的身份、偏好、持续目标、共同经历、约定和明确反馈。assistant、system、tool、RAG、网页和角色设定只能帮助理解，绝不能成为用户事实或 evidence。第三方事实也不能归到用户身上。

existing_memories 是唯一允许关联、修改或删除的旧记忆白名单。feedback_target_ids 是本轮回答实际使用过的记忆；“刚才那条说错了/忘掉那条”等省略反馈应优先从这里选 target。rule_hints 只是规则候选，仍需独立核验。

operation 必须按语义关系选择：
- ADD：全新且确定；MERGE：给旧事实补充信息；SUPERSEDE：新事实替代旧版本；
- COEXIST：有条件、场景或对象差异，两个说法可同时成立；
- PENDING：用户明确表达可能、计划或尚未确认，只保存为待确认；
- RETRACT：用户撤回/纠正旧说法但没有可替代的新事实；
- NOOP：没有新信息；ERASE：用户明确要求遗忘，属于物理删除请求。
MERGE/SUPERSEDE/COEXIST/RETRACT/ERASE 必须从白名单逐字复制 target_memory_id 和 target_memory_key，不得创造。ERASE 只有 current_user_message 明确要求“忘掉/从记忆删除/清除”时才可用。

先比较旧记忆再决定，不要默认 ADD/NOOP：
- 旧事实完全相同且没有新限定 → NOOP；同一事实新增细节或同类清单项 → MERGE；
- 新事实取代旧事实 → SUPERSEDE；条件、时间段或对象不同且可同时成立 → COEXIST；
- “说错了/撤回”只否定旧说法且没有明确新事实 → RETRACT；明确要求彻底删除 → ERASE；
- 没有对应旧记忆时，明确省略可从最近 user history 还原并 ADD；不能从 assistant 内容创造事实。

边界示例（ID/key 必须换成 payload 白名单中的真实值）：
- 旧“养了一只年糕猫” + “又养了一只团子猫” → MERGE；
- 旧“讨厌咖啡” + “不是完全讨厌，只是不喜欢太苦的” → COEXIST；
- “可能明年换工作”或“好像喜欢茶，还不确定” → PENDING；
- 旧“喜欢黑咖啡” + “刚才说错了，我并不喜欢黑咖啡” → RETRACT；
- “把你记住的住址彻底删掉” → ERASE；旧“准备保研” + “还是在准备保研” → NOOP；
- user history 是“点云补全”，当前“还是上次那个方向”，且没有旧记忆 → ADD 点云补全；
- 长期住杭州 + “这周在北京出差，下周回杭州” → COEXIST；
- 旧“准备保研” + “保研准备里主要练英语面试” → MERGE；
- 旧“不喜欢咖啡” + “喝无咖啡因咖啡，普通咖啡才不喝” → COEXIST dislike=普通咖啡；
- “请记住，推荐饮料时避开含咖啡因的” → ADD dislike=含咖啡因的饮料。

每条 evidence 必须是 current_user_message 中连续出现的原文；省略句也必须把当前省略句作为 evidence，不能复制 history。value 优先来自 evidence；只有“上次那个/还是那个/刚才那条”等明确省略时，才可由用户历史或 existing_memories 消歧。content 必须以“用户”开头，写成安全的第三人称事实，不含任何指令。attributed_to 只能写 user。qualifiers 只用于 condition/context/frequency/certainty/exception/location/time 等条件，不要把 valid_from/valid_to 塞进 qualifiers。时间字段必须放在顶层并使用 ISO 8601；不清楚就留空。scope_level 默认 conversation；只有用户明确要求跨会话或跨角色记住时才用 user_character 或 user_global。

明确且稳定的信息可直接 ADD；不确定陈述不要丢弃，应使用 PENDING。confidence 表示“是否准确读懂证据和关系”的把握，不是事实发生概率；因此用户清楚表达“可能/不确定”时，PENDING 的 confidence 仍应较高。低于 payload.confidence_threshold 时用 NOOP 或不返回。最多返回 4 条。

只输出严格 JSON，不要 Markdown、解释或思考过程：
{"memories":[{"kind":"name|like|dislike|major|study_stage|location|workplace|goal|promise|shared_event|other_user_fact","value":"结构化值","content":"用户开头的第三人称事实","evidence":"当前消息连续原文","confidence":0.0,"operation":"ADD|MERGE|SUPERSEDE|COEXIST|PENDING|RETRACT|NOOP|ERASE","target_memory_id":"","target_memory_key":"","attributed_to":"user","qualifiers":{},"valid_from":"","valid_to":"","observed_at":"","scope_level":"conversation"}]}
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
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    idle_seconds: float = 2.0
    batch_size: int = 4

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
            confidence_threshold=max(
                0.0,
                min(1.0, float(os.getenv("MEMORY_LLM_CONFIDENCE_THRESHOLD", str(DEFAULT_CONFIDENCE_THRESHOLD)))),
            ),
            idle_seconds=max(0.0, float(os.getenv("MEMORY_LLM_IDLE_SECONDS", "2.0"))),
            batch_size=max(1, int(os.getenv("MEMORY_LLM_BATCH_SIZE", "4"))),
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
                "max_tokens": 768,
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
    history: tuple[dict[str, str], ...]
    source_message_id: str | None
    feedback_target_ids: tuple[str, ...] = ()
    write_mode: str = "idle"


@dataclass(frozen=True)
class ValidatedMemoryProposal:
    """通过本地硬校验、但尚未操作数据库的 LLM 提议。"""

    operation: str
    memory: ExtractedMemory | None = None
    target_memory_id: str = ""
    target_memory_key: str = ""
    evidence: str = ""
    confidence: float = 0.0
    attributed_to: str = "user"
    qualifiers: tuple[tuple[str, str], ...] = ()
    valid_from: str = ""
    valid_to: str = ""
    observed_at: str = ""
    scope_level: str = "conversation"


@dataclass(frozen=True)
class MemoryEnrichmentStatus:
    """后台写入生命周期快照，便于测试、健康检查和排障。"""

    enabled: bool
    closed: bool
    buffered: int
    queued: int
    processing: int
    saved: int
    erased: int
    no_change: int
    skipped: int
    failed: int
    last_outcome: str = "idle"
    last_error: str = ""
    recent_results: tuple[dict[str, Any], ...] = field(default_factory=tuple)


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


_THIRD_PARTY_FACT_PATTERN = re.compile(
    r"(?:我(?:的)?(?:朋友|同学|室友|同事|老师|家人|父母|爸爸|妈妈)|"
    r"(?:他|她|他们|她们|朋友|同学|室友|同事))[^，。！？,!?]{0,12}(?:喜欢|讨厌|叫|是|在|准备|工作)"
)
_NAMED_THIRD_PARTY_PATTERN = re.compile(
    r"(?:^|[，。；;])(?:小[\u4e00-\u9fff]|老[\u4e00-\u9fff]|[A-Za-z]{2,16})"
    r"[^，。！？,!?]{0,12}(?:喜欢|讨厌|叫|是|在|准备|工作)"
)
_ELLIPSIS_REFERENCE_PATTERN = re.compile(r"(?:还是|上次|之前|原来|刚才|那条|这条|那个|这件事|照旧|继续)")
_EXPLICIT_REMEMBER_PATTERN = re.compile(r"(?:请|要|务必|一定|帮我)?(?:记住|记一下|记下来|记录下来|存到记忆|保存到记忆)")
_CORRECTION_PATTERN = re.compile(
    r"(?:我(?:刚才|之前|上次)?说错了|刚才那条不对|更正|纠正|改成|撤回|不是[^，。！？]{0,24}(?:而是|只是)|不再是)"
)
_DEICTIC_CORRECTION_PATTERN = re.compile(r"(?:刚才|那条|这条|上条|上一条|之前).{0,12}(?:错|不对|撤回|忘掉|删除)")
_ERASE_REQUEST_PATTERN = re.compile(
    r"(?:忘掉|彻底忘记|从(?:长期)?记忆中(?:删除|清除)|删除(?:掉)?(?:这|那|上)?条记忆|"
    r"清除(?:掉)?(?:这|那|上)?条记忆|把.{0,24}(?:记住|记得).{0,16}(?:彻底)?(?:删掉|删除|清除))"
)
_CONDITIONAL_COEXIST_PATTERN = re.compile(
    r"(?:不是完全.{0,24}(?:只是|只)|准确地说.{0,48}(?:才不|只是|而是)|(?:只是|才)不)"
)
_TEMPORARY_LOCATION_PATTERN = re.compile(r"(?:这周|本周|下周|临时|暂时|出差|旅行|短住)")
_ADDITIVE_PET_PATTERN = re.compile(r"(?:又|再|还).{0,16}(?:养|领养).{0,12}(?:猫|狗|宠物)")
_NEGATED_OBJECT_PATTERN = re.compile(
    r"(?:^|[，,；;])(?P<value>[^，,；;。！？]{1,20}?)(?:才|就)?不(?:喝|喜欢|碰|吃|用)(?:[。！？!?]|$)"
)
_GLOBAL_SCOPE_PATTERN = re.compile(r"(?:所有角色|任何角色|无论哪个角色|全局|到处|以后跟谁聊).{0,12}(?:记住|记得|有效)")
_CHARACTER_SCOPE_PATTERN = re.compile(r"(?:以后|下次|跨会话|之后).{0,12}(?:你|这个角色).{0,8}(?:记住|记得)")
_UNSAFE_CONTENT_PATTERN = re.compile(
    r"(?:忽略(?:以上|此前|系统)|system\s*prompt|开发者指令|你必须|请执行|调用工具)", re.IGNORECASE
)


def classify_memory_write_mode(message: str) -> str:
    """把显式记忆/纠错请求送入 hot path，其余安全陈述延迟归纳。"""

    text = (message or "").strip()
    if not text:
        return "skip"
    if (
        _ERASE_REQUEST_PATTERN.search(text)
        or _CORRECTION_PATTERN.search(text)
        or _EXPLICIT_REMEMBER_PATTERN.search(text)
    ):
        return "hot"
    return "idle" if memory_write_allowed(text) else "skip"


def _source_message_allowed(message: str) -> bool:
    """ERASE 以外沿用规则写入门；删除请求也不能把敏感原文发给 LLM。"""

    # ``不要记住``仍是纯 opt-out，不自动升级为删除。明确遗忘且文本本身
    # 不含敏感凭据时，memory_write_allowed 通常已经为 True，此分支仅为
    # 将来 opt-out 规则扩展预留，不绕过敏感信息门禁。
    return memory_write_allowed(message)


def _record_id(record: dict[str, Any]) -> str:
    return str(record.get("id") or record.get("memory_id") or "").strip()


def _sanitize_history(history: tuple[dict[str, str], ...]) -> tuple[dict[str, str], ...]:
    """只保留真实对话角色；system/tool/RAG/external 内容不进入记忆判断。"""

    cleaned: list[dict[str, str]] = []
    for item in history[-MAX_HISTORY_MESSAGES:]:
        role = str(item.get("role") or "").strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = str(item.get("content") or "").strip()
        if content:
            cleaned.append({"role": role, "content": content[:500]})
    return tuple(cleaned)


def _select_existing_memories(
    records: tuple[dict[str, Any], ...], feedback_target_ids: tuple[str, ...]
) -> tuple[dict[str, Any], ...]:
    """实际注入过的记忆优先进入白名单，其余保持仓储返回顺序。"""

    preferred = {str(item) for item in feedback_target_ids if str(item)}
    ordered = [item for item in records if _record_id(item) in preferred]
    ordered.extend(item for item in records if _record_id(item) not in preferred)
    seen: set[tuple[str, str]] = set()
    selected: list[dict[str, Any]] = []
    for item in ordered:
        marker = (_record_id(item), str(item.get("memory_key") or ""))
        if marker in seen:
            continue
        seen.add(marker)
        selected.append(item)
        if len(selected) >= MAX_EXISTING_MEMORIES:
            break
    return tuple(selected)


def _normalize_attributed_to(value: Any) -> str:
    normalized = str(value or "user").strip().lower()
    return "user" if normalized in {"user", "self", "用户", "本人"} else ""


def _sanitize_qualifiers(raw: Any, *, evidence: str) -> tuple[tuple[str, str], ...] | None:
    if raw in (None, "", {}, []):
        return ()
    items: list[tuple[str, Any]]
    if isinstance(raw, dict):
        items = list(raw.items())
    elif isinstance(raw, list):
        items = [("context", value) for value in raw]
    else:
        return None
    result: list[tuple[str, str]] = []
    normalized_evidence = _normalize(evidence)
    for key, value in items[:_MAX_QUALIFIERS]:
        normalized_key = str(key or "").strip().lower()
        if normalized_key not in _ALLOWED_QUALIFIER_KEYS:
            return None
        if isinstance(value, bool):
            normalized_value = "true" if value else "false"
        elif isinstance(value, (str, int, float)):
            normalized_value = re.sub(r"\s+", " ", str(value)).strip()
        else:
            return None
        if not normalized_value or len(normalized_value) > _MAX_QUALIFIER_CHARS:
            return None
        # 自然语言限定条件必须能在当前证据中找到；结构化 certainty 布尔值
        # 由 operation=PENDING 已表达，不要求逐字出现。
        if normalized_value not in {"true", "false"} and _normalize(normalized_value) not in normalized_evidence:
            return None
        result.append((normalized_key, normalized_value))
    return tuple(result)


def _normalize_iso_time(raw: Any) -> str | None:
    text = str(raw or "").strip()
    if not text:
        return ""
    try:
        value: datetime
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
            parsed_date = date.fromisoformat(text)
            value = datetime.combine(parsed_date, datetime.min.time(), tzinfo=timezone.utc)
        else:
            value = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            else:
                value = value.astimezone(timezone.utc)
        return value.isoformat()
    except (TypeError, ValueError, OverflowError):
        return None


def _resolve_target_record(
    *,
    target_memory_id: str,
    target_memory_key: str,
    existing_memories: tuple[dict[str, Any], ...],
) -> dict[str, Any] | None:
    by_id = [item for item in existing_memories if target_memory_id and _record_id(item) == target_memory_id]
    by_key = [
        item
        for item in existing_memories
        if target_memory_key and str(item.get("memory_key") or "") == target_memory_key
    ]
    if target_memory_id and not by_id:
        return None
    if target_memory_key and not by_key:
        return None
    if target_memory_id and target_memory_key:
        return next(
            (item for item in by_id if str(item.get("memory_key") or "") == target_memory_key),
            None,
        )
    candidates = by_id or by_key
    return next(
        (item for item in candidates if str(item.get("status") or "active") in {"active", "pending"}),
        candidates[0] if candidates else None,
    )


def _infer_add_relation_target(
    kind: str,
    evidence: str,
    existing_memories: tuple[dict[str, Any], ...],
) -> tuple[str, dict[str, Any]] | None:
    """只对两个高精度 ADD 误判场景关联同槽旧记忆。"""

    active = [
        item
        for item in existing_memories
        if str(item.get("status") or "active") in {"active", "pending"}
    ]
    if kind == "location" and _TEMPORARY_LOCATION_PATTERN.search(evidence):
        target = next(
            (item for item in active if str(item.get("memory_key") or "") == "user_location"),
            None,
        )
        if target is not None:
            return "COEXIST", target
    if kind == "other_user_fact" and _ADDITIVE_PET_PATTERN.search(evidence):
        target = next(
            (
                item
                for item in active
                if str(item.get("memory_key") or "").startswith("fact_")
                and re.search(r"猫|狗|宠物", str(item.get("content") or ""))
            ),
            None,
        )
        if target is not None:
            return "MERGE", target
    return None


def _negative_object_from_evidence(evidence: str) -> str:
    match = _NEGATED_OBJECT_PATTERN.search(evidence)
    if match is None:
        return ""
    return re.sub(r"^(?:准确地说|其实|但是|不过)", "", match.group("value")).strip()


def _grounded_value(
    value: str,
    *,
    evidence: str,
    history: tuple[dict[str, str], ...],
    existing_memories: tuple[dict[str, Any], ...],
) -> bool:
    """值必须来自当前证据；明确省略时才允许由给定上下文消歧。"""
    normalized_value = _normalize(value)
    normalized_evidence = _normalize(evidence)
    if normalized_value and normalized_value in normalized_evidence:
        return True
    # 结构化值可能把证据中的修饰语重新排序，例如
    # “推荐饮料时避开含咖啡因的”→“含咖啡因的饮料”。要求至少 75%
    # 的 value bigram 逐字存在于当前证据，允许重排但不允许补造实体。
    if len(normalized_value) >= 4:
        value_bigrams = {
            normalized_value[index : index + 2] for index in range(len(normalized_value) - 1)
        }
        evidence_bigrams = {
            normalized_evidence[index : index + 2] for index in range(len(normalized_evidence) - 1)
        }
        if value_bigrams and len(value_bigrams & evidence_bigrams) / len(value_bigrams) >= 0.75:
            return True
    if not _ELLIPSIS_REFERENCE_PATTERN.search(evidence):
        return False
    grounding_texts = [str(item.get("content") or "") for item in existing_memories]
    # assistant/RAG 只能帮助理解对话结构，不能提供待晋升的事实值。
    grounding_texts.extend(
        str(item.get("content") or "") for item in history if str(item.get("role") or "").strip().lower() == "user"
    )
    return any(normalized_value and normalized_value in _normalize(text) for text in grounding_texts)


def _target_key_matches_kind(kind: str, target_memory_key: str) -> bool:
    expected_prefix = {
        "name": "user_name",
        "like": "preference_",
        "dislike": "preference_",
        "major": "user_major",
        "study_stage": "user_study_stage",
        "location": "user_location",
        "workplace": "user_workplace",
        "goal": "goal_",
        "promise": "promise_",
        "shared_event": "event_",
        "other_user_fact": "fact_",
    }[kind]
    return target_memory_key.startswith(expected_prefix)


def _normalize_kind_and_value(kind: str, value: str, evidence: str) -> tuple[str, str]:
    """把常见模型表述归一成数据库模板需要的结构化值。"""
    normalized_kind = kind
    normalized_value = value.strip()
    if kind == "like" and re.search(r"(?:不喜欢|不碰|不喝|避开)", f"{value} {evidence}"):
        normalized_kind = "dislike"
    if normalized_kind == "goal":
        normalized_value = re.sub(r"^(?:正在|最近|目前|主要|在)*(?:准备|备考)", "", normalized_value).strip()
    if normalized_kind == "dislike":
        normalized_value = re.sub(r"^(?:平时)?(?:不喜欢|不碰|不喝|避开)", "", normalized_value).strip()
    return normalized_kind, normalized_value


def _infer_kind_from_target(record: dict[str, Any]) -> str:
    key = str(record.get("memory_key") or "")
    content = str(record.get("content") or "")
    if key == "user_name":
        return "name"
    if key.startswith("preference_"):
        return "dislike" if "不喜欢" in content else "like"
    if key == "user_major":
        return "major"
    if key == "user_study_stage":
        return "study_stage"
    if key == "user_location":
        return "location"
    if key == "user_workplace":
        return "workplace"
    if key.startswith("goal_"):
        return "goal"
    if key.startswith("promise_"):
        return "promise"
    if key.startswith("event_"):
        return "shared_event"
    return "other_user_fact"


def _scope_level_for_message(raw: Any, source_message: str) -> str:
    requested = str(raw or "conversation").strip().lower()
    if requested not in _ALLOWED_SCOPE_LEVELS or requested == "conversation":
        return "conversation"
    # 作用域晋升属于权限，而不是语义猜测：没有用户原文授权就降级。
    if requested == "user_global" and _GLOBAL_SCOPE_PATTERN.search(source_message):
        return requested
    if (
        requested == "user_character"
        and _EXPLICIT_REMEMBER_PATTERN.search(source_message)
        and _CHARACTER_SCOPE_PATTERN.search(source_message)
    ):
        return requested
    return "conversation"


def _canonical_memory_fields(kind: str, value: str, evidence: str) -> tuple[str, str, str, float] | None:
    if kind == "name":
        if not memory_name_allowed(value):
            return None
        return "user_fact", "user_name", f"用户说自己叫{value}", 0.9
    if kind in {"like", "dislike"}:
        return (
            "user_fact",
            f"preference_{value[:20]}",
            f"用户说{'喜欢' if kind == 'like' else '不喜欢'}{value}",
            0.6 if kind == "like" else 0.5,
        )
    if kind in {"major", "study_stage", "location", "workplace"}:
        field = {
            "major": ("user_major", "用户说自己的专业是", 0.8),
            "study_stage": ("user_study_stage", "用户说自己是", 0.7),
            "location": ("user_location", "用户说自己来自或居住在", 0.6),
            "workplace": ("user_workplace", "用户说自己在", 0.7),
        }[kind]
        suffix = "工作" if kind == "workplace" else ""
        return "user_fact", field[0], f"{field[1]}{value}{suffix}", field[2]
    if kind == "goal":
        return "shared_event", f"goal_{value[:24]}", f"用户正在进行或准备：{value}", 0.7
    if kind == "promise":
        return "promise", f"promise_{value[:20]}", f"用户提到约定：{evidence}", 0.8
    if kind == "shared_event":
        return "shared_event", f"event_{value[:24]}", f"用户提到共同经历：{evidence}", 0.7
    if kind == "other_user_fact":
        return "user_fact", f"fact_{value[:24]}", f"用户明确提到：{evidence}", 0.6
    return None


def _candidate_to_proposal(
    raw: Any,
    *,
    source_message: str,
    history: tuple[dict[str, str], ...],
    existing_memories: tuple[dict[str, Any], ...],
    confidence_threshold: float,
    feedback_target_ids: tuple[str, ...] = (),
) -> ValidatedMemoryProposal | None:
    if not isinstance(raw, dict):
        return None
    kind = str(raw.get("kind") or "").strip()
    value = re.sub(r"\s+", " ", str(raw.get("value") or "")).strip()
    evidence = re.sub(r"\s+", " ", str(raw.get("evidence") or "")).strip()
    proposed_content = re.sub(r"\s+", " ", str(raw.get("content") or "")).strip()
    if proposed_content in {"用户开头的第三人称事实", "第三人称安全描述"}:
        proposed_content = ""
    operation = str(raw.get("operation") or "ADD").strip().upper()
    if operation == "IGNORE":
        return None
    # UPDATE 是旧公开解析契约；保留返回值，但持久化时按 SUPERSEDE 执行。
    semantic_operation = "SUPERSEDE" if operation == "UPDATE" else operation
    target_memory_id = str(raw.get("target_memory_id") or "").strip()
    target_memory_key = str(raw.get("target_memory_key") or "").strip()

    normalized_source = _normalize(source_message)
    normalized_evidence = _normalize(evidence)
    if (
        normalized_evidence
        and normalized_evidence not in normalized_source
        and _ELLIPSIS_REFERENCE_PATTERN.search(source_message)
        and any(
            normalized_evidence in _normalize(str(item.get("content") or ""))
            for item in history
            if str(item.get("role") or "").strip().lower() == "user"
        )
    ):
        # 模型有时会把用于消歧的 user history 复制为 evidence。保留它
        # 提议的 value，但 evidence 必须改回当前明确省略句。
        evidence = re.sub(r"\s+", " ", source_message).strip()
        normalized_evidence = normalized_source
    try:
        confidence = float(raw.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return None
    if semantic_operation not in _SEMANTIC_OPERATIONS or not 0.0 <= confidence <= 1.0:
        return None
    required_confidence = (
        min(confidence_threshold, PENDING_CONFIDENCE_THRESHOLD)
        if semantic_operation == "PENDING"
        else confidence_threshold
    )
    if semantic_operation != "NOOP" and confidence < required_confidence:
        return None
    if not evidence or len(evidence) > _MAX_EVIDENCE_CHARS:
        return None
    if _THIRD_PARTY_FACT_PATTERN.search(evidence) or _NAMED_THIRD_PARTY_PATTERN.search(evidence):
        return None
    if not _normalize_attributed_to(raw.get("attributed_to")):
        return None

    if semantic_operation in {"PENDING", "RETRACT", "ERASE"}:
        evidence_allowed = memory_write_allowed(evidence) and "?" not in evidence and "？" not in evidence
    else:
        evidence_allowed = memory_evidence_allowed(evidence)
    if not evidence_allowed:
        return None

    if not normalized_evidence or normalized_evidence not in normalized_source:
        return None

    if (
        semantic_operation in {"MERGE", "SUPERSEDE"}
        and target_memory_id
        and _CONDITIONAL_COEXIST_PATTERN.search(source_message)
    ):
        semantic_operation = "COEXIST"
        operation = "COEXIST"

    if semantic_operation == "ADD" and not (target_memory_id or target_memory_key):
        inferred = _infer_add_relation_target(kind, evidence, existing_memories)
        if inferred is not None:
            semantic_operation, inferred_target = inferred
            operation = semantic_operation
            target_memory_id = _record_id(inferred_target)
            target_memory_key = str(inferred_target.get("memory_key") or "")

    target_record = _resolve_target_record(
        target_memory_id=target_memory_id,
        target_memory_key=target_memory_key,
        existing_memories=existing_memories,
    )
    if semantic_operation in _TARGET_REQUIRED_OPERATIONS and target_record is None:
        return None
    if target_record is not None:
        target_memory_id = _record_id(target_record)
        target_memory_key = str(target_record.get("memory_key") or "")
        if semantic_operation != "ERASE" and str(target_record.get("status") or "active") not in {
            "active",
            "pending",
        }:
            return None
    elif semantic_operation == "ADD":
        target_memory_id = ""
        target_memory_key = ""

    if (
        semantic_operation == "MERGE"
        and target_record is not None
        and value
        and re.sub(r"[，。！？,!?：:；;]", "", _normalize(value))
        in re.sub(
            r"[，。！？,!?：:；;]",
            "",
            _normalize(str(target_record.get("content") or "")),
        )
        and raw.get("qualifiers") in (None, "", {}, [])
    ):
        return ValidatedMemoryProposal(
            operation="NOOP",
            target_memory_id=target_memory_id,
            target_memory_key=target_memory_key,
            evidence=evidence,
            confidence=confidence,
            observed_at=_normalize_iso_time(raw.get("observed_at")) or "",
        )

    if semantic_operation == "ERASE" and not _ERASE_REQUEST_PATTERN.search(source_message):
        return None
    if semantic_operation == "RETRACT" and not _CORRECTION_PATTERN.search(source_message):
        return None
    if (
        semantic_operation in _TARGET_REQUIRED_OPERATIONS
        and feedback_target_ids
        and _DEICTIC_CORRECTION_PATTERN.search(source_message)
        and target_memory_id not in {str(item) for item in feedback_target_ids}
    ):
        return None

    if semantic_operation == "NOOP":
        return ValidatedMemoryProposal(
            operation="NOOP",
            evidence=evidence,
            confidence=confidence,
            observed_at=_normalize_iso_time(raw.get("observed_at")) or "",
        )

    if kind not in _ALLOWED_KINDS:
        if target_record is None:
            return None
        kind = _infer_kind_from_target(target_record)
    if not value and target_record is not None and semantic_operation in {"RETRACT", "ERASE"}:
        value = target_memory_key or "target"
    if not value or (len(value) > _MAX_VALUE_CHARS and semantic_operation not in {"RETRACT", "ERASE"}):
        return None
    if semantic_operation not in {"RETRACT", "ERASE"} and not _grounded_value(
        value,
        evidence=evidence,
        history=history,
        existing_memories=existing_memories,
    ):
        grounded_negative = (
            _negative_object_from_evidence(evidence) if kind in {"like", "dislike"} else ""
        )
        if not grounded_negative:
            return None
        kind = "dislike"
        value = grounded_negative
        # 模型的抽象 content 可能不再对应证据中更精确的对象；回退到
        # 本地 canonical 模板，避免保留未经证据支持的泛化。
        proposed_content = ""

    kind, value = _normalize_kind_and_value(kind, value, evidence)
    if not value:
        return None

    if (
        target_record is not None
        and semantic_operation not in {"RETRACT", "ERASE"}
        and not _target_key_matches_kind(kind, target_memory_key)
    ):
        return None

    raw_qualifiers = raw.get("qualifiers")
    misplaced_validity: dict[str, Any] = {}
    if isinstance(raw_qualifiers, dict) and raw_qualifiers and set(raw_qualifiers) <= {
        "valid_from",
        "valid_to",
    }:
        misplaced_validity = raw_qualifiers
        raw_qualifiers = {}
    qualifiers = _sanitize_qualifiers(raw_qualifiers, evidence=evidence)
    if qualifiers is None:
        return None
    valid_from = _normalize_iso_time(
        raw.get("valid_from", raw.get("valid_at")) or misplaced_validity.get("valid_from")
    )
    valid_to = _normalize_iso_time(
        raw.get("valid_to", raw.get("invalid_at")) or misplaced_validity.get("valid_to")
    )
    observed_at = _normalize_iso_time(raw.get("observed_at"))
    if valid_from is None or valid_to is None or observed_at is None:
        return None
    if valid_from and valid_to and datetime.fromisoformat(valid_from) > datetime.fromisoformat(valid_to):
        return None

    if target_record is not None and semantic_operation in {"RETRACT", "ERASE"}:
        memory_type = str(target_record.get("memory_type") or "user_fact")
        if memory_type not in {"user_fact", "shared_event", "promise", "conversation_summary"}:
            memory_type = "user_fact"
        key = target_memory_key
        canonical_content = str(target_record.get("content") or f"用户撤回记忆：{target_memory_key}")
        importance = float(target_record.get("importance") or 0.5)
    else:
        canonical = _canonical_memory_fields(kind, value, evidence)
        if canonical is None:
            return None
        memory_type, key, canonical_content, importance = canonical

    # 旧 UPDATE 入口一直承诺由本地模板生成 content；新关系操作才允许
    # 使用已通过主体/证据硬校验的 LLM 自包含表述。
    if proposed_content and operation != "UPDATE":
        if (
            len(proposed_content) > MAX_MEMORY_CONTENT_CHARS
            or not proposed_content.startswith("用户")
            or _UNSAFE_CONTENT_PATTERN.search(proposed_content)
            or (
                _normalize(value) not in _normalize(proposed_content) and semantic_operation not in {"RETRACT", "ERASE"}
            )
        ):
            return None
        content = proposed_content
    else:
        content = canonical_content

    if qualifiers and (not proposed_content or operation == "UPDATE"):
        qualifier_text = "；".join(f"{key_name}={item_value}" for key_name, item_value in qualifiers)
        content = f"{content}（{qualifier_text}）"
    if semantic_operation == "PENDING" and not content.startswith("待确认："):
        content = f"待确认：{content}"
    if semantic_operation == "MERGE" and target_record is not None:
        previous = str(target_record.get("content") or "").strip()
        if previous and _normalize(previous) != _normalize(content):
            content = f"{previous}；补充：{content}"

    memory_key = target_memory_key if target_record is not None else key
    memory = ExtractedMemory(
        memory_type=memory_type,
        memory_key=_truncate(memory_key, 60),
        content=_truncate(content, MAX_MEMORY_CONTENT_CHARS),
        importance=importance,
    )
    return ValidatedMemoryProposal(
        operation=operation,
        memory=memory,
        target_memory_id=target_memory_id,
        target_memory_key=target_memory_key,
        evidence=evidence,
        confidence=confidence,
        attributed_to="user",
        qualifiers=qualifiers,
        valid_from=valid_from,
        valid_to=valid_to,
        observed_at=observed_at,
        scope_level=_scope_level_for_message(raw.get("scope_level"), source_message),
    )


def parse_llm_proposals(
    text: str,
    *,
    source_message: str,
    history: tuple[dict[str, str], ...] = (),
    existing_memories: tuple[dict[str, Any], ...] = (),
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    feedback_target_ids: tuple[str, ...] = (),
    source_type: str = "user",
) -> list[ValidatedMemoryProposal]:
    """解析并硬校验证据、旧 ID、主体、时间、作用域与删除权限。"""
    if source_type.strip().lower() != "user" or not _source_message_allowed(source_message):
        return []
    raw_memories = _extract_json(text).get("memories", [])
    if not isinstance(raw_memories, list):
        raise ValueError("记忆 LLM 的 memories 必须是数组")
    by_key: dict[tuple[str, str, str], ValidatedMemoryProposal] = {}
    for raw in raw_memories[: MAX_EXTRACTED_MEMORIES * 2]:
        proposal = _candidate_to_proposal(
            raw,
            source_message=source_message,
            history=history,
            existing_memories=existing_memories,
            confidence_threshold=confidence_threshold,
            feedback_target_ids=feedback_target_ids,
        )
        if proposal is not None:
            memory_key = proposal.memory.memory_key if proposal.memory is not None else ""
            by_key[(proposal.operation, proposal.target_memory_id, memory_key)] = proposal
    return sorted(
        by_key.values(),
        key=lambda item: item.memory.importance if item.memory is not None else 0.0,
        reverse=True,
    )[:MAX_EXTRACTED_MEMORIES]


def parse_llm_memories(text: str, *, source_message: str) -> list[ExtractedMemory]:
    """向后兼容入口：解析无上下文的 ADD 候选。"""
    return [
        item.memory
        for item in parse_llm_proposals(text, source_message=source_message)
        if item.memory is not None and item.operation not in {"NOOP", "RETRACT", "ERASE"}
    ]


def build_memory_llm_messages(
    message: str,
    rule_hints: tuple[ExtractedMemory, ...],
    history: tuple[dict[str, str], ...],
    existing_memories: tuple[dict[str, Any], ...],
    max_input_chars: int,
    confidence_threshold: float,
    feedback_target_ids: tuple[str, ...] = (),
    write_mode: str = "idle",
) -> list[dict[str, str]]:
    safe_history = _sanitize_history(history)
    selected_memories = _select_existing_memories(existing_memories, feedback_target_ids)
    valid_feedback_ids = {
        item
        for item in (str(value) for value in feedback_target_ids)
        if any(_record_id(record) == item for record in selected_memories)
    }
    payload = {
        "current_user_message": message[:max_input_chars],
        "recent_history": [
            {
                "role": str(item.get("role") or "")[:16],
                "content": str(item.get("content") or "")[:500],
                "eligible_as_memory_evidence": False,
            }
            for item in safe_history
        ],
        "rule_hints": [
            {
                "memory_type": item.memory_type,
                "memory_key": item.memory_key,
                "content": item.content,
            }
            for item in rule_hints
        ],
        "existing_memories": [
            {
                "memory_id": _record_id(item),
                "memory_key": str(item.get("memory_key") or "")[:60],
                "memory_type": str(item.get("memory_type") or "")[:32],
                "content": str(item.get("content") or "")[:MAX_MEMORY_CONTENT_CHARS],
                "status": str(item.get("status") or "active")[:24],
                "valid_from": str(item.get("valid_from") or "")[:40],
                "valid_to": str(item.get("valid_to") or "")[:40],
                "was_injected_in_last_reply": _record_id(item) in valid_feedback_ids,
            }
            for item in selected_memories
        ],
        "feedback_target_ids": sorted(valid_feedback_ids),
        "write_mode": write_mode,
        "confidence_threshold": confidence_threshold,
        "current_time_utc": datetime.now(timezone.utc).isoformat(),
    }
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


class MemoryEnrichmentScheduler:
    """显式反馈走 hot path，隐式事实按 scope 在 idle 后批量归纳。"""

    def __init__(
        self,
        *,
        config: MemoryLlmConfig,
        completion: MemoryCompletion,
    ) -> None:
        self.enabled = config.enabled
        self._completion = completion
        self._max_input_chars = config.max_input_chars
        self._confidence_threshold = config.confidence_threshold
        self._idle_seconds = config.idle_seconds
        self._batch_size = config.batch_size
        self._capacity = config.queue_size
        self._queue: asyncio.Queue[tuple[_MemoryJob, ...]] = asyncio.Queue(config.queue_size)
        self._pending: dict[tuple[Any, ...], list[_MemoryJob]] = {}
        self._idle_tasks: dict[tuple[Any, ...], asyncio.Task[None]] = {}
        self._worker: asyncio.Task[None] | None = None
        self._closed = False
        self._inflight = 0
        self._queued_jobs = 0
        self._processing = 0
        self._saved = 0
        self._erased = 0
        self._no_change = 0
        self._skipped = 0
        self._failed = 0
        self._last_outcome = "idle"
        self._last_error = ""
        self._recent_results: deque[dict[str, Any]] = deque(maxlen=32)

    @property
    def status(self) -> MemoryEnrichmentStatus:
        return MemoryEnrichmentStatus(
            enabled=self.enabled,
            closed=self._closed,
            buffered=sum(len(items) for items in self._pending.values()),
            queued=self._queued_jobs,
            processing=self._processing,
            saved=self._saved,
            erased=self._erased,
            no_change=self._no_change,
            skipped=self._skipped,
            failed=self._failed,
            last_outcome=self._last_outcome,
            last_error=self._last_error,
            recent_results=tuple(self._recent_results),
        )

    def _scope_key(
        self,
        repository: CharacterMemoryRepository,
        character_id: str,
        user_scope: UserScope,
    ) -> tuple[Any, ...]:
        return (id(repository), character_id, *user_scope.memory_scope_key)

    def _ensure_worker(self) -> None:
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run(), name="character-memory-llm-worker")

    def _remember_schedule_skip(self, reason: str) -> None:
        self._skipped += 1
        self._last_outcome = "skipped"
        self._recent_results.append({"status": "skipped", "reason": reason})

    def schedule(
        self,
        *,
        repository: CharacterMemoryRepository,
        character_id: str,
        user_scope: UserScope,
        message: str,
        rule_hints: list[ExtractedMemory],
        history: tuple[dict[str, str], ...] = (),
        source_message_id: str | None = None,
        feedback_target_ids: tuple[str, ...] = (),
        source_type: str = "user",
        immediate: bool | None = None,
    ) -> bool:
        if self._closed:
            self._remember_schedule_skip("closed")
            return False
        if not self.enabled:
            self._remember_schedule_skip("disabled")
            return False
        if source_type.strip().lower() != "user":
            self._remember_schedule_skip("non_user_source")
            return False
        mode = "hot" if immediate is True else classify_memory_write_mode(message)
        if immediate is False and mode != "skip":
            mode = "idle"
        if mode == "skip" or not _source_message_allowed(message):
            self._remember_schedule_skip("write_gate")
            return False
        if self._inflight >= self._capacity:
            logger.warning("后台记忆判断容量已满，本轮不写入长期记忆")
            self._remember_schedule_skip("capacity")
            return False

        job = _MemoryJob(
            repository=repository,
            character_id=character_id,
            user_scope=user_scope,
            message=message,
            rule_hints=tuple(rule_hints),
            history=_sanitize_history(history),
            source_message_id=source_message_id,
            feedback_target_ids=tuple(str(item) for item in feedback_target_ids if str(item)),
            write_mode=mode,
        )
        self._inflight += 1
        scope_key = self._scope_key(repository, character_id, user_scope)
        if mode == "hot":
            # 先提交同一 scope 已缓冲的旧事实，保证后续纠错不会越过它。
            self._flush_scope(scope_key)
            if not self._enqueue((job,)):
                self._inflight -= 1
                self._remember_schedule_skip("queue_full")
                return False
            self._last_outcome = "queued_hot"
            return True

        pending = self._pending.setdefault(scope_key, [])
        pending.append(job)
        previous_timer = self._idle_tasks.pop(scope_key, None)
        if previous_timer is not None:
            previous_timer.cancel()
        if len(pending) >= self._batch_size or self._idle_seconds <= 0:
            self._flush_scope(scope_key)
            self._last_outcome = "queued_batch"
        else:
            self._idle_tasks[scope_key] = asyncio.create_task(
                self._flush_after_idle(scope_key),
                name="character-memory-idle-consolidation",
            )
            self._last_outcome = "buffered_idle"
        return True

    def _enqueue(self, jobs: tuple[_MemoryJob, ...]) -> bool:
        if not jobs:
            return True
        self._ensure_worker()
        try:
            self._queue.put_nowait(jobs)
            self._queued_jobs += len(jobs)
            return True
        except asyncio.QueueFull:
            logger.warning("后台记忆判断队列已满，本批次跳过")
            return False

    def _flush_scope(self, scope_key: tuple[Any, ...]) -> None:
        timer = self._idle_tasks.pop(scope_key, None)
        current = asyncio.current_task()
        if timer is not None and timer is not current:
            timer.cancel()
        jobs = tuple(self._pending.pop(scope_key, ()))
        if jobs and not self._enqueue(jobs):
            self._inflight = max(0, self._inflight - len(jobs))
            self._skipped += len(jobs)
            self._last_outcome = "skipped"
            for job in jobs:
                self._recent_results.append(
                    {"source_message_id": job.source_message_id or "", "status": "skipped", "reason": "queue_full"}
                )

    async def _flush_after_idle(self, scope_key: tuple[Any, ...]) -> None:
        try:
            await asyncio.sleep(self._idle_seconds)
            self._flush_scope(scope_key)
        except asyncio.CancelledError:
            return

    async def flush_memory(self, timeout: float = 10.0) -> bool:
        """立即提交所有 idle 缓冲并等待完成；不关闭客户端。"""

        timers = tuple(self._idle_tasks.values())
        self._idle_tasks.clear()
        for timer in timers:
            timer.cancel()
        if timers:
            await asyncio.gather(*timers, return_exceptions=True)
        for scope_key in tuple(self._pending):
            self._flush_scope(scope_key)
        if self._queued_jobs == 0 and self._processing == 0:
            return True
        try:
            await asyncio.wait_for(self._queue.join(), timeout=max(timeout, 0.0))
            return True
        except TimeoutError:
            return False

    async def _run(self) -> None:
        while True:
            jobs = await self._queue.get()
            try:
                self._queued_jobs = max(0, self._queued_jobs - len(jobs))
                self._processing += len(jobs)
                for job in jobs:
                    await self._process_job(job)
            except asyncio.CancelledError:
                raise
            finally:
                self._processing = max(0, self._processing - len(jobs))
                self._inflight = max(0, self._inflight - len(jobs))
                self._queue.task_done()

    async def _process_job(self, job: _MemoryJob) -> None:
        result = {
            "source_message_id": job.source_message_id or "",
            "write_mode": job.write_mode,
            "status": "no_change",
            "accepted": 0,
            "persisted": 0,
        }
        try:
            # 实际注入 ID 可能不在默认最新 10 条中；先读一个有界大窗口，
            # 再由 _select_existing_memories 压回提示词预算。
            fetch_limit = 100 if job.feedback_target_ids else MAX_EXISTING_MEMORIES
            records = tuple(
                await job.repository.list_memory_records(
                    job.character_id,
                    job.user_scope,
                    limit=fetch_limit,
                )
            )
            existing_memories = _select_existing_memories(records, job.feedback_target_ids)
            response = await self._completion.complete(
                build_memory_llm_messages(
                    job.message,
                    job.rule_hints,
                    job.history,
                    existing_memories,
                    self._max_input_chars,
                    self._confidence_threshold,
                    feedback_target_ids=job.feedback_target_ids,
                    write_mode=job.write_mode,
                )
            )
            proposals = parse_llm_proposals(
                response,
                source_message=job.message,
                history=job.history,
                existing_memories=existing_memories,
                confidence_threshold=self._confidence_threshold,
                feedback_target_ids=job.feedback_target_ids,
                source_type="user",
            )
            result["accepted"] = len(proposals)
            outcomes: list[str] = []
            for proposal in proposals:
                try:
                    outcomes.append(await self._persist_proposal(job, proposal))
                except Exception as exc:
                    self._failed += 1
                    outcomes.append("failed")
                    self._last_error = _truncate(str(exc), 240)
                    logger.warning(
                        "后台记忆单条写入失败 character=%s operation=%s",
                        job.character_id,
                        proposal.operation,
                        exc_info=True,
                    )

            saved = outcomes.count("saved")
            erased = outcomes.count("erased")
            result["persisted"] = saved + erased
            if "failed" in outcomes and saved + erased:
                status = "partial"
            elif "failed" in outcomes:
                status = "failed"
            elif erased:
                status = "erased"
            elif saved:
                status = "saved"
            elif "skipped" in outcomes:
                status = "skipped"
                self._skipped += 1
            else:
                status = "no_change"
                self._no_change += 1
            result["status"] = status
            self._last_outcome = status
            logger.info(
                "后台记忆判断完成 character=%s accepted=%d persisted=%d status=%s",
                job.character_id,
                len(proposals),
                saved + erased,
                status,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._failed += 1
            self._last_outcome = "failed"
            self._last_error = _truncate(str(exc), 240)
            result["status"] = "failed"
            result["error"] = self._last_error
            logger.warning("后台记忆判断失败，本轮跳过写入", exc_info=True)
        finally:
            self._recent_results.append(result)

    async def _persist_proposal(self, job: _MemoryJob, proposal: ValidatedMemoryProposal) -> str:
        semantic_operation = "SUPERSEDE" if proposal.operation == "UPDATE" else proposal.operation
        if semantic_operation == "NOOP":
            return "no_change"

        target_id: int | str | None = None
        if proposal.target_memory_id:
            target_id = (
                int(proposal.target_memory_id) if proposal.target_memory_id.isdigit() else proposal.target_memory_id
            )

        if semantic_operation == "ERASE":
            eraser = getattr(job.repository, "erase_memory", None)
            if callable(eraser):
                deleted = await eraser(
                    job.character_id,
                    job.user_scope,
                    memory_id=target_id,
                    memory_key=proposal.target_memory_key or None,
                    scope_level=proposal.scope_level,
                )
            else:
                legacy_delete = getattr(job.repository, "delete_memory", None)
                if not callable(legacy_delete) or not isinstance(target_id, int):
                    return "skipped"
                deleted = await legacy_delete(target_id, job.character_id, job.user_scope)
            if int(deleted or 0) > 0:
                self._erased += 1
                return "erased"
            return "no_change"

        item = proposal.memory
        if item is None:
            return "no_change"
        observed_at = proposal.observed_at or datetime.now(timezone.utc).isoformat()
        memory = MemoryItem(
            memory_id="",
            memory_type=item.memory_type,  # type: ignore[arg-type]
            content=item.content,
            importance=item.importance,
            evidence=(proposal.evidence,),
            valid_from=proposal.valid_from,
            valid_to=proposal.valid_to,
            confidence=proposal.confidence,
            status="pending" if semantic_operation == "PENDING" else "active",  # type: ignore[arg-type]
            relation_type=semantic_operation,
            source_message_ids=(job.source_message_id,) if job.source_message_id else (),
        )

        append_claim = getattr(job.repository, "append_claim", None)
        if callable(append_claim):
            parent_id = target_id if semantic_operation in {"MERGE", "COEXIST"} else None
            # MERGE 的新 claim 已在 parser 中聚合旧内容，因此它是新的
            # canonical active 版本；旧版本要进入 superseded，不能继续以
            # active 重复参与检索。parent 同时保留可追溯关系。
            supersedes_id = target_id if semantic_operation in {"MERGE", "SUPERSEDE", "RETRACT"} else None
            metadata = {
                "qualifiers": dict(proposal.qualifiers),
                "target_memory_key": proposal.target_memory_key,
                "write_mode": job.write_mode,
                "feedback_target_ids": list(job.feedback_target_ids),
            }
            record = await append_claim(
                job.character_id,
                job.user_scope,
                memory,
                memory_key=item.memory_key,
                relation_type=semantic_operation,
                scope_level=proposal.scope_level,
                status="pending" if semantic_operation == "PENDING" else None,
                parent_memory_id=parent_id,
                supersedes_memory_id=supersedes_id,
                evidence=(proposal.evidence,),
                confidence=proposal.confidence,
                attributed_to=proposal.attributed_to,
                valid_from=proposal.valid_from or None,
                valid_to=proposal.valid_to or None,
                observed_at=observed_at,
                source_message_id=job.source_message_id,
                source_message_ids=(job.source_message_id,) if job.source_message_id else (),
                metadata=metadata,
            )
            if isinstance(record, dict) and record.get("persisted") is False:
                return "no_change"
            self._saved += 1
            return "saved"

        # 旧仓储兼容：确定事实仍可工作；PENDING 不降级成 active，避免把
        # “可能”错误注入。RETRACT 在无版本能力时删除旧 active 记录。
        if semantic_operation == "PENDING":
            return "skipped"
        if semantic_operation == "RETRACT":
            legacy_delete = getattr(job.repository, "delete_memory", None)
            if callable(legacy_delete) and isinstance(target_id, int):
                deleted = await legacy_delete(target_id, job.character_id, job.user_scope)
                if deleted:
                    self._saved += 1
                    return "saved"
                return "no_change"
            return "skipped"
        legacy_write = getattr(job.repository, "add_or_update_memory", None)
        if not callable(legacy_write):
            return "skipped"
        memory_key = item.memory_key
        if semantic_operation == "COEXIST" and proposal.qualifiers:
            suffix = "_".join(value for _key, value in proposal.qualifiers)[:16]
            memory_key = _truncate(f"{memory_key}__{suffix}", 60)
        await legacy_write(
            job.character_id,
            job.user_scope,
            memory,
            memory_key=memory_key,
            source_message_id=job.source_message_id,
        )
        self._saved += 1
        return "saved"

    async def shutdown(self, timeout: float = 10.0) -> None:
        self._closed = True
        await self.flush_memory(timeout=timeout)
        if self._worker is not None and not self._worker.done():
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
