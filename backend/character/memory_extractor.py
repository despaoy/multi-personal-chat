"""规则版记忆提取与关系推进。

只从用户消息中提取"明确说出来"的信息，宁缺毋滥：
- 自我介绍（"我叫X"）→ user_fact / user_name；
- 称呼偏好（"叫我X"）→ 关系 preferred_address；
- 好恶表达（"我喜欢/讨厌X"）→ user_fact / preference_*（喜欢与
  厌恶共用同一 key，最新表态覆盖旧表态，避免"喜欢咖啡"与
  "讨厌咖啡"并存的事实冲突）；
- 承诺约定（"下次一定X"、"答应你X"）→ promise。时间词后必须
  跟意愿动词，"明天天气怎么样"这类询问不会被误判为承诺。

关系阶段只按交互轮数单向推进（stranger → acquaintance → familiar），
永不自动回退；close 属于高亲密阶段，仅凭消息数量不足以判定，
只能由管理员通过管理接口手动设置；回退同样只能由管理员手动修改。

不调用 LLM：全部为正则/关键词规则，可独立测试。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from character.models import RelationshipStage

# 关系自动推进阈值（交互轮数，含边界）。
# close 不设自动阈值：亲密阶段由管理员手动确认。
_STAGE_THRESHOLDS: tuple[tuple[int, RelationshipStage], ...] = (
    (50, "familiar"),
    (10, "acquaintance"),
)

_STAGE_ORDER: tuple[RelationshipStage, ...] = (
    "stranger",
    "acquaintance",
    "familiar",
    "close",
)

MAX_MEMORY_CONTENT_CHARS = 120

# "我叫X" / "我是X" / "我的名字是X"
_NAME_PATTERNS = (
    re.compile(r"我叫(?P<name>[\w\u4e00-\u9fff]{1,12})"),
    re.compile(r"我的名字(?:是|叫)(?P<name>[\w\u4e00-\u9fff]{1,12})"),
    re.compile(r"(?:我是|我叫)(?P<name>[A-Za-z][A-Za-z0-9_ ]{0,15})"),
)
# "叫我X" / "称呼我为X"
_ADDRESS_PATTERNS = (
    re.compile(r"(?:叫我|称呼我为|称呼我叫)(?P<name>[\w\u4e00-\u9fff]{1,12})"),
)
# "我喜欢X" / "我爱X" / "我不喜欢X" / "我讨厌X"
_PREFERENCE_PATTERNS = (
    re.compile(r"我(?:很)?喜欢(?P<subject>[^，。！？,!?]{1,20})"),
    re.compile(r"我(?:超|最)?爱(?:吃|喝|看|玩)?(?P<subject>[^，。！？,!?]{1,20})"),
)
_DISLIKE_PATTERNS = (
    re.compile(r"我不喜欢(?P<subject>[^，。！？,!?]{1,20})"),
    re.compile(r"我(?:很)?讨厌(?P<subject>[^，。！？,!?]{1,20})"),
)
# "下次一定X" / "明天我会X" / "答应你X" / "约好了X"
# 时间词后必须紧跟意愿动词（一定/我会/要/会/带/陪/帮/请/一起），
# 否则"明天天气怎么样""以后怎么办"这类普通询问会被误判为承诺。
_PROMISE_PATTERNS = (
    re.compile(
        r"(?:下次|明天|回头|以后)(?:一定|我会|我要|要|会|带|陪|帮|请|给你|一起)"
        r"(?P<subject>[^，。！？,!?]{2,30})"
    ),
    re.compile(r"(?:答应你|约好了?|说好了?)(?P<subject>[^，。！？,!?]{2,30})"),
)
# 承诺内容中出现疑问词时视为询问而非承诺（如"明天会下雨吗"）
_QUESTION_MARKERS = ("怎么", "什么", "为什么", "如何", "吗", "呢", "多少", "几")


@dataclass(frozen=True)
class ExtractedMemory:
    """一条从用户消息中提取出的记忆（未入库）。"""

    memory_type: str
    memory_key: str
    content: str
    importance: float


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().strip("的了")


def extract_memories(message: str) -> list[ExtractedMemory]:
    """从用户消息中提取长期记忆，无命中时返回空列表。

    同一条消息最多产出：1 条名字 + 2 条偏好 + 1 条承诺，防止刷屏。
    内容以第三人称描述存储（"用户说自己叫X"），避免把用户原话
    当成可执行指令注入参考区。
    """
    text = (message or "").strip()
    if not text:
        return []

    extracted: list[ExtractedMemory] = []

    name = _first_match(_NAME_PATTERNS, text)
    if name:
        extracted.append(
            ExtractedMemory(
                memory_type="user_fact",
                memory_key="user_name",
                content=f"用户说自己叫{_clean(name)}",
                importance=0.9,
            )
        )

    preferences = _all_matches(_PREFERENCE_PATTERNS, text, limit=2)
    for subject in preferences:
        cleaned = _clean(subject)
        if cleaned:
            extracted.append(
                ExtractedMemory(
                    memory_type="user_fact",
                    memory_key=f"preference_{cleaned[:20]}",
                    content=f"用户说喜欢{cleaned}",
                    importance=0.6,
                )
            )
    if not preferences:
        dislikes = _all_matches(_DISLIKE_PATTERNS, text, limit=1)
        for subject in dislikes:
            cleaned = _clean(subject)
            if cleaned:
                # 喜欢与厌恶共用 preference_* key：对同一事物的最新
                # 表态覆盖旧表态，避免两条互相矛盾的记忆并存。
                extracted.append(
                    ExtractedMemory(
                        memory_type="user_fact",
                        memory_key=f"preference_{cleaned[:20]}",
                        content=f"用户说不喜欢{cleaned}",
                        importance=0.5,
                    )
                )

    promise = _first_match(_PROMISE_PATTERNS, text)
    if promise:
        cleaned = _clean(promise)
        # 疑问句不是承诺："明天会下雨吗""下次带我去哪"等
        if cleaned and not any(marker in cleaned for marker in _QUESTION_MARKERS):
            extracted.append(
                ExtractedMemory(
                    memory_type="promise",
                    memory_key=f"promise_{cleaned[:20]}",
                    content=f"用户提到：{_truncate(cleaned, MAX_MEMORY_CONTENT_CHARS - 6)}",
                    importance=0.8,
                )
            )

    # 统一截断，防止单条记忆过长
    return [
        ExtractedMemory(
            memory_type=item.memory_type,
            memory_key=item.memory_key[:60],
            content=_truncate(item.content, MAX_MEMORY_CONTENT_CHARS),
            importance=item.importance,
        )
        for item in extracted
    ]


def extract_preferred_address(message: str) -> str | None:
    """从"叫我X"类表达中提取用户偏好的称呼。"""
    address = _first_match(_ADDRESS_PATTERNS, (message or "").strip())
    if not address:
        return None
    cleaned = _clean(address)
    return _truncate(cleaned, 20) if cleaned else None


def next_relationship_stage(
    current_stage: RelationshipStage, interaction_count: int
) -> RelationshipStage:
    """按交互轮数计算目标关系阶段（只前进不后退）。

    自动推进上限为 familiar：close 仅凭消息数量不足以判定亲密程度，
    只能由管理员通过管理接口手动设置；已处于 close 的关系也不会
    因计数回落而自动降级。
    """
    target: RelationshipStage = "stranger"
    for threshold, stage in _STAGE_THRESHOLDS:
        if interaction_count >= threshold:
            target = stage
            break
    try:
        current_index = _STAGE_ORDER.index(current_stage)
    except ValueError:
        current_index = 0
    target_index = _STAGE_ORDER.index(target)
    # 只允许前进：计数回落（测试/重置）不自动降级
    return _STAGE_ORDER[max(current_index, target_index)]


def _first_match(patterns: tuple[re.Pattern[str], ...], text: str) -> str | None:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match.group("name") if "name" in pattern.groupindex else match.group("subject")
    return None


def _all_matches(
    patterns: tuple[re.Pattern[str], ...], text: str, *, limit: int
) -> list[str]:
    results: list[str] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            value = (
                match.group("name") if "name" in pattern.groupindex else match.group("subject")
            )
            if value and value not in results:
                results.append(value)
            if len(results) >= limit:
                return results
    return results


def _truncate(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"
