"""长期记忆检索与排序服务。

职责：从仓储读取最近记忆，按"相关度 + 重要性 + 新近度"的固定权重
打分，选出最多 MAX_MEMORY_ITEMS 条交给上下文编译。相关度低于
MIN_RELEVANCE 的记忆直接淘汰——重要性/新近度只对相关记忆加分，
不能把无关记忆"顶"进上下文。

不调用 LLM、不做向量检索：第一版使用规则打分（中文二元字符匹配），
保证可独立测试和可解释。权重与候选上限是常量，不暴露成每请求可调参数。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from character.context_builder import MAX_MEMORY_ITEMS
from character.models import MemoryItem, UserScope
from repositories.character_memory import CharacterMemoryRepository

# 综合得分权重：相关度 60% + 重要性 30% + 新近度 10%
WEIGHT_RELEVANCE = 0.6
WEIGHT_IMPORTANCE = 0.3
WEIGHT_RECENCY = 0.1

# 相关度硬门槛：低于该值的记忆一律不入选，即使重要度/新近度很高。
# 防止"考试怎么样"因为重要度得分而注入"用户喜欢咖啡"这类无关记忆。
MIN_RELEVANCE = 0.05

# 结构化意图兜底的相关度保底分：bigram 无法关联"我叫什么名字"与
# "用户说自己叫小明"这类"问题→事实"的语义对（相关度 0），按问题
# 意图直接召回对应 memory_key / memory_type 的记忆，相关度保底，
# 保证通过 MIN_RELEVANCE 门槛并在排序中优先于弱词面匹配。
INTENT_RELEVANCE_FLOOR = 0.4

# 意图识别：按强标点把消息拆成子句，子句内定位全部话题词，
# 逐个取话题词前最近的代词作为提问主体。
# 固定字符距离（主体.{0,2}话题）会被稍长的语句绕过：
# "你平时最喜欢什么"中"平时最"超出窗口，偏好记忆经词面匹配泄漏。
# 空白不能作为子句边界（"你 平时最喜欢什么"按空白拆分后，话题
# 子句失去主体代词，抑制失效；"你知道我 叫什么名字吗"同理丢失
# 用户主体无法召回），子句内部空白先归一化移除。
# 主体判定规则：
# - 用户主体（我/我们/咱/咱们）+ 话题 → 正向意图，兜底召回；
# - 非用户主体（你/您/她/他/它等）+ 话题 → 抑制对应用户私人记忆；
# - 同一问题里存在任一用户主体话题时，同类记忆不抑制
#   （"你叫什么名字以及我叫什么名字"，需遍历全部话题词而非首个）。
_CLAUSE_SPLIT_PATTERN = re.compile(r"[，。！？；、,.!?;：:]+")
# 多字代词在前，避免"你们"被截断成"你"
_PRONOUN_PATTERN = re.compile(
    r"我们|咱们|你们|她们|他们|它们|咱|您|你|她|他|它|我"
)
_USER_SUBJECTS = frozenset(("我", "我们", "咱", "咱们"))
_NAME_TOPIC_PATTERN = re.compile(r"名字|叫什么|叫啥|是谁")
_PREFERENCE_TOPIC_PATTERN = re.compile(r"喜欢|讨厌|钟意|爱好|偏好|爱")
# 约定是双方共同记忆，主体允许"我/我们/咱/你"（"你答应过我什么"）；
# 话题限定"约好/说好/答应过/承诺过/约定"，抽象提问（"什么是承诺"）无主体不触发
_PROMISE_INTENT_PATTERN = re.compile(
    r"(?:我们|咱|我|你).{0,6}(?:约好|说好|答应过|承诺过|许诺过|约定)"
)

# 新近度半衰期（天）：30 天前的记忆新近度得分约为一半
RECENCY_HALF_LIFE_DAYS = 30.0

# 每次排序的候选上限（读取最近 N 条进入打分）
CANDIDATE_LIMIT = 30

_VALID_MEMORY_TYPES = ("user_fact", "shared_event", "promise", "conversation_summary")


def _parse_timestamp(value: Any) -> datetime | None:
    """解析数据库中的 ISO 文本时间戳，失败时返回 None。"""
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _bigrams(text: str) -> frozenset[str]:
    """提取文本的二元字符（bigram）集合。

    中文没有分词时，二元字符是最小可用语义单元："咖啡好喝" 与
    "用户说喜欢咖啡" 可以通过 "咖啡" 这个 bigram 建立关联，而单字
    集合会因"的/了/吗"等高频字产生大量假阳性关联。
    单字符文本退化为单字集合。
    """
    normalized = "".join(text.split())
    if not normalized:
        return frozenset()
    if len(normalized) == 1:
        return frozenset((normalized,))
    return frozenset(
        normalized[i : i + 2] for i in range(len(normalized) - 1)
    )


def _relevance(query_grams: frozenset[str], content: str) -> float:
    """查询与记忆内容的二元字符重合率（Jaccard）。"""
    if not query_grams or not content:
        return 0.0
    content_grams = _bigrams(content)
    if not content_grams:
        return 0.0
    overlap = len(query_grams & content_grams)
    union = len(query_grams | content_grams)
    return overlap / union if union else 0.0


@dataclass(frozen=True)
class _MemoryIntents:
    """从当前消息识别出的记忆询问意图。

    name/preference/promise 为正向意图（兜底召回对应的用户记忆）；
    suppress_* 为非用户主体抑制（询问角色/第三方的名字或喜好时，
    对应用户私人记忆即使词面匹配也整体跳过）。
    """

    name: bool = False
    preference: bool = False
    promise: bool = False
    suppress_name: bool = False
    suppress_preference: bool = False


def _topic_subjects(clause: str, topic_pattern: re.Pattern[str]) -> list[str]:
    """判断子句中每个话题词的提问主体：'user' / 'non_user' 列表。

    遍历子句内全部话题词（search 只取首个会漏掉
    "你叫什么名字以及我叫什么名字"中第二个话题的用户主体），
    逐个取话题词之前最近的代词作为主体（汉语主体通常位于话题前，
    中间可夹杂"平时最"等任意长度的状语，不能依赖固定字符距离）：
    - "你平时最喜欢什么" → 主体"你"（非用户）
    - "你问我喜欢什么"   → 主体"我"（用户，最近的代词）
    话题词前没有代词时不判定（如"这个群叫什么名字"）。
    """
    subjects: list[str] = []
    for match in topic_pattern.finditer(clause):
        last_pronoun = None
        for found in _PRONOUN_PATTERN.finditer(clause[: match.start()]):
            last_pronoun = found.group()
        if last_pronoun is None:
            continue
        subjects.append(
            "user" if last_pronoun in _USER_SUBJECTS else "non_user"
        )
    return subjects


def _detect_memory_intents(query: str) -> _MemoryIntents:
    """识别"询问关于我/我们的记忆"的意图（名字/偏好/约定）。

    "我叫什么名字"与存储内容"用户说自己叫小明"没有任何公共
    bigram，纯词面匹配相关度为 0；这类问题→事实的语义对只能靠
    意图兜底召回。意图与抑制都按"子句内话题词的主体代词"判定：
    - 用户主体（我/我们/咱）→ 正向意图；
    - 非用户主体（你/您/她/他/它）→ 抑制对应用户私人记忆
      （询问角色或第三方的名字/喜好，即使词面 bigram 命中也不注入）；
    - 抽象概念提问（"什么是承诺"）无主体，不触发任何意图。
    """
    text = (query or "").strip()
    if not text:
        return _MemoryIntents()
    name_user = name_other = False
    pref_user = pref_other = False
    for clause in _CLAUSE_SPLIT_PATTERN.split(text):
        # 子句内部空白归一化：空白不是子句边界，留在原位会割裂
        # 主体与话题词（"你知道我 叫什么名字吗"）
        clause = "".join(clause.split())
        if not clause:
            continue
        for subject in _topic_subjects(clause, _NAME_TOPIC_PATTERN):
            if subject == "user":
                name_user = True
            else:
                name_other = True
        for subject in _topic_subjects(clause, _PREFERENCE_TOPIC_PATTERN):
            if subject == "user":
                pref_user = True
            else:
                pref_other = True
    return _MemoryIntents(
        name=name_user,
        preference=pref_user,
        promise=bool(_PROMISE_INTENT_PATTERN.search(text)),
        # 同类问题里存在任一用户主体话题时不抑制
        # （"你叫什么名字？我叫什么名字？"仍可召回名字记忆）
        suppress_name=name_other and not name_user,
        suppress_preference=pref_other and not pref_user,
    )


def _matches_intent(
    row: dict[str, Any], intents: _MemoryIntents
) -> bool:
    """判断记忆行是否命中当前询问意图（按 memory_key / memory_type）。"""
    memory_key = str(row.get("memory_key") or "")
    memory_type = str(row.get("memory_type") or "")
    if intents.name and memory_key.startswith("user_name"):
        return True
    if intents.preference and memory_key.startswith("preference_"):
        return True
    if intents.promise and memory_type == "promise":
        return True
    return False


def _is_suppressed(row: dict[str, Any], intents: _MemoryIntents) -> bool:
    """非用户主体问题：跳过对应的用户私人记忆。

    "你叫什么名字""你喜欢什么"询问的是角色自身，即使词面 bigram
    命中用户名字/偏好记忆（共享"名字""喜欢"等常见词），也不得
    把用户私人记忆注入角色的自我描述。
    """
    memory_key = str(row.get("memory_key") or "")
    if intents.suppress_name and memory_key.startswith("user_name"):
        return True
    if intents.suppress_preference and memory_key.startswith("preference_"):
        return True
    return False


def _recency(updated_at: Any, now: datetime) -> float:
    """新近度得分：30 天半衰期，越新得分越高，范围 [0, 1]。"""
    parsed = _parse_timestamp(updated_at)
    if parsed is None:
        return 0.0
    age_days = max(0.0, (now - parsed).total_seconds() / 86400.0)
    if age_days <= 0:
        return 1.0
    return 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)


class CharacterMemoryService:
    """长期记忆检索与排序（规则打分，无 LLM、无向量检索）。"""

    def __init__(self, repository: CharacterMemoryRepository) -> None:
        self._repo = repository

    async def load_relevant_memories(
        self, character_id: str, user_scope: UserScope, query: str
    ) -> tuple[tuple[MemoryItem, ...], int]:
        """选出与当前消息最相关的记忆。

        返回 (选中的记忆元组按相关度降序, 候选总数)。
        没有记忆时返回 ((), 0)。

        相关度使用二元字符（bigram）Jaccard 相似度，并施加
        MIN_RELEVANCE 硬门槛：与当前话题无关的记忆即使重要度
        再高也不注入，避免答非所问的"记忆污染"。

        结构化意图兜底：名字/偏好/约定类问题（"我叫什么名字"）
        与存储事实（"用户说自己叫小明"）无公共 bigram，纯词面
        匹配相关度为 0；意图命中（memory_key / memory_type 匹配）
        的记忆相关度保底 INTENT_RELEVANCE_FLOOR，保证可召回。
        询问角色自身（"你叫什么名字""你喜欢什么"）时，对应用户
        私人记忆被抑制，即使词面匹配也不注入。
        """
        records = await self._repo.list_memory_records(
            character_id, user_scope, limit=CANDIDATE_LIMIT
        )
        if not records:
            return (), 0

        query_grams = _bigrams(query)
        intents = _detect_memory_intents(query)
        now = datetime.now(timezone.utc)
        scored: list[tuple[float, MemoryItem]] = []
        for row in records:
            content = str(row.get("content") or "").strip()
            if not content:
                continue
            if _is_suppressed(row, intents):
                continue
            relevance = _relevance(query_grams, content)
            if _matches_intent(row, intents):
                relevance = max(relevance, INTENT_RELEVANCE_FLOOR)
            elif relevance < MIN_RELEVANCE:
                # 无关记忆：重要度/新近度不得替代话题相关性
                continue
            importance = _clamp01(float(row.get("importance") or 0.0))
            score = (
                WEIGHT_RELEVANCE * relevance
                + WEIGHT_IMPORTANCE * importance
                + WEIGHT_RECENCY * _recency(row.get("updated_at"), now)
            )
            memory_type = row.get("memory_type", "user_fact")
            if memory_type not in _VALID_MEMORY_TYPES:
                memory_type = "user_fact"
            scored.append(
                (
                    score,
                    MemoryItem(
                        memory_id=str(row.get("id", "")),
                        memory_type=memory_type,  # type: ignore[arg-type]
                        content=content,
                        importance=importance,
                    ),
                )
            )

        # 得分降序；得分相同保持仓储返回的"最近优先"顺序
        scored.sort(key=lambda pair: pair[0], reverse=True)
        selected = tuple(item for _score, item in scored[:MAX_MEMORY_ITEMS])
        return selected, len(records)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
