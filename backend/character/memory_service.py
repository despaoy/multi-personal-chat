"""长期记忆检索与排序服务。

CAHM 在保留中文 bigram、重要性、新近度、结构化意图和主体抑制的基础上，
加入进程内缓存的句向量余弦相似度与最低混合分门控。provider 不可用时
明确降级为原词面排序，保证记忆增强失败不影响在线回复。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import threading
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np

from character.context_builder import MAX_MEMORY_ITEMS
from character.models import MemoryItem, UserScope

if TYPE_CHECKING:
    from knowledge.retrieval_core.embedding import EmbeddingProvider
    from repositories.character_memory import CharacterMemoryRepository

logger = logging.getLogger(__name__)

# 综合得分权重：相关度 60% + 重要性 30% + 新近度 10%
WEIGHT_RELEVANCE = 0.6
WEIGHT_IMPORTANCE = 0.3
WEIGHT_RECENCY = 0.1

# CAHM 混合分固定权重（显式常量，便于消融）。
HYBRID_WEIGHT_SEMANTIC = 0.65
HYBRID_WEIGHT_LEXICAL = 0.15
HYBRID_WEIGHT_IMPORTANCE = 0.15
HYBRID_WEIGHT_RECENCY = 0.05

# 排名融合只使用各检索通道内的相对次序，避免把余弦、Jaccard、重要度
# 等不同标尺硬塞进一个绝对分数。权重仍保留原 CAHM 的四类信号占比。
RRF_K = 60
# 意图是“无词面重合时的召回兜底”，不能压过精确词面命中。
INTENT_RRF_WEIGHT = 0.2

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
_PRONOUN_PATTERN = re.compile(r"我们|咱们|你们|她们|他们|它们|咱|您|你|她|他|它|我")
_USER_SUBJECTS = frozenset(("我", "我们", "咱", "咱们"))
_NAME_TOPIC_PATTERN = re.compile(r"名字|叫什么|叫啥|是谁")
_PREFERENCE_TOPIC_PATTERN = re.compile(r"喜欢|讨厌|钟意|爱好|偏好|爱")
_GOAL_INTENT_PATTERN = re.compile(
    r"科研方向|研究方向|升学申请|学习目标|当前.{0,4}方向|最近.{0,4}准备|"
    r"保研|推免|升学|项目进度"
)
# 约定是双方共同记忆，主体允许"我/我们/咱/你"（"你答应过我什么"）；
# 话题限定"约好/说好/答应过/承诺过/约定"，抽象提问（"什么是承诺"）无主体不触发
_PROMISE_INTENT_PATTERN = re.compile(r"(?:我们|咱|我|你).{0,6}(?:约好|说好|答应过|承诺过|许诺过|约定)")

# 新近度半衰期（天）：30 天前的记忆新近度得分约为一半
RECENCY_HALF_LIFE_DAYS = 30.0

# 每次排序的候选上限（读取最近 N 条进入打分）
CANDIDATE_LIMIT = 30
SEMANTIC_MEMORY_CANDIDATE_LIMIT = max(1, int(os.getenv("SEMANTIC_MEMORY_CANDIDATE_LIMIT", "100")))
MIN_HYBRID_MEMORY_SCORE = max(0.0, min(1.0, float(os.getenv("MIN_HYBRID_MEMORY_SCORE", "0.35"))))
INTENT_HYBRID_SCORE_FLOOR = 0.4
MIN_CLAIM_CONFIDENCE = max(0.0, min(1.0, float(os.getenv("MIN_MEMORY_CLAIM_CONFIDENCE", "0.45"))))
PENDING_STATUS_FACTOR = 0.35

_VALID_MEMORY_TYPES = ("user_fact", "shared_event", "promise", "conversation_summary")
_CURRENT_MEMORY_STATUSES = frozenset(("active", "current"))
_NON_CURRENT_MEMORY_STATUSES = frozenset(("superseded", "retracted", "archived", "erased", "deleted"))
_NON_RETRIEVABLE_RELATIONS = frozenset(("RETRACT", "NOOP", "ERASE"))

# 历史问答只在查询含有明确过去时间指示时启用。具体月份会解析为
# 半开时间窗 [month_start, next_month_start)，其余历史表达允许检索完整
# 版本链，再交给相关度排序选择。默认当前问答仍只读取 active claim。
_HISTORICAL_QUERY_PATTERN = re.compile(
    r"以前|之前|过去|当时|曾经|上次|原来|去年|前年|\d{4}年|今年[一二三四五六七八九十\d]{1,3}月"
)
_YEAR_MONTH_PATTERN = re.compile(
    r"(?:(?P<year>\d{4})年|(?P<relative>今年|去年|前年))"
    r"(?P<month>[一二三四五六七八九十\d]{1,3})月"
)
_CHINESE_MONTHS = {
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
    "十一": 11,
    "十二": 12,
}

# 轻量 query expansion：只抽取原消息中已有实体焦点，并把时间/主题表达
# 映射为少量稳定检索词；不生成新事实，也不调用第二个 LLM。
_QUOTED_ENTITY_PATTERN = re.compile(r"[《“\"']([^》”\"']{1,32})[》”\"']")
_ABOUT_ENTITY_PATTERN = re.compile(r"(?:关于|说到|提到|上次那个|之前那个|那个)([\w\u4e00-\u9fff-]{2,24})")
_QUERY_FILLERS = tuple(
    sorted(
        (
            "你还记得",
            "还记得",
            "你知道",
            "能不能告诉我",
            "告诉我",
            "请问",
            "怎么样",
            "怎么回事",
            "是什么",
            "什么来着",
            "什么",
            "来着",
        ),
        key=len,
        reverse=True,
    )
)
_TIME_EXPANSIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"上次|之前|以前|过去|当时|曾经"), "历史 上次 之前 过去 当时"),
    (re.compile(r"最近|现在|目前|如今|当前"), "当前 最近 现在 目前"),
    (re.compile(r"以后|未来|将来|明年|打算|准备"), "未来 计划 打算 准备"),
    (re.compile(r"昨天|今天|明天|哪天|什么时候|多久"), "时间 日期 昨天 今天 明天"),
)
_TOPIC_EXPANSIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"名字|叫什么|叫啥|姓名"), "姓名 名字 称呼"),
    (re.compile(r"喜欢|讨厌|偏好|爱好|钟意"), "偏好 喜欢 不喜欢 讨厌 爱好"),
    (re.compile(r"约好|说好|答应|承诺|约定|许诺"), "约定 承诺 答应 计划"),
    (re.compile(r"保研|推免|升学|申请"), "保研 推免 升学 申请"),
    (re.compile(r"科研|研究|方向|项目"), "科研 研究 方向 项目"),
    (re.compile(r"目标|计划|进度|准备"), "目标 计划 进度 准备"),
    (re.compile(r"学习|考试|复习"), "学习 考试 复习"),
    (re.compile(r"工作|上班|职业|职场"), "工作 上班 职业 职场 公司"),
    (re.compile(r"住哪|来自|家乡|城市|学校|公司|地点|地方"), "地点 来自 居住 学校 公司"),
    (re.compile(r"猫|狗|宠物"), "宠物 猫 狗 饲养"),
)

# 数据库存储使用第三人称安全描述，防止用户原话被当作指令注入模型。
# 这些固定包装词不属于事实主题，检索时应去掉；否则多条记忆会因为
# 共享“用户说/用户提到”而产生无意义的词面重合。
_MEMORY_CONTENT_PREFIXES = tuple(
    sorted(
        (
            "用户说自己来自或居住在",
            "用户正在进行或准备：",
            "用户正在准备或学习",
            "用户提到共同经历：",
            "用户说自己的专业是",
            "用户明确提到：",
            "用户提到约定：",
            "用户说自己叫",
            "用户的名字是",
            "用户说不喜欢",
            "用户说喜欢",
            "用户说自己在",
            "用户说自己是",
            "用户提到：",
        ),
        key=len,
        reverse=True,
    )
)


def _parse_timestamp(value: Any) -> datetime | None:
    """解析数据库中的 ISO 文本时间戳，失败时返回 None。"""
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
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


@dataclass(frozen=True)
class _HistoricalWindow:
    """历史查询的目标时间窗；两端均空表示泛指过去。"""

    start: datetime | None = None
    end: datetime | None = None


def _month_number(value: str) -> int | None:
    text = value.strip()
    if text.isdigit():
        month = int(text)
    else:
        month = _CHINESE_MONTHS.get(text, 0)
    return month if 1 <= month <= 12 else None


def _next_month(value: datetime) -> datetime:
    if value.month == 12:
        return value.replace(year=value.year + 1, month=1)
    return value.replace(month=value.month + 1)


def _historical_query_window(query: str, now: datetime) -> _HistoricalWindow | None:
    """识别中文历史表达，并尽可能收窄到年/月时间窗。"""

    text = "".join((query or "").split())
    if not text or not _HISTORICAL_QUERY_PATTERN.search(text):
        return None

    year_month = _YEAR_MONTH_PATTERN.search(text)
    if year_month:
        month = _month_number(year_month.group("month"))
        if month is not None:
            if year_month.group("year"):
                year = int(year_month.group("year"))
            else:
                offset = {"今年": 0, "去年": -1, "前年": -2}[year_month.group("relative")]
                year = now.year + offset
            try:
                start = datetime(year, month, 1, tzinfo=now.tzinfo or timezone.utc)
            except ValueError:
                return _HistoricalWindow()
            return _HistoricalWindow(start=start, end=_next_month(start))

    explicit_year = re.search(r"(?P<year>\d{4})年", text)
    if explicit_year:
        year = int(explicit_year.group("year"))
        start = datetime(year, 1, 1, tzinfo=now.tzinfo or timezone.utc)
        return _HistoricalWindow(start=start, end=start.replace(year=year + 1))
    if "去年" in text:
        start = datetime(now.year - 1, 1, 1, tzinfo=now.tzinfo or timezone.utc)
        return _HistoricalWindow(start=start, end=start.replace(year=now.year))
    if "前年" in text:
        start = datetime(now.year - 2, 1, 1, tzinfo=now.tzinfo or timezone.utc)
        return _HistoricalWindow(start=start, end=start.replace(year=now.year - 1))
    return _HistoricalWindow()


def _is_historical_record(
    row: dict[str, Any],
    window: _HistoricalWindow,
    *,
    include_pending: bool,
) -> bool:
    """历史模式允许已替代/归档版本，但仍拒绝撤回、删除和待确认事实。"""

    status = _memory_status(row)
    if status in {"retracted", "erased", "deleted"}:
        return False
    if status == "pending" and not include_pending:
        return False
    if status not in {*_CURRENT_MEMORY_STATUSES, "superseded", "archived", "pending"}:
        return False
    relation_type = str(row.get("relation_type") or row.get("relation") or "ADD").strip().upper()
    if relation_type in _NON_RETRIEVABLE_RELATIONS:
        return False

    valid_from = _parse_timestamp(row.get("valid_from") or row.get("valid_at"))
    valid_to = _parse_timestamp(row.get("valid_to") or row.get("invalid_at"))
    # 半开区间重叠：[claim_from, claim_to) ∩ [query_from, query_to)
    if window.end is not None and valid_from is not None and valid_from >= window.end:
        return False
    return not (window.start is not None and valid_to is not None and valid_to <= window.start)


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
    return frozenset(normalized[i : i + 2] for i in range(len(normalized) - 1))


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


def _retrieval_text(content: str) -> str:
    """返回用于相关度计算的事实主体，保留未知/历史格式原文。"""
    text = (content or "").strip()
    for prefix in _MEMORY_CONTENT_PREFIXES:
        if text.startswith(prefix):
            subject = text[len(prefix) :].strip()
            return subject or text
    return text


def _expand_memory_query(query: str, intents: _MemoryIntents | None = None) -> tuple[str, ...]:
    """生成原查询、实体焦点、时间别名和主题别名组成的检索视图。

    扩展只影响召回排序，不会写回长期记忆。第一项始终保留原查询，
    因而旧 embedding provider 与缓存行为保持兼容。
    """

    text = (query or "").strip()
    if not text:
        return ()
    values: list[str] = [text]

    compact = _CLAUSE_SPLIT_PATTERN.sub("", "".join(text.split()))
    core = compact
    for filler in _QUERY_FILLERS:
        core = core.replace(filler, "")
    core = re.sub(r"^(?:我|我们|咱|咱们|你|您)", "", core)
    core = re.sub(r"[吗呢呀啊吧么]$", "", core)
    if len(core) >= 2:
        values.append(core)

    values.extend(match.group(1).strip() for match in _QUOTED_ENTITY_PATTERN.finditer(text))
    values.extend(match.group(1).strip() for match in _ABOUT_ENTITY_PATTERN.finditer(compact))

    for pattern, expansion in _TIME_EXPANSIONS:
        if pattern.search(text):
            values.append(expansion)
    for pattern, expansion in _TOPIC_EXPANSIONS:
        if pattern.search(text):
            values.append(expansion)

    # 已识别的主体意图是比通用话题关键词更可靠的 query rewrite。
    resolved_intents = intents or _detect_memory_intents(text)
    if resolved_intents.name:
        values.append("用户 姓名 名字 称呼")
    if resolved_intents.preference:
        values.append("用户 偏好 喜欢 不喜欢 讨厌 爱好")
    if resolved_intents.promise:
        values.append("共同 约定 承诺 答应 计划")
    if resolved_intents.goal:
        values.append("目标 方向 计划 项目")

    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip()
        key = "".join(normalized.split())
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(normalized)
    return tuple(unique[:8])


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
    goal: bool = False
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
        subjects.append("user" if last_pronoun in _USER_SUBJECTS else "non_user")
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
        goal=bool(_GOAL_INTENT_PATTERN.search(text)),
        # 同类问题里存在任一用户主体话题时不抑制
        # （"你叫什么名字？我叫什么名字？"仍可召回名字记忆）
        suppress_name=name_other and not name_user,
        suppress_preference=pref_other and not pref_user,
    )


def _matches_intent(row: dict[str, Any], intents: _MemoryIntents) -> bool:
    """判断记忆行是否命中当前询问意图（按 memory_key / memory_type）。"""
    memory_key = str(row.get("memory_key") or "")
    memory_type = str(row.get("memory_type") or "")
    if intents.name and memory_key.startswith("user_name"):
        return True
    if intents.preference and memory_key.startswith("preference_"):
        return True
    return bool(intents.promise and memory_type == "promise")


def _is_suppressed(row: dict[str, Any], intents: _MemoryIntents) -> bool:
    """非用户主体问题：跳过对应的用户私人记忆。

    "你叫什么名字""你喜欢什么"询问的是角色自身，即使词面 bigram
    命中用户名字/偏好记忆（共享"名字""喜欢"等常见词），也不得
    把用户私人记忆注入角色的自我描述。
    """
    memory_key = str(row.get("memory_key") or "")
    # 明确的研究/升学目标问答采用结构化类别路由，避免 embedding 把
    # “科研方向”误吸到“工作单位”等同现词但不同槽位的用户事实。
    if intents.goal and not memory_key.startswith("goal_"):
        return True
    if intents.suppress_name and memory_key.startswith("user_name"):
        return True
    return bool(intents.suppress_preference and memory_key.startswith("preference_"))


def _recency(updated_at: Any, now: datetime) -> float:
    """新近度得分：30 天半衰期，越新得分越高，范围 [0, 1]。"""
    parsed = _parse_timestamp(updated_at)
    if parsed is None:
        return 0.0
    age_days = max(0.0, (now - parsed).total_seconds() / 86400.0)
    if age_days <= 0:
        return 1.0
    return 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _memory_status(row: dict[str, Any]) -> str:
    """读取新旧 schema 的 claim 状态；旧记录视为 active。"""

    return str(row.get("status") or row.get("memory_status") or "active").strip().lower()


def _is_current_record(row: dict[str, Any], now: datetime, *, include_pending: bool) -> bool:
    """仅允许当前有效 claim 进入召回池。"""

    status = _memory_status(row)
    if status in _NON_CURRENT_MEMORY_STATUSES:
        return False
    if status == "pending" and not include_pending:
        return False
    if status not in _CURRENT_MEMORY_STATUSES and status != "pending":
        return False
    relation_type = str(row.get("relation_type") or row.get("relation") or "ADD").strip().upper()
    if relation_type in _NON_RETRIEVABLE_RELATIONS:
        return False

    valid_from = _parse_timestamp(row.get("valid_from") or row.get("valid_at"))
    valid_to = _parse_timestamp(row.get("valid_to") or row.get("invalid_at"))
    if valid_from is not None and valid_from > now:
        return False
    return valid_to is None or valid_to > now


def _json_value(value: Any) -> Any:
    """兼容仓储已解码值和旧/测试仓储直接返回的 JSON 文本。"""

    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{\"":
        return value
    try:
        return json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return value


def _evidence_texts(value: Any) -> tuple[str, ...]:
    """把 evidence/source event 的常见表示规范化为短文本元组。"""

    decoded = _json_value(value)
    if decoded is None:
        return ()
    if isinstance(decoded, dict):
        for key in ("text", "content", "quote", "evidence", "summary"):
            text = str(decoded.get(key) or "").strip()
            if text:
                return (text,)
        return ()
    if isinstance(decoded, (list, tuple, set)):
        result: list[str] = []
        for item in decoded:
            result.extend(_evidence_texts(item))
        return tuple(result)
    text = str(decoded).strip()
    return (text,) if text else ()


def _source_ids(row: dict[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for key in ("source_message_ids", "source_message_ids_json", "source_event_ids"):
        decoded = _json_value(row.get(key))
        if isinstance(decoded, (list, tuple, set)):
            values.extend(str(item).strip() for item in decoded if str(item).strip())
        elif decoded is not None and str(decoded).strip():
            values.append(str(decoded).strip())
    source_message_id = str(row.get("source_message_id") or "").strip()
    if source_message_id:
        values.append(source_message_id)
    return tuple(dict.fromkeys(values))


def _row_evidence(row: dict[str, Any], rows_by_id: dict[str, dict[str, Any]]) -> tuple[str, ...]:
    """提取 claim 自带 evidence，并按显式 source event ID 附一跳事件。"""

    values: list[str] = []
    for key in (
        "evidence",
        "evidence_json",
        "source_event",
        "source_events",
        "adjacent_event",
        "adjacent_events",
    ):
        values.extend(_evidence_texts(row.get(key)))

    explicit_event_ids = _json_value(row.get("source_event_ids"))
    if isinstance(explicit_event_ids, (list, tuple, set)):
        for event_id in explicit_event_ids:
            related = rows_by_id.get(str(event_id))
            if related is None:
                continue
            content = str(related.get("content") or "").strip()
            if content:
                values.append(content)

    # 最多携带四条紧邻证据；完整原始记录仍可通过 source IDs 追溯。
    return tuple(dict.fromkeys(value for value in values if value))[:4]


def _rank_route(values: dict[int, float], eligible: set[int]) -> dict[int, float]:
    """去除非正分后返回一个 RRF 排名通道。"""

    return {index: score for index, score in values.items() if index in eligible and score > 0.0}


def _reciprocal_rank_fusion(
    eligible: set[int],
    routes: list[tuple[float, dict[int, float]]],
) -> dict[int, float]:
    fused = {index: 0.0 for index in eligible}
    for weight, route in routes:
        if weight <= 0.0 or not route:
            continue
        ordered = sorted(route.items(), key=lambda pair: pair[1], reverse=True)
        for rank, (index, _score) in enumerate(ordered, start=1):
            fused[index] += weight / (RRF_K + rank)
    return fused


class CharacterMemoryService:
    """CAHM 长期记忆检索；可切换 lexical baseline 做消融。"""

    def __init__(
        self,
        repository: CharacterMemoryRepository,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        semantic_enabled: bool | None = None,
        gate_enabled: bool = True,
        min_hybrid_score: float | None = None,
        candidate_limit: int | None = None,
        include_pending: bool = False,
        rrf_enabled: bool | None = None,
        query_expansion_enabled: bool | None = None,
        version_filter_enabled: bool | None = None,
        evidence_enabled: bool | None = None,
    ) -> None:
        self._repo = repository
        if semantic_enabled is None:
            semantic_enabled = os.getenv("CAHM_SEMANTIC_MEMORY_ENABLED", "true").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        self._semantic_enabled = bool(semantic_enabled)
        self._embedding_provider = embedding_provider
        self._gate_enabled = bool(gate_enabled)
        self._min_hybrid_score = (
            MIN_HYBRID_MEMORY_SCORE
            if min_hybrid_score is None
            else max(0.0, min(1.0, float(min_hybrid_score)))
        )
        self._candidate_limit = max(
            1,
            int(
                candidate_limit
                if candidate_limit is not None
                else (SEMANTIC_MEMORY_CANDIDATE_LIMIT if self._semantic_enabled else CANDIDATE_LIMIT)
            ),
        )
        self._embedding_cache: dict[tuple[str, str, str], np.ndarray] = {}
        self._embedding_lock = threading.Lock()
        self._semantic_failure_logged = False
        self._include_pending = bool(include_pending)
        self._rrf_enabled = _env_bool("CAHM_RRF_ENABLED", True) if rrf_enabled is None else bool(rrf_enabled)
        self._query_expansion_enabled = (
            _env_bool("CAHM_QUERY_EXPANSION_ENABLED", True)
            if query_expansion_enabled is None
            else bool(query_expansion_enabled)
        )
        self._version_filter_enabled = (
            _env_bool("CAHM_VERSION_FILTER_ENABLED", True)
            if version_filter_enabled is None
            else bool(version_filter_enabled)
        )
        self._evidence_enabled = (
            _env_bool("CAHM_EVIDENCE_ENABLED", True)
            if evidence_enabled is None
            else bool(evidence_enabled)
        )

    async def load_relevant_memories(
        self,
        character_id: str,
        user_scope: UserScope,
        query: str,
        *,
        include_historical: bool | None = None,
    ) -> tuple[tuple[MemoryItem, ...], int]:
        """选出与当前消息最相关的记忆。

        返回 (选中的记忆元组按相关度降序, 候选总数)。
        没有记忆时返回 ((), 0)。

        仅 active/current 且处于有效时间窗的 claim 参与检索；pending
        默认排除，superseded/retracted/archived 永远排除。语义、多个
        query expansion 的 bigram 词面排名、重要度和新近度分别排序，
        再用 RRF 融合，避免不同标尺被一个固定加权阈值误判。

        结构化意图兜底：名字/偏好/约定类问题（"我叫什么名字"）
        与存储事实（"用户说自己叫小明"）无公共 bigram，纯词面
        匹配相关度为 0；意图命中（memory_key / memory_type 匹配）
        的记忆相关度保底 INTENT_RELEVANCE_FLOOR，保证可召回。
        询问角色自身（"你叫什么名字""你喜欢什么"）时，对应用户
        私人记忆被抑制，即使词面匹配也不注入。

        ``include_historical=None`` 时自动识别明确过去表达；具体年/月
        按 claim 有效期重叠筛选，泛指过去读取完整版本链。显式 False
        可强制保持当前版本模式，True 可强制读取历史候选。
        """
        now = datetime.now(timezone.utc)
        historical_window = _historical_query_window(query, now)
        historical_requested = historical_window is not None
        if include_historical is not None:
            historical_requested = bool(include_historical)
            if historical_requested and historical_window is None:
                historical_window = _HistoricalWindow()
            elif not historical_requested:
                historical_window = None

        try:
            records = await self._repo.list_memory_records(
                character_id,
                user_scope,
                limit=self._candidate_limit,
                include_inactive=historical_requested,
            )
        except TypeError as exc:
            # 兼容旧仓储与轻量测试替身；仓储内部自身抛出的 TypeError 不吞掉。
            message = str(exc)
            if "include_inactive" not in message and "unexpected keyword" not in message:
                raise
            records = await self._repo.list_memory_records(
                character_id,
                user_scope,
                limit=self._candidate_limit,
            )
        if not records:
            return (), 0

        intents = _detect_memory_intents(query)
        if not self._query_expansion_enabled and intents.goal:
            # goal 类别路由属于平衡版 query expansion，不污染关闭该开关
            # 的 legacy 消融路径；姓名/偏好/promise 的既有安全意图不变。
            intents = replace(intents, goal=False)
        usable_records = [
            row
            for row in records
            if (
                not self._version_filter_enabled
                or (
                    _is_historical_record(
                        row,
                        historical_window,
                        include_pending=self._include_pending,
                    )
                    if historical_window is not None
                    else _is_current_record(row, now, include_pending=self._include_pending)
                )
            )
            and not _is_suppressed(row, intents)
            and str(row.get("content") or "").strip()
        ]
        if not usable_records:
            return (), len(records)

        semantic_scores: dict[int, float] | None = None
        if self._semantic_enabled:
            try:
                semantic_scores = await asyncio.to_thread(self._semantic_similarities, query, usable_records)
            except Exception as exc:  # 记忆增强失败不得影响回复
                if not self._semantic_failure_logged:
                    logger.warning("CAHM 语义检索不可用，降级到 bigram baseline: %s", exc)
                    self._semantic_failure_logged = True

        query_views = _expand_memory_query(query, intents) if self._query_expansion_enabled else ((query or "").strip(),)
        query_views = tuple(view for view in query_views if view)
        lexical_routes: list[dict[int, float]] = [dict() for _view in query_views]
        lexical_max_scores: dict[int, float] = {}
        importance_scores: dict[int, float] = {}
        recency_scores: dict[int, float] = {}
        intent_scores: dict[int, float] = {}
        confidence_scores: dict[int, float] = {}
        eligible: set[int] = set()

        for index, row in enumerate(usable_records):
            content = str(row.get("content") or "").strip()
            retrieval_text = _retrieval_text(content)
            relevances: list[float] = []
            for route, view in zip(lexical_routes, query_views, strict=True):
                relevance = _relevance(_bigrams(view), retrieval_text)
                route[index] = relevance
                relevances.append(relevance)
            max_relevance = max(relevances, default=0.0)
            lexical_max_scores[index] = max_relevance
            intent_match = _matches_intent(row, intents)
            if intent_match:
                intent_scores[index] = INTENT_RELEVANCE_FLOOR

            importance = _clamp01(float(row.get("importance") or 0.0))
            recency = _recency(row.get("updated_at"), now)
            confidence = _clamp01(_safe_float(row.get("confidence"), 1.0))
            if self._version_filter_enabled and confidence < MIN_CLAIM_CONFIDENCE:
                # 证据不足的 claim 不因重要度或新近度进入上下文。
                continue

            if semantic_scores is None:
                is_eligible = intent_match or max_relevance >= MIN_RELEVANCE
            elif not self._gate_enabled:
                # 保留无门控消融：仍受生命周期、主体和 claim 置信度约束。
                is_eligible = True
            elif self._rrf_enabled:
                # 任一可靠通道即可通过：结构化意图、明确词面重合或语义
                # 相似度。重要度/新近度只参与已相关候选的 RRF 排序。
                is_eligible = (
                    intent_match
                    or max_relevance >= MIN_RELEVANCE
                    or semantic_scores.get(index, 0.0) >= self._min_hybrid_score
                )
            else:
                # legacy hybrid 会在固定加权分计算后应用原门槛。
                is_eligible = True
            if not is_eligible:
                continue

            eligible.add(index)
            importance_scores[index] = importance
            recency_scores[index] = recency
            confidence_scores[index] = confidence

        if not eligible:
            return (), len(records)

        if self._rrf_enabled:
            routes: list[tuple[float, dict[int, float]]] = []
            if semantic_scores is None:
                lexical_weight = WEIGHT_RELEVANCE
                importance_weight = WEIGHT_IMPORTANCE
                recency_weight = WEIGHT_RECENCY
            else:
                routes.append((HYBRID_WEIGHT_SEMANTIC, _rank_route(semantic_scores, eligible)))
                lexical_weight = HYBRID_WEIGHT_LEXICAL
                importance_weight = HYBRID_WEIGHT_IMPORTANCE
                recency_weight = HYBRID_WEIGHT_RECENCY

            if lexical_routes:
                per_route_weight = lexical_weight / len(lexical_routes)
                routes.extend((per_route_weight, _rank_route(route, eligible)) for route in lexical_routes)
            routes.extend(
                (
                    (importance_weight, _rank_route(importance_scores, eligible)),
                    (recency_weight, _rank_route(recency_scores, eligible)),
                    (INTENT_RRF_WEIGHT, _rank_route(intent_scores, eligible)),
                )
            )
            fused_scores = _reciprocal_rank_fusion(eligible, routes)
            for index in eligible:
                # claim 自身置信度只作温和校准；legacy 记录默认 1.0。
                fused_scores[index] *= 0.5 + 0.5 * confidence_scores[index]
                if _memory_status(usable_records[index]) == "pending":
                    fused_scores[index] *= PENDING_STATUS_FACTOR
        else:
            # 与原 CAHM 一致的固定加权路径，专供消融；默认不走此分支。
            fused_scores: dict[int, float] = {}
            for index in eligible:
                relevance = lexical_max_scores.get(index, 0.0)
                intent_match = index in intent_scores
                if intent_match:
                    relevance = max(relevance, INTENT_RELEVANCE_FLOOR)
                if semantic_scores is None:
                    score = (
                        WEIGHT_RELEVANCE * relevance
                        + WEIGHT_IMPORTANCE * importance_scores[index]
                        + WEIGHT_RECENCY * recency_scores[index]
                    )
                else:
                    score = (
                        HYBRID_WEIGHT_SEMANTIC * semantic_scores.get(index, 0.0)
                        + HYBRID_WEIGHT_LEXICAL * relevance
                        + HYBRID_WEIGHT_IMPORTANCE * importance_scores[index]
                        + HYBRID_WEIGHT_RECENCY * recency_scores[index]
                    )
                    if intent_match:
                        score = max(score, INTENT_HYBRID_SCORE_FLOOR)
                    if self._gate_enabled and score < self._min_hybrid_score:
                        continue
                fused_scores[index] = score
            eligible = set(fused_scores)
            if not eligible:
                return (), len(records)

        rows_by_id = {str(row.get("id")): row for row in records if row.get("id") is not None}
        scored: list[tuple[float, MemoryItem]] = []
        for index in sorted(eligible):
            row = usable_records[index]
            content = str(row.get("content") or "").strip()
            memory_type = row.get("memory_type", "user_fact")
            if memory_type not in _VALID_MEMORY_TYPES:
                memory_type = "user_fact"
            importance = importance_scores[index]
            status = _memory_status(row) if self._version_filter_enabled else "active"
            relation_type = str(row.get("relation_type") or row.get("relation") or "ADD").upper()
            if not self._evidence_enabled:
                relation_type = "ADD"
            scored.append(
                (
                    fused_scores[index],
                    MemoryItem(
                        memory_id=str(row.get("id", "")),
                        memory_type=memory_type,  # type: ignore[arg-type]
                        content=content,
                        importance=importance,
                        evidence=_row_evidence(row, rows_by_id) if self._evidence_enabled else (),
                        valid_from=(
                            str(row.get("valid_from") or row.get("valid_at") or "")
                            if self._version_filter_enabled
                            else ""
                        ),
                        valid_to=(
                            str(row.get("valid_to") or row.get("invalid_at") or "")
                            if self._version_filter_enabled
                            else ""
                        ),
                        confidence=confidence_scores[index] if self._version_filter_enabled else 1.0,
                        status=status,  # type: ignore[arg-type]
                        relation_type=relation_type,
                        source_message_ids=_source_ids(row) if self._evidence_enabled else (),
                        historical=historical_window is not None,
                    ),
                )
            )

        # RRF 得分降序；得分相同保持仓储返回的“最近优先”顺序。
        scored.sort(key=lambda pair: pair[0], reverse=True)
        selected = tuple(item for _score, item in scored[:MAX_MEMORY_ITEMS])
        return selected, len(records)

    def _semantic_similarities(self, query: str, records: list[dict[str, Any]]) -> dict[int, float]:
        """批量计算余弦相似度；记忆向量按 id/updated_at/content hash 缓存。"""
        with self._embedding_lock:
            if self._embedding_provider is None:
                from knowledge.retrieval_core.embedding import get_default_embedding_provider

                self._embedding_provider = get_default_embedding_provider()
            provider = self._embedding_provider
            missing_indices: list[int] = []
            missing_texts: list[str] = []
            vectors: dict[int, np.ndarray] = {}
            keys: dict[int, tuple[str, str, str]] = {}
            for index, row in enumerate(records):
                content = str(row.get("content") or "").strip()
                content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
                key = (str(row.get("id") or ""), str(row.get("updated_at") or ""), content_hash)
                keys[index] = key
                cached = self._embedding_cache.get(key)
                if cached is None:
                    missing_indices.append(index)
                    missing_texts.append(_retrieval_text(content))
                else:
                    vectors[index] = cached

            encoded = provider.embed_texts([query, *missing_texts])
            matrix = np.asarray(encoded, dtype=np.float32)
            if matrix.ndim != 2 or matrix.shape[0] != len(missing_texts) + 1:
                raise ValueError("embedding provider 返回形状不正确")
            query_vector = _normalized_vector(matrix[0])
            for offset, index in enumerate(missing_indices, start=1):
                vector = _normalized_vector(matrix[offset])
                self._embedding_cache[keys[index]] = vector
                vectors[index] = vector

            return {
                index: _clamp01(float(np.dot(query_vector, _normalized_vector(vector))))
                for index, vector in vectors.items()
            }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _normalized_vector(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    return value if norm <= 0.0 else value / norm
