"""Shared query analysis and alias normalization.

轻量规则分析（不调用 LLM 做实时分类）：
- 实体识别与 domain 自动选择（alias 归一）
- 文档类型倾向（关系/事件/事实）
- 叙事层意图（reality/temporal/scope 词）
- 卷名/故事标题命中
- 扩展关键词

分析结果用于：扩展关键词、metadata 过滤建议、召回权重调整、
结果解释。不用于直接生成答案。

层级词表为通用中文语义词（非作品专属）；domain 可通过
alias/story_titles 注入专属词表。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .loaders import AliasEntityNormalizer

if TYPE_CHECKING:
    from collections.abc import Sequence

# -- 通用词表（领域无关） -------------------------------------------------

RELATION_QUERY_WORDS = [
    "什么关系",
    "的什么人",
    "关系",
    "亲属",
    "兄妹",
    "姐妹",
    "兄弟",
    "母女",
    "父女",
    "父子",
    "主仆",
    "主人",
    "女仆",
    "仆人",
    "朋友",
    "好友",
    "恋人",
    "情侣",
    "夫妻",
    "师徒",
    "姐姐",
    "妹妹",
    "哥哥",
    "弟弟",
    "妈妈",
    "母亲",
    "父亲",
    "家人",
    "亲戚",
]

EVENT_QUERY_WORDS = [
    "事件",
    "发生了什么",
    "发生了",
    "经过",
    "遭遇",
    "做了什么",
    "起因",
    "原因",
    "结果",
    "后来",
    "怎么样了",
    "如何",
    "为什么",
    "为何",
    "怎么会",
    "导致",
    "造成",
]

FACT_QUERY_WORDS = [
    "是什么",
    "是什么人",
    "是谁",
    "设定",
    "什么身份",
    "什么目标",
    "什么状态",
    "什么性格",
    "多少",
    "哪一",
    "哪个",
    "哪些",
    "什么类型",
    "什么属性",
    "死因",
    "怎么死",
    "如何死",
    "如何死亡",
    "因何而死",
    "死亡原因",
]

DEATH_QUERY_WORDS = [
    "死因",
    "怎么死",
    "如何死",
    "如何死亡",
    "因何而死",
    "死亡原因",
    "自缢",
    "上吊",
    "自杀",
    "身亡",
]
FINALITY_QUERY_WORDS = ["最终", "最后", "到底", "真相", "实际"]

PREDICATE_QUERY_WORDS: dict[str, list[str]] = {
    "身份": ["身份", "是谁", "是什么人", "做什么工作"],
    "目标": ["目标", "目的", "想要做什么"],
    "状态": ["状态", "现状", "怎么样", "变成", "成为", "恢复"],
    "性格": ["性格", "什么样的人"],
    "死因": DEATH_QUERY_WORDS,
    "设定": ["设定", "有什么作用", "什么效果"],
    "经历": ["经历", "遭遇", "发生过什么"],
    "偏好": ["喜欢什么", "偏好", "爱好"],
    "能力": ["能力", "会什么", "能做什么"],
    "外貌": ["外貌", "长什么样", "样貌"],
}

RELATION_TYPE_QUERY_WORDS: dict[str, list[str]] = {
    "哥哥": ["哥哥", "兄长"],
    "妹妹": ["妹妹"],
    "姐姐": ["姐姐"],
    "弟弟": ["弟弟"],
    "母亲": ["母亲", "妈妈"],
    "父亲": ["父亲", "爸爸"],
    "朋友": ["朋友", "好友"],
    "恋人": ["恋人", "情侣", "交往"],
    "主人": ["主人", "主仆", "仆人", "女仆"],
    "创造者": ["创造者", "创造了", "创造的"],
    "敌对": ["敌对", "敌人"],
}

LEXICAL_QUERY_EXPANSIONS: dict[str, list[str]] = {
    "上吊": ["自缢"],
    "自杀": ["自缢", "轻生", "主动赴死"],
    "车祸": ["交通事故"],
}

REALITY_WORDS: dict[str, list[str]] = {
    "objective": ["真相", "真实", "实际上", "事实上", "客观", "现实中", "真的"],
    "fictional": ["梦", "梦境", "幻想", "故事里", "书中世界", "虚构", "梦中"],
    "character_claim": ["声称", "主张", "认为", "所说", "说过", "自称", "嘴上"],
    "inferred": ["推断", "推测", "猜测", "猜测是", "大概"],
}

TEMPORAL_WORDS: dict[str, list[str]] = {
    "flashback": ["过去", "以前", "当年", "小时候", "曾经", "回忆", "往昔", "童年"],
    "current": ["现在", "如今", "目前", "当下"],
    "hypothetical": ["如果", "假如", "假设", "要是", "假若"],
    "reconstruction": ["重构", "重演", "重现", "再演"],
}

SCOPE_WORDS: dict[str, list[str]] = {
    "main_story": ["主线", "本篇", "正篇"],
    "bonus_story": ["番外", "追加", "日后谈", "特典", "外传"],
}


@dataclass
class QueryAnalysis:
    """查询分析结果（供检索层消费）。"""

    original_query: str
    normalized_query: str
    entities: list[str] = field(default_factory=list)
    matched_domains: list[str] = field(default_factory=list)
    expanded_keywords: list[str] = field(default_factory=list)
    lexical_expansions: list[str] = field(default_factory=list)
    doc_type_preferences: list[str] = field(default_factory=list)
    predicate_preferences: list[str] = field(default_factory=list)
    relation_type_preferences: list[str] = field(default_factory=list)
    reality_preferences: list[str] = field(default_factory=list)
    temporal_preferences: list[str] = field(default_factory=list)
    scope_preferences: list[str] = field(default_factory=list)
    story_hits: list[str] = field(default_factory=list)
    wants_real_history: bool = False
    causal_intent: bool = False
    # domain 命中的信号强度（诊断用）
    domain_hit_reasons: dict[str, str] = field(default_factory=dict)

    def preferred_filters(self) -> dict[str, list[str]]:
        """由分析结果推导的软过滤建议（检索层做加权，不做硬过滤）。"""
        filters: dict[str, list[str]] = {}
        if self.doc_type_preferences:
            filters["document_type"] = list(self.doc_type_preferences)
        return filters


def _contains_any(query: str, words: Sequence[str]) -> list[str]:
    return [w for w in words if w in query]


def _has_intent_words(query: str) -> bool:
    """查询是否包含文档类型意图词（关系/事件/事实问法）。"""
    return bool(
        _contains_any(query, RELATION_QUERY_WORDS)
        or _contains_any(query, EVENT_QUERY_WORDS)
        or _contains_any(query, FACT_QUERY_WORDS)
    )


class QueryAnalyzer:
    """按已注册 domain 的 alias 表做实体归一与域匹配。"""

    def __init__(self, domain_configs: Sequence):
        self._normalizers: dict[str, AliasEntityNormalizer] = {}
        self._story_titles: dict[str, list[str]] = {}
        for config in domain_configs:
            self._normalizers[config.domain_id] = AliasEntityNormalizer(config.aliases)
            self._story_titles[config.domain_id] = list(config.story_titles)

    # -- 实体识别 ---------------------------------------------------------
    def _scan_entities(self, domain_id: str, query: str) -> tuple:
        """返回 (规范实体列表, 命中的原始 token 列表)。"""
        normalizer = self._normalizers[domain_id]
        entities: list[str] = []
        matched_tokens: list[str] = []
        for token, canonical in normalizer.tokens():
            if token and token in query:
                if canonical not in entities:
                    entities.append(canonical)
                if token not in matched_tokens:
                    matched_tokens.append(token)
        return entities, matched_tokens

    def _normalize_query(self, domain_id: str, query: str) -> str:
        """把查询中的别名替换为规范名（长词优先）。"""
        normalizer = self._normalizers[domain_id]
        text = query
        for token, canonical in normalizer.tokens():
            if token in text:
                text = text.replace(token, canonical)
        return text

    # -- 域自动选择 ---------------------------------------------------------
    def _match_domain(
        self,
        domain_id: str,
        entities: list[str],
        matched_tokens: list[str],
        story_hits: list[str],
        query: str,
    ) -> str | None:
        """域门控信号（通用规则，无作品特例）：

        1. 故事标题命中
        2. ≥2 个不同实体命中
        3. ≥2 字的别名 token 命中（多字名/全名）
        4. 单字实体命中 + 查询含类型意图词（关系/事件/事实问法）
           ——单字名单独出现不触发（避免"王妃"类误路由），
           但带明确知识问法的单字名查询（"妃的哥哥是谁"）触发
        """
        if story_hits:
            return f"story:{','.join(story_hits[:2])}"
        if len(entities) >= 2:
            return f"entities:{','.join(entities[:3])}"
        multi_char_tokens = [t for t in matched_tokens if len(t) >= 2]
        if multi_char_tokens:
            return f"entity:{multi_char_tokens[0]}"
        if matched_tokens and _has_intent_words(query):
            return f"entity_intent:{matched_tokens[0]}"
        return None

    # -- 主入口 -------------------------------------------------------------
    def analyze(self, query: str, domain_id: str | None = None) -> QueryAnalysis:
        query = (query or "").strip()
        if not query:
            return QueryAnalysis(original_query=query, normalized_query=query)

        candidate_domains = (
            [domain_id] if domain_id and domain_id in self._normalizers else list(self._normalizers.keys())
        )

        all_entities: list[str] = []
        matched_domains: list[str] = []
        domain_reasons: dict[str, str] = {}
        story_hits: list[str] = []
        normalized_query = query

        for did in candidate_domains:
            entities, matched_tokens = self._scan_entities(did, query)
            stories = [title for title in self._story_titles.get(did, []) if title in query]
            reason = self._match_domain(did, entities, matched_tokens, stories, query)
            if reason:
                matched_domains.append(did)
                domain_reasons[did] = reason
                for entity in entities:
                    if entity not in all_entities:
                        all_entities.append(entity)
                for story in stories:
                    if story not in story_hits:
                        story_hits.append(story)
                if did == domain_id:
                    normalized_query = self._normalize_query(did, query)

        # 无显式 domain 时，用命中的第一个域做归一（多域时保留原文，
        # 由各域分别归一——retrieval 层按域处理）
        if domain_id is None and matched_domains:
            normalized_query = self._normalize_query(matched_domains[0], query)

        doc_type_prefs: list[str] = []
        if _contains_any(query, RELATION_QUERY_WORDS):
            doc_type_prefs.append("relation")
        if _contains_any(query, EVENT_QUERY_WORDS):
            doc_type_prefs.append("event")
        if _contains_any(query, FACT_QUERY_WORDS):
            doc_type_prefs.append("fact")
        if _contains_any(query, DEATH_QUERY_WORDS):
            doc_type_prefs.append("fact")
        # 去重且保持稳定顺序
        doc_type_prefs = list(dict.fromkeys(doc_type_prefs))

        causal_intent = bool(_contains_any(query, ["为什么", "为何", "怎么会", "导致", "造成"]))
        if causal_intent:
            doc_type_prefs = list(dict.fromkeys([*doc_type_prefs, "fact"]))

        predicate_prefs = [
            predicate for predicate, words in PREDICATE_QUERY_WORDS.items() if _contains_any(query, words)
        ]
        if "是什么" in query and not predicate_prefs:
            predicate_prefs.append("设定")
        relation_type_prefs = [
            relation for relation, words in RELATION_TYPE_QUERY_WORDS.items() if _contains_any(query, words)
        ]

        reality_prefs: list[str] = []
        for value, words in REALITY_WORDS.items():
            if _contains_any(query, words):
                reality_prefs.append(value)
        if (
            _contains_any(query, DEATH_QUERY_WORDS)
            and _contains_any(query, FINALITY_QUERY_WORDS)
            and "objective" not in reality_prefs
        ):
            reality_prefs.insert(0, "objective")
        temporal_prefs: list[str] = []
        for value, words in TEMPORAL_WORDS.items():
            if _contains_any(query, words):
                temporal_prefs.append(value)
        scope_prefs: list[str] = []
        for value, words in SCOPE_WORDS.items():
            if _contains_any(query, words):
                scope_prefs.append(value)

        wants_real_history = bool(re.search(r"现实(中|里|世界)|历史(上|中)|真实存在", query))

        expanded = list(all_entities) + [t for t in story_hits]
        for word in _contains_any(query, RELATION_QUERY_WORDS + EVENT_QUERY_WORDS + FACT_QUERY_WORDS):
            if word not in expanded:
                expanded.append(word)
        lexical_expansions: list[str] = []
        for trigger, alternatives in LEXICAL_QUERY_EXPANSIONS.items():
            if trigger in query:
                lexical_expansions.extend(alternatives)
        lexical_expansions = list(dict.fromkeys(lexical_expansions))

        return QueryAnalysis(
            original_query=query,
            normalized_query=normalized_query,
            entities=all_entities,
            matched_domains=matched_domains,
            expanded_keywords=expanded[:12],
            lexical_expansions=lexical_expansions,
            doc_type_preferences=doc_type_prefs,
            predicate_preferences=predicate_prefs,
            relation_type_preferences=relation_type_prefs,
            reality_preferences=reality_prefs,
            temporal_preferences=temporal_prefs,
            scope_preferences=scope_prefs,
            story_hits=story_hits,
            wants_real_history=wants_real_history,
            causal_intent=causal_intent,
            domain_hit_reasons=domain_reasons,
        )
