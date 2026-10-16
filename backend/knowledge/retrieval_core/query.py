"""Extract registered names and literal query data for retrieval.

Semantic task planning belongs to retrieval_query_plan. This module does not
infer document types, narrative intent or substitute synonym queries. The
legacy single-character alias eligibility gate remains a separate domain rule.
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
    "哪本",
    "主人公",
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


@dataclass
class QueryAnalysis:
    """查询分析结果（供检索层消费）。"""

    original_query: str
    normalized_query: str
    entities: list[str] = field(default_factory=list)
    matched_domains: list[str] = field(default_factory=list)
    story_hits: list[str] = field(default_factory=list)
    # domain 命中的信号强度（诊断用）
    domain_hit_reasons: dict[str, str] = field(default_factory=dict)


def _contains_any(query: str, words: Sequence[str]) -> list[str]:
    # 主人公 is a narrative role, not a master/servant relation. Preserve a
    # separate occurrence of 主人 in mixed queries rather than masking all of it.
    return [w for w in words if (bool(re.search(r"主人(?!公)", query)) if w == "主人" else w in query)]


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
        if domain_id is not None and domain_id not in self._normalizers:
            raise ValueError(f"Unknown query domain: {domain_id}")
        if not query.strip():
            return QueryAnalysis(original_query=query, normalized_query=query)

        candidates = [domain_id] if domain_id is not None else list(self._normalizers)
        entities: list[str] = []
        domains: list[str] = []
        reasons: dict[str, str] = {}
        stories: list[str] = []
        normalized_views: dict[str, str] = {}
        for did in candidates:
            normalizer = self._normalizers[did]
            matches = normalizer.match_spans(query)
            names = list(dict.fromkeys(canonical for _, _, _, canonical in matches))
            tokens = list(dict.fromkeys(token for _, _, token, _ in matches))
            titles = [title for title in self._story_titles[did] if title in query]
            reason = (
                "explicit_domain" if domain_id is not None else self._match_domain(did, names, tokens, titles, query)
            )
            if reason is None:
                continue
            domains.append(did)
            reasons[did] = reason
            entities.extend(name for name in names if name not in entities)
            stories.extend(title for title in titles if title not in stories)
            normalized_views[did] = normalizer.normalize_matches(query, matches)
        return QueryAnalysis(
            original_query=query,
            normalized_query=normalized_views[domains[0]] if len(domains) == 1 else query,
            entities=entities,
            matched_domains=domains,
            story_hits=stories,
            domain_hit_reasons=reasons,
        )
