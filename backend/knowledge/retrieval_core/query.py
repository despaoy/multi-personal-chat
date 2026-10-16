"""Extract registered names and literal query data for retrieval.

Semantic task planning belongs to retrieval_query_plan. This module does not
infer document types, narrative intent or substitute synonym queries. The
single-character names require an explicitly selected or semantically proven domain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .loaders import AliasEntityNormalizer

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class QueryAnalysis:
    """查询分析结果（供检索层消费）。"""

    original_query: str
    normalized_query: str
    entities: list[str] = field(default_factory=list)
    matched_domains: list[str] = field(default_factory=list)
    story_hits: list[str] = field(default_factory=list)
    registered_aliases: dict[str, str] = field(default_factory=dict)
    # domain 命中的信号强度（诊断用）
    domain_hit_reasons: dict[str, str] = field(default_factory=dict)


class QueryAnalyzer:
    """按已注册 domain 的 alias 表做实体归一与域匹配。"""

    def __init__(self, domain_configs: Sequence):
        self._normalizers: dict[str, AliasEntityNormalizer] = {}
        self._story_titles: dict[str, list[str]] = {}
        for config in domain_configs:
            self._normalizers[config.domain_id] = AliasEntityNormalizer(config.aliases)
            self._story_titles[config.domain_id] = list(config.story_titles)

    # -- 域自动选择 ---------------------------------------------------------
    def _match_domain(
        self,
        entities: list[str],
        matched_tokens: list[str],
        story_hits: list[str],
    ) -> str | None:
        """域门控信号（通用规则，无作品特例）：

        1. 故事标题命中
        2. ≥2 个不同实体命中
        3. ≥2 字的别名 token 命中（多字名/全名）
        """
        if story_hits:
            return f"story:{','.join(story_hits[:2])}"
        if len(entities) >= 2:
            return f"entities:{','.join(entities[:3])}"
        multi_char_tokens = [t for t in matched_tokens if len(t) >= 2]
        if multi_char_tokens:
            return f"entity:{multi_char_tokens[0]}"
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
        alias_views: dict[str, dict[str, str]] = {}
        for did in candidates:
            normalizer = self._normalizers[did]
            matches = normalizer.match_spans(query)
            names = list(dict.fromkeys(canonical for _, _, _, canonical in matches))
            tokens = list(dict.fromkeys(token for _, _, token, _ in matches))
            titles = [title for title in self._story_titles[did] if title in query]
            reason = "explicit_domain" if domain_id is not None else self._match_domain(names, tokens, titles)
            if reason is None:
                continue
            domains.append(did)
            reasons[did] = reason
            entities.extend(name for name in names if name not in entities)
            stories.extend(title for title in titles if title not in stories)
            normalized_views[did] = normalizer.normalize_matches(query, matches)
            alias_views[did] = {token: canonical for _, _, token, canonical in matches}
        return QueryAnalysis(
            original_query=query,
            normalized_query=normalized_views[domains[0]] if len(domains) == 1 else query,
            entities=entities,
            matched_domains=domains,
            story_hits=stories,
            domain_hit_reasons=reasons,
            registered_aliases=alias_views[domains[0]] if len(domains) == 1 else {},
        )
