"""
RAG辅助工具模块 - 优化版
作为向量数据库和重排器的上层封装，提供查询扩展、多查询融合召回、两阶段检索（粗排+精排）
以及上下文构建等一站式RAG服务。
改进：查询扩展、元数据过滤、分数归一化、查询缓存、多查询检索
"""

import copy
import hashlib
import json
import logging
import math
import os
import re
import time
from collections import OrderedDict
from threading import RLock
from typing import Any

from infra.environment import read_bool

try:
    from nonebot.log import logger
except ImportError:
    logger = logging.getLogger(__name__)

from .reranker import get_reranker
from .vector_db import get_vector_db


class DomainProfile:
    """单个知识域的查询扩展配置。

    把领域词表收敛为普通配置对象，由 QueryExpander 按域加载：
    默认加载原神域以保持现有行为，其他知识域（如月社妃 RAG）
    可注入独立的人物别名和领域词表，互不影响。
    """

    def __init__(
        self,
        name: str,
        synonym_map: dict[str, list[str]] | None = None,
        region_keywords: dict[str, list[str]] | None = None,
        domain_keywords: dict[str, list[str]] | None = None,
        category_map: dict[str, str] | None = None,
        regions: list[str] | None = None,
        filter_anchors: list[str] | None = None,
    ):
        self.name = name
        self.synonym_map: dict[str, list[str]] = synonym_map or {}
        self.region_keywords: dict[str, list[str]] = region_keywords or {}
        self.domain_keywords: dict[str, list[str]] = domain_keywords or {}
        self.category_map: dict[str, str] = category_map or {}
        self.regions: list[str] = regions or []
        self.filter_anchors = tuple(filter_anchors if filter_anchors is not None else [name])


def genshin_profile() -> DomainProfile:
    """默认原神知识域配置，词表内容与历史版本保持一致。"""
    return DomainProfile(
        name="genshin",
        filter_anchors=["原神", "Genshin", "胡桃", "钟离", "七七", "魈", "璃月", "蒙德", "稻妻", "须弥"],
        synonym_map={
            "胡桃": ["胡桃", "往生堂堂主", "七十七代堂主", "火系主C"],
            "钟离": ["钟离", "岩王帝君", "摩拉克斯", "岩神"],
            "七七": ["七七", "不卜庐", "僵尸"],
            "魈": ["魈", "降魔大圣", "护法夜叉"],
            "原神": ["原神", "Genshin", "Genshin Impact"],
            "圣遗物": ["圣遗物", "遗器", "artifact"],
            "命座": ["命座", "星座", "constellation", "命之座"],
            "天赋": ["天赋", "技能", "talent"],
            "配队": ["配队", "阵容", "队伍搭配", "team comp"],
            "元素反应": ["元素反应", "反应", "elemental reaction"],
            "深渊": ["深渊", "深渊螺旋", "spiral abyss"],
            "突破": ["突破", "ascension"],
            "副本": ["副本", "秘境", "domain"],
            "璃月": ["璃月", "璃月港", "岩之国", "契约之国"],
            "旅行者": ["旅行者", "主角", "空", "荧"],
        },
        region_keywords={
            "璃月": ["璃月", "岩", "摩拉克斯", "七星", "千岩军", "奥赛尔", "漩涡之魔神"],
            "蒙德": ["蒙德", "风", "巴巴托斯", "特瓦林", "骑士团", "西风"],
            "稻妻": ["稻妻", "雷", "雷电将军", "眼狩令", "幕府"],
            "须弥": ["须弥", "草", "大慈树王", "小吉祥草王", "世界树", "虚空"],
        },
        domain_keywords={
            "角色": ["角色", "人物", "hero", "character"],
            "武器": ["武器", "weapon", "剑", "弓", "法器", "长柄", "双手剑"],
            "玩法": ["玩法", "攻略", "怎么打", "如何打", "技巧"],
            "剧情": ["剧情", "故事", "任务", "传说", "经历了", "冒险"],
            "系统": ["系统", "机制", "规则", "怎么算"],
        },
        category_map={
            "角色": "角色",
            "武器": "武器",
            "圣遗物": "圣遗物",
            "副本": "副本",
            "剧情": "事件",
            "事件": "事件",
            "世界": "世界",
            # 兼容英文目录名
            "characters": "角色",
            "weapons": "武器",
            "artifacts": "圣遗物",
            "domains": "副本",
            "events": "事件",
        },
        regions=["璃月", "蒙德", "稻妻", "须弥"],
    )


def _stable_result_key(result: dict[str, Any]) -> str:
    """生成检索结果的稳定融合标识。

    优先使用稳定 ID（id / chunk_id / document_id）识别同一文档；
    没有稳定 ID 时退化为 title+content 的 sha256 摘要，
    避免旧的 title + content[:100] 前缀容易碰撞的问题。
    """
    for field in ("id", "chunk_id", "document_id"):
        value = result.get(field)
        if value is not None and value != "":
            return f"{field}:{value}"
    title = result.get("title") or ""
    content = result.get("content") or ""
    digest = hashlib.sha256(f"{title}\n{content}".encode()).hexdigest()
    return f"content:{digest}"


class QueryExpander:
    """查询扩展器，按已注册知识域的同义词、区域关键词、领域关键词
    对查询做替换和扩展，生成多个变体查询以提高向量检索的召回率。

    原始查询始终保留且排在第一位；扩展结果稳定去重，最多保留 5 条。
    """

    def __init__(self, profiles: list[DomainProfile] | None = None):
        """初始化查询扩展器。

        Args:
            profiles: 知识域配置列表；缺省时加载原神域以保持现有行为
        """
        self.profiles: list[DomainProfile] = [genshin_profile()] if profiles is None else list(profiles)
        # 兼容旧属性：合并所有已注册域的区域词表（区域加权仍按此查询）
        self.region_keywords: dict[str, list[str]] = {}
        for profile in self.profiles:
            self.region_keywords.update(profile.region_keywords)

    def add_profile(self, profile: DomainProfile) -> None:
        """追加一个知识域配置（如月社妃 RAG 的人物别名/领域词表）。"""
        self.profiles.append(profile)
        self.region_keywords.update(profile.region_keywords)

    def expand_query(self, query: str) -> list[str]:
        """对用户查询进行扩展，生成多个变体查询。

        依次进行同义词替换、领域关键词追加、区域限定查询生成；
        原始查询始终排在第一项，结果稳定去重，最多 5 条。

        Args:
            query: 原始用户查询

        Returns:
            最多5个扩展查询变体列表，首项为原始查询
        """
        expanded: list[str] = [query]
        seen = {query}

        def _add(candidate: str) -> None:
            if candidate and candidate not in seen:
                seen.add(candidate)
                expanded.append(candidate)

        for profile in self.profiles:
            self._expand_with_profile(query, profile, _add)

        return expanded[:5]

    def _expand_with_profile(
        self,
        query: str,
        profile: DomainProfile,
        add,
    ) -> None:
        """按单个知识域生成变体：同义词替换、领域前缀、区域限定查询。"""
        for key, synonyms in profile.synonym_map.items():
            if key in query:
                for syn in synonyms:
                    add(query.replace(key, syn))

        for domain, keywords in profile.domain_keywords.items():
            for kw in keywords:
                if kw in query:
                    add(f"{domain} {query}")
                    break

        # 区域限定查询：如果提到具体地区则生成精确检索变体
        for region, region_kws in profile.region_keywords.items():
            for kw in region_kws:
                if kw in query:
                    add(f"{region} 剧情 {query}")
                    break

    def extract_filters(self, query: str) -> dict[str, Any]:
        """从查询中提取元数据过滤条件，如类别和区域。

        按已注册知识域的 category_map / regions 依次匹配。

        Args:
            query: 用户查询

        Returns:
            过滤条件字典，可包含category和region字段
        """
        from knowledge.source_expansion import requested_document_titles

        # A complete named-source read defines its own evidence targets.
        # Incidental domain words in other subtasks cannot exclude those sources.
        if requested_document_titles(query):
            return {}

        # Category labels belong to their declared domain. Incidental words
        # such as "character" in a mixed personal/public question cannot
        # exclude an unrelated knowledge base before semantic retrieval.
        from character.quoted_erasure_authority import masked_quotes

        try:
            scope = masked_quotes(query)[0]
        except ValueError:
            return {}
        profiles = [
            profile for profile in self.profiles
            if any(
                re.search(r"(?<![A-Za-z0-9_])" + re.escape(anchor) + r"(?![A-Za-z0-9_])", scope, re.I)
                if anchor.isascii() else anchor in scope
                for anchor in profile.filter_anchors if anchor
            )
        ]
        filters: dict[str, Any] = {}

        for profile in profiles:
            for cn_name, en_category in profile.category_map.items():
                if cn_name in scope:
                    filters["category"] = en_category
                    break
            if "category" in filters:
                break

        # 检测地区关键词，用于后过滤
        for profile in profiles:
            region = next((r for r in profile.regions if r in scope), None)
            if region:
                filters["region"] = region
                break

        return filters


class RAGHelper:
    """RAG辅助类，提供完整的检索增强生成流程。

    核心流程为两阶段检索：第一阶段用混合检索（向量+BM25）做粗排召回，
    第二阶段用Cross-Encoder重排器做精排。支持查询缓存、多查询融合、
    区域加权等优化策略。
    """

    def __init__(self):
        """初始化RAG辅助类，自动加载向量数据库和重排器（若可用）。"""
        self.min_score_threshold = 0.05
        self.max_context_length = 2000
        self.top_k = 5

        self.reranker = None
        self.enable_reranking = read_bool(os.environ, "RERANKER_ENABLED")
        self.recall_multiplier = 4

        self.query_expander = QueryExpander()
        self.enable_query_expansion = True
        self.enable_multi_query = True

        self._query_cache: OrderedDict[str, tuple[float, list[dict[str, Any]]]] = OrderedDict()
        self._cache_lock = RLock()
        self._cache_max_size = 100
        self._cache_ttl = max(1, int(os.getenv("RAG_RETRIEVAL_CACHE_TTL", "60")))

        if self.enable_reranking:
            self.reranker = get_reranker()

    def _get_from_cache(self, query: str) -> list[dict[str, Any]] | None:
        with self._cache_lock:
            cached = self._query_cache.get(query)
            if cached is None:
                return None
            expires_at, results = cached
            if expires_at <= time.monotonic():
                self._query_cache.pop(query, None)
                return None
            self._query_cache.move_to_end(query)
            return copy.deepcopy(results)

    def _add_to_cache(self, query: str, results: list[dict[str, Any]]) -> None:
        with self._cache_lock:
            self._query_cache[query] = (
                time.monotonic() + self._cache_ttl,
                copy.deepcopy(results),
            )
            self._query_cache.move_to_end(query)
            while len(self._query_cache) > self._cache_max_size:
                self._query_cache.popitem(last=False)

    def _normalize_scores(self, results: list[dict[str, Any]], score_key: str = "score") -> list[dict[str, Any]]:
        if not results:
            return results

        scores = [r.get(score_key, 0) for r in results]
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score

        if score_range == 0:
            for r in results:
                r["normalized_score"] = 1.0 if max_score > 0 else 0.0
        else:
            for r in results:
                r["normalized_score"] = (r.get(score_key, 0) - min_score) / score_range

        return results

    @staticmethod
    def _merge_result(all_results: dict[str, dict[str, Any]], result: dict[str, Any]) -> None:
        """把单条检索结果融合进多查询结果集。

        以稳定标识（id/chunk_id/document_id，缺失时用内容摘要）识别同一文档；
        融合前复制结果，不修改 vector_db 返回的原始字典；
        合并时保留最高相关分数，并累计多查询命中次数。
        """
        key = _stable_result_key(result)
        existing = all_results.get(key)
        if existing is None:
            merged = dict(result)
            merged["query_count"] = 1
            all_results[key] = merged
            return
        for field in ("score", "fused_score"):
            values = [v for v in (existing.get(field), result.get(field)) if v is not None]
            if values:
                existing[field] = max(values)
        existing["query_count"] = existing.get("query_count", 1) + 1

    def retrieve_context(
        self,
        query: str,
        top_k: int | None = None,
        enable_rerank: bool = True,
        filters: dict[str, Any] | None = None,
        use_cache: bool = True,
        additional_queries: tuple[str, ...] = (),
        infer_filters: bool = True,
    ) -> list[dict[str, Any]]:
        """检索与查询最相关的知识库文档。

        流程：查询扩展 -> 多查询混合检索 -> 分数融合（含区域加权） -> Cross-Encoder重排 -> 分数归一化。
        支持查询缓存，重复查询直接返回缓存结果。

        Args:
            query: 用户查询文本
            top_k: 返回结果数量，默认使用self.top_k（5）
            enable_rerank: 是否启用重排器精排
            filters: 元数据过滤条件
            use_cache: 是否使用查询缓存

        Returns:
            按相关性降序排列的文档列表，每项包含normalized_score等字段
        """
        views = self._validated_task_views(query, additional_queries)
        if views:
            return self._task_candidate_plan(
                query, views, top_k=top_k, enable_rerank=enable_rerank,
                filters=filters, use_cache=use_cache,
            ).results


        # 先确定实际 filters（未显式传入时从查询中提取），再生成缓存键，
        # 确保缓存键包含所有真正影响检索结果的参数
        use_expansion = self.enable_multi_query and self.enable_query_expansion
        if use_expansion and not filters and infer_filters:
            filters = self.query_expander.extract_filters(query)
        search_filters = {k: v for k, v in (filters or {}).items() if k != "region"} or None

        vector_db = get_vector_db()
        vector_generation = vector_db.cache_generation
        serialized_filters = json.dumps(filters or {}, ensure_ascii=False, sort_keys=True, default=str)
        cache_key = (
            f"{query}|{top_k or self.top_k}|rerank={enable_rerank and self.enable_reranking}|"
            f"expansion={use_expansion}|threshold={self.min_score_threshold}|"
            f"recall_multiplier={self.recall_multiplier}|filters={serialized_filters}|"
            f"generation={vector_generation}|infer_filters={infer_filters}"
        )
        if use_cache:
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                logger.info("RAG缓存命中")
                return cached

        final_top_k = top_k or self.top_k
        all_results: dict[str, dict[str, Any]] = {}

        expanded_queries = self.query_expander.expand_query(query) if use_expansion else [query]
        for q in expanded_queries:
            recall_top_k = final_top_k * self.recall_multiplier
            recall_results = vector_db.hybrid_search(
                q, top_k=recall_top_k, threshold=self.min_score_threshold,
                keyword_weight=0.3, filters=search_filters,
            )
            for result in recall_results:
                self._merge_result(all_results, result)

        if not all_results:
            logger.info("RAG检索未找到相关文档")
            return []

        # 地区加权：如果查询指定了地区，提升包含该地区关键词的文档
        region_filter = (filters or {}).get("region", "")
        if region_filter and region_filter in self.query_expander.region_keywords:
            region_kws = self.query_expander.region_keywords[region_filter]
            for result in all_results.values():
                content = result.get("content", "") + result.get("title", "")
                if any(kw in content for kw in region_kws):
                    result["region_boost"] = 0.5
                else:
                    result["region_boost"] = -0.5  # 惩罚非目标地区文档

        for result in all_results.values():
            bonus = result.get("query_count", 1) * 0.05
            region_bonus = result.get("region_boost", 0)
            base_score = result.get("fused_score", result.get("score", 0)) or 0
            result["final_score"] = base_score + bonus + region_bonus

        # 确定性排序：final_score 降序，同分按稳定融合标识键升序
        recall_results = [
            item[1]
            for item in sorted(
                all_results.items(),
                key=lambda item: (-item[1]["final_score"], item[0]),
            )
        ]

        logger.info(f"第一阶段混合检索完成: {len(recall_results)} 个候选文档")

        if enable_rerank and self.enable_reranking:
            results = self.reranker.rerank(query, recall_results, top_k=final_top_k)
            if not results:
                raise RuntimeError("RAG reranker returned no results for non-empty candidates")
            results = self._normalize_scores(results, "rerank_score")
        else:
            results = self._normalize_scores(recall_results[:final_top_k], "final_score")
        logger.info("检索完成")
        if use_cache:
            self._add_to_cache(cache_key, results)
        return results

    @staticmethod
    def _validated_task_views(query: str, additional_queries: tuple[str, ...], *, question_binding=None) -> tuple[str, ...]:
        reference_names = ()
        if question_binding is not None:
            from knowledge.public_question_binding import bound_object_search_views

            reference_names = bound_object_search_views(question_binding, query)
        if (
            not isinstance(additional_queries, tuple) or len(additional_queries) > 4
            or any(
                not isinstance(view, str) or not view.strip() or len(view) > 1024
                or (view not in reference_names and any(span not in query for span in view.split()))
                for view in additional_queries
            )
        ):
            raise ValueError("Additional query views must be bounded literal input spans")
        return tuple(dict.fromkeys(view for view in additional_queries if view != query))

    def _task_candidate_plan(self, query, views, *, top_k, enable_rerank, filters, use_cache, threshold=0.3, question_binding=None):
        from knowledge.task_retrieval import collect_task_candidates

        resolved_filters = copy.deepcopy(filters)
        if self.enable_multi_query and self.enable_query_expansion and not resolved_filters:
            resolved_filters = self.query_expander.extract_filters(query)

        def snapshot():
            index = get_vector_db()
            return id(index), index.cache_generation

        def retrieve(question, *, top_k, filters):
            # Only the original question resolves implicit scope. A task view
            # cannot infer a new category or change an explicit KB filter.
            return self.retrieve_context(
                question, top_k=top_k, enable_rerank=enable_rerank,
                filters=filters, use_cache=use_cache, infer_filters=False,
            )

        return collect_task_candidates(
            query, views, top_k=top_k or self.top_k, filters=resolved_filters,
            retrieve=retrieve, confidence=self.compute_confidence,
            stable_key=_stable_result_key, snapshot=snapshot, threshold=threshold, question_binding=question_binding,
        )

    @staticmethod
    def _absolute_score(result: dict[str, Any]) -> float:
        """Return a cross-query comparable score instead of per-result-list normalization."""
        value = result.get("score", result.get("fused_score", result.get("final_score")))
        message = "RAG retrieval score must be a finite number"
        try:
            score = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(message) from exc
        if isinstance(value, bool) or not math.isfinite(score):
            raise ValueError(message)
        return max(0.0, min(1.0, score))

    def compute_confidence(self, results: list[dict[str, Any]]) -> float:
        """Estimate confidence from absolute retrieval scores, not min-max rank scores."""
        if not results:
            return 0.0
        scores = [self._absolute_score(result) for result in results[:3]]
        top_score = scores[0]
        support_score = sum(scores) / len(scores)
        return round(0.7 * top_score + 0.3 * support_score, 4)

    def build_citations(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """从检索结果构建引用列表，供前端展示证据来源。"""
        citations: list[dict[str, Any]] = []
        for r in results:
            content = r.get("content", "")
            citations.append(
                {
                    "source_id": str(r.get("id", r.get("chunk_id", ""))),
                    "source_title": r.get("title", r.get("original_title", "未知来源")),
                    "evidence_excerpt": content[:200] + ("..." if len(content) > 200 else ""),
                    "score": round(r.get("normalized_score", r.get("score", 0)), 4),
                    "kb_revision": r.get("kb_revision", ""),
                    "source_path": r.get("source_path", ""),
                    "source_line": r.get("source_line"),
                    "source_event_ids": list(r.get("source_event_ids", [])),
                    "source_lineage": list(r.get("source_lineage", [])),
                    "section": r.get("section", r.get("category", "")),
                    "version": r.get("version", "1.0"),
                }
            )
        return citations

    def should_abstain(self, confidence: float, threshold: float = 0.3) -> bool:
        """判断是否应弃答（置信度低于阈值时返回 True）。"""
        return confidence < threshold

    def retrieve_with_citations(
        self,
        query: str,
        top_k: int | None = None,
        threshold: float = 0.3,
        filters: dict[str, Any] | None = None,
        additional_queries: tuple[str, ...] = (),
        question_binding: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """检索并返回带引用和置信度的结构化结果。

        Returns:
            {results, citations, confidence, abstained}
        """
        views = self._validated_task_views(query, additional_queries, question_binding=question_binding)
        task_coverage = ()
        if views:
            plan = self._task_candidate_plan(
                query, views, top_k=top_k, enable_rerank=True,
                filters=filters, use_cache=True, threshold=threshold,
                **({"question_binding": question_binding} if question_binding is not None else {}),
            )
            results, task_coverage = plan.results, plan.coverage
        else:
            results = self.retrieve_context(query, top_k=top_k, filters=filters)
        confidence = self.compute_confidence(results)
        abstained = self.should_abstain(confidence, threshold)
        citations = self.build_citations(results) if not abstained else []
        return {
            "results": results,
            "citations": citations,
            "confidence": confidence,
            "abstained": abstained,
            **({"task_candidate_coverage": task_coverage} if views else {}),
        }

    def format_context_results(self, results: list[dict[str, Any]]) -> str:
        """把已经检索到的结果格式化为模型上下文，不再次执行检索。"""
        if not results:
            return ""

        context_parts = []
        total_length = 0
        for result in results:
            doc_content = result.get("content", "")
            doc_title = result.get("title", "")
            score = result.get("normalized_score", result.get("score", 0))
            doc_text = f"【相关文档: {doc_title}（相关度: {score:.2f}）】\n{doc_content}\n"

            if total_length + len(doc_text) > self.max_context_length:
                remaining = self.max_context_length - total_length
                if remaining > 100:
                    doc_text = f"【相关文档: {doc_title}（相关度: {score:.2f}）】\n{doc_content[:remaining]}...\n"
                    context_parts.append(doc_text)
                    total_length += len(doc_text)
                break

            context_parts.append(doc_text)
            total_length += len(doc_text)

        if not context_parts:
            return ""

        logger.info("构建RAG上下文，长度: %d，文档数: %d", total_length, len(context_parts))
        return "\n".join(context_parts)

    def build_context_prompt(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> str:
        """检索一次并构建模型上下文；已有结果应调用 format_context_results。"""
        results = self.retrieve_context(query, top_k, filters=filters)
        return self.format_context_results(results)

    def format_results_for_display(self, results: list[dict[str, Any]]) -> str:
        if not results:
            return ""

        parts = []
        for i, result in enumerate(results, 1):
            title = result.get("title", "无标题")
            content = result.get("content", "")
            score = result.get("normalized_score", result.get("score", 0))

            if len(content) > 512:
                content = content[:512] + "..."

            parts.append(f"{i}. {title} (相关度: {score:.2f})\n   {content}\n")

        return "\n".join(parts)

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._query_cache.clear()
        logger.info("RAG查询缓存已清除")


_rag_helper: RAGHelper | None = None


def get_rag_helper() -> RAGHelper:
    global _rag_helper
    if _rag_helper is None:
        _rag_helper = RAGHelper()
    return _rag_helper


def rag_retrieve(
    query: str, top_k: int = 10, enable_rerank: bool = True, filters: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """便捷函数：检索与查询相关的知识库文档。

    Args:
        query: 用户查询文本
        top_k: 返回结果数量，默认10
        enable_rerank: 是否启用重排，默认True
        filters: 元数据过滤条件

    Returns:
        按相关性降序排列的文档列表
    """
    helper = get_rag_helper()
    return helper.retrieve_context(query, top_k, enable_rerank, filters=filters)


def rag_build_prompt(query: str, top_k: int = 10, filters: dict[str, Any] | None = None) -> str:
    """便捷函数：检索知识库并构建上下文提示文本。

    Args:
        query: 用户查询, top_k: 检索文档数量，默认10
        filters: 元数据过滤条件

    Returns:
        格式化后的上下文字符串，无匹配文档时返回空字符串
    """
    logger.info(f"rag_build_prompt被调用: query={query}, top_k={top_k}")
    helper = get_rag_helper()
    result = helper.build_context_prompt(query, top_k, filters=filters)
    logger.info(f"rag_build_prompt返回: 长度={len(result)}")
    return result
