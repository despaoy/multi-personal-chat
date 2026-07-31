"""
RAG辅助工具模块 - 优化版
作为向量数据库和重排器的上层封装，提供查询扩展、多查询融合召回、两阶段检索（粗排+精排）
以及上下文构建等一站式RAG服务。
改进：查询扩展、元数据过滤、分数归一化、查询缓存、多查询检索
"""

import copy
import json
import logging
import os
import time
import re
from collections import OrderedDict
from threading import RLock
from typing import List, Dict, Any, Optional

try:
    from nonebot.log import logger
except ImportError:
    logger = logging.getLogger(__name__)

VECTOR_DB_AVAILABLE = False
_vector_db = None

try:
    from .vector_db import get_vector_db
    VECTOR_DB_AVAILABLE = True
    logger.info("RAG辅助工具: 向量数据库模块加载成功")
except ImportError as e:
    logger.warning(f"RAG辅助工具: 向量数据库模块不可用: {e}")

RERANKER_AVAILABLE = False
try:
    from .reranker import get_reranker
    RERANKER_AVAILABLE = True
    logger.info("RAG辅助工具: Cross-Encoder重排器模块加载成功")
except ImportError as e:
    logger.warning(f"RAG辅助工具: Cross-Encoder重排器模块不可用: {e}")


class QueryExpander:
    """查询扩展器，通过对原神领域同义词、区域关键词、领域关键词的替换和扩展，
    生成多个变体查询以提高向量检索的召回率。"""

    def __init__(self):
        """初始化查询扩展器，加载原神领域同义词表、区域关键词表和领域关键词表。"""
        self.synonym_map = {
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
        }

        self.region_keywords = {
            "璃月": ["璃月", "岩", "摩拉克斯", "七星", "千岩军", "奥赛尔", "漩涡之魔神"],
            "蒙德": ["蒙德", "风", "巴巴托斯", "特瓦林", "骑士团", "西风"],
            "稻妻": ["稻妻", "雷", "雷电将军", "眼狩令", "幕府"],
            "须弥": ["须弥", "草", "大慈树王", "小吉祥草王", "世界树", "虚空"],
        }

        self.domain_keywords = {
            "角色": ["角色", "人物", "hero", "character"],
            "武器": ["武器", "weapon", "剑", "弓", "法器", "长柄", "双手剑"],
            "玩法": ["玩法", "攻略", "怎么打", "如何打", "技巧"],
            "剧情": ["剧情", "故事", "任务", "传说", "经历了", "冒险"],
            "系统": ["系统", "机制", "规则", "怎么算"],
        }

    def expand_query(self, query: str) -> List[str]:
        """对用户查询进行扩展，生成多个变体查询。

        依次进行同义词替换、领域关键词追加、区域限定查询生成。

        Args:
            query: 原始用户查询

        Returns:
            最多5个扩展查询变体列表，包含原始查询
        """
        expanded = [query]

        for key, synonyms in self.synonym_map.items():
            if key in query:
                for syn in synonyms:
                    expanded_query = query.replace(key, syn)
                    if expanded_query != query:
                        expanded.append(expanded_query)

        for domain, keywords in self.domain_keywords.items():
            for kw in keywords:
                if kw in query:
                    domain_query = f"{domain} {query}"
                    if domain_query not in expanded:
                        expanded.append(domain_query)
                    break

        # 添加区域限定查询 - 如果提到具体地区则精确检索
        for region, region_kws in self.region_keywords.items():
            for kw in region_kws:
                if kw in query:
                    region_query = f"{region} 剧情 {query}"
                    if region_query not in expanded:
                        expanded.insert(0, region_query)  # 优先检索
                    break

        return expanded[:5]

    def extract_filters(self, query: str) -> Dict[str, Any]:
        """从查询中提取元数据过滤条件，如类别和区域。

        Args:
            query: 用户查询

        Returns:
            过滤条件字典，可包含category和region字段
        """
        filters = {}

        category_map = {
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
        }

        for cn_name, en_category in category_map.items():
            if cn_name in query:
                filters["category"] = en_category
                break

        # 检测地区关键词，用于后过滤
        for region in ["璃月", "蒙德", "稻妻", "须弥"]:
            if region in query:
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
        self.use_vector_db = VECTOR_DB_AVAILABLE
        self.min_score_threshold = 0.05
        self.max_context_length = 2000
        self.top_k = 5

        self.enable_reranking = RERANKER_AVAILABLE
        self.reranker = None
        self.enable_reranking = self.enable_reranking and os.getenv("RERANKER_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
        self.recall_multiplier = 4
        self.rerank_top_k = 5

        self.query_expander = QueryExpander()
        self.enable_query_expansion = True
        self.enable_multi_query = True

        self._query_cache: OrderedDict[str, tuple[float, List[Dict[str, Any]]]] = OrderedDict()
        self._cache_lock = RLock()
        self._cache_max_size = 100
        self._cache_ttl = max(1, int(os.getenv("RAG_RETRIEVAL_CACHE_TTL", "60")))

        if self.enable_reranking:
            try:
                self.reranker = get_reranker()
                logger.info("RAGHelper: Cross-Encoder重排器初始化成功")
            except Exception as e:
                logger.error(f"RAGHelper: 重排器初始化失败: {e}")
                self.enable_reranking = False

    def _get_from_cache(self, query: str) -> Optional[List[Dict[str, Any]]]:
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

    def _add_to_cache(self, query: str, results: List[Dict[str, Any]]) -> None:
        with self._cache_lock:
            self._query_cache[query] = (
                time.monotonic() + self._cache_ttl,
                copy.deepcopy(results),
            )
            self._query_cache.move_to_end(query)
            while len(self._query_cache) > self._cache_max_size:
                self._query_cache.popitem(last=False)

    def _normalize_scores(self, results: List[Dict[str, Any]], score_key: str = "score") -> List[Dict[str, Any]]:
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

    def retrieve_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        enable_rerank: bool = True,
        filters: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
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
        if not self.use_vector_db:
            return []

        try:
            start_time = time.time()

            vector_generation = get_vector_db().cache_generation
            serialized_filters = json.dumps(
                filters or {}, ensure_ascii=False, sort_keys=True, default=str
            )
            cache_key = (
                f"{query}|{top_k or self.top_k}|rerank={enable_rerank}|"
                f"filters={serialized_filters}|generation={vector_generation}"
            )
            if use_cache:
                cached = self._get_from_cache(cache_key)
                if cached is not None:
                    logger.info(f"RAG缓存命中: {query}")
                    return cached

            final_top_k = top_k or self.top_k
            all_results: Dict[str, Dict[str, Any]] = {}

            if self.enable_multi_query and self.enable_query_expansion:
                expanded_queries = self.query_expander.expand_query(query)
                if not filters:
                    filters = self.query_expander.extract_filters(query)

                search_filters = {k: v for k, v in filters.items() if k != "region"} if filters else None

                for q in expanded_queries:
                    recall_top_k = final_top_k * self.recall_multiplier
                    vector_db = get_vector_db()
                    recall_results = vector_db.hybrid_search(
                        q,
                        top_k=recall_top_k,
                        threshold=self.min_score_threshold,
                        keyword_weight=0.3,
                        filters=search_filters,
                    )

                    for result in recall_results:
                        content = result.get("content", "")
                        title = result.get("title", "")
                        doc_key = f"{title}_{content[:100]}"
                        if doc_key in all_results:
                            existing = all_results[doc_key]
                            existing["score"] = max(existing.get("score", 0), result.get("score", 0))
                            existing["fused_score"] = max(existing.get("fused_score", 0), result.get("fused_score", 0))
                            if "query_count" in existing:
                                existing["query_count"] += 1
                            else:
                                existing["query_count"] = 2
                        else:
                            result["query_count"] = 1
                            all_results[doc_key] = result
            else:
                recall_top_k = final_top_k * self.recall_multiplier
                vector_db = get_vector_db()
                search_filters = {k: v for k, v in (filters or {}).items() if k != "region"} or None
                recall_results = vector_db.hybrid_search(
                    query,
                    top_k=recall_top_k,
                    threshold=self.min_score_threshold,
                    keyword_weight=0.3,
                    filters=search_filters,
                )
                for result in recall_results:
                    content = result.get("content", "")
                    title = result.get("title", "")
                    doc_key = f"{title}_{content[:100]}"
                    all_results[doc_key] = result

            recall_results = list(all_results.values())

            if not recall_results:
                logger.info("RAG检索未找到相关文档")
                return []

            # 地区加权：如果查询指定了地区，提升包含该地区关键词的文档
            region_filter = (filters or {}).get("region", "")
            if region_filter and region_filter in self.query_expander.region_keywords:
                region_kws = self.query_expander.region_keywords[region_filter]
                for result in recall_results:
                    content = result.get("content", "") + result.get("title", "")
                    if any(kw in content for kw in region_kws):
                        result["region_boost"] = 0.5
                    else:
                        result["region_boost"] = -0.5  # 惩罚非目标地区文档

            for result in recall_results:
                bonus = result.get("query_count", 1) * 0.05
                region_bonus = result.get("region_boost", 0)
                base_score = result.get("fused_score", result.get("score", 0))
                result["final_score"] = base_score + bonus + region_bonus

            recall_results.sort(key=lambda x: x.get("final_score", 0), reverse=True)

            logger.info(f"第一阶段混合检索完成: {len(recall_results)} 个候选文档")

            if enable_rerank and self.enable_reranking and self.reranker and len(recall_results) > 1:
                try:
                    rerank_start = time.time()
                    reranked_results = self.reranker.rerank(query, recall_results, top_k=final_top_k)
                    rerank_time = time.time() - rerank_start

                    if reranked_results and len(reranked_results) > 0:
                        reranked_results = self._normalize_scores(reranked_results)
                        logger.info(f"第二阶段重排完成: {len(recall_results)} -> {len(reranked_results)} 文档, "
                                    f"重排耗时={rerank_time:.3f}s")
                        total_time = time.time() - start_time
                        logger.info(f"两阶段检索总耗时: {total_time:.3f}s")

                        if use_cache:
                            self._add_to_cache(cache_key, reranked_results)
                        return reranked_results
                    else:
                        logger.warning("重排返回空结果，使用原始召回结果")
                except Exception as e:
                    logger.error(f"重排失败，使用原始召回结果: {e}")

            results = recall_results[:final_top_k]
            results = self._normalize_scores(results, "final_score")
            total_time = time.time() - start_time
            logger.info(f"检索完成（未使用重排）: {len(results)} 个文档, 耗时={total_time:.3f}s")

            if use_cache:
                self._add_to_cache(cache_key, results)
            return results

        except Exception as e:
            logger.error(f"RAG检索失败: {e}")
            return []

    @staticmethod
    def _absolute_score(result: Dict[str, Any]) -> float:
        """Return a cross-query comparable score instead of per-result-list normalization."""
        value = result.get("score", result.get("fused_score", result.get("final_score", 0.0)))
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    def compute_confidence(self, results: List[Dict[str, Any]]) -> float:
        """Estimate confidence from absolute retrieval scores, not min-max rank scores."""
        if not results:
            return 0.0
        scores = [self._absolute_score(result) for result in results[:3]]
        top_score = scores[0]
        support_score = sum(scores) / len(scores)
        return round(0.7 * top_score + 0.3 * support_score, 4)

    def build_citations(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """从检索结果构建引用列表，供前端展示证据来源。"""
        citations: List[Dict[str, Any]] = []
        for r in results:
            content = r.get("content", "")
            citations.append({
                "source_id": str(r.get("id", r.get("chunk_id", ""))),
                "source_title": r.get("title", r.get("original_title", "未知来源")),
                "evidence_excerpt": content[:200] + ("..." if len(content) > 200 else ""),
                "score": round(r.get("normalized_score", r.get("score", 0)), 4),
                "content_hash": r.get("content_hash", ""),
                "kb_revision": r.get("kb_revision", ""),
                "section": r.get("section", r.get("category", "")),
                "version": r.get("version", "1.0"),
            })
        return citations

    def should_abstain(self, confidence: float, threshold: float = 0.3) -> bool:
        """判断是否应弃答（置信度低于阈值时返回 True）。"""
        return confidence < threshold

    def retrieve_with_citations(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: float = 0.3,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """检索并返回带引用和置信度的结构化结果。

        Returns:
            {results, citations, confidence, abstained}
        """
        results = self.retrieve_context(query, top_k=top_k, filters=filters)
        confidence = self.compute_confidence(results)
        abstained = self.should_abstain(confidence, threshold)
        citations = self.build_citations(results) if not abstained else []
        return {
            "results": results,
            "citations": citations,
            "confidence": confidence,
            "abstained": abstained,
        }

    def format_context_results(self, results: List[Dict[str, Any]]) -> str:
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
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """检索一次并构建模型上下文；已有结果应调用 format_context_results。"""
        results = self.retrieve_context(query, top_k, filters=filters)
        return self.format_context_results(results)
    def format_results_for_display(self, results: List[Dict[str, Any]]) -> str:
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


_rag_helper: Optional[RAGHelper] = None


def get_rag_helper() -> RAGHelper:
    global _rag_helper
    if _rag_helper is None:
        _rag_helper = RAGHelper()
    return _rag_helper


def rag_retrieve(query: str, top_k: int = 10, enable_rerank: bool = True,
                 filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
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


def rag_build_prompt(query: str, top_k: int = 10,
                     filters: Optional[Dict[str, Any]] = None) -> str:
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
