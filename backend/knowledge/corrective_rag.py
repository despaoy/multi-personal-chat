"""纠正性 RAG - 低置信度时重写查询并重试检索，仍低则弃答。

遵循路线图 guardrail：
- 默认重试次数限制为 1 次（max_retries=1），由构造参数真实控制
- 由环境变量 CORRECTIVE_RAG_ENABLED 控制（默认 false，生产/实验分离）
- 复用 RAGHelper 的 retrieve_with_citations / compute_confidence / should_abstain
- 查询重写仅基于关键词提取（使用必需的 jieba 分词），不调用 LLM 或外部服务
"""

from __future__ import annotations

import logging
from typing import Any

from .retrieval_core.tokenization import segment

logger = logging.getLogger(__name__)

# 简易中文停用词表（用于查询重写时去停用词）
_STOPWORDS = {
    "的",
    "了",
    "是",
    "在",
    "我",
    "你",
    "他",
    "她",
    "它",
    "们",
    "这",
    "那",
    "怎么",
    "什么",
    "为什么",
    "哪里",
    "哪个",
    "请问",
    "一下",
    "可能",
    "应该",
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "what",
    "how",
    "why",
}

class CorrectiveRAG:
    """纠正性 RAG：retrieve → confidence check → reformulate → re-retrieve → abstain。

    流程：
    1. 首次检索 retrieve_with_citations
    2. 若置信度低于阈值，从 top 结果提取关键词重写查询并重试（最多 max_retries 轮）
    3. 重写没有新增信息时提前停止，不做无意义的重复检索
    4. 若重试后仍低于阈值，弃答
    """

    def __init__(self, rag_helper, threshold: float = 0.3, max_retries: int = 1):
        self.rag_helper = rag_helper
        self.threshold = threshold
        self.max_retries = max(0, int(max_retries))

    def reformulate_query(self, query: str, top_results: list[dict[str, Any]]) -> str:
        """从 top 结果提取关键词，追加到原查询形成重写查询。"""
        keywords: list[str] = []
        query_folded = query.casefold()
        for result in top_results[:3]:
            content = result.get("content", "")
            title = result.get("title", "")
            text = f"{title} {content}"
            for tok in segment(text):
                if (
                    len(tok) > 1
                    and tok not in keywords
                    and tok not in _STOPWORDS
                    and tok.casefold() not in query_folded
                ):
                    keywords.append(tok)
            if len(keywords) >= 8:
                break

        if not keywords:
            return query

        # 追加最多 5 个关键词到原查询
        extra = " ".join(keywords[:5])
        return f"{query} {extra}"

    def retrieve_with_correction(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """纠正性检索：首次检索 → 低置信度则重写重试（最多 max_retries 轮）→ 仍低则弃答。

        Returns:
            {results, citations, confidence, abstained, reformulated,
             original_query, reformulated_query, rounds}
            rounds 记录每轮使用的 query、confidence 和 abstained 结果。
        """
        rounds: list[dict[str, Any]] = []
        current_query = query
        for attempt in range(self.max_retries + 1):
            response = self.rag_helper.retrieve_with_citations(
                current_query, top_k=top_k, threshold=self.threshold, filters=filters
            )
            rounds.append({
                "query": current_query,
                "confidence": response["confidence"],
                "abstained": response["abstained"],
            })
            if not response["abstained"] or attempt == self.max_retries:
                break
            candidate = self.reformulate_query(current_query, response["results"])
            if candidate == current_query:
                logger.info("纠正性RAG: 重写查询无新增信息，停止重试")
                break
            current_query = candidate
            logger.info("纠正性RAG: 执行第%d次查询重写", attempt + 1)

        if response["abstained"]:
            response = {
                "results": [],
                "citations": [],
                "confidence": response["confidence"],
                "abstained": True,
            }
        reformulated = current_query != query
        return {
            **response,
            "reformulated": reformulated,
            "original_query": query,
            "reformulated_query": current_query if reformulated else None,
            "rounds": rounds,
        }


_corrective_rag: CorrectiveRAG | None = None


def get_corrective_rag(threshold: float = 0.3, max_retries: int = 1) -> CorrectiveRAG:
    """获取 CorrectiveRAG 单例。"""
    global _corrective_rag
    if _corrective_rag is None:
        from .rag_helper import get_rag_helper

        _corrective_rag = CorrectiveRAG(get_rag_helper(), threshold=threshold, max_retries=max_retries)
    return _corrective_rag
