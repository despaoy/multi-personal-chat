"""纠正性 RAG - 低置信度时重写查询并重试检索，仍低则弃答。

遵循路线图 guardrail：
- 默认重试次数限制为 1 次（max_retries=1），由构造参数真实控制
- 由环境变量 CORRECTIVE_RAG_ENABLED 控制（默认 false，生产/实验分离）
- 复用 RAGHelper 的 retrieve_with_citations / compute_confidence / should_abstain
- 查询重写仅基于关键词提取（优先 jieba，缺失时确定性回退），不调用 LLM 或外部服务
"""

from __future__ import annotations

import logging
import re
from typing import Any

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

try:
    import jieba

    _JIEBA_AVAILABLE = True
except ImportError:  # pragma: no cover - 测试环境强制走回退分支
    jieba = None
    _JIEBA_AVAILABLE = False


def _tokenize(text: str) -> list[str]:
    """中文分词：优先项目已有的 jieba；不可用时回退为确定性的
    中文连续片段 + 英文单词切分。"""
    if _JIEBA_AVAILABLE:
        tokens = [t.strip() for t in jieba.cut(text)]
        return [t for t in tokens if t]
    return re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+", text)


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
            for tok in _tokenize(text):
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

        def _record(current_query: str, response: dict[str, Any]) -> None:
            rounds.append(
                {
                    "query": current_query,
                    "confidence": response.get("confidence", 0.0),
                    "abstained": bool(response.get("abstained", False)),
                }
            )

        current_query = query
        response = self.rag_helper.retrieve_with_citations(
            current_query, top_k=top_k, threshold=self.threshold, filters=filters
        )
        _record(current_query, response)

        if not response.get("abstained", False):
            # 置信度足够，直接返回
            return {
                **response,
                "reformulated": False,
                "original_query": query,
                "reformulated_query": None,
                "rounds": rounds,
            }

        # 低置信度，尝试重写查询重试（max_retries 真实控制重试轮数）
        logger.info(f"纠正性RAG: 首次置信度 {response.get('confidence', 0.0)} < {self.threshold}，尝试查询重写")
        reformulated_query: str | None = None

        for attempt in range(self.max_retries):
            candidate = self.reformulate_query(current_query, response.get("results", []))
            if candidate == current_query:
                # 重写查询没有新增信息，不做无意义的第二次检索
                logger.info("纠正性RAG: 重写查询无新增信息，停止重试")
                break
            reformulated_query = candidate
            current_query = candidate
            logger.info(f"纠正性RAG: 第{attempt + 1}次重写查询 -> {current_query}")

            response = self.rag_helper.retrieve_with_citations(
                current_query, top_k=top_k, threshold=self.threshold, filters=filters
            )
            _record(current_query, response)

            if not response.get("abstained", False):
                logger.info(f"纠正性RAG: 重写后置信度 {response.get('confidence', 0.0)} >= {self.threshold}，成功")
                return {
                    **response,
                    "reformulated": True,
                    "original_query": query,
                    "reformulated_query": reformulated_query,
                    "rounds": rounds,
                }

        # 重试耗尽或无有效重写，弃答
        final_confidence = rounds[-1]["confidence"] if rounds else 0.0
        logger.info(f"纠正性RAG: 最终置信度 {final_confidence} 仍低，弃答")
        return {
            "results": [],
            "citations": [],
            "confidence": final_confidence,
            "abstained": True,
            "reformulated": reformulated_query is not None,
            "original_query": query,
            "reformulated_query": reformulated_query,
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
