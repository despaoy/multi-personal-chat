"""P7 corrective RAG 适配（与既有 CorrectiveRAG 协调，不重写）。

与既有 knowledge/corrective_rag.py 的关系：
- 既有 CorrectiveRAG 服务 legacy RAGHelper 链路（api/generate
  回退分支），保持不动
- 本模块服务 P6 pipeline bundle：首次检索不足 → 关键词改写
  （jieba，无 LLM）→ 二次检索 → 证据合并 → 最终 abstention

guardrail（沿用既有路线图约束）：
- max_retries 真实控制重试轮数（默认 1），无无限重试
- 重写无新增信息时提前停止
- 合并去重（document id），citations 契约不变（单一 bundle 形态）
- 不改变用户原意：只在原查询后追加检索关键词
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


# 与既有 CorrectiveRAG 相同的停用词（避免两处漂移，直接对齐）
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
except ImportError:  # pragma: no cover - 测试环境无 jieba 时走确定性回退
    jieba = None
    _JIEBA_AVAILABLE = False


def _tokenize(text: str) -> list[str]:
    if _JIEBA_AVAILABLE:
        tokens = [t.strip() for t in jieba.cut(text)]
        return [t for t in tokens if t]
    return re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+", text)


# 检索函数协议：query, top_k, filters → bundle | None
RetrieveFn = Callable[..., Any]


class CorrectiveRetrievalAdapter:
    """P6 bundle 的纠正性检索：改写 → 重试 → 合并。"""

    def __init__(self, retrieve: RetrieveFn, max_retries: int = 1):
        self._retrieve = retrieve
        self.max_retries = max(0, int(max_retries))

    def reformulate_query(self, query: str, bundle: dict[str, Any]) -> str:
        """从 top 结果提取关键词追加到原查询（不改变原意）。"""
        keywords: list[str] = []
        query_folded = query.casefold()
        for result in (bundle.get("results") or [])[:3]:
            text = "{} {}".format(result.get("title") or "", result.get("summary") or "")
            for token in _tokenize(text):
                if (
                    len(token) > 1
                    and token not in keywords
                    and token not in _STOPWORDS
                    and token.casefold() not in query_folded
                ):
                    keywords.append(token)
            if len(keywords) >= 8:
                break
        if not keywords:
            return query
        return f"{query} {' '.join(keywords[:5])}"

    def retrieve_with_correction(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """返回 (最终 bundle, 改写信息)。

        bundle 为 None 表示未命中域（调用方回退）；
        改写信息记录 rounds / reformulated_query，进 retrieval_metadata。
        """
        info: dict[str, Any] = {"reformulated": False, "reformulated_query": None, "rounds": []}
        bundle = self._retrieve(query, top_k=top_k, filters=filters)

        def _record(current: str, b: dict[str, Any] | None) -> None:
            info["rounds"].append(
                {
                    "query": current,
                    "confidence": float(b.get("confidence") or 0.0) if b else 0.0,
                    "abstained": bool(b.get("abstained")) if b else True,
                }
            )

        _record(query, bundle)

        if bundle is None or not bundle.get("abstained"):
            return bundle, info

        current_query = query
        for _attempt in range(self.max_retries):
            candidate = self.reformulate_query(current_query, bundle or {})
            if candidate == current_query:
                logger.info("P7 corrective: 重写查询无新增信息，停止重试")
                break
            current_query = candidate
            retry_bundle = self._retrieve(current_query, top_k=top_k, filters=filters)
            _record(current_query, retry_bundle)

            if retry_bundle is None:
                # 改写后未命中域：保留首轮结果（域门控更可信）
                break
            if not retry_bundle.get("abstained"):
                info["reformulated"] = True
                info["reformulated_query"] = candidate
                merged = self._merge_bundles(bundle, retry_bundle)
                return merged, info
            bundle = retry_bundle

        logger.info("P7 corrective: 重试后仍低置信度，交由 abstention 策略裁决")
        return bundle, info

    @staticmethod
    def _merge_bundles(primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
        """按 document id 去重合并两轮证据（主轮在前，保排名稳定）。"""
        merged_results: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for original in list(primary.get("results") or []) + list(secondary.get("results") or []):
            result = dict(original)
            doc_id = str(result.get("id"))
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            merged_results.append(result)

        citation_by_id: dict[str, dict[str, Any]] = {}
        for citation in list(primary.get("citations") or []) + list(secondary.get("citations") or []):
            cite_id = str(citation.get("source_id"))
            citation_by_id.setdefault(cite_id, dict(citation))

        # 重排名次重排（合并后按融合分数稳定排序）
        merged_results.sort(key=lambda r: (-float(r.get("fused_score") or 0.0), str(r.get("id"))))
        for rank, result in enumerate(merged_results, 1):
            result["retrieval_rank"] = rank

        # citation 顺序必须跟随合并后的检索排名，EvidencePacket 才能稳定
        # 分配 S1..Sn，而不是沿用两轮调用的偶然拼接顺序。
        merged_citations = [
            citation_by_id[doc_id] for result in merged_results if (doc_id := str(result.get("id"))) in citation_by_id
        ]

        return {
            **secondary,  # 以改写轮 bundle 为基础（其 confidence 已达标）
            "results": merged_results,
            "citations": merged_citations,
        }
