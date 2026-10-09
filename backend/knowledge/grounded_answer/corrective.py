"""Corrective RAG 适配（与通用知识库 CorrectiveRAG 协调，不重写）。

与既有 knowledge/corrective_rag.py 的关系：
- 既有 CorrectiveRAG 服务 legacy RAGHelper 链路（api/generate
  回退分支），保持不动
- 本模块服务结构化检索 bundle：首次检索不足 → 关键词改写
  （jieba，无 LLM）→ 二次检索 → 证据合并 → 最终 abstention

guardrail（沿用既有路线图约束）：
- max_retries 真实控制重试轮数（默认 1），无无限重试
- 重写无新增信息时提前停止
- 合并去重（document id），citations 契约不变（单一 bundle 形态）
- 不改变用户原意：只在原查询后追加检索关键词
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from ..query_reformulation import append_retrieval_keywords

logger = logging.getLogger(__name__)


# 检索函数协议：query, top_k, filters → bundle | None
RetrieveFn = Callable[..., Any]


class CorrectiveRetrievalAdapter:
    """结构化 bundle 的纠正性检索：改写 → 重试 → 合并。"""

    def __init__(self, retrieve: RetrieveFn, max_retries: int = 1):
        self._retrieve = retrieve
        if type(max_retries) is not int or max_retries < 0:
            raise ValueError("CorrectiveRetrievalAdapter.max_retries must be a non-negative integer")
        self.max_retries = max_retries

    def reformulate_query(self, query: str, bundle: dict[str, Any]) -> str:
        """从 top 结果提取关键词追加到原查询（不改变原意）。"""
        return append_retrieval_keywords(query, (
            "{} {}".format(result.get("title") or "", result.get("summary") or "")
            for result in (bundle.get("results") or [])[:3]
        ))

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
        bundle = None
        current_query = query
        for attempt in range(self.max_retries + 1):
            current = self._retrieve(current_query, top_k=top_k, filters=filters)
            info["rounds"].append({
                "query": current_query,
                "confidence": current["confidence"] if current is not None else 0.0,
                "abstained": current["abstained"] if current is not None else True,
            })
            if current is None:
                # A rewritten query outside the domain cannot replace its last in-domain evidence.
                break
            if not current["abstained"]:
                bundle = self._merge_bundles(bundle, current) if bundle is not None else current
                break
            bundle = current
            if attempt == self.max_retries:
                break
            candidate = self.reformulate_query(current_query, bundle)
            if candidate == current_query:
                logger.info("Grounded Answer corrective: 重写查询无新增信息，停止重试")
                break
            current_query = candidate
            info["reformulated"] = True
            info["reformulated_query"] = candidate
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
