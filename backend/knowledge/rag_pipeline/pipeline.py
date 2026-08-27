"""RAG 管线编排（P6）。

单次检索流程：
query → QueryAnalyzer（实体/别名/叙事层意图/域选择）
      → HybridRetriever（多域：sparse + vector + entity + RRF 融合）
      → PipelineReranker（CrossEncoder 或确定性降级）
      → ContextBuilder（预算内组装 + 引用）
      → RetrievalBundle（兼容现有 generate.py 的 bundle 契约）

索引不可用/未命中域/查询为空 → 清晰降级（None 或空结果），
不导致上层聊天失败。
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

from .context import ContextBuilder, RetrievedContext
from .index import DomainIndex
from .query import QueryAnalyzer
from .rerank import PipelineReranker
from .retrieval import HybridRetriever

if TYPE_CHECKING:
    from .embedding import EmbeddingProvider
    from .query import QueryAnalysis
    from .registry import DomainRegistry, KnowledgeDomainConfig
    from .retrieval import RetrievalCandidate

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5
DEFAULT_ABSTAIN_THRESHOLD = 0.25


class DomainRuntime:
    """单域运行时：配置 + 已加载索引 + 检索器。"""

    def __init__(
        self,
        config: KnowledgeDomainConfig,
        index: DomainIndex,
        embedding_provider: EmbeddingProvider,
    ):
        self.config = config
        self.index = index
        self.retriever = HybridRetriever(config, index, embedding_provider)


class RagPipeline:
    """跨域统一检索管线。"""

    def __init__(
        self,
        registry: DomainRegistry,
        embedding_provider: EmbeddingProvider,
        reranker: PipelineReranker | None = None,
        context_builder: ContextBuilder | None = None,
        abstain_threshold: float = DEFAULT_ABSTAIN_THRESHOLD,
    ):
        self.registry = registry
        self.embedding_provider = embedding_provider
        self.reranker = reranker or PipelineReranker()
        self.context_builder = context_builder or ContextBuilder()
        self.abstain_threshold = abstain_threshold
        self._query_analyzer = QueryAnalyzer(registry.list_domains())
        self._runtimes: dict[str, DomainRuntime] = {}
        self._load_attempted = False
        self._available = False
        self._load_lock = threading.Lock()

    # -- 索引加载 -----------------------------------------------------------
    def load_indexes(self, force: bool = False) -> bool:
        """加载所有启用域的持久化索引；至少一域成功即 available。

        线程安全：后台预热与首次业务检索并发时，后来者阻塞等待
        加载完成而不是读到中间态。
        """
        with self._load_lock:
            if self._load_attempted and not force:
                return self._available
            self._runtimes = {}
            loaded_any = False
            for config in self.registry.list_domains(enabled_only=True):
                index = DomainIndex(
                    domain_id=config.domain_id,
                    index_root=config.resolve_index_root(),
                    dimension=self.embedding_provider.dimension,
                )
                if index.load() and index.count() > 0:
                    self._runtimes[config.domain_id] = DomainRuntime(config, index, self.embedding_provider)
                    loaded_any = True
                else:
                    logger.warning(
                        "知识域 %s 索引不可用（%s）——该域检索降级",
                        config.domain_id,
                        config.resolve_index_root(),
                    )
            # 加载完成后才置位，避免并发调用读到"已尝试但未完成"的中间态
            self._load_attempted = True
            self._available = loaded_any
            if loaded_any:
                logger.info("P6 管线就绪，已加载域: %s", list(self._runtimes.keys()))
            return loaded_any

    def is_available(self) -> bool:
        return self._available and bool(self._runtimes)

    def domain_stats(self) -> dict[str, Any]:
        return {domain_id: runtime.index.stats() for domain_id, runtime in self._runtimes.items()}

    def warmup_embedding(self) -> bool:
        """预加载 embedding 模型（后台/启动时调用，避免首查超时）。"""
        try:
            self.embedding_provider.embed_query("预热查询")
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("embedding 预热失败: %s", e)
            return False

    # -- 检索 ---------------------------------------------------------------
    def analyze(self, query: str, domain_id: str | None = None) -> QueryAnalysis:
        return self._query_analyzer.analyze(query, domain_id=domain_id)

    def _select_domains(
        self,
        analysis: QueryAnalysis,
        domain_id: str | None,
    ) -> list[str]:
        if domain_id:
            if domain_id not in self._runtimes:
                return []
            return [domain_id]
        return [did for did in analysis.matched_domains if did in self._runtimes]

    def retrieve(
        self,
        query: str,
        domain_id: str | None = None,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        mode: str = "hybrid",
        use_rerank: bool = True,
        use_context: bool = True,
    ) -> dict[str, Any] | None:
        """执行一次完整检索。

        Returns:
            RetrievalBundle dict（results/citations/confidence/abstained/
            context_text/domains/query_analysis）；
            未命中任何启用域（auto 选择无实体信号）时返回 None，
            调用方应回退到既有检索链路。
        """
        if not self.is_available():
            self.load_indexes()
            if not self.is_available():
                return None

        analysis = self.analyze(query, domain_id=domain_id)
        selected = self._select_domains(analysis, domain_id)
        if not selected:
            return None

        final_top_k = top_k or DEFAULT_TOP_K
        merged: list[RetrievalCandidate] = []
        for did in selected:
            runtime = self._runtimes[did]
            # 召回窗口显著大于最终 top_k：混合排名靠后的精确命中
            # （如关系卡）需要进入重排阶段才能被特征救回
            per_domain_k = max(final_top_k * 4, 30)
            candidates = runtime.retriever.search(analysis, top_k=per_domain_k, filters=filters, mode=mode)
            merged.extend(candidates)

        merged.sort(key=lambda c: (-c.fused_score, c.row))

        cross_encoder_used = False
        if use_rerank and merged:
            rerank_result = self.reranker.rerank(analysis, merged, top_k=max(final_top_k * 2, final_top_k))
            top_candidates = rerank_result[:final_top_k]
            cross_encoder_used = self.reranker.uses_cross_encoder
        else:
            top_candidates = merged[:final_top_k]

        context: RetrievedContext | None = None
        if use_context:
            context = self.context_builder.build(
                analysis, top_candidates, selected, cross_encoder_scores=cross_encoder_used
            )

        results = []
        for rank, candidate in enumerate(top_candidates, 1):
            item = candidate.to_dict()
            item["retrieval_rank"] = rank
            results.append(item)

        confidence = context.confidence if context else self._fallback_confidence(top_candidates)
        abstained = confidence < self.abstain_threshold or not results

        return {
            "results": results,
            "citations": context.citations if context else [],
            "confidence": round(confidence, 4),
            "abstained": abstained,
            "context_text": context.context_text if context else "",
            "domains": selected,
            "query_analysis": {
                "entities": analysis.entities,
                "matched_domains": analysis.matched_domains,
                "doc_type_preferences": analysis.doc_type_preferences,
                "reality_preferences": analysis.reality_preferences,
                "story_hits": analysis.story_hits,
                "domain_hit_reasons": analysis.domain_hit_reasons,
            },
        }

    @staticmethod
    def _fallback_confidence(candidates: list[RetrievalCandidate]) -> float:
        """未使用重排时的近似置信度（RRF 分数量级放大归一）。"""
        if not candidates:
            return 0.0
        return min(1.0, candidates[0].fused_score * 8.0)
