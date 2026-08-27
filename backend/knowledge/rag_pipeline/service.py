"""P6 管线服务门面：业务接入的稳定入口。

- 全局单例 get_rag_pipeline_service()，惰性构建
- retrieve_with_citations：与现有 RAGHelper bundle 契约兼容
  （results/citations/confidence/abstained + context_text）
- 域门控：未命中任何启用域（查询无实体/故事信号）返回 None，
  调用方回退既有检索链路——普通聊天不会加载游戏知识库
- 索引/模型不可用时清晰降级（is_available=False，retrieve 返回 None）

角色长期记忆（character 包）与作品知识索引完全隔离：
记忆仓储不走本服务；共享的只有 embedding provider 约定，
索引与命名空间互不重叠，避免用户记忆与作品设定互相污染。
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

from .embedding import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_EMBEDDING_MODEL_ID,
    SentenceTransformerEmbeddingProvider,
)
from .pipeline import RagPipeline
from .registry import get_default_registry

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


class RagPipelineService:
    """P6 统一检索服务。"""

    def __init__(self, pipeline: RagPipeline):
        self.pipeline = pipeline
        self._warmup_started = False

    # -- 状态 ---------------------------------------------------------------
    def is_available(self) -> bool:
        return self.pipeline.is_available()

    def stats(self) -> dict[str, Any]:
        return {
            "available": self.is_available(),
            "domains": self.pipeline.domain_stats(),
            "registry": self.pipeline.registry.snapshot(),
        }

    def warmup_async(self) -> None:
        """后台线程预热 embedding 模型（幂等，daemon）。"""
        if self._warmup_started:
            return
        self._warmup_started = True
        if not _env_flag("RAG_PIPELINE_WARMUP", "true"):
            return

        def _warm():
            try:
                self.pipeline.load_indexes()
                self.pipeline.warmup_embedding()
                logger.info("P6 管线预热完成")
            except Exception as e:  # noqa: BLE001 - 预热失败不阻塞主流程
                logger.warning("P6 管线预热失败（运行时按需重试）: %s", e)

        thread = threading.Thread(target=_warm, name="rag-pipeline-warmup", daemon=True)
        thread.start()

    # -- 检索 ---------------------------------------------------------------
    def retrieve_with_citations(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        domain_id: str | None = None,
    ) -> dict[str, Any] | None:
        """检索并返回带引用的 bundle；未命中域返回 None（回退既有链路）。

        任何内部异常都被捕获并记录，返回 None 而不是让上层聊天失败。
        """
        try:
            return self.pipeline.retrieve(
                query,
                domain_id=domain_id,
                top_k=top_k,
                filters=filters,
                mode="hybrid",
                use_rerank=True,
                use_context=True,
            )
        except Exception as e:  # noqa: BLE001 - 索引故障不能拖垮聊天
            logger.warning("P6 检索异常（降级回既有链路）: %s", e)
            return None


_service: RagPipelineService | None = None
_service_lock = threading.Lock()


def build_default_service() -> RagPipelineService:
    """按环境配置构建默认服务实例。"""
    provider = SentenceTransformerEmbeddingProvider(
        model_path=os.getenv("EMBEDDING_MODEL_PATH", "").strip() or None,
        model_id=os.getenv("RAG_PIPELINE_EMBEDDING_MODEL_ID", DEFAULT_EMBEDDING_MODEL_ID),
        expected_dim=_env_int("RAG_PIPELINE_EMBEDDING_DIM", DEFAULT_EMBEDDING_DIM),
        batch_size=_env_int("RAG_PIPELINE_EMBEDDING_BATCH", 32),
        timeout_seconds=float(os.getenv("RAG_PIPELINE_EMBEDDING_TIMEOUT", "300")),
    )
    pipeline = RagPipeline(
        registry=get_default_registry(),
        embedding_provider=provider,
    )
    return RagPipelineService(pipeline)


def get_rag_pipeline_service() -> RagPipelineService:
    """获取全局服务单例（线程安全，惰性构建）。"""
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = build_default_service()
                _service.warmup_async()
    return _service


def reset_rag_pipeline_service() -> None:
    """重置单例（测试用）。"""
    global _service
    with _service_lock:
        _service = None
