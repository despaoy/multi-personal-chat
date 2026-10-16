"""Focused tests for production multi-scale character knowledge retrieval."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np
import pytest

from knowledge.multiscale_rag.runtime import MultiScaleRagRuntime
from knowledge.multiscale_rag.service import CARD_TYPES
from knowledge.multiscale_rag.source_text import OriginalTextExtractor

if TYPE_CHECKING:
    from pathlib import Path


class FakeQueryEmbeddingProvider:
    dimension = 384
    model_id = "paraphrase-multilingual-MiniLM-L12-v2"
    encoding_contract = "complete-token-windows-weighted-v1"

    def __init__(self) -> None:
        self._model = object()

    def embed_query(self, query: str) -> np.ndarray:
        digest = hashlib.sha256(query.encode("utf-8")).digest()
        raw = np.resize(np.frombuffer(digest, dtype=np.uint8).astype(np.float32), self.dimension)
        norm = np.linalg.norm(raw)
        return raw / norm if norm else raw


def test_runtime_loads_all_four_scales_and_returns_bundle(monkeypatch):
    from knowledge.multiscale_rag.vector_runtime import LocalMeanPoolingEmbeddingProvider

    # Preserve the real provider/model identity required by these artifacts;
    # only query encoding is a declared synthetic control in this unit test.
    provider = LocalMeanPoolingEmbeddingProvider(model_path="synthetic-no-model-assets")
    monkeypatch.setattr(provider, "embed_query", FakeQueryEmbeddingProvider().embed_query)
    monkeypatch.setattr("knowledge.multiscale_rag.runtime.LocalMeanPoolingEmbeddingProvider", lambda **_kwargs: provider)
    monkeypatch.setenv("CHARACTER_RAG_RERANKER_ENABLED", "false")
    runtime = MultiScaleRagRuntime()

    assert runtime.is_available() is True
    stats = runtime.stats()
    assert stats["documents"] == 1098
    assert stats["document_type_counts"] == {
        "fact": 218,
        "relation": 32,
        "event": 159,
        "story": 18,
        "scene": 262,
        "evidence": 409,
    }

    bundle = runtime.retrieve_with_citations("妃和琉璃是什么关系？", top_k=3)

    assert bundle is not None
    assert bundle["results"]
    assert bundle["domains"] == ["tsukiyashiro_kisaki"]
    assert bundle["query_analysis"]["route_types"] == sorted(CARD_TYPES)
    assert bundle["context_trust"] == "untrusted_retrieved_evidence"
    assert isinstance(bundle["abstained"], bool)


def test_runtime_passes_cloud_budget_to_loaded_retriever():
    runtime = MultiScaleRagRuntime(context_max_chars=16384)
    runtime._provider = FakeQueryEmbeddingProvider()
    assert runtime.is_available()
    assert runtime._service.context_max_chars == 16384
    bundle = runtime.retrieve_with_citations('妃和琉璃是什么关系？', top_k=3)
    assert bundle['context_budget']['max_chars'] == 16384
    assert MultiScaleRagRuntime().context_max_chars == 6000


@pytest.mark.parametrize('value', [0, -1, True, 1.5])
def test_runtime_rejects_invalid_evidence_budget(value):
    with pytest.raises(ValueError, match='context_max_chars'):
        MultiScaleRagRuntime(context_max_chars=value)


def test_runtime_keeps_unrelated_chat_out_of_game_domain():
    runtime = MultiScaleRagRuntime()
    runtime._provider = FakeQueryEmbeddingProvider()

    assert runtime.retrieve_with_citations("今天天气怎么样", top_k=3) is None




def test_runtime_does_not_ignore_generic_kb_filter():
    runtime = MultiScaleRagRuntime()
    runtime._provider = FakeQueryEmbeddingProvider()

    assert (
        runtime.retrieve_with_citations(
            "妃和琉璃是什么关系？",
            top_k=3,
            filters={"knowledge_base_id": 7},
        )
        is None
    )




def test_source_extractor_rejects_path_traversal(tmp_path: Path):
    extractor = OriginalTextExtractor(tmp_path)
    with pytest.raises(ValueError, match="超出语料根目录"):
        extractor.resolve_path("../outside.txt")


def ranking_score_runtime(score, method="heuristic", *, empty=False):
    from types import SimpleNamespace

    runtime = object.__new__(MultiScaleRagRuntime)
    runtime._base_config = SimpleNamespace(domain_id="synthetic")
    runtime._gate = SimpleNamespace(analyze=lambda query: SimpleNamespace(entities=[], matched_domains=["synthetic"]))
    results = [] if empty else [dict(id="synthetic", title="合成资料", content="只能离线验收，禁止联网。",
                                    rerank_score=score, fused_score=0.99, rerank_method=method)]
    runtime._service = SimpleNamespace(retrieve=lambda *a, **kw: dict(
        results=results, context_text="只能离线验收，禁止联网。" if results else "",
        citations=[dict(source_id="synthetic")] if results else [],
    ))
    return runtime


@pytest.mark.parametrize("score", [0.0, -0.1, 0.24, 0.25, 0.8])
def test_ranking_score_heuristic_never_substitutes_high_recall_score(monkeypatch, score):
    monkeypatch.setenv("CHARACTER_RAG_ABSTAIN_THRESHOLD", "0.25")
    runtime = ranking_score_runtime(score)
    bundle = runtime.retrieve_with_citations("核对验收限制", domain_id=runtime.config.domain_id)
    assert bundle["confidence"] == score and bundle["abstained"] is (score < 0.25)


@pytest.mark.parametrize("score", [-1.0, 0.0, 1.0])
def test_ranking_score_cross_encoder_keeps_logit_threshold(monkeypatch, score):
    import math

    monkeypatch.setenv("CHARACTER_RAG_CROSS_ENCODER_MIN_LOGIT", "0")
    runtime = ranking_score_runtime(score, "cross_encoder")
    bundle = runtime.retrieve_with_citations("核对验收限制", domain_id=runtime.config.domain_id)
    assert bundle["confidence"] == round(1 / (1 + math.exp(-score)), 4)
    assert bundle["abstained"] is (score < 0)


@pytest.mark.parametrize("score", [None, "invalid", True, float("nan"), float("inf")])
@pytest.mark.parametrize("method", ["heuristic", "cross_encoder"])
def test_ranking_score_corruption_is_not_low_confidence_abstention(score, method):
    runtime = ranking_score_runtime(score, method)
    with pytest.raises((ValueError, TypeError)):
        runtime.retrieve_with_citations("核对验收限制", domain_id=runtime.config.domain_id)


def test_ranking_score_empty_results_remain_normal_abstention():
    runtime = ranking_score_runtime(None, empty=True)
    bundle = runtime.retrieve_with_citations("核对验收限制", domain_id=runtime.config.domain_id)
    assert bundle["confidence"] == 0 and bundle["abstained"] and not bundle["results"]


async def test_ranking_score_zero_prevents_answer_model_despite_high_recall_score():
    from knowledge.grounded_answer.service import GroundedAnswerService

    runtime = ranking_score_runtime(0.0)
    service = GroundedAnswerService(retriever=runtime.retrieve_with_citations)

    async def generate(**kwargs):
        raise AssertionError("High recall score must not rescue a rejected ranking score")

    result = await service.answer("核对验收限制", domain_id=runtime.config.domain_id, generate=generate)
    assert result.abstained and not result.model_invoked and result.confidence == 0


@pytest.mark.parametrize("field,value", [
    ("max_lines", 0), ("max_lines", True), ("max_lines", "2"), ("max_lines", 1.5),
    ("max_chars", 99), ("max_chars", True), ("max_chars", "100"), ("max_chars", 1.5),
])
def test_source_extractor_configuration_is_not_implicitly_corrected(tmp_path, field, value):
    with pytest.raises(ValueError, match=field):
        OriginalTextExtractor(tmp_path, **{field: value})


@pytest.mark.parametrize("field", ["line_start", "line_end"])
@pytest.mark.parametrize("value", [None, True, "1", 1.5])
def test_source_extractor_line_identity_is_not_coerced(tmp_path, field, value):
    from knowledge.retrieval_core.documents import SourceReference

    (tmp_path / "source.txt").write_text("完整原文。", encoding="utf-8")
    kwargs = dict(source_path="source.txt", line_start=1, line_end=1)
    kwargs[field] = value
    with pytest.raises(ValueError, match="来源行号"):
        OriginalTextExtractor(tmp_path).extract(SourceReference(**kwargs))


@pytest.mark.parametrize("start,end", [(1, 4), (4, 4), (3, 2)])
def test_source_extractor_missing_declared_lines_are_not_partial_success(tmp_path, start, end):
    from knowledge.retrieval_core.documents import SourceReference

    (tmp_path / "source.txt").write_text("第一条。\n完整条件：仅离线。\n最终限制：禁止联网。", encoding="utf-8")
    with pytest.raises(ValueError, match="来源行号"):
        OriginalTextExtractor(tmp_path).extract(SourceReference("source.txt", start, end))


@pytest.mark.parametrize("limit", ["lines", "chars"])
def test_source_extractor_limits_fail_instead_of_losing_tail(tmp_path, limit):
    from knowledge.retrieval_core.documents import SourceReference

    body = ("完整资料。\n" * 2 if limit == "lines" else "完整资料。" * 25) + "最终限制：禁止联网。"
    (tmp_path / "source.txt").write_text(body, encoding="utf-8")
    source = SourceReference("source.txt", 1, len(body.splitlines()))
    extractor = OriginalTextExtractor(tmp_path, **({"max_lines": 2} if limit == "lines" else {"max_chars": 100}))
    with pytest.raises(ValueError, match="max_" + limit):
        extractor.extract(source)


def test_source_extractor_exact_limit_keeps_full_tail_and_metadata(tmp_path):
    from knowledge.retrieval_core.documents import SourceReference

    body = "完整资料。" * 25 + "\n最终限制：禁止联网。"
    (tmp_path / "source.txt").write_text(body, encoding="utf-8")
    excerpt = OriginalTextExtractor(tmp_path, max_lines=2, max_chars=len(body)).extract(SourceReference("source.txt", 1, 2))
    assert excerpt.text == body and not excerpt.truncated
    assert (excerpt.line_start, excerpt.line_end) == (1, 2)


@pytest.mark.parametrize("failure", ["missing_file", "over_limit"])
def test_source_extractor_tightening_does_not_hide_read_failures(tmp_path, failure):
    from knowledge.retrieval_core.documents import SourceReference

    source = SourceReference("source.txt", 1, 2)
    if failure == "over_limit":
        (tmp_path / "source.txt").write_text("完整证据。\n最终限制：禁止联网。", encoding="utf-8")
    extractor = OriginalTextExtractor(tmp_path, max_lines=1 if failure == "over_limit" else 2)
    with pytest.raises(ValueError if failure == "over_limit" else FileNotFoundError):
        extractor.tighten_to_evidence(source, "完整证据。")


def test_source_extractor_tightening_matches_complete_span_without_widening(tmp_path):
    from knowledge.retrieval_core.documents import SourceReference

    (tmp_path / "source.txt").write_text("第一条。\n完整证据：禁止联网。\n第三条。", encoding="utf-8")
    extractor = OriginalTextExtractor(tmp_path)
    source = SourceReference("source.txt", 1, 3)
    narrowed, status = extractor.tighten_to_evidence(source, "完整证据：禁止联网。")
    assert status == "exact_source_match" and (narrowed.line_start, narrowed.line_end) == (2, 2)
    unmatched, status = extractor.tighten_to_evidence(source, "未在来源出现的文字")
    assert unmatched is source and status == "approved_evidence_fallback"


@pytest.mark.parametrize("contract", [None, "prefix-only-v1"])
def test_runtime_rejects_incompatible_encoding_before_exposing_a_service(monkeypatch, tmp_path, contract):
    from knowledge.multiscale_rag import runtime as module

    provider = FakeQueryEmbeddingProvider()
    manifest = dict(embedding_model=provider.model_id, embedding_encoding=contract)
    monkeypatch.setattr(module, "_load_bundle", lambda root: ([], np.empty((0, 384)), manifest))
    monkeypatch.setattr(module, "LocalMeanPoolingEmbeddingProvider", lambda **kwargs: provider)
    runtime = MultiScaleRagRuntime(index_root=tmp_path)
    with pytest.raises(ValueError, match="encoding contract mismatch.*rebuild"):
        runtime._ensure_loaded()
    assert runtime._service is None
