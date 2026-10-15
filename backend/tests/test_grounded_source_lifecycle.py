"""The default grounded-answer source-snapshot lifecycle.

The three vector-artifact partitions, runtime, routed retriever, source reader,
packet/prompt/citation path and answer cache are real. Embeddings, ranking and
model output are explicit deterministic controls; this does not evaluate model
semantics and performs no paid model calls.
"""

from dataclasses import replace
from html import unescape
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from knowledge import curated_sources
from knowledge.curated_sources import CuratedSourceChangedError
from knowledge.grounded_answer.service import build_default_grounded_answer_service
from knowledge.multiscale_rag import runtime as runtime_module
from knowledge.multiscale_rag.constants import EMBEDDING_TEXT_VERSION, INDEX_FORMAT_VERSION
from knowledge.multiscale_rag.service import RoutedMultiScaleService
from knowledge.multiscale_rag.source_text import OriginalTextExtractor
from knowledge.multiscale_rag.vector_runtime import LocalMeanPoolingEmbeddingProvider, write_vector_artifacts
from knowledge.retrieval_core.documents import KnowledgeIndexDocument, SourceReference

DOMAIN = "tsukiyashiro_kisaki"
QUERY = "请给出虚构星禾档案馆通行安排的完整原文，逐项保留日期、时段、否定、条件和例外。"
BODY_A = (
    "2026年10月16日馆方通告：虚构星禾档案馆每周三17:00至18:00只办理归还，不办理续借。\n"
    "仅已登记的预约者可进门；雨天暂停对外开放，预约不能豁免雨天限制。"
)
BODY_B = (
    "2026年10月17日馆方通告：虚构星禾档案馆每周五16:00至17:00只办理续借，不办理归还。\n"
    "仅已登记的预约者可进门；大风天暂停对外开放，预约不能豁免大风限制。"
)
FACT_ID = "fiction:archive:fact"
EVIDENCE_ID = "fiction:archive:evidence"
INDEX_DESCRIPTION = "虚构星禾档案馆有一段两行通行安排声明；须查当前完整原文，本索引描述不证明日期、时段或例外。"


def arrange_grounded_sources(tmp_path, monkeypatch):
    """Return an importable real runtime/default-service rig without asking it.

    Useful to a separate baseline preflight: call ``rig.service.answer`` with
    ``generate=rig.generate`` and ``domain_id=DOMAIN``. ``rig.on_generate`` may
    be a synchronous source-change callback. Source files and indexes contain
    only the complete fictional data above. No real model asset is loaded.
    """
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    path = corpus / "fiction-archive.txt"
    path.write_bytes(BODY_A.encode("utf-8"))
    index_root = tmp_path / "indexes"
    reference = SourceReference(source_path=path.name, line_start=1, line_end=2)

    def document(identity, kind, metadata):
        return KnowledgeIndexDocument(
            id=identity,
            domain_id=DOMAIN,
            document_type=kind,
            title="虚构星禾档案馆通行安排",
            summary=INDEX_DESCRIPTION,
            content=INDEX_DESCRIPTION,
            embedding_text=INDEX_DESCRIPTION,
            keywords=["虚构星禾档案馆", "通行安排", "日期", "时段", "否定", "条件", "例外"],
            source=reference,
            metadata={"embedding_text_version": EMBEDDING_TEXT_VERSION, **metadata},
            reality_status="fictional",
            temporal_scope="current_statement",
            content_scope="main_story",
            index_version=INDEX_FORMAT_VERSION,
        )

    fact = document(FACT_ID, "fact", {})
    evidence = document(EVIDENCE_ID, "evidence", {"parent_id": FACT_ID})
    vector = np.ones(384, dtype=np.float32)
    vector /= np.linalg.norm(vector)
    provider = LocalMeanPoolingEmbeddingProvider(model_path="synthetic-no-model-assets")
    embedding_calls = []

    def controlled_embedding(query):
        embedding_calls.append(query)
        return vector.copy()

    monkeypatch.setattr(provider, "embed_query", controlled_embedding)
    for partition, documents in (
        ("card_index", [fact]),
        ("scene_story_index", []),
        ("evidence_index", [evidence]),
    ):
        vectors = np.repeat(vector[None, :], len(documents), axis=0)
        write_vector_artifacts(
            index_root / partition,
            documents,
            vectors,
            {
                "index_version": INDEX_FORMAT_VERSION,
                "embedding_text_version": EMBEDDING_TEXT_VERSION,
                "embedding_model": provider.model_id,
                "document_count": len(documents),
                "vector_dimension": 384,
                "normalized": True,
            },
        )

    # Keep ambient deployment configuration out of this isolated rig.
    for variable in ("CHARACTER_RAG_INDEX_ROOT", "MULTISCALE_RAG_INDEX_ROOT"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("CHARACTER_RAG_RERANKER_ENABLED", "false")
    monkeypatch.setenv("RERANKER_ENABLED", "false")
    monkeypatch.setenv("CHARACTER_RAG_CROSS_ENCODER_MIN_LOGIT", "0")
    monkeypatch.setenv("GROUNDED_ANSWER_CORRECTIVE", "false")
    monkeypatch.setenv("GROUNDED_GENERATION_TIMEOUT", "5")
    monkeypatch.setenv("GROUNDED_ANSWER_MAX_CHARS", "4000")
    monkeypatch.setattr(runtime_module, "_REPO_ROOT", corpus)
    monkeypatch.setattr(runtime_module, "LocalMeanPoolingEmbeddingProvider", lambda **_kwargs: provider)
    runtime = runtime_module.MultiScaleRagRuntime(index_root=index_root)
    assert runtime.is_available()
    assert isinstance(runtime._service, RoutedMultiScaleService)
    extractor = runtime._service.extractor
    assert isinstance(extractor, OriginalTextExtractor)
    monkeypatch.setattr(curated_sources, "_extractor", lambda: extractor)
    monkeypatch.setattr(runtime_module, "get_multiscale_rag_service", lambda: runtime)
    rerank_calls = []

    class ControlledRanking:
        text_view = "content"

        def rerank(self, analysis, candidates, *, top_k):
            rerank_calls.append(analysis.original_query)
            assert len(candidates) == 1 and candidates[0].document.id == FACT_ID
            return [replace(candidates[0], rerank_score=.95, rerank_method="cross_encoder")][:top_k]

    runtime._service.reranker = ControlledRanking()
    service = build_default_grounded_answer_service()
    assert service.cache.ttl > 0
    rig = SimpleNamespace(
        service=service,
        runtime=runtime,
        extractor=extractor,
        path=path,
        corpus=corpus,
        index_root=index_root,
        generate_calls=[],
        embedding_calls=embedding_calls,
        rerank_calls=rerank_calls,
        on_generate=None,
        query=QUERY,
        domain_id=DOMAIN,
    )

    async def controlled_model(**kwargs):
        visible = unescape(kwargs["messages"][-1]["content"])
        admitted = [body for body in (BODY_A, BODY_B) if body in visible]
        assert len(admitted) == 1, "The model must receive one complete actual source snapshot"
        assert '<user_query>\n' + QUERY + '\n</user_query>' in visible
        body = admitted[0]
        rig.generate_calls.append({"kwargs": kwargs, "source_body": body})
        if rig.on_generate is not None:
            rig.on_generate()
        return body + " [S1]"

    rig.generate = controlled_model
    rig.model_callback = controlled_model
    return rig


async def ask(rig):
    return await rig.service.answer(
        QUERY, domain_id=DOMAIN, top_k=1, generate=rig.generate, model_id="synthetic-source-model",
    )


def assert_success(result, body):
    assert result.answer == body
    assert result.domain_id == DOMAIN and not result.abstained and result.failure_kind == ""
    assert result.citations and len(result.citations) == 1
    assert result.citations[0]["document_id"] == FACT_ID
    assert body in result.citations[0]["evidence_excerpt"]


async def test_unchanged_curated_source_reuses_default_answer_cache(tmp_path, monkeypatch):
    rig = arrange_grounded_sources(tmp_path, monkeypatch)
    first = await ask(rig)
    assert_success(first, BODY_A)
    assert first.model_invoked and not first.cache_hit
    assert first.curated_source_receipts
    assert "curated_source_receipts=" not in repr(first)
    cached = await ask(rig)
    assert_success(cached, BODY_A)
    assert cached.cache_hit and not cached.model_invoked
    assert len(rig.generate_calls) == len(rig.embedding_calls) == len(rig.rerank_calls) == 1
    assert rig.service.cache.stats()["entries"] == 1


async def test_same_declared_span_replacement_is_fresh_miss_then_cacheable(tmp_path, monkeypatch):
    rig = arrange_grounded_sources(tmp_path, monkeypatch)
    assert_success(await ask(rig), BODY_A)
    rig.path.write_bytes(BODY_B.encode("utf-8"))  # Same path and valid two-line span.
    fresh = await ask(rig)
    assert_success(fresh, BODY_B)
    assert fresh.model_invoked and not fresh.cache_hit
    assert [call["source_body"] for call in rig.generate_calls] == [BODY_A, BODY_B]
    cached = await ask(rig)
    assert_success(cached, BODY_B)
    assert cached.cache_hit and not cached.model_invoked and len(rig.generate_calls) == 2


@pytest.mark.parametrize("failure", ["deleted", "permission"])
async def test_cached_answer_cannot_hide_original_io_failure_and_recovers(tmp_path, monkeypatch, failure):
    rig = arrange_grounded_sources(tmp_path, monkeypatch)
    assert_success(await ask(rig), BODY_A)
    with monkeypatch.context() as patch:
        if failure == "deleted":
            rig.path.unlink()
            error_type = FileNotFoundError
        else:
            # Portable permission failure: only this exact synthetic source's
            # real read is denied; the rest of Path I/O stays unchanged.
            original_read = Path.read_bytes

            def denied(path):
                if path == rig.path:
                    raise PermissionError("Synthetic curated original is unreadable")
                return original_read(path)

            patch.setattr(Path, "read_bytes", denied)
            error_type = PermissionError
        with pytest.raises(error_type):
            await ask(rig)
        assert len(rig.generate_calls) == 1  # No fresh model call or cached success.
    rig.path.write_bytes(BODY_A.encode("utf-8"))
    assert_success(await ask(rig), BODY_A)


@pytest.mark.parametrize("transport", ["normal", "sse"])
async def test_model_time_source_change_cannot_finish_or_cache_success(tmp_path, monkeypatch, transport):
    rig = arrange_grounded_sources(tmp_path, monkeypatch)
    rig.on_generate = lambda: rig.path.write_bytes(BODY_B.encode("utf-8"))
    events = []
    with pytest.raises(CuratedSourceChangedError):
        if transport == "normal":
            await ask(rig)
        else:
            stream = rig.service.answer_stream(
                QUERY, domain_id=DOMAIN, top_k=1,
                generate=rig.generate, model_id="synthetic-source-model",
            )
            try:
                async for event in stream:
                    events.append(event)
            finally:
                await stream.aclose()
    assert [event.type for event in events] == (["meta"] if transport == "sse" else [])
    assert len(rig.generate_calls) == 1 and rig.generate_calls[0]["source_body"] == BODY_A
    assert rig.service.cache.stats()["entries"] == 0
    rig.on_generate = None
    recovered = await ask(rig)
    assert_success(recovered, BODY_B)
    assert recovered.model_invoked and not recovered.cache_hit and len(rig.generate_calls) == 2


async def test_raw_excerpt_and_catalog_cannot_bind_two_different_file_snapshots(tmp_path, monkeypatch):
    rig = arrange_grounded_sources(tmp_path, monkeypatch)
    original_extract = rig.extractor.extract
    reads = []

    def change_after_first_extract(source, *, snapshot=None):
        excerpt = original_extract(source, snapshot=snapshot)
        reads.append(excerpt.text)
        if len(reads) == 1:
            rig.path.write_bytes(BODY_B.encode("utf-8"))
        return excerpt

    with monkeypatch.context() as patch:
        patch.setattr(rig.extractor, "extract", change_after_first_extract)
        # Old retrieval reads raw=A, then successfully collects receipt=B.
        # One shared catalog snapshot must reject a change within that read.
        with pytest.raises(ValueError, match="changed"):
            rig.runtime.retrieve_with_citations(QUERY, domain_id=DOMAIN, top_k=1)
    assert reads == [BODY_A]
    assert not rig.generate_calls and rig.service.cache.stats()["entries"] == 0
    assert_success(await ask(rig), BODY_B)


async def test_source_changed_after_prompt_preparation_stops_before_model(tmp_path, monkeypatch):
    rig = arrange_grounded_sources(tmp_path, monkeypatch)
    build = rig.service.prompt_builder.build_messages

    def change_after_preparation(*args, **kwargs):
        messages = build(*args, **kwargs)
        rig.path.write_bytes(BODY_B.encode("utf-8"))
        return messages

    monkeypatch.setattr(rig.service.prompt_builder, "build_messages", change_after_preparation)
    with pytest.raises(CuratedSourceChangedError):
        await ask(rig)
    assert not rig.generate_calls and rig.service.cache.stats()["entries"] == 0


async def test_present_malformed_catalog_is_not_an_optional_empty_dependency(tmp_path, monkeypatch):
    rig = arrange_grounded_sources(tmp_path, monkeypatch)
    retrieve = rig.service.retriever

    def malformed_catalog(*args, **kwargs):
        bundle = retrieve(*args, **kwargs)
        bundle["curated_source_catalog"]["sources"] = ""
        return bundle

    monkeypatch.setattr(rig.service, "retriever", malformed_catalog)
    with pytest.raises(ValueError, match="catalogue records"):
        await ask(rig)
    assert not rig.generate_calls and rig.service.cache.stats()["entries"] == 0
