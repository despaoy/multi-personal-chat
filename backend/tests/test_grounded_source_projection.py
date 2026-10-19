"""Current curated text is the ordinary fact answer's only evidence.

Reuse the real isolated vector artifacts/runtime/default-service fixture. Its
embedding, ranking and model are deterministic controls, not a model-quality
evaluation. The indexed fact/summary/evidence intentionally remain A while the
complete declared original is B.
"""

import json
from html import unescape

import pytest
import test_grounded_source_lifecycle as source_fixture
from knowledge.grounded_answer.packet import EvidencePacketBuilder
from knowledge.grounded_answer.prompt import GroundedPromptBuilder

DOMAIN = source_fixture.DOMAIN
BODY_A = source_fixture.BODY_A
BODY_B = source_fixture.BODY_B
FACT_ID = source_fixture.FACT_ID
QUERY = "虚构星禾档案馆目前哪天哪个时段办理什么业务，谁能进门，恶劣天气和预约分别有哪些限制？"
MODEL_ID = "synthetic-source-projection"

# Source freshness, cache and transport contracts control query preparation.
pytestmark = pytest.mark.usefixtures("literal_grounded_query")


def arrange_grounded_projection(tmp_path, monkeypatch):
    """Return a rig usable by an old-version preflight without assuming repair.

    QUERY is an ordinary multi-fact question; it never requests source display.
    A remains in the real loaded fact summary/content and evidence content.
    Only the actual original file is replaced with valid same-span B. The
    existing controlled model reports whichever full body it actually sees.
    """
    monkeypatch.setattr(source_fixture, "QUERY", QUERY)
    monkeypatch.setattr(source_fixture, "INDEX_DESCRIPTION", BODY_A)
    rig = source_fixture.arrange_grounded_sources(tmp_path, monkeypatch)
    fact = rig.runtime._service.by_id[FACT_ID]
    evidence = rig.runtime._service.evidence_by_parent[FACT_ID]
    assert fact.summary == fact.content == evidence.content == BODY_A
    assert rig.path.read_text(encoding="utf-8") == BODY_A
    rig.path.write_bytes(BODY_B.encode("utf-8"))
    rig.query = QUERY
    return rig


async def ask_projection(rig):
    return await rig.service.answer(
        QUERY, domain_id=DOMAIN, top_k=1, generate=rig.generate, model_id=MODEL_ID,
    )


def assert_current_citation(citations, rig):
    assert len(citations) == 1
    citation = citations[0]
    assert citation["document_id"] == citation["source_id"] == FACT_ID
    assert citation["domain_id"] == DOMAIN
    assert citation["document_type"] == "evidence"
    assert citation["source_title"] == rig.path.name
    assert citation["source_path"] == rig.path.name
    assert citation["source_line"] == 1 and citation["source_line_end"] == 2
    assert citation["summary"] == ""
    for field in ("reality_status", "temporal_scope", "content_scope"):
        assert citation[field] == "unknown"
    assert citation["story_unit_id"] == ""
    assert BODY_B in citation["evidence_excerpt"]
    assert BODY_A not in json.dumps(citations, ensure_ascii=False)


def assert_current_result(result, rig):
    assert result.answer == BODY_B
    assert not result.abstained and result.failure_kind == "" and result.domain_id == DOMAIN
    assert_current_citation(result.citations, rig)
    assert len(result.curated_source_receipts) == 1
    assert result.curated_source_receipts[0]["excerpt"]["text"] == BODY_B


@pytest.mark.parametrize("transport", ["normal", "sse"])
async def test_ordinary_fact_answer_uses_only_fresh_declared_text_and_cache(tmp_path, monkeypatch, transport):
    rig = arrange_grounded_projection(tmp_path, monkeypatch)
    if transport == "normal":
        fresh = await ask_projection(rig)
    else:
        stream = rig.service.answer_stream(
            QUERY, domain_id=DOMAIN, top_k=1, generate=rig.generate, model_id=MODEL_ID,
        )
        try:
            events = [event async for event in stream]
        finally:
            await stream.aclose()
        assert [event.type for event in events] == ["meta", "delta", "citations", "done"]
        assert events[1].data["text"] == events[-1].data["answer"] == BODY_B
        assert not events[-1].data["abstained"] and events[-1].data["model_invoked"]
        assert_current_citation(events[2].data["citations"], rig)
        assert BODY_A not in json.dumps([event.data for event in events], ensure_ascii=False)
        # Streaming remains uncached. The normal service request can now fill
        # its existing cache with the same current evidence projection.
        assert rig.service.cache.stats()["entries"] == 0
        fresh = await ask_projection(rig)
    assert_current_result(fresh, rig)
    assert fresh.model_invoked and not fresh.cache_hit
    assert len(rig.generate_calls) == (1 if transport == "normal" else 2)
    for call in rig.generate_calls:
        wire = unescape("\n".join(message["content"] for message in call["kwargs"]["messages"]))
        assert call["source_body"] == BODY_B and BODY_B in wire
        assert BODY_A not in wire and "每周三17:00至18:00" not in wire
    count = len(rig.generate_calls)
    cached = await ask_projection(rig)
    assert_current_result(cached, rig)
    assert cached.cache_hit and not cached.model_invoked and len(rig.generate_calls) == count
    assert rig.service.cache.stats()["entries"] == 1


async def test_declared_unavailable_original_cannot_be_replaced_by_old_index_claim(tmp_path, monkeypatch):
    rig = arrange_grounded_projection(tmp_path, monkeypatch)
    # Real collection produces its own explicit no_visible_source record;
    # no forged bundle is needed to test the missing-evidence relationship.
    rig.runtime._service.evidence_by_parent.clear()
    result = await ask_projection(rig)
    assert result.abstained and not result.model_invoked and not result.cache_hit
    assert not result.citations and BODY_A not in result.answer
    assert not rig.generate_calls and rig.service.cache.stats()["entries"] == 0
    assert len(rig.embedding_calls) == len(rig.rerank_calls) == 1


@pytest.mark.parametrize("mismatch", ["parent_id", "domain_id"])
async def test_invalid_fresh_source_binding_fails_before_model(tmp_path, monkeypatch, mismatch):
    rig = arrange_grounded_projection(tmp_path, monkeypatch)
    actual_retriever = rig.service.retriever

    def wrong_binding(*args, **kwargs):
        bundle = actual_retriever(*args, **kwargs)
        assert len(bundle["curated_source_catalog"]["sources"]) == 1
        receipt = bundle["curated_source_catalog"]["sources"][0]
        assert receipt["excerpt"]["text"] == BODY_B
        receipt[mismatch] = "fiction:unrelated-parent" if mismatch == "parent_id" else "unrelated-domain"
        return bundle

    monkeypatch.setattr(rig.service, "retriever", wrong_binding)
    with pytest.raises(ValueError):
        await ask_projection(rig)
    assert not rig.generate_calls and rig.service.cache.stats()["entries"] == 0
    assert len(rig.embedding_calls) == len(rig.rerank_calls) == 1


async def test_complete_curated_source_over_grounded_budget_fails_without_partial_answer(tmp_path, monkeypatch):
    rig = arrange_grounded_projection(tmp_path, monkeypatch)
    rig.service.packet_builder = EvidencePacketBuilder(6000)
    rig.service.prompt_builder = GroundedPromptBuilder(6000)
    first, second = BODY_B.splitlines()
    body = first + "\n" + second + "另外：仍须保留完整日期、时段和天气限制；" * 450
    assert len(body.splitlines()) == 2
    assert rig.service.packet_builder.evidence_budget == 6000 < len(body) < rig.extractor.max_chars
    rig.path.write_bytes(body.encode("utf-8"))
    with pytest.raises(ValueError, match="budget"):
        await ask_projection(rig)
    assert not rig.generate_calls and rig.service.cache.stats()["entries"] == 0
    assert len(rig.embedding_calls) == len(rig.rerank_calls) == 1


async def test_old_index_budget_cannot_discard_legal_short_fresh_source(tmp_path, monkeypatch):
    rig = arrange_grounded_projection(tmp_path, monkeypatch)
    # Change the actual loaded evidence document, while the short declared
    # original remains B. Existing indexed-context admission must omit this
    # oversized block; it does not own admission of the fresh source snapshot.
    indexed = rig.runtime._service.evidence_by_parent[FACT_ID]
    indexed.content = BODY_A + "\n" + "这是索引中的重复旧描述，不能替代完整当前声明。" * 500
    assert len(indexed.content) > rig.runtime.context_max_chars
    assert rig.path.read_text(encoding="utf-8") == BODY_B
    bundle = rig.runtime.retrieve_with_citations(QUERY, domain_id=DOMAIN, top_k=1)
    assert bundle["results"] and bundle["results"][0]["id"] == FACT_ID
    assert bundle["citations"] == [] and bundle["context_budget"]["skipped_blocks"] == 1
    assert len(bundle["curated_source_catalog"]["sources"]) == 1
    assert bundle["curated_source_catalog"]["sources"][0]["excerpt"]["text"] == BODY_B
    assert not bundle["curated_source_catalog"]["unavailable"]
    result = await ask_projection(rig)
    assert_current_result(result, rig)
    assert result.model_invoked and not result.cache_hit
    assert len(rig.generate_calls) == 1 and rig.generate_calls[0]["source_body"] == BODY_B
    wire = unescape("\n".join(
        message["content"] for message in rig.generate_calls[0]["kwargs"]["messages"]
    ))
    assert BODY_B in wire and BODY_A not in wire
    assert rig.service.cache.stats()["entries"] == 1


async def test_low_confidence_with_legal_large_original_abstains_without_generation_budget_failure(tmp_path, monkeypatch):
    from dataclasses import replace

    rig = arrange_grounded_projection(tmp_path, monkeypatch)
    rig.service.packet_builder = EvidencePacketBuilder(6000)
    rig.service.prompt_builder = GroundedPromptBuilder(6000)
    first, second = BODY_B.splitlines()
    body = first + "\n" + second + "另外：仍须保留完整日期、时段和天气限制；" * 450
    assert len(body.splitlines()) == 2
    assert rig.service.packet_builder.evidence_budget == 6000 < len(body) < rig.extractor.max_chars
    rig.path.write_bytes(body.encode("utf-8"))

    class LowConfidenceRanking:
        text_view = "content"

        def rerank(self, analysis, candidates, *, top_k):
            rig.rerank_calls.append(analysis.original_query)
            assert len(candidates) == 1 and candidates[0].document.id == FACT_ID
            # The real runtime compares this negative logit with its configured
            # zero threshold. No retrieval/collection/abstention result is faked.
            return [replace(candidates[0], rerank_score=-1.0, rerank_method="cross_encoder")][:top_k]

    rig.runtime._service.reranker = LowConfidenceRanking()
    actual_retriever = rig.service.retriever
    observed = []

    def observe_retrieval(*args, **kwargs):
        bundle = actual_retriever(*args, **kwargs)
        observed.append(bundle)
        return bundle

    monkeypatch.setattr(rig.service, "retriever", observe_retrieval)
    result = await ask_projection(rig)
    assert len(observed) == 1
    bundle = observed[0]
    assert bundle["abstained"] is True and bundle["results"][0]["rerank_score"] == -1.0
    assert len(bundle["curated_source_catalog"]["sources"]) == 1
    assert bundle["curated_source_catalog"]["sources"][0]["excerpt"]["text"] == body
    assert not bundle["curated_source_catalog"]["unavailable"]
    assert result.abstained and not result.model_invoked and not result.cache_hit
    assert result.answer_mode.value == "abstention" and result.failure_kind == ""
    assert not result.citations and result.retrieval_metadata["evidence_count"] == 0
    assert BODY_A not in result.answer and BODY_B not in result.answer
    assert not rig.generate_calls and rig.service.cache.stats()["entries"] == 0
    assert len(rig.embedding_calls) == len(rig.rerank_calls) == 1

