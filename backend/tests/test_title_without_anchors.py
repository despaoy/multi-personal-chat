"""Exact named reads use verified source authority, never weak semantic candidates."""

from threading import RLock

import pytest

from inference.evidence_coverage import requested_source_lookups
from knowledge.original_sources import attach_original_sources
from knowledge.source_expansion import expand_source_context
from knowledge.vector_db import VectorDatabase


def row(parent, title, text="完整原文", kb=7, index=0):
    return dict(
        id=f"doc_{parent}_chunk_{index}",
        document_id=parent,
        chunk_index=index,
        knowledge_base_id=kb,
        title=title,
        category="规程",
        content=text,
    )


def vector(records):
    v = VectorDatabase.__new__(VectorDatabase)
    v._lock = RLock()
    v._cache_generation = 11
    v.snapshot_validated = True
    v.metadata = records
    return v


def expand(v, bundle=None, query="读取《A》。", filters=None, budget=65536):
    return expand_source_context(
        bundle if bundle is not None else dict(results=[], citations=[], confidence=0.0, abstained=True),
        v,
        expected_generation=11,
        source_budget_tokens=budget,
        filters=filters,
        query=query,
    )


@pytest.mark.parametrize("candidates,abstained", [([], True), ([], False), ([row(9, "相似资料")], True)])
def test_exact_title_without_reliable_anchor_rejects_other_candidates(candidates, abstained):
    original = row(1, "A", "费用8元，完整条件与例外。")
    bundle = dict(results=candidates, citations=[dict(source_id="doc_9_chunk_0")], confidence=0.1, abstained=abstained)
    result = expand(vector([original, row(9, "相似资料")]), bundle)
    assert not result["abstained"] and result["confidence"] == 0.1
    assert result["citations"] == [] and result["semantic_candidates_discarded"]
    assert [r["id"] for r in result["results"]] == [original["id"]]
    assert result["results"][0]["retrieval_role"] == "requested_source"
    assert result["results"][0]["score"] == 0 and candidates == bundle["results"]
    assert requested_source_lookups(result, "读取《A》") == (
        dict(title="A", lookup_status="matched_in_index_scope", source_ids=["doc_1"]),
    )
    fresh = {**original, "id": 1, "content": "新鲜完整原文：费用8元，例外另有条件。"}
    attached = attach_original_sources(
        result, lambda identity: fresh if identity == 1 else None, source_budget_tokens=65536, authority_revision=77
    )
    assert attached["original_source_packets"][0]["original_body"] == fresh["content"]


@pytest.mark.parametrize(
    "query", ["A怎么办？", "资料说‘读取《A》’，请解释引用。", "不要读取《A》。", "读取《A》，但别读取这份。"]
)
def test_no_complete_positive_read_keeps_abstention(query):
    bundle = dict(results=[row(9, "相似资料")], abstained=True, confidence=0.1)
    assert expand(vector([row(1, "A")]), bundle, query=query) is bundle


@pytest.mark.parametrize("invalid", ["unvalidated", "generation", "duplicate", "chunk_identity"])
def test_named_read_requires_unchanged_validated_index(invalid):
    v = vector([row(1, "A")])
    if invalid == "unvalidated":
        v.snapshot_validated = False
    elif invalid == "generation":
        v._cache_generation += 1
    elif invalid == "duplicate":
        v.metadata.append(row(1, "A", "同身份冒用正文"))
    else:
        v.metadata[0]["id"] = "doc_2_chunk_0"
    with pytest.raises(RuntimeError):
        expand(v)


def test_all_titles_absent_in_original_scope_keep_abstention_and_scoped_receipts():
    result = expand(vector([row(1, "A", kb=8)]), filters={"knowledge_base_id": 7})
    assert result["abstained"] and result["results"] == []
    assert requested_source_lookups(result, "读取《A》") == (
        dict(title="A", lookup_status="not_found_in_index_scope", source_ids=[]),
    )


def test_no_anchors_keeps_same_title_versions_as_independent_roots():
    result = expand(vector([row(1, "A", "2025旧版费用8"), row(2, "A", "2026新版费用11")]))
    assert result["ambiguous_requested_titles"] == ["A"]
    assert [r["document_id"] for r in result["results"]] == [1, 2]
    assert requested_source_lookups(result, "读取《A》")[0]["source_ids"] == ["doc_1", "doc_2"]


def test_budget_exhausted_is_located_but_not_visible_or_usable_evidence():
    result = expand(vector([row(1, "A", "必要条件" * 200)]), budget=1)
    assert result["abstained"] and result["results"] == []
    assert result["source_coverage"][0]["retrieved_document_ids"] == []
    assert requested_source_lookups(result, "读取《A》")[0]["lookup_status"] == "matched_in_index_scope"
    assert (
        attach_original_sources(
            result, lambda _: pytest.fail("No body read is authorized"), source_budget_tokens=1, authority_revision=77
        )
        is result
    )


async def test_named_read_rejects_original_authority_revision_race(monkeypatch):
    from api import generate, knowledge
    from knowledge import rag_helper, vector_db

    v = vector([row(1, "A")])
    revision = [77]
    monkeypatch.setattr(knowledge, "_ensure_vector_index", lambda: True)
    monkeypatch.setattr(knowledge, "_vector_index_revision", 77)
    monkeypatch.setattr(knowledge, "_get_rebuild_revision", lambda: revision[0])
    monkeypatch.setattr(vector_db, "get_vector_db", lambda: v)

    class Helper:
        @staticmethod
        def retrieve_with_citations(*args, **kwargs):
            revision[0] += 1
            return dict(results=[], confidence=0.0, abstained=True)

    monkeypatch.setattr(rag_helper, "get_rag_helper", Helper)
    monkeypatch.setenv("CORRECTIVE_RAG_ENABLED", "false")
    with pytest.raises(RuntimeError, match="authority changed"):
        await generate._retrieve_rag_bundle("读取《A》", 3, {"knowledge_base_id": 7})
