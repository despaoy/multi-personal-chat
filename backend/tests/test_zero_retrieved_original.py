"""Budget-omitted indexed roots do not invalidate independently authorized originals."""

import copy
from threading import RLock

import pytest

from inference.evidence_coverage import requested_source_lookups
from inference.generation_request import GenerationRequest, RetrievalResult, build_generation_request
from knowledge.evidence_packets import document_evidence_packets
from knowledge.original_sources import attach_original_sources
from knowledge.source_expansion import expand_source_context
from knowledge.vector_db import VectorDatabase


def row(parent, title, text):
    return dict(
        id=f"doc_{parent}_chunk_0",
        document_id=parent,
        chunk_index=0,
        knowledge_base_id=7,
        title=title,
        category="规程",
        content=text,
    )


def expanded(query="分别读取《A》《B》。", same_title=False):
    v = VectorDatabase.__new__(VectorDatabase)
    v._lock = RLock()
    v._cache_generation = 11
    v.snapshot_validated = True
    v.metadata = [row(1, "A", "旧版独立超长正文。" * 1000), row(2, "A" if same_title else "B", "费用8元，时限5小时。")]
    return expand_source_context(
        dict(results=[], confidence=0.0, abstained=True),
        v,
        expected_generation=11,
        source_budget_tokens=100,
        query=query,
    )


def attach(result):
    reads = []

    def fresh(identity):
        reads.append(identity)
        assert identity == 2
        return dict(
            id=2,
            title=result["results"][0]["title"],
            knowledge_base_id=7,
            category="规程",
            content="费用8元，时限5小时。",
        )

    return attach_original_sources(result, fresh, source_budget_tokens=100, authority_revision=77), reads


def compiled(result, query):
    packets = document_evidence_packets(result["results"]) + tuple(result["original_source_packets"])
    return build_generation_request(
        GenerationRequest(
            message=query,
            context_window_tokens=8192,
            max_tokens=512,
            evidence_max_chars=0,
            retrieval=RetrievalResult(
                status="ok",
                evidence="\n".join(p["text"] for p in packets),
                evidence_packets=packets,
                source_coverage=result["source_coverage"],
                requested_sources=requested_source_lookups(result, query),
            ),
        )
    )


@pytest.mark.parametrize("query", ["分别读取《A》《B》。", "分别读取《B》《A》。"])
def test_true_mixed_budget_preserves_zero_coverage_and_other_fresh_original(query):
    before = expanded(query)
    snapshot = copy.deepcopy(before)
    result, reads = attach(before)
    assert reads == [2] and before == snapshot
    plan = compiled(result, query)
    scopes = {r["source_id"]: r for r in plan.retrieval.source_coverage}
    absent = scopes["doc_1"]
    assert absent["retrieved_chunk_count"] == absent["admitted_chunk_count"] == 0
    assert absent["status"] == "partial" and absent["original_status"] == "not_verified"
    assert absent["omission_reasons"] == ["source_context_budget"]
    assert "original_source_receipt" not in absent
    assert scopes["doc_2"]["original_status"] == "verified_original_body_admitted"
    assert "费用8元，时限5小时。" in plan.messages[-1]["content"]
    assert "旧版独立超长正文" not in plan.messages[-1]["content"]
    assert all(r["lookup_status"] == "matched_in_index_scope" for r in plan.retrieval.requested_sources)


def test_omitted_same_title_version_does_not_hide_available_version():
    query = "读取《A》，分别核对版本。"
    result, reads = attach(expanded(query, same_title=True))
    assert reads == [2]
    plan = compiled(result, query)
    assert plan.retrieval.requested_sources[0]["source_ids"] == ["doc_1", "doc_2"]
    assert plan.retrieval.source_coverage[0]["original_status"] == "not_verified"
    assert plan.retrieval.source_coverage[1]["original_status"] == "verified_original_body_admitted"


@pytest.mark.parametrize(
    "malformed",
    [
        "empty_index",
        "duplicate",
        "other_parent",
        "bad_index",
        "retrieved_nonempty",
        "retrieved_missing",
        "receipt",
        "source_identity",
    ],
)
def test_unexplained_or_forged_zero_source_never_authorizes_reads(malformed):
    result = expanded()
    missing = result["source_coverage"][0]
    if malformed == "empty_index":
        missing["indexed_document_ids"] = []
    elif malformed == "duplicate":
        missing["indexed_document_ids"] *= 2
    elif malformed == "other_parent":
        missing["indexed_document_ids"] = ["doc_2_chunk_0"]
    elif malformed == "bad_index":
        missing["indexed_document_ids"] = ["doc_1_chunk_-1"]
    elif malformed == "retrieved_nonempty":
        missing["retrieved_document_ids"] = ["doc_1_chunk_0"]
    elif malformed == "retrieved_missing":
        missing.pop("retrieved_document_ids")
    elif malformed == "receipt":
        missing["original_source_receipt"] = {"authority": "fresh_knowledge_document_read"}
    else:
        missing["source_id"] = "doc_0"
    with pytest.raises(ValueError):
        attach_original_sources(
            result,
            lambda _: pytest.fail("Invalid zero source must fail before reads"),
            source_budget_tokens=100,
            authority_revision=77,
        )
