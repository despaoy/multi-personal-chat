"""Final visibility is packet-bound and does not establish semantic adequacy."""

from copy import deepcopy

import pytest

from inference.task_evidence_coverage import settle_task_coverage, validate_task_candidates
from knowledge.original_sources import attach_original_sources

QUERY = "核对青川规则和蓝溪规则"


def record(i, query, ids):
    return dict(
        task_index=i,
        kind="original_question" if i == 0 else "public_task",
        query=query,
        index_generation=7,
        candidate_count=len(ids),
        retained_candidate_count=len(ids),
        confidence=0.8,
        status="candidates_retrieved",
        semantic_coverage="unverified",
        candidate_source_links=tuple(dict(document_id=x, source_id=x.split("_chunk_")[0]) for x in ids),
    )


def receipts():
    return (
        record(0, QUERY, ["doc_1_chunk_0", "doc_2_chunk_0"]),
        record(1, "青川规则", ["doc_1_chunk_0"]),
        record(2, "蓝溪规则", ["doc_2_chunk_0"]),
    )


def baseline():
    records = receipts()
    validate_task_candidates(records, query=QUERY)
    rows = settle_task_coverage(records, {"doc_1_chunk_0", "doc_2_chunk_0"})
    assert all(x["status"] == "candidate_evidence_admitted" and x["semantic_coverage"] == "unverified" for x in rows)
    return records


def test_each_task_tracks_its_own_actual_packet_visibility():
    records = baseline()
    rows = settle_task_coverage(records, {"doc_1_chunk_0"})
    assert rows[0]["status"] == "partial_candidate_evidence_admitted"
    assert rows[1]["status"] == "candidate_evidence_admitted"
    assert rows[2]["status"] == "not_admitted" and rows[2]["admitted_candidate_count"] == 0
    assert rows[2]["semantic_coverage"] == "not_established"


def test_unknown_or_weak_task_never_borrows_another_task_packet():
    records = list(baseline())
    records[2] = dict(
        records[2],
        candidate_count=0,
        retained_candidate_count=0,
        confidence=0,
        status="no_candidates",
        semantic_coverage="not_established",
        candidate_source_links=(),
    )
    rows = settle_task_coverage(tuple(records), {"doc_1_chunk_0"})
    assert rows[2]["status"] == "no_reliable_candidates" and rows[2]["admitted_candidate_count"] == 0
    assert rows[1]["status"] == "candidate_evidence_admitted"


def test_source_and_index_generation_are_bound_but_not_fact_certificates():
    rows = settle_task_coverage(baseline(), {"doc_1_chunk_0", "doc_2_chunk_0"})
    assert all(x["index_generation"] == 7 and x["verified_original_source_count"] == 0 for x in rows)
    assert all(x["semantic_coverage"] == "unverified" for x in rows)


def original_packet():
    doc = dict(
        id=1,
        title="青川规则",
        content="完整原文：材料齐全仍须核验，不能跳过审查。",
        category="public",
        knowledge_base_id=7,
    )
    bundle = dict(
        results=[
            dict(
                id="doc_1_chunk_0",
                document_id=1,
                title=doc["title"],
                content="索引片段",
                category="public",
                knowledge_base_id=7,
            )
        ],
        source_coverage=(
            dict(
                source_id="doc_1",
                source_title=doc["title"],
                indexed_document_ids=["doc_1_chunk_0", "doc_1_chunk_1"],
                retrieved_document_ids=["doc_1_chunk_0"],
            ),
        ),
    )
    attached = attach_original_sources(bundle, lambda identity: doc, source_budget_tokens=65536, authority_revision=11)
    return attached["source_coverage"], attached["original_source_packets"]


def test_verified_whole_original_counts_even_when_its_chunk_is_not_admitted():
    records = baseline()
    sources, packets = original_packet()
    rows = settle_task_coverage(records, {"doc_1_original"}, admitted_packets=packets, source_coverage=sources)
    assert rows[1]["status"] == "candidate_evidence_admitted" and rows[1]["admitted_chunk_count"] == 0
    assert rows[1]["verified_original_source_count"] == 1 and rows[1]["original_authority_revisions"] == (11,)
    assert rows[1]["semantic_coverage"] == "unverified" and rows[2]["status"] == "not_admitted"


@pytest.mark.parametrize("change", ["body", "text", "identity", "parent"])
def test_tampered_or_unrelated_original_never_grants_task_visibility(change):
    records = baseline()
    sources, packets = original_packet()
    positive = settle_task_coverage(records, {"doc_1_original"}, admitted_packets=packets, source_coverage=sources)
    assert positive[1]["verified_original_source_count"] == 1
    packet = deepcopy(packets[0])
    if change == "body":
        packet["original_body"] = "Missing limitation"
    elif change == "text":
        packet["text"] = "Altered text"
    elif change == "identity":
        packet["document_ids"] = ["doc_2_original"]
    else:
        packet["original_source_id"] = "doc_2"
    rows = settle_task_coverage(records, {"doc_1_original"}, admitted_packets=(packet,), source_coverage=sources)
    assert rows[1]["status"] == "not_admitted" and rows[1]["verified_original_source_count"] == 0


def test_original_identity_without_exact_packet_does_not_grant_visibility():
    records = baseline()
    sources, packets = original_packet()
    rows = settle_task_coverage(records, {"doc_1_original"}, source_coverage=sources)
    assert rows[1]["status"] == "not_admitted" and rows[1]["verified_original_source_count"] == 0


@pytest.mark.parametrize(
    "change",
    [
        "boolean_index",
        "wrong_parent",
        "generation",
        "outside_query",
        "confidence",
        "semantic_grant",
        "counts",
        "duplicate_identity",
    ],
)
def test_invalid_producer_receipts_rejected_after_valid_positive_baseline(change):
    records = deepcopy(list(baseline()))
    row = records[1]
    if change == "boolean_index":
        row["task_index"] = True
    elif change == "wrong_parent":
        row["candidate_source_links"][0]["source_id"] = "doc_2"
    elif change == "generation":
        row["index_generation"] = 8
    elif change == "outside_query":
        row["query"] = "Invented object"
    elif change == "confidence":
        row["confidence"] = float("nan")
    elif change == "semantic_grant":
        row["semantic_coverage"] = "verified"
    elif change == "counts":
        row["retained_candidate_count"] = True
    else:
        row["candidate_source_links"] *= 2
        row["retained_candidate_count"] = row["candidate_count"] = 2
    with pytest.raises(ValueError):
        validate_task_candidates(records, query=QUERY)


def test_foreign_root_question_cannot_rebind_a_receipt():
    with pytest.raises(ValueError):
        validate_task_candidates(baseline(), query="different full query")
