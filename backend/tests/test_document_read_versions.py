"""Explicit independent source roots retain same-title versions without exclusion false positives."""

from threading import RLock

import pytest

from inference.generation_request import GenerationRequest, RetrievalResult, build_generation_request
from knowledge.evidence_packets import document_evidence_packets
from knowledge.source_expansion import expand_source_context, requested_document_titles
from knowledge.vector_db import VectorDatabase


@pytest.mark.parametrize(
    "suffix",
    [
        "按正文版本分别列出。",
        "请分别列出版本，不要合并。",
        "请逐一核对后分别列出费用。",
        "按两个版本分别读取全文。",
    ],
)
def test_positive_separate_read_is_not_negative_bie(suffix):
    assert requested_document_titles("逐项比较《A》《B》。" + suffix) == ("A", "B")


@pytest.mark.parametrize(
    "suffix",
    [
        "别列出第一份。",
        "请别读取第一份。",
        "另外别阅读第一份。",
        "并且别列出第一份。",
        "但别查看第一份。",
        "不要分别列出第一份。",
        "只分别读取第一份。",
        "请不必逐项比较第一份。",
    ],
)
def test_true_negative_read_clause_still_defers(suffix):
    assert requested_document_titles("逐项比较《A》《B》。" + suffix) == ()


def row(parent, title, text, kb=7, index=0, category="配送"):
    return dict(
        id=f"doc_{parent}_chunk_{index}",
        document_id=parent,
        chunk_index=index,
        knowledge_base_id=kb,
        title=title,
        category=category,
        content=text,
    )


def expand(records, anchors, query="逐项比较《A》《B》。两个版本分别列出。", filters=None, budget=65536):
    v = VectorDatabase.__new__(VectorDatabase)
    v._lock = RLock()
    v._cache_generation = 11
    v.snapshot_validated = True
    v.metadata = records
    return expand_source_context(
        dict(results=anchors, confidence=0.8, abstained=False),
        v,
        expected_generation=11,
        source_budget_tokens=budget,
        filters=filters,
        query=query,
    )


def plan(result, window=65536):
    packets = document_evidence_packets(result["results"])
    return build_generation_request(
        GenerationRequest(
            message="逐项比较各独立版本",
            context_window_tokens=window,
            evidence_max_chars=0,
            retrieval=RetrievalResult(
                status="ok",
                evidence="\n".join(p["text"] for p in packets),
                evidence_packets=packets,
                source_coverage=result["source_coverage"],
            ),
        )
    )


def test_same_title_two_versions_are_independent_whole_task_roots():
    anchor = row(1, "A", "A2026费用1")
    old = row(2, "B", "B2025费用8")
    new = row(3, "B", "B2026费用11")
    result = expand([anchor, old, new], [anchor])
    assert result["results"][0] == anchor and result["confidence"] == 0.8
    assert result["ambiguous_requested_titles"] == ["B"] and result["unresolved_requested_titles"] == []
    assert [r["id"] for r in result["results"]] == [anchor["id"], old["id"], new["id"]]
    assert all(
        r["retrieval_role"] == "requested_source" and r["score"] == 0 and "supporting_document_ids" not in r
        for r in result["results"][1:]
    )
    wire = plan(result).messages[-1]["content"]
    assert all(r["content"] in wire for r in [anchor, old, new])
    assert {c["source_id"] for c in result["source_coverage"]} == {"doc_1", "doc_2", "doc_3"}


def test_ranked_one_version_does_not_hide_another_or_independent_requested_doc():
    old = row(1, "A", "2025编号951费用8")
    new = row(2, "A", "2026编号952费用11")
    other = row(3, "B", "独立项目编号953")
    result = expand([old, new, other], [new])
    assert result["results"][0] == new and result["ambiguous_requested_titles"] == ["A"]
    assert len(result["results"]) == 3 and all(
        r["content"] in plan(result).messages[-1]["content"] for r in [old, new, other]
    )


@pytest.mark.parametrize("outside", ["knowledge_base", "category"])
def test_same_title_versions_never_bypass_existing_scope(outside):
    anchor = row(1, "A", "锚点")
    allowed = row(2, "B", "允许版本")
    other = row(
        3,
        "B",
        "范围外版本",
        kb=8 if outside == "knowledge_base" else 7,
        category="其他" if outside == "category" else "配送",
    )
    result = expand([anchor, allowed, other], [anchor], filters={"category": "配送"} if outside == "category" else None)
    assert (
        result["results"][1]["id"] == allowed["id"]
        and len(result["results"]) == 2
        and result.get("ambiguous_requested_titles") == []
    )


def test_duplicate_title_multi_chunk_versions_keep_separate_coverage():
    anchor = row(1, "A", "A原文")
    records = [
        anchor,
        row(2, "B", "2025开头"),
        row(2, "B", "2025限制", index=1),
        row(3, "B", "2026开头"),
        row(3, "B", "2026例外", index=1),
    ]
    result = expand(records, [anchor])
    compiled = plan(result)
    assert all(r["content"] in compiled.messages[-1]["content"] for r in records)
    scopes = {s["source_id"]: s for s in compiled.retrieval.source_coverage}
    assert scopes["doc_2"]["admitted_chunk_count"] == scopes["doc_3"]["admitted_chunk_count"] == 2


def test_budget_omission_does_not_certify_all_versions_or_drop_independent_small_source():
    anchor = row(1, "A", "锚点")
    huge = row(2, "B", "旧版超长正文" * 5000)
    small = row(3, "B", "新版完整参数")
    missing = row(4, "C", "独立完整参数")
    result = expand([anchor, huge, small, missing], [anchor], query="分别读取《A》《B》《C》", budget=100)
    assert (
        huge["content"] not in plan(result).messages[-1]["content"]
        and small["content"] in plan(result).messages[-1]["content"]
        and missing["content"] in plan(result).messages[-1]["content"]
    )
    scopes = {s["source_id"]: s for s in plan(result).retrieval.source_coverage}
    assert scopes["doc_2"]["admitted_chunk_count"] == 0 and scopes["doc_2"]["status"] != "complete"


def test_duplicate_index_identity_is_still_invalid_authority():
    anchor = row(1, "A", "第一份")
    duplicate = {**anchor, "content": "冒用同一索引身份"}
    with pytest.raises(RuntimeError, match="Ambiguous indexed source identity"):
        expand([anchor, duplicate], [anchor])
