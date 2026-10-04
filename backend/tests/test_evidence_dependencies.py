"""Whole-source dependencies follow actual admitted referring evidence."""

from copy import deepcopy
from dataclasses import replace
from html import unescape
from threading import RLock

import pytest

from inference.evidence_dependencies import dependency_satisfied
from inference.generation_request import GenerationRequest, RetrievalResult, build_generation_request
from knowledge.evidence_packets import document_evidence_packets
from knowledge.original_sources import attach_original_sources
from knowledge.source_expansion import expand_source_context
from knowledge.vector_db import VectorDatabase


def scenario(large=False, independent_target=False):
    def row(parent, title, text, index=0):
        return dict(
            id=f"doc_{parent}_chunk_{index}",
            document_id=parent,
            chunk_index=index,
            knowledge_base_id=7,
            title=title,
            category="规程",
            content=text,
        )

    head = row(1, "青川主规程", "独立基础办理费3元。")
    tail = row(
        1, "青川主规程", ("完整独立背景。" * 50000 if large else "完整独立背景。") + "具体附加条件见《蓝溪补充》。", 1
    )
    target = row(2, "蓝溪补充", "关联完整条件：附加费用8元，材料与审核均通过才可使用。")
    records = [head, tail, target]
    v = VectorDatabase.__new__(VectorDatabase)
    v._lock = RLock()
    v._cache_generation = 11
    v.snapshot_validated = True
    v.metadata = records
    roots = [head, target] if independent_target else [head]
    data = expand_source_context(
        dict(results=roots, confidence=0.8, abstained=False),
        v,
        expected_generation=11,
        source_budget_tokens=65536,
        filters={"knowledge_base_id": 7},
        query=(
            "分别读取《青川主规程》《蓝溪补充》。"
            if independent_target
            else "读取《青川主规程》的基础费用与引用附加条件。"
        ),
    )
    docs = {
        1: dict(
            id=1,
            title=head["title"],
            category=head["category"],
            knowledge_base_id=7,
            content=head["content"] + "\n" + tail["content"],
        ),
        2: dict(
            id=2, title=target["title"], category=target["category"], knowledge_base_id=7, content=target["content"]
        ),
    }
    data = attach_original_sources(data, docs.get, source_budget_tokens=65536, authority_revision=77)
    packets = document_evidence_packets(data["results"]) + tuple(data["original_source_packets"])
    request = GenerationRequest(
        message="核对基础费用及附加条件；原文未完整可见时明确不能核对引用关系。",
        history=(
            {"role": "user", "content": "独立前任务条件已完整提供。"},
            {"role": "assistant", "content": "前任务已结束。"},
        ),
        context_window_tokens=65536,
        max_tokens=2048,
        evidence_max_chars=0,
        retrieval=RetrievalResult(
            status="ok",
            evidence="\n".join(p["text"] for p in packets),
            evidence_packets=packets,
            source_coverage=data["source_coverage"],
            source_references=data.get("source_references", ()),
        ),
    )
    return request, data, head, tail, target


def positive():
    request, data, head, tail, target = scenario()
    plan = build_generation_request(request)
    assert all(row["content"] in unescape(plan.messages[-1]["content"]) for row in [head, tail, target])
    assert all(r["original_status"] == "verified_original_body_admitted" for r in plan.retrieval.source_coverage)
    assert plan.generation["max_tokens"] == 2048
    return request, data, head, tail, target


def only_originals(request):
    packets = tuple(p for p in request.retrieval.evidence_packets if p.get("original_source_id"))
    assert len(packets) == 2
    return replace(request, retrieval=replace(request.retrieval, evidence_packets=packets))


def test_complete_referenced_original_preserves_its_actual_dependency():
    request, data, head, tail, target = positive()
    source = next(p for p in data["original_source_packets"] if p["original_source_id"] == "doc_2")
    assert source["kind"] == "background" and source["supporting_document_ids"] == [tail["id"]]
    assert source["supporting_source_refs"][0]["text"] == tail["content"]


def test_omitted_referring_chunk_and_original_cannot_leave_orphan_original():
    positive()
    request, data, head, tail, target = scenario(True)
    plan = build_generation_request(request)
    text = unescape(plan.messages[-1]["content"])
    assert head["content"] in text and tail["content"] not in text and target["content"] not in text
    assert all(r["original_status"] != "verified_original_body_admitted" for r in plan.retrieval.source_coverage)
    assert plan.generation["max_tokens"] == 2048


def test_actual_independent_target_is_not_dropped_for_an_unavailable_parent():
    positive()
    request, data, head, tail, target = scenario(True, True)
    plan = build_generation_request(request)
    assert target["content"] in unescape(plan.messages[-1]["content"])
    packet = next(p for p in data["original_source_packets"] if p["original_source_id"] == "doc_2")
    assert packet["kind"] == "evidence" and "supporting_document_ids" not in packet


def test_verified_original_parent_can_satisfy_missing_chunk_dependency():
    request, *_ = positive()
    request = only_originals(request)
    plan = build_generation_request(request)
    assert len(plan.retrieval.evidence_packets) == 2
    assert all(r["original_status"] == "verified_original_body_admitted" for r in plan.retrieval.source_coverage)


def test_deferred_child_is_reconsidered_after_verified_parent_arrives():
    request, *_ = positive()
    request = only_originals(request)
    packets = tuple(reversed(request.retrieval.evidence_packets))
    plan = build_generation_request(replace(request, retrieval=replace(request.retrieval, evidence_packets=packets)))
    assert [p["original_source_id"] for p in plan.retrieval.evidence_packets] == ["doc_1", "doc_2"]


@pytest.mark.parametrize("change", ["body", "text", "id", "source", "kb", "missing_index"])
def test_unverifiable_whole_parent_never_grants_a_dependency(change):
    request, *_ = positive()
    request = only_originals(request)
    parent, child = deepcopy(request.retrieval.evidence_packets)
    assert dependency_satisfied(child, [parent], request.retrieval.source_coverage)
    coverage = deepcopy(request.retrieval.source_coverage)
    if change == "body":
        parent["original_body"] = "其他未提及引用的正文"
    elif change == "text":
        parent["text"] = "其他正文"
    elif change == "id":
        parent["document_ids"] = ["doc_3_original"]
    elif change == "source":
        parent["original_source_id"] = "doc_3"
    elif change == "kb":
        parent["knowledge_base_id"] = 8
    else:
        coverage[0]["indexed_document_ids"] = ["doc_1_chunk_0"]
        coverage[0]["retrieved_document_ids"] = ["doc_1_chunk_0"]
    assert not dependency_satisfied(child, [parent], coverage)


@pytest.mark.parametrize(
    "change", ["parent_id", "kb", "relation", "missing_title", "support", "missing_text", "conflict"]
)
def test_invalid_dependency_receipt_is_rejected_from_a_valid_baseline(change):
    request, *_ = positive()
    request = only_originals(request)
    parent, child = deepcopy(request.retrieval.evidence_packets)
    assert dependency_satisfied(child, [parent], request.retrieval.source_coverage)
    ref = child["supporting_source_refs"][0]
    if change == "parent_id":
        ref["source_id"] = "doc_99"
    elif change == "kb":
        ref["knowledge_base_id"] = 8
    elif change == "relation":
        ref["relation"] = "same_source"
    elif change == "missing_title":
        ref["target_title"] = "另一个未引用的标题"
    elif change == "support":
        child["supporting_document_ids"] = ["doc_1_chunk_99"]
    elif change == "missing_text":
        ref["text"] = ""
    else:
        child["supporting_source_refs"] = (*child["supporting_source_refs"], dict(ref, text="另一段《蓝溪补充》。"))
    with pytest.raises(ValueError):
        dependency_satisfied(child, [parent], request.retrieval.source_coverage)


def test_unrooted_cycle_cannot_admit_itself_or_loop():
    request, *_ = positive()
    request = only_originals(request)
    parent, child = deepcopy(request.retrieval.evidence_packets)
    parent.update(kind="background", supporting_document_ids=child["document_ids"])
    plan = build_generation_request(
        replace(request, retrieval=replace(request.retrieval, evidence_packets=(parent, child)))
    )
    assert not plan.retrieval.evidence_packets and plan.retrieval.status == "character_abstention"


def test_stored_parent_id_without_actual_packets_is_not_dependency_evidence():
    request, *_ = positive()
    request = only_originals(request)
    parent, child = request.retrieval.evidence_packets
    assert dependency_satisfied(child, [parent], request.retrieval.source_coverage)
    assert not dependency_satisfied(child, [], request.retrieval.source_coverage)
