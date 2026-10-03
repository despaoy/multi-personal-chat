"""Explicit cross-document references under snapshot, scope and packet budgets."""

from threading import RLock

import pytest

from inference.generation_request import GenerationRequest, RetrievalResult, build_generation_request
from knowledge.evidence_packets import document_evidence_packets
from knowledge.source_expansion import expand_source_context
from knowledge.vector_db import VectorDatabase


def row(parent, title, text, *, index=0, kb=7, category="规程"):
    return dict(
        id=f"doc_{parent}_chunk_{index}",
        document_id=parent,
        chunk_index=index,
        knowledge_base_id=kb,
        title=title,
        category=category,
        content=text,
    )


def expand(records, anchors=None, *, budget=65536, filters=None):
    v = VectorDatabase.__new__(VectorDatabase)
    v._lock = RLock()
    v._cache_generation = 11
    v.snapshot_validated = True
    v.metadata = records
    return expand_source_context(
        dict(results=anchors or [records[0]], confidence=0.8, abstained=False),
        v,
        expected_generation=11,
        source_budget_tokens=budget,
        filters=filters,
    )


def plan(result, window=65536):
    packets = document_evidence_packets(result["results"])
    return build_generation_request(
        GenerationRequest(
            message="核对当前参数",
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


def test_exact_current_reference_adds_whole_source_without_changing_rank_or_confidence():
    main = row(1, "主规程", "历史种子31；须结合《现行修订单 A2》。")
    current = row(2, "现行修订单 A2", "现行种子97")
    tail = row(2, "现行修订单 A2", "批大小10", index=1)
    old = row(3, "历史修订单 A1", "历史种子53")
    result = expand([main, current, tail, old])
    assert result["results"][0] == main and result["confidence"] == 0.8
    assert [r["id"] for r in result["results"]] == [main["id"], current["id"], tail["id"]]
    assert all(
        r["score"] == r["normalized_score"] == 0 and r["supporting_document_ids"] == [main["id"]]
        for r in result["results"][1:]
    )
    actual = plan(result)
    assert current["content"] in actual.messages[-1]["content"] and tail["content"] in actual.messages[-1]["content"]
    target = next(r for r in actual.retrieval.source_coverage if r["source_id"] == "doc_2")
    assert target["indexed_chunk_count"] == target["admitted_chunk_count"] == 2


@pytest.mark.parametrize("scope", ["other_kb", "category"])
def test_reference_never_crosses_knowledge_base_or_original_filter(scope):
    main = row(1, "主规程", "须结合《同名修订》。")
    target = row(
        2,
        "同名修订",
        "范围外参数",
        kb=8 if scope == "other_kb" else 7,
        category="其他" if scope == "category" else "规程",
    )
    result = expand([main, target], filters={"knowledge_base_id": 7, "category": "规程"})
    assert result["results"] == [main] and len(result["source_coverage"]) == 1


def test_duplicate_exact_titles_do_not_choose_arbitrary_authority():
    main = row(1, "主规程", "关联《修订》。")
    result = expand([main, row(2, "修订", "值10"), row(3, "修订", "值20")])
    assert result["results"] == [main] and result["confidence"] == 0.8
    assert result["source_references"] == (dict(
        referring_source_id="doc_1", referring_source_title="主规程", referenced_title="修订",
        lookup_status="ambiguous_in_referring_scope", source_ids=["doc_2", "doc_3"],
    ),)


def test_nested_reference_cycle_terminates_and_keeps_actual_referring_packet_dependencies():
    main = row(1, "主规程", "关联《修订》。")
    a = row(2, "修订", "需《补充》，并参照《主规程》。")
    b = row(3, "补充", "参数97，参照《修订》。")
    result = expand([main, a, b])
    assert [r["id"] for r in result["results"]] == [main["id"], a["id"], b["id"]]
    assert result["results"][1]["supporting_document_ids"] == [main["id"]] and result["results"][2][
        "supporting_document_ids"
    ] == [a["id"]]
    assert len(plan(result).retrieval.evidence_packets) == 3


def test_producer_budget_omitted_reference_retains_truthful_partial_scope():
    main = row(1, "主规程", "关联《修订》。")
    target = row(2, "修订", "完整大条款" * 2000)
    result = expand([main, target], budget=100)
    assert result["results"] == [main]
    target_scope = next(r for r in plan(result).retrieval.source_coverage if r["source_id"] == "doc_2")
    assert (
        target_scope["indexed_chunk_count"] == 1
        and target_scope["admitted_chunk_count"] == 0
        and target_scope["status"] == "partial"
        and target_scope["omission_reasons"] == ["source_context_budget"]
    )


def test_final_budget_cannot_admit_reference_when_actual_referring_chunk_was_rejected():
    main = row(1, "主规程", "关联《修订》。" + "完整必要正文" * 3000)
    target = row(2, "修订", "不能无来源独立接纳的参数97")
    other = row(3, "其他选中资料", "有效独立设备事实")
    result = expand([main, target, other], anchors=[main, other])
    actual = plan(result, 8192)
    assert (
        other["content"] in actual.messages[-1]["content"] and target["content"] not in actual.messages[-1]["content"]
    )
    assert next(r for r in actual.retrieval.source_coverage if r["source_id"] == "doc_2")["admitted_chunk_count"] == 0


def test_malformed_target_identity_is_rejected_before_use():
    main = row(1, "主规程", "关联《修订》。")
    bad = {**row(2, "修订", "当前参数97"), "id": "doc_99_chunk_0"}
    with pytest.raises(RuntimeError, match="Invalid indexed sibling identity"):
        expand([main, bad])


def test_unresolved_or_unquoted_names_do_not_authorize_unrelated_documents():
    main = row(1, "主规程", "未找到《缺失修订》，历史档案有未引用资料的名称。")
    other = row(2, "未引用资料", "不相关参数")
    result = expand([main, other])
    assert result["results"] == [main] and len(result["source_coverage"]) == 1
