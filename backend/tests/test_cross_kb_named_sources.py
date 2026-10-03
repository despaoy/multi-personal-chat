"""Direct named-source reads use explicit scope, independently of ranked KBs."""

import copy
import json
from pathlib import Path
from threading import RLock

import pytest

from inference.generation_request import GenerationRequest, RetrievalResult, build_generation_request
from knowledge.evidence_packets import document_evidence_packets
from knowledge.source_expansion import expand_source_context
from knowledge.vector_db import VectorDatabase

CASE = json.loads(
    (Path(__file__).parent / "fixtures/deepseek_cross_kb_named_source_case.json").read_text(encoding="utf-8")
)


def expand(records=None, anchors=None, *, filters=None, query=None, budget=65536, abstained=False):
    index = VectorDatabase.__new__(VectorDatabase)
    index._lock = RLock()
    index._cache_generation = 11
    index.snapshot_validated = True
    index.metadata = copy.deepcopy(CASE["frozen_indexed_records"] if records is None else records)
    bundle = dict(
        results=copy.deepcopy(CASE["frozen_ranked_anchors"] if anchors is None else anchors),
        confidence=0.5748,
        abstained=abstained,
    )
    original = copy.deepcopy(bundle)
    result = expand_source_context(
        bundle,
        index,
        expected_generation=11,
        source_budget_tokens=budget,
        filters=filters,
        query=CASE["question"] if query is None else query,
    )
    assert bundle == original
    return result


def plan(result, *, window=65536):
    packets = document_evidence_packets(result["results"])
    return build_generation_request(
        GenerationRequest(
            message=CASE["question"],
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


def test_full_actual_four_kb_question_preserves_original_ranking_and_all_sources():
    result = expand()
    assert result["results"][:3] == CASE["frozen_ranked_anchors"]
    assert result["confidence"] == 0.5748 and not result["unresolved_requested_titles"]
    assert {x["knowledge_base_id"] for x in result["results"]} == {3, 4, 5, 6}
    missing = result["results"][-1]
    assert missing["id"] == "doc_507_chunk_0" and missing["retrieval_role"] == "requested_source"
    assert missing["score"] == 0 and "supporting_document_ids" not in missing
    wire = plan(result).messages[-1]["content"]
    assert all(x["content"] in wire for x in CASE["documents"])


@pytest.mark.parametrize("kb", [3, 4, 5, 6])
def test_one_explicit_kb_cannot_read_named_docs_outside_its_original_filter(kb):
    rows = [x for x in CASE["frozen_indexed_records"] if x["knowledge_base_id"] == kb]
    result = expand(anchors=rows, filters={"knowledge_base_id": kb})
    assert len(result["results"]) == 1 and result["results"][0]["knowledge_base_id"] == kb
    assert set(result["unresolved_requested_titles"]) == {
        x["title"] for x in CASE["frozen_indexed_records"] if x["knowledge_base_id"] != kb
    }
    assert result["ambiguous_requested_titles"] == []


def test_same_title_in_other_unrestricted_kb_is_independent_root_with_own_identity():
    records = copy.deepcopy(CASE["frozen_indexed_records"])
    second = dict(
        records[-1],
        id="doc_508_chunk_0",
        document_id=508,
        knowledge_base_id=77,
        content="完整公共寄件说明另一版本：编号LZ-885，费用15元，时限72小时，必要条件是有效地址；地址无效须暂停。",
    )
    result = expand(records=[*records, second])
    assert result["ambiguous_requested_titles"] == [second["title"]]
    assert {x["id"] for x in result["results"]} == {x["id"] for x in [*records, second]}
    assert all(x["content"] in plan(result).messages[-1]["content"] for x in [records[-1], second])


def test_explicit_category_filter_is_not_replaced_by_cross_kb_title_match():
    records = copy.deepcopy(CASE["frozen_indexed_records"])
    records[-1]["category"] = "不在本轮分类"
    result = expand(records=records, filters={"category": "办理"})
    assert all(x["category"] == "办理" for x in result["results"])
    assert result["unresolved_requested_titles"] == ["绿泽寄件受理"]


def test_direct_root_does_not_authorize_its_cross_kb_internal_reference():
    a = dict(CASE["frozen_indexed_records"][0], content="完整正文：内部另引用《未点名跨库补充》。")
    b = dict(CASE["frozen_indexed_records"][1], title="未点名跨库补充", content="内部引用不能扩到这个不同库。")
    result = expand(records=[a, b], anchors=[a], query="分别读取《绿泽窗口受理》这一份说明。")
    assert result["results"] == [a] and len(result["source_coverage"]) == 1


def test_missing_or_excluded_read_does_not_expand_cross_kb_sources():
    result = expand(query="不要分别读取《绿泽寄件受理》，先核对已检索材料。")
    assert result["results"] == CASE["frozen_ranked_anchors"]
    assert "requested_source_titles" not in result


def test_cross_kb_root_can_survive_unrelated_rejected_large_rank_anchor():
    a = dict(CASE["frozen_indexed_records"][0], content="完整大型原文" * 9000)
    b = CASE["frozen_indexed_records"][-1]
    result = expand(records=[a, b], anchors=[a], query="分别读取《绿泽窗口受理》《绿泽寄件受理》这二份说明。")
    wire = plan(result, window=8192).messages[-1]["content"]
    assert b["content"] in wire and a["content"] not in wire


def test_unvalidated_rank_authority_is_not_rescued_by_direct_titles():
    stale = copy.deepcopy(CASE["frozen_ranked_anchors"])
    stale[0]["content"] = "过期正文"
    with pytest.raises(RuntimeError, match="no longer matches indexed authority"):
        expand(anchors=stale)


def test_abstained_bundle_does_not_gain_cross_kb_authority():
    result = expand(abstained=True)
    assert result["results"] == CASE["frozen_ranked_anchors"] and result["abstained"]
