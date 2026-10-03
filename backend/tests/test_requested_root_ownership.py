"""Complete synthetic source ownership, not native private fixtures."""

import copy
from types import SimpleNamespace

import pytest

from inference.generation_request import GenerationRequest, RetrievalResult, build_generation_request
from knowledge.evidence_packets import document_evidence_packets
from knowledge.rag_helper import RAGHelper
from knowledge.source_expansion import expand_source_context
from tests.test_document_read_versions import row
from tests.test_title_without_anchors import vector


def expand(records, anchors, query="读取《紫茶规程》《本人版本说明》。", **changes):
    bundle = (
        dict(results=anchors, confidence=0.9, abstained=False, citations=[dict(source_id=a["id"]) for a in anchors])
        | changes
    )
    old, corpus = copy.deepcopy(bundle), copy.deepcopy(records)
    result = expand_source_context(
        bundle, vector(records), expected_generation=11, source_budget_tokens=65536, query=query
    )
    assert bundle == old and records == corpus
    return result


def plan(result, window=65536):
    packets = document_evidence_packets(result["results"])
    return build_generation_request(
        GenerationRequest(
            message="读取完整指定原文并核对所有条件。",
            max_tokens=64,
            context_window_tokens=window,
            evidence_max_chars=0,
            retrieval=RetrievalResult(
                status="ok",
                evidence_packets=packets,
                source_coverage=result.get("source_coverage", ()),
                evidence="\n".join(p["text"] for p in packets),
            ),
        )
    )


@pytest.mark.parametrize("case", ["mixed", "only_unrelated", "missing_requested"])
def test_unrelated_ranked_root_cannot_supply_declared_document_task(case):
    a = row(1, "紫茶规程", "登记通过才使用；预约不免核验。")
    b = row(2, "本人版本说明", "本人条件为仓位已登记且签章校验通过。")
    c = row(9, "紫茶代办", "代办费用30；参照《代办补充》。")
    d = row(10, "代办补充", "代办需要委托书，不能当本人的条件。")
    records = [a, b, c, d] if case != "missing_requested" else [c, d]
    anchors = [dict(c, score=0.95)]
    if case == "mixed":
        anchors.append(dict(a, score=0.2))
    result = expand(records, anchors)
    assert {r["document_id"] for r in result["results"]} == ({1, 2} if case != "missing_requested" else set())
    assert result["confidence"] <= (0.2 if case == "mixed" else 0.0)
    assert all(x["source_id"] != c["id"] for x in result["citations"])
    if case == "missing_requested":
        assert result["abstained"] and result["unresolved_requested_titles"] == [a["title"], b["title"]]
    else:
        wire = plan(result).messages[-1]["content"]
        assert a["content"] in wire and b["content"] in wire
        assert c["content"] not in wire and d["content"] not in wire


def test_referenced_ranked_neighbor_is_context_with_real_referring_dependency():
    a = row(1, "紫茶规程", "完整条件还需《补充条款》。")
    b = row(2, "本人版本说明", "自己的条件不等于公定条件。")
    c = row(9, "补充条款", "核验失败走普通流程，另参照《末尾例外》。")
    d = row(10, "末尾例外", "预约不豁免核验。")
    result = expand([a, b, c, d], [dict(c, score=0.99), dict(a, score=0.3)])
    context = next(r for r in result["results"] if r["document_id"] == 9)
    assert context["retrieval_role"] == "source_context" and context["score"] == 0
    assert context["supporting_document_ids"] == [a["id"]]
    tail = next(r for r in result["results"] if r["document_id"] == 10)
    assert tail["supporting_document_ids"] == [c["id"]]
    assert all(r["content"] in plan(result).messages[-1]["content"] for r in [a, b, c, d])


def test_requested_anchor_order_scores_and_all_versions_survive_scoping():
    old = row(1, "紫茶规程", "2025费用8元；旧条件。")
    current = row(2, "紫茶规程", "2026费用11元；新条件。")
    b = row(3, "本人版本说明", "所有版本独立列出。")
    c = row(9, "紫茶代办", "不属于指定原文。")
    ranked = [dict(b, score=0.7), dict(c, score=0.95), dict(current, score=0.6)]
    result = expand([old, current, b, c], ranked)
    assert result["results"][:2] == [ranked[0], ranked[2]]
    assert result["ambiguous_requested_titles"] == ["紫茶规程"]
    assert {r["document_id"] for r in result["results"]} == {1, 2, 3}
    assert result["confidence"] <= 0.7


def test_scoping_does_not_upgrade_already_lower_original_confidence():
    a = row(1, "紫茶规程", "完整条件。")
    b = row(2, "本人版本说明", "完整版本。")
    c = row(9, "代办", "无关。")
    result = expand([a, b, c], [dict(c, score=0.99), dict(a, score=0.8)], confidence=0.1)
    assert result["confidence"] <= 0.1 and not result["abstained"]


def test_generic_query_keeps_ranked_root_and_its_reference_contract():
    a = row(1, "代办", "请结合《补充》。")
    b = row(2, "补充", "完整委托书条件。")
    anchor = dict(a, score=0.95)
    result = expand([a, b], [anchor], query="代办的完整条件是什么？")
    assert result["results"][0] == anchor and result["confidence"] == 0.9
    assert result["results"][1]["supporting_document_ids"] == [a["id"]]


@pytest.mark.parametrize("scope", ["kb", "category"])
def test_declared_roots_still_obey_original_filter_after_unrelated_rank_exclusion(scope):
    a = row(1, "紫茶规程", "范围内完整原文。")
    b = row(
        2,
        "本人版本说明",
        "范围外完整原文。",
        kb=8 if scope == "kb" else 7,
        category="其他" if scope == "category" else "配送",
    )
    c = row(9, "代办", "范围内但未请求。")
    v = vector([a, b, c])
    filters = {"knowledge_base_id": 7} if scope == "kb" else {"category": "配送"}
    result = expand_source_context(
        dict(results=[dict(c, score=0.99)], confidence=0.9, abstained=False),
        v,
        expected_generation=11,
        source_budget_tokens=65536,
        filters=filters,
        query="读取《紫茶规程》《本人版本说明》。",
    )
    assert {r["document_id"] for r in result["results"]} == {1}
    assert result["unresolved_requested_titles"] == [b["title"]]


@pytest.mark.parametrize("broken", ["stale_content", "identity", "generation"])
def test_unrequested_anchor_corruption_is_not_hidden_by_scope_filter(broken):
    a = row(1, "紫茶规程", "有效原文。")
    c = row(9, "代办", "原索引内容。")
    anchor = dict(c, score=0.9)
    if broken == "stale_content":
        anchor["content"] = "冒用内容。"
    if broken == "identity":
        c["id"] = anchor["id"] = "doc_8_chunk_0"
    v = vector([a, c])
    if broken == "generation":
        v._cache_generation = 12
    with pytest.raises(RuntimeError):
        expand_source_context(
            dict(results=[anchor], confidence=0.9, abstained=False),
            v,
            expected_generation=11,
            source_budget_tokens=65536,
            query="读取《紫茶规程》。",
        )


def test_nondocument_rank_does_not_authorize_a_named_read():
    a = row(1, "紫茶规程", "完整原文。")
    c = dict(
        id="raw_9",
        document_id=None,
        chunk_index=None,
        knowledge_base_id=7,
        title="代办散文",
        category="配送",
        content="不属于命名原文。",
    )
    result = expand([a, c], [dict(c, score=0.99)], query="读取《紫茶规程》。")
    assert [r["document_id"] for r in result["results"]] == [1] and result["confidence"] == 0.0


def test_ranked_ambiguous_reference_cannot_pick_a_version_outside_direct_roots():
    a = row(1, "紫茶规程", "需结合《补充》。")
    b = row(2, "补充", "旧版本条件。")
    c = row(3, "补充", "新版本条件。")
    result = expand([a, b, c], [dict(b, score=0.99)], query="读取《紫茶规程》。")
    assert {r["document_id"] for r in result["results"]} == {1}
    assert result["source_references"][0]["lookup_status"] == "ambiguous_in_referring_scope"


def test_missing_requested_source_cannot_be_replaced_by_unrelated_strong_neighbor():
    a = row(1, "紫茶规程", "完整公定条件。")
    c = row(9, "代办", "看似相似的个人说明。")
    result = expand([a, c], [dict(c, score=0.99), dict(a, score=0.3)])
    assert result["unresolved_requested_titles"] == ["本人版本说明"]
    assert {x["source_id"] for x in result["source_coverage"]} == {"doc_1"}


def test_literal_reference_cycle_terminates_under_declared_root():
    a = row(1, "紫茶规程", "参照《补充》。")
    b = row(2, "补充", "完整条件，参照《紫茶规程》。")
    result = expand([a, b], [dict(b, score=0.99)], query="读取《紫茶规程》。")
    assert [r["document_id"] for r in result["results"]] == [1, 2]
    assert result["results"][1]["supporting_document_ids"] == [a["id"]]


def test_referenced_context_cannot_survive_omitted_root_in_generation_budget():
    a = row(1, "紫茶规程", "很长原文条件" * 6000 + "参照《补充》。")
    b = row(2, "本人版本说明", "独立完整个人版本。")
    c = row(9, "补充", "仅为长文支持，不是独立权限。")
    result = expand([a, b, c], [dict(c, score=0.99), dict(a, score=0.2)])
    wire = plan(result, window=8192).messages[-1]["content"]
    assert a["content"] not in wire and c["content"] not in wire and b["content"] in wire


async def test_real_retrieval_assembly_attaches_only_scoped_complete_original_bodies(monkeypatch):
    from api import generate, knowledge
    from knowledge import rag_helper, vector_db

    docs = [
        dict(id=i, title=t, content=body, knowledge_base_id=7, category="配送")
        for i, t, body in [
            (1, "紫茶规程", "完整公定条件及末尾预约例外。"),
            (2, "本人版本说明", "完整个人条件，不证明实际操作。"),
            (9, "代办", "完整委托规则，但不是这次来源。"),
        ]
    ]
    records = [d | dict(id=f"doc_{d['id']}_chunk_0", document_id=d["id"], chunk_index=0) for d in docs]
    indexed = vector(records)
    monkeypatch.setattr(knowledge, "_ensure_vector_index", lambda: True)
    monkeypatch.setattr(knowledge, "_vector_index_revision", 77)
    monkeypatch.setattr(knowledge, "_get_rebuild_revision", lambda: 77)
    monkeypatch.setattr(vector_db, "get_vector_db", lambda: indexed)
    monkeypatch.setattr(
        generate, "db", SimpleNamespace(get_knowledge_document=lambda i: next(d for d in docs if d["id"] == i))
    )
    monkeypatch.setenv("CORRECTIVE_RAG_ENABLED", "false")
    query = "读取《紫茶规程》《本人版本说明》。分别核对完整条件与主体。"

    def retrieve(actual, **kwargs):
        assert actual == query and kwargs["filters"] is None
        return dict(
            results=[dict(records[2], score=0.99), dict(records[0], score=0.3)], confidence=0.9, abstained=False
        )

    helper = SimpleNamespace(
        retrieve_with_citations=retrieve,
        compute_confidence=lambda results: RAGHelper.compute_confidence(object.__new__(RAGHelper), results),
        build_citations=lambda results: [dict(source_id=r["id"]) for r in results],
    )
    monkeypatch.setattr(rag_helper, "get_rag_helper", lambda: helper)
    result = await generate._retrieve_rag_bundle(query, 3, None)
    assert {p["original_body"] for p in result["original_source_packets"]} == {d["content"] for d in docs[:2]}
    assert all("doc_9" not in c["source_id"] for c in result["citations"])
