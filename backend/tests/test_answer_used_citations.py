"""Citations follow answer use and whole-source admission, never rank alone."""

import json
import re
from dataclasses import replace

import pytest

from inference.answer_citations import citation_marker, finalize_answer_citations, prepare_answer_citations
from inference.generation_request import (
    GenerationRequest,
    GenerationResult,
    RetrievalResult,
    build_generation_request,
    generate_character_response,
)


def owned_template(raw, kwargs):
    namespace = re.search(r"\[\[cite:([0-9a-f]{12}):S1\]\]", kwargs["messages"][0]["content"]).group(1)
    return re.sub(r"\[S(\d{1,2})\]", lambda m: citation_marker(namespace, "S" + m.group(1)), raw)


def bundle():
    docs = ({"id": "course", "title": "课程规则", "content": "课程编号YY-573-R，普通雨天照常开课。"},
            {"id": "repair", "title": "维修规则", "content": "维修编号WX-204，维修预约不能代替课程预约。"},
            {"id": "paint", "title": "绘画规则", "content": "其他工坊编号ZY-882，不是纸雕课程。"})
    citations = tuple({"source_id": d["id"], "source_title": "不得沿用未匹配标题", "source_path": d["id"] + ".txt", "score": 0.0} for d in docs)
    return RetrievalResult(status="ok", evidence="原始检索格式", documents=docs, citations=citations)


def test_all_complete_sources_and_late_qualifications_are_offered_even_at_zero_score():
    original = bundle()
    docs = (dict(original.documents[0], content="背景" * 1200 + "末尾限定：没有确认不能预约。"), *original.documents[1:])
    prepared = prepare_answer_citations(replace(original, documents=docs))
    assert [c["source_id"] for c in prepared.citations] == ["course", "repair", "paint"]
    assert all(d["content"] in prepared.evidence for d in docs)
    assert prepared.citations[0]["source_title"] == "课程规则"
    assert original.citations[0]["source_title"] == "不得沿用未匹配标题"
    assert prepare_answer_citations(prepared) is prepared


@pytest.mark.parametrize("invalid", [{"id": "repair", "source_id": "course"}, {"source_id": "missing"}])
def test_unknown_or_conflicting_source_metadata_raises(invalid):
    original = replace(bundle(), citations=(invalid,))
    with pytest.raises(ValueError, match="Answer citation source"):
        prepare_answer_citations(original)
    assert original.citations == (invalid,) and not original.answer_citations_bound


@pytest.mark.parametrize("raw,expected", [("课程YY-573-R[S1]", ["course"]),
    ("维修WX-204[S2]，课程YY-573-R[S1]，再次课程[S1]", ["repair", "course"]),
    ("课程YY-573-R，没有标记。", []), ("YY-573-R[S99]", [])])
async def test_actual_generated_answer_binds_only_valid_used_keys_in_first_use_order(raw, expected):
    calls = []

    async def generate(**kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            assert kwargs["max_tokens"] == 512 and not expected
            return json.dumps(dict(citations=[]))
        assert "来源标记" in kwargs["messages"][0]["content"]
        assert all(d["content"] in kwargs["messages"][-1]["content"] for d in bundle().documents)
        return owned_template(raw, kwargs)

    result = await generate_character_response(GenerationRequest(message="查课程或维修编号", retrieval=bundle(), max_tokens=128), generate)
    assert [c["source_id"] for c in result.response_citations] == expected
    assert all(c["source_path"] == c["source_id"] + ".txt" for c in result.response_citations)
    assert "[S" not in result.reply and len(calls) == (1 if expected else 2)


async def test_budget_rejected_source_cannot_be_restored_by_a_model_marker():
    original = bundle()
    original = replace(original, documents=(original.documents[0], dict(original.documents[1], content="长" * 8000)), citations=original.citations[:2])

    async def generate(**kwargs):
        assert "WX-204" not in kwargs["messages"][-1]["content"]
        return owned_template("课程YY-573-R[S1]，伪造维修[S2]", kwargs)

    result = await generate_character_response(GenerationRequest(message="查编号", retrieval=original, max_tokens=100, context_window_tokens=2400), generate)
    assert [c["source_id"] for c in result.plan.retrieval.citations] == ["course"]
    assert [c["source_id"] for c in result.response_citations] == ["course"]


def test_native_packet_background_dependency_is_preserved():
    original = bundle()
    packets = ({"kind": "evidence", "text": original.documents[0]["content"], "document_ids": ["course"]},
               {"kind": "background", "text": "场景只是背景", "document_ids": [], "supporting_document_ids": ["course"]})
    prepared = prepare_answer_citations(replace(original, evidence_packets=packets))
    assert prepared.evidence_packets[1] == packets[1]
    assert prepared.evidence_packets[0]["document_ids"] == ["course"]
    assert prepared.evidence_packets[0]["text"].endswith(packets[0]["text"])
    plan = build_generation_request(GenerationRequest(message="课程号", retrieval=prepared))
    assert [c["source_id"] for c in plan.retrieval.citations] == ["course"]


async def test_failed_guard_never_produces_a_citable_reply():
    from character.output_guard import ReplyGuard
    from inference.generation_request import ReplyValidationError

    prepared = prepare_answer_citations(bundle())
    calls = []

    async def generate(**kwargs):
        calls.append(kwargs)
        return "你一直喜欢喝咖啡。" + citation_marker(prepared.citation_namespace, "S1")

    request = GenerationRequest(message="课程编号是什么？", retrieval=prepared,
        reply_guard=ReplyGuard(forbid_unsupported_user_fact=True), reply_guard_mode="strict")
    with pytest.raises(ReplyValidationError) as error:
        await generate_character_response(request, generate)
    assert error.value.violations == ("unsupported_user_fact",) and len(calls) == 2


def test_explicit_source_lookup_and_non_rag_literal_markers_stay_unchanged():
    lookup = replace(bundle(), source_lookup=True)
    assert prepare_answer_citations(lookup) is lookup
    plan = build_generation_request(GenerationRequest(message="解释符号[S1]"))
    result = GenerationResult(reply="符号[S1]是题面文字。", plan=plan)
    assert finalize_answer_citations(result) is result


async def test_api_propagates_budget_abstention_instead_of_claiming_grounded_answer(monkeypatch, semantic_plan, controlled_public_bundle):
    from api import generate
    from db.schemas import MessageRequest
    from inference.provider_context import ProviderContextBudget

    class Client:
        async def generate(self, **kwargs):
            assert "<retrieved_evidence" not in kwargs["messages"][-1]["content"]
            return "目前无法核实课程编号。"

    async def retrieve(query, top_k, filters):
        return {"results": [{"id": "oversized", "content": "完整原始资料" * 2000}],
                "citations": [{"source_id": "oversized"}], "confidence": 0.9, "abstained": False}

    monkeypatch.setattr(generate, "_retrieve_rag_bundle", retrieve)
    monkeypatch.setattr(generate, "_get_system_prompt", lambda _: "persona")
    monkeypatch.setattr(generate, "get_provider_context_budget", lambda: ProviderContextBudget(8192))
    semantic_plan("课程编号", {"public_knowledge": [0]})
    _, _, meta = await generate._generate_with_retrieval(MessageRequest(message="课程编号"), None,
        runtime_config={"useKnowledgeBase": True, "maxTokens": 128}, model_generate=Client().generate)
    assert meta["abstained"] is True
    assert meta["answerMode"] == "abstention"
    assert meta["citations"] == []


def test_citation_preparation_keeps_uncited_and_uncitable_source_bodies():
    original = bundle()
    extra = (dict(content="附录完整限定：预约不代表已获得实际执行授权。"),
             dict(id=True, content="未编号原文：周末关闭，未确认不得办理。"))
    original = replace(original, documents=(*original.documents, *extra), citations=original.citations[:1])
    prepared = prepare_answer_citations(original)
    plan = build_generation_request(GenerationRequest(message="保留所有完整规则及附录限定。", retrieval=prepared))
    assert len(prepared.evidence_packets) == 5 and [c["source_id"] for c in prepared.citations] == ["course"]
    assert all(row["content"] in prepared.evidence and row["content"] in plan.messages[-1]["content"]
               for row in original.documents)
    assert all(packet["document_ids"] == [] for packet in prepared.evidence_packets[-2:])
    assert all("来源标记：" not in packet["text"] for packet in prepared.evidence_packets[1:])


def test_citation_preparation_reuses_background_dependencies_for_budget_pruning():
    original = bundle()
    anchor = dict(original.documents[0], content="完整课程背景" * 8000 + "末尾：未确认不得执行。")
    background = dict(original.documents[1], retrieval_role="source_context", supporting_document_ids=["course"])
    original = replace(original, documents=(anchor, background, original.documents[2]))
    prepared = prepare_answer_citations(original)
    assert prepared.evidence_packets[1]["kind"] == "background"
    assert prepared.evidence_packets[1]["supporting_document_ids"] == ["course"]
    plan = build_generation_request(GenerationRequest(
        message="核对完整条款并保留独立资料。", retrieval=prepared, context_window_tokens=2400, max_tokens=128,
    ))
    assert [c["source_id"] for c in plan.retrieval.citations] == ["paint"]
    assert original.documents[2]["content"] in plan.messages[-1]["content"]
    assert background["content"] not in plan.messages[-1]["content"] and "未确认不得执行" not in plan.messages[-1]["content"]


@pytest.mark.parametrize("citation", [dict(id=7), dict(source_id="7"), dict(id=7, source_id="7")])
def test_citation_preparation_normalizes_native_and_legacy_ids_once(citation):
    doc = dict(chunk_id=7, original_title="完整补充资料", content="周四办理，雨天暂停，预约不能免除原件。")
    original = RetrievalResult(status="ok", evidence=doc["content"], documents=(doc,), citations=(citation,))
    prepared = prepare_answer_citations(original)
    assert prepared.evidence_packets[0]["document_ids"] == ["7"]
    assert prepared.citations[0] == dict(citation, id="7", source_id="7", source_title=doc["original_title"], key="S1")
    assert original.citations == (citation,) and original.documents == (doc,)


@pytest.mark.asyncio
@pytest.mark.parametrize("defect", [
    "duplicate_citation", "duplicate_document", "normalized_duplicate_document", "invalid_identity",
    "source_capacity", "invalid_packet_ids", "invalid_packet_text",
])
async def test_citation_preparation_failure_stops_before_model(defect):
    original = bundle()
    if defect == "duplicate_citation":
        original = replace(original, citations=(original.citations[0], original.citations[0]))
    elif defect == "duplicate_document":
        original = replace(original, documents=(*original.documents, original.documents[0]))
    elif defect == "normalized_duplicate_document":
        original = replace(original, documents=(dict(original.documents[0], id=1), dict(original.documents[1], id="1")),
                           citations=(dict(source_id="1"),))
    elif defect == "invalid_identity":
        original = replace(original, citations=(dict(source_id=True),))
    elif defect == "source_capacity":
        docs = tuple(dict(id=f"record-{i}", title=f"完整资料{i}", content=f"第{i}项规则：周四办理，雨天暂停。") for i in range(100))
        original = replace(original, documents=docs, citations=tuple(dict(source_id=doc["id"]) for doc in docs))
    else:
        packet = dict(document_ids="course" if defect == "invalid_packet_ids" else ["course"],
                      text=None if defect == "invalid_packet_text" else original.documents[0]["content"])
        original = replace(original, evidence_packets=(packet,))
    calls = []

    async def model(**kwargs):
        calls.append(kwargs)
        raise AssertionError("Citation preparation failure must precede model generation")

    reason = "Invalid generation evidence packet" if defect.startswith("invalid_packet_") else "Answer citation"
    with pytest.raises(ValueError, match=reason):
        await generate_character_response(GenerationRequest(
            message="核对全部完整来源和限制。", retrieval=original, context_window_tokens=65536,
        ), model)
    assert not calls and not original.answer_citations_bound and not original.citation_namespace
    if defect == "source_capacity":
        assert len(original.documents) == len(original.citations) == 100
