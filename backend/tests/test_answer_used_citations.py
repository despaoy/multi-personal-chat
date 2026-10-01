"""Citations follow answer use and whole-source admission, never rank alone."""

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
def test_unknown_or_conflicting_source_metadata_never_binds(invalid):
    prepared = prepare_answer_citations(replace(bundle(), citations=(invalid,)))
    assert not prepared.citations
    result = finalize_answer_citations(GenerationResult(reply="课程编号[S1]", plan=build_generation_request(GenerationRequest(message="问", retrieval=prepared))))
    assert not result.response_citations


@pytest.mark.parametrize("raw,expected", [("课程YY-573-R[S1]", ["course"]),
    ("维修WX-204[S2]，课程YY-573-R[S1]，再次课程[S1]", ["repair", "course"]),
    ("课程YY-573-R，没有标记。", []), ("YY-573-R[S99]", [])])
async def test_actual_generated_answer_binds_only_valid_used_keys_in_first_use_order(raw, expected):
    async def generate(**kwargs):
        assert "来源标记" in kwargs["messages"][0]["content"]
        assert all(d["content"] in kwargs["messages"][-1]["content"] for d in bundle().documents)
        return owned_template(raw, kwargs)

    result = await generate_character_response(GenerationRequest(message="查课程或维修编号", retrieval=bundle(), max_tokens=128), generate)
    assert [c["source_id"] for c in result.response_citations] == expected
    assert all(c["source_path"] == c["source_id"] + ".txt" for c in result.response_citations)
    assert "[S" not in result.reply


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


def test_guard_fallback_cannot_claim_any_retrieved_source():
    prepared = prepare_answer_citations(bundle())
    plan = build_generation_request(GenerationRequest(message="课程号", retrieval=prepared))
    result = finalize_answer_citations(GenerationResult(reply="保守回应" + citation_marker(prepared.citation_namespace, "S1"), plan=plan, guard_fallback="unsupported_user_fact"))
    assert result.response_citations == ()
    assert result.reply == "保守回应"


def test_explicit_source_lookup_and_non_rag_literal_markers_stay_unchanged():
    lookup = replace(bundle(), source_lookup=True)
    assert prepare_answer_citations(lookup) is lookup
    plan = build_generation_request(GenerationRequest(message="解释符号[S1]"))
    result = GenerationResult(reply="符号[S1]是题面文字。", plan=plan)
    assert finalize_answer_citations(result) is result


async def test_api_propagates_budget_abstention_instead_of_claiming_grounded_answer(monkeypatch):
    from types import SimpleNamespace

    from api import generate
    from db.schemas import MessageRequest
    from knowledge import intent_detector, rag_helper

    class Client:
        async def generate(self, **kwargs):
            assert "<retrieved_evidence" not in kwargs["messages"][-1]["content"]
            return "目前无法核实课程编号。"

    async def retrieve(query, top_k, filters):
        return {"results": [{"id": "oversized", "content": "完整原始资料" * 2000}],
                "citations": [{"source_id": "oversized"}], "confidence": 0.9, "abstained": False}

    monkeypatch.setattr(intent_detector, "needs_rag", lambda _: (True, "fact", None))
    monkeypatch.setattr(generate, "_retrieve_rag_bundle", retrieve)
    monkeypatch.setattr(generate, "_get_system_prompt", lambda _: "persona")
    monkeypatch.setattr(generate, "_vllm_client", Client())
    monkeypatch.setattr(rag_helper, "get_rag_helper", lambda: SimpleNamespace(format_context_results=lambda _: "候选原始格式"))
    _, _, meta = await generate._generate_with_vllm(MessageRequest(message="课程编号"), None,
        runtime_config={"useKnowledgeBase": True, "maxTokens": 128})
    assert meta["abstained"] is True
    assert meta["answerMode"] == "abstention"
    assert meta["citations"] == []
