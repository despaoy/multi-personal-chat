"""Citation presentation never clips or changes selected source evidence."""

import json
from html import unescape
from pathlib import Path

import pytest

from inference.answer_citations import prepare_answer_citations
from inference.context_budget import estimated_tokens
from inference.generation_request import GenerationRequest, RetrievalResult, build_generation_request
from knowledge.evidence_packets import document_evidence_packets

pytestmark = pytest.mark.usefixtures("no_live_context_review")


def test_complete_chunks_keep_content_and_native_or_legacy_identity():
    body = "已撤销草案。" + "完整库存记录" * 600 + "最终：禁止训练和联网。"
    docs = [
        {"id": "a", "title": "完整规程", "content": body},
        {"chunk_id": 2, "original_title": "补充规定", "content": "最终例外：仍需审批。"},
    ]
    packets = document_evidence_packets(docs)
    assert body in packets[0]["text"] and packets[0]["document_ids"] == ["a"]
    assert packets[1]["document_ids"] == ["2"] and "补充规定" in packets[1]["text"]


def test_uncitable_chunk_stays_whole_without_inventing_source_identity():
    packets = document_evidence_packets(
        [
            {"content": "完整已检索资料"},
            {"id": True, "content": "没有合法来源编号"},
        ]
    )
    assert len(packets) == 2 and all(p["document_ids"] == [] for p in packets)
    assert packets[0]["text"].endswith("完整已检索资料")


@pytest.mark.parametrize("citations", [(), ({"source_id": "wrong"},)])
def test_no_bound_source_keeps_packets_without_marker_policy(citations):
    docs = ({"id": "valid", "content": "原文末尾：不允许微调。"},)
    packets = document_evidence_packets(docs)
    original = RetrievalResult(
        status="ok", evidence=packets[0]["text"], evidence_packets=packets, documents=docs, citations=citations
    )
    if citations:
        with pytest.raises(ValueError, match="missing from retrieved documents"):
            prepare_answer_citations(original)
        assert original.evidence_packets == packets and original.citations == citations
        return
    prepared = prepare_answer_citations(original)
    assert (
        prepared.evidence_packets == packets
        and not prepared.answer_citations_bound
        and not prepared.citation_namespace
        and not prepared.citations
    )
    plan = build_generation_request(GenerationRequest(message="解释资料", retrieval=prepared))
    assert "本轮来源标记示例：" not in "\n".join(m["content"] for m in plan.messages)


def test_enabled_citations_label_exact_existing_packet_without_rebuilding_body():
    docs = ({"id": "public", "title": "真实标题", "content": "文档投影中的另一个文本"},)
    packet = {"kind": "evidence", "document_ids": ["public"], "text": "完整已选片段：最终不允许训练。"}
    original = RetrievalResult(
        status="ok",
        evidence=packet["text"],
        evidence_packets=(packet,),
        documents=docs,
        citations=({"source_id": "public"},),
    )
    prepared = prepare_answer_citations(original)
    assert prepared.answer_citations_bound and prepared.evidence_packets[0]["text"].endswith(packet["text"])
    assert docs[0]["content"] not in prepared.evidence and prepared.citations[0]["source_title"] == "真实标题"


async def test_citations_off_reaches_model_once_with_complete_evidence(monkeypatch, semantic_plan):
    from types import SimpleNamespace

    from api import generate
    from db.schemas import MessageRequest
    from inference.context_budget import ReviewContextBudget
    from knowledge import public_question_binding, public_task_evidence

    fixture = json.loads(
        (Path(__file__).parent / "fixtures/deepseek_citation_independent_evidence_cases.json").read_text(encoding="utf-8")
    )
    document = dict(id="one-complete-chunk", title=fixture["document"]["title"], content=fixture["document"]["content"])
    planning = semantic_plan(fixture["question"], dict(public_knowledge=[0]))

    async def retrieve(*args, **kwargs):
        return dict(results=[document], citations=[dict(source_id=document["id"])], confidence=0.9, abstained=False)

    async def binding(*args, **kwargs):
        return None

    async def transmission_control(bundle, *args, **kwargs):
        assert bundle["results"] == [document]
        return bundle

    calls = []

    async def model(**kwargs):
        calls.append(kwargs)
        wire = unescape(kwargs["messages"][-1]["content"])
        assert document["content"] in wire and "最终有效条件：" in wire
        assert "本轮来源标记示例：" not in "\n".join(m["content"] for m in kwargs["messages"])
        return "规程 XL-936-N 的最终限制禁止联网、训练和微调。"

    monkeypatch.setenv("RAG_CITATIONS_ENABLED", "false")
    monkeypatch.setattr(generate, "retrieve_chat_knowledge", retrieve)
    monkeypatch.setattr(generate, "_get_system_prompt", lambda _: "人物规则")
    monkeypatch.setattr(public_question_binding, "resolve_question_binding", binding)
    monkeypatch.setattr(public_task_evidence, "review_public_candidates", transmission_control)
    monkeypatch.setattr(generate, "get_provider_context_budget", lambda: SimpleNamespace(
        review=ReviewContextBudget(65536), window_tokens=65536, evidence_max_chars=0,
    ))
    reply, used, meta = await generate._generate_with_retrieval(
        MessageRequest(message=fixture["question"]), None,
        runtime_config=dict(maxTokens=1024, useKnowledgeBase=True), model_generate=model,
    )
    assert len(planning) == 1 and used and not meta["abstained"] and meta["citations"] == []
    assert len(calls) == 1 and "XL-936-N" in reply


def test_true_token_overflow_excludes_entire_chunk_and_dependent_citation():
    docs = (
        {"id": "too-long", "content": "大" * 70000 + "最后限制"},
        {"id": "fits", "content": "完整较小证据：仍需审批。"},
    )
    packets = document_evidence_packets(docs)
    assert estimated_tokens(packets[0]["text"]) > 8192
    retrieval = prepare_answer_citations(
        RetrievalResult(
            status="ok",
            evidence="\n\n".join(p["text"] for p in packets),
            evidence_packets=packets,
            documents=docs,
            citations=({"source_id": "too-long"}, {"source_id": "fits"}),
        )
    )
    result = build_generation_request(
        GenerationRequest(message="询问规程", context_window_tokens=8192, evidence_max_chars=0, retrieval=retrieval)
    )
    assert [p["document_ids"] for p in result.retrieval.evidence_packets] == [["fits"]]
    assert [c["source_id"] for c in result.retrieval.citations] == ["fits"]
    assert "最后限制" not in result.retrieval.evidence and "仍需审批" in result.retrieval.evidence


@pytest.mark.parametrize("corrupt", [
    {"content": None}, {"content": " "},
    {"retrieval_role": "source_context"},
    {"retrieval_role": "source_context", "supporting_document_ids": []},
    {"retrieval_role": "source_context", "supporting_document_ids": "anchor"},
    {"retrieval_role": "source_context", "supporting_document_ids": [""]},
])
async def test_corrupt_retrieved_document_returns_error_before_model(monkeypatch, corrupt, semantic_plan, controlled_public_bundle):
    from fastapi import HTTPException

    from api import generate
    from db.schemas import MessageRequest

    planning = semantic_plan("按规程说明验收时间和全部限制。", dict(public_knowledge=[0]))
    valid = {"id": "anchor", "title": "合成验收规程", "content": "周五验收；只允许离线测试，禁止联网。"}
    broken = {**valid, "id": "other", **corrupt}
    calls = []

    async def retrieve(*args, **kwargs):
        return {"results": [valid, broken], "citations": [{"source_id": "anchor"}],
                "confidence": 0.9, "abstained": False}

    async def model(**kwargs):
        calls.append(kwargs)
        return "must not silently omit invalid source evidence"

    monkeypatch.setenv("RAG_CITATIONS_ENABLED", "false")
    monkeypatch.setattr(generate, "retrieve_chat_knowledge", retrieve)
    monkeypatch.setattr(generate, "_get_system_prompt", lambda _: "人物规则")
    with pytest.raises(HTTPException) as error:
        await generate._generate_with_retrieval(
            MessageRequest(message="按规程说明验收时间和全部限制。"), None,
            runtime_config={"maxTokens": 1024, "useKnowledgeBase": True}, model_generate=model,
        )
    assert error.value.status_code == 503 and isinstance(error.value.__cause__, ValueError)
    assert len(planning) == 1 and not calls
