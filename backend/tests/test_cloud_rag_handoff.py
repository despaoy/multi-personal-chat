"""Whole public packets share the final budget with private speech."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from character.models import CompiledCharacterContext
from character.source_memory import compile_sources
from inference.generation_request import GenerationRequest, RetrievalResult, build_generation_request
from inference.provider_context import ProviderContextBudget, get_provider_context_budget


@pytest.mark.parametrize(
    "provider,window,expected",
    [("openai_compat", 65536, (None, 0)), ("openai_compat", 8192, (None, 0)), ("vllm", 8192, (6000, 6000))],
)
def test_provider_atomic_rag_allowance(provider, window, expected):
    budget = get_provider_context_budget(
        SimpleNamespace(_current_provider=SimpleNamespace(value=provider)),
        env={"OPENAI_COMPAT_CONTEXT_WINDOW_TOKENS": str(window), "VLLM_MAX_MODEL_LEN": str(window)},
    )
    assert (budget.rag_max_chars, budget.evidence_max_chars) == expected


def test_factory_refreshes_budget_on_provider_change(monkeypatch):
    import inference.provider_context as provider
    from knowledge.multiscale_rag import runtime

    budget = ProviderContextBudget(65536, rag_max_chars=None, evidence_max_chars=0)
    calls = []

    class Runtime:
        def __init__(self, *, context_max_chars):
            self.context_max_chars = context_max_chars
            calls.append(context_max_chars)

        def warmup_async(self):
            pass

    monkeypatch.setattr(runtime, "_runtime", None)
    monkeypatch.setattr(runtime, "MultiScaleRagRuntime", Runtime)
    monkeypatch.setattr(provider, "get_provider_context_budget", lambda: budget)
    first = runtime.get_multiscale_rag_service()
    assert runtime.get_multiscale_rag_service() is first
    budget = ProviderContextBudget(8192)
    second = runtime.get_multiscale_rag_service()
    assert second is not first and second.context_max_chars == 6000
    assert calls == [None, 6000]


def private_context():
    source = {
        "source_message_id": "private-owner-source",
        "body": "Private inventory receipt; no permission.\n" * 600 + "最终：只能离线，不得训练。",
        "observed_at": "2026-10-02T09:00:00+08:00",
    }
    packet = compile_sources([source], max_chars=None).context
    return CompiledCharacterContext(
        profile_context="",
        dynamic_context="",
        reference_context="",
        source_candidate_context=packet,
        memory_source_status="budget_omitted",
    )


def retrieval(texts):
    return RetrievalResult(
        status="ok",
        evidence="\n\n".join(texts),
        evidence_packets=tuple({"kind": "evidence", "text": t, "document_ids": [str(i)]} for i, t in enumerate(texts)),
        citations=tuple({"id": str(i), "source_path": str(i) + ".txt"} for i in range(len(texts))),
    )


def test_long_public_tail_and_private_whole_packet_share_actual_budget():
    public = "Public stock count; no enrollment permission.\n" * 500 + "最终公开限制：禁止微调，编号 WX-804-Q。"
    request = GenerationRequest(
        message="分别回答私人条件和公开规程。",
        character_context=private_context(),
        retrieval=retrieval([public]),
        context_window_tokens=65536,
        evidence_max_chars=0,
        max_tokens=1024,
    )
    plan = build_generation_request(request)
    assert plan.retrieval.evidence == public and plan.retrieval.citations == request.retrieval.citations
    assert (
        plan.character_context.memory_source_status == "available"
        and not plan.character_context.source_candidate_context
    )
    packet = json.loads(plan.character_context.episodic_reference_context)
    assert packet["records"][0]["text"].endswith("最终：只能离线，不得训练。")
    assert public in plan.messages[-1]["content"]


def test_real_token_overflow_omits_whole_public_preserving_private_decision():
    large = "超" * 70000 + "不得联网"
    good = "完整较小公开规程：正常开课不证明用户报名。"
    supplied = retrieval([large, good])
    packets = (
        *supplied.evidence_packets,
        {"kind": "background", "text": "排除来源的场景不得残留", "document_ids": [], "supporting_document_ids": ["0"]},
    )
    plan = build_generation_request(
        GenerationRequest(
            message="分别说明资料。",
            character_context=private_context(),
            retrieval=replace(supplied, evidence_packets=packets),
            context_window_tokens=65536,
            evidence_max_chars=0,
            max_tokens=1024,
        )
    )
    assert plan.retrieval.evidence == good and [c["id"] for c in plan.retrieval.citations] == ["1"]
    assert len(plan.retrieval.evidence_packets) == 1
    assert (
        plan.character_context.memory_source_status == "available"
        and not plan.character_context.source_candidate_context
    )


def test_local_character_cap_still_omits_whole_source():
    text = "本地资料" * 2000 + "仅适用于先验审批通过者。"
    plan = build_generation_request(
        GenerationRequest(message="询问资料", retrieval=retrieval([text]), context_window_tokens=65536)
    )
    assert (
        plan.retrieval.status == "character_abstention" and not plan.retrieval.evidence and not plan.retrieval.citations
    )


def test_public_service_none_keeps_whole_packet(monkeypatch):
    from knowledge.multiscale_rag import service
    from knowledge.retrieval_core.documents import SourceReference

    text = "完整公开库存资料。" * 2500 + "最终资格限制：尚未获准。"
    doc = SimpleNamespace(
        id="public-card",
        document_type="fact",
        title="完整资料",
        summary="错误的片段摘要",
        metadata={},
        source=SourceReference("synthetic.txt", 1, 2510),
    )
    candidate = SimpleNamespace(document=doc, to_dict=lambda: {"id": doc.id})
    instance = object.__new__(service.RoutedMultiScaleService)
    instance.context_max_chars = None
    instance.identity_coverage = False
    instance.config = SimpleNamespace(domain_id="test")
    instance.analyzer = instance.reranker = instance.extractor = None
    instance.indexes = {}
    instance.by_id = {}
    instance.retrievers = {service.CARD_TYPES: SimpleNamespace(search=lambda *a, **kw: [candidate])}
    instance.evidence_by_parent = {doc.id: SimpleNamespace(content=text)}
    monkeypatch.setattr(service, "analyze_explicit_domain", lambda *a: SimpleNamespace(entities=[]))
    monkeypatch.setattr(service, "choose_card_types", lambda *a: service.CARD_TYPES)
    monkeypatch.setattr(service, "rerank_with_title_frames", lambda *a, **kw: [candidate])
    result = instance.retrieve("资料")
    assert text in result["context_text"] and result["evidence_packets"][0]["text"] == result["context_text"]
    assert result["context_budget"]["max_chars"] is None and result["context_budget"]["skipped_blocks"] == 0
    assert [c["id"] for c in result["citations"]] == [doc.id]
