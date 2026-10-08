from types import SimpleNamespace

import pytest

from db.schemas import MessageRequest
from inference.generation_request import GenerationRequest, build_generation_request


@pytest.mark.asyncio
async def test_unavailable_character_domain_does_not_switch_to_generic_kb(monkeypatch):
    from api import generate
    from knowledge.multiscale_rag import runtime
    from knowledge.retrieval_core import query

    service = SimpleNamespace(config=object(), retrieve_with_citations=lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime, "get_multiscale_rag_service", lambda: service)
    monkeypatch.setattr(query, "QueryAnalyzer", lambda _: SimpleNamespace(
        analyze=lambda _: SimpleNamespace(matched_domains=["character"])))
    with pytest.raises(RuntimeError, match="domain is unavailable"):
        await generate._retrieve_rag_bundle("人物关系是什么", 3, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["vllm", "openai_compat"])
@pytest.mark.parametrize("output_invalid", [False, True])
async def test_generation_failure_is_reported_and_recorded(monkeypatch, provider, output_invalid):
    from api import generate
    from inference import model_manager

    records, counters = [], []
    from unittest.mock import Mock

    switch = Mock(side_effect=AssertionError("A failed request must not switch the configured provider"))
    manager = SimpleNamespace(_current_provider=SimpleNamespace(value=provider),
                              set_lora_adapter=lambda _: None, set_provider=switch)
    monkeypatch.setattr(model_manager, "get_model_manager", lambda: manager)
    monkeypatch.setattr(generate, "db", SimpleNamespace(config={}, loras=[]))
    monkeypatch.setattr(generate, "INPUT_VALIDATOR_AVAILABLE", False)
    monkeypatch.setattr(generate, "response_cache", None)
    monkeypatch.setattr(generate, "_vllm_client", object())
    monkeypatch.setattr(generate, "set_consecutive", lambda *args: counters.append(args))

    async def available():
        return True

    from inference.generation_request import ReplyValidationError
    failure = ReplyValidationError(("unsupported_user_fact",)) if output_invalid else RuntimeError("model unavailable")

    async def failed_expression(*args, **kwargs):
        raise failure

    async def record(*args, **kwargs):
        records.append(kwargs)

    monkeypatch.setattr(generate, "_ensure_vllm", available)
    monkeypatch.setattr(generate, "circuit_breaker_registry", None)
    monkeypatch.setattr(generate, "_generate_with_retrieval", failed_expression)
    monkeypatch.setattr(generate, "_generate_with_vllm", failed_expression)
    monkeypatch.setattr(generate, "_record_model_invocation", record)
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as caught:
        await generate._generate_reply_impl(MessageRequest(message="问题"), persist_message=False)
    assert caught.value.status_code == (503 if provider == "vllm" or output_invalid else 500)
    if output_invalid:
        assert caught.value.detail["stage"] == "reply_validation"
        assert caught.value.detail["violations"] == ["unsupported_user_fact"]
    assert caught.value.__cause__ is failure
    assert records[0]["error_type"] == type(failure).__name__
    switch.assert_not_called()
    assert ("model_failure", False) in counters
    assert ("model_failure", True) not in counters


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError("retrieval unavailable"), TimeoutError("timeout")])
async def test_retrieval_errors_stop_before_model_generation(monkeypatch, failure):
    from api import generate
    from knowledge import intent_detector

    captured = {}

    async def broken_retrieval(*args):
        raise failure

    async def model(**kwargs):
        captured.update(kwargs)
        return "暂时还不能确定。"

    monkeypatch.setattr(intent_detector, "needs_rag", lambda _: (True, "fact", None))
    monkeypatch.setattr(generate, "_retrieve_rag_bundle", broken_retrieval)
    monkeypatch.setattr(generate, "_get_system_prompt", lambda _: "persona")
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as caught:
        await generate._generate_with_vllm(
            MessageRequest(message="人物关系是什么"), None,
            runtime_config={"useKnowledgeBase": True}, model_generate=model,
        )
    assert caught.value.status_code == 503
    assert caught.value.__cause__ is failure
    assert not captured


@pytest.mark.parametrize("status", ["abstained", "error", "character_abstention"])
def test_only_character_abstention_allows_uncertainty_generation(status):
    from inference.generation_request import RetrievalResult

    plan = build_generation_request(GenerationRequest(
        message="question",
        persona_prompt="persona",
        apply_prompt_policy=False,
        retrieval=RetrievalResult(status=status, evidence="UNRELIABLE_FACT"),
    ))
    assert plan.should_generate is (status == "character_abstention")
    assert plan.retrieval.has_evidence is False
    assert "UNRELIABLE_FACT" not in str(plan.messages)
    if status == "character_abstention":
        assert "【本轮证据不足】" in plan.messages[0]["content"]


@pytest.mark.asyncio
async def test_production_and_character_benchmark_share_model_request(monkeypatch):
    from api import generate
    from evaluation import character_benchmark_v3

    production_call = {}

    class Client:
        async def generate(self, **kwargs):
            production_call.update(kwargs)
            return "production reply"

    monkeypatch.setattr(generate, "_get_system_prompt", lambda _: "persona")
    monkeypatch.setattr(generate, "_vllm_client", Client())
    await generate._generate_with_vllm(
        MessageRequest(message="question", senderName="琉璃"),
        None,
        runtime_config={
            "useKnowledgeBase": False,
            "temperature": 0.0,
            "maxTokens": 256,
            "topP": 0.9,
        },
    )

    benchmark_call = {}

    def fake_call(base_url, model, messages, generation, timeout):
        benchmark_call.update(messages=messages, **generation)
        return "benchmark reply", 1.0, ""

    monkeypatch.setattr(character_benchmark_v3, "_call", fake_call)
    character_benchmark_v3._call_conversation(
        "http://test",
        "model",
        "persona",
        ["question"],
        {
            "temperature": 0.0,
            "max_tokens": 256,
            "top_p": 0.9,
            "repetition_penalty": 1.0,
            "frequency_penalty": 0.0,
            "enable_thinking": False,
        },
        1.0,
        interlocutor="琉璃",
    )

    assert production_call["messages"] == benchmark_call["messages"]
    for key in (
        "temperature",
        "max_tokens",
        "top_p",
        "repetition_penalty",
        "frequency_penalty",
        "enable_thinking",
    ):
        assert production_call[key] == benchmark_call[key]


@pytest.mark.asyncio
async def test_production_rag_uses_shared_grounded_request(monkeypatch):
    import re

    from api import generate
    from inference import answer_citations
    from inference.answer_citations import prepare_answer_citations
    from knowledge import intent_detector

    monkeypatch.setattr(answer_citations.secrets, "token_hex", lambda _size: "123456abcdef")
    captured = {}

    class Client:
        async def generate(self, **kwargs):
            captured.update(kwargs)
            return "answer" + re.search(r"\[\[cite:[0-9a-f]{12}:S1\]\]", kwargs["messages"][0]["content"]).group(0)

    async def retrieve(query, top_k, filters):
        return {
            "results": [{"id": "doc-1", "content": "evidence"}],
            "citations": [{"source_id": "doc-1"}],
            "confidence": 0.9,
            "abstained": False,
        }

    monkeypatch.setattr(intent_detector, "needs_rag", lambda _: (True, "fact", None))
    monkeypatch.setattr(generate, "_retrieve_rag_bundle", retrieve)
    monkeypatch.setattr(generate, "_get_system_prompt", lambda _: "persona")
    monkeypatch.setattr(generate, "_vllm_client", Client())

    reply, _, meta = await generate._generate_with_vllm(
        MessageRequest(message="question", senderName="琉璃"),
        None,
        runtime_config={"useKnowledgeBase": True, "temperature": 0.7},
    )
    expected = build_generation_request(
        GenerationRequest(
            message="question",
            persona_prompt="persona",
            interlocutor="琉璃",
            retrieval=prepare_answer_citations(generate.RetrievalResult(status="ok", evidence="evidence",
                documents=({"id": "doc-1", "content": "evidence"},), citations=({"source_id": "doc-1"},),
                evidence_packets=({"kind": "evidence", "document_ids": ["doc-1"],
                    "text": "【检索资料片段: 未命名资料】\nevidence"},))),
            temperature=0.7,
        )
    )

    assert reply == "answer"
    assert [c["source_id"] for c in meta["citations"]] == ["doc-1"]
    assert captured["messages"] == [dict(message) for message in expected.messages]
    assert captured["temperature"] == expected.generation["temperature"] == 0.5
    assert expected.prompt_policy_version


@pytest.mark.asyncio
async def test_configuration_read_failure_stops_before_model_or_save(monkeypatch):
    from unittest.mock import AsyncMock

    from fastapi import HTTPException

    from api import generate
    from inference import model_manager

    failure = OSError("database disconnected")

    class BrokenDatabase:
        loras = []

        @property
        def config(self):
            raise failure

    monkeypatch.setattr(model_manager, "get_model_manager", lambda: SimpleNamespace(
        _current_provider=SimpleNamespace(value="openai_compat")))
    monkeypatch.setattr(generate, "INPUT_VALIDATOR_AVAILABLE", False)
    model, save = AsyncMock(), AsyncMock()
    monkeypatch.setattr(generate, "_generate_with_retrieval", model)
    monkeypatch.setattr(generate, "_save_message", save)
    with pytest.raises(HTTPException) as caught:
        await generate._generate_reply_impl(MessageRequest(message="完整测试问题"), message_db=BrokenDatabase())
    assert caught.value.status_code == 503 and caught.value.__cause__ is failure
    model.assert_not_awaited()
    save.assert_not_awaited()


@pytest.mark.asyncio
async def test_native_timeout_cancels_work_and_releases_capacity_without_breaker(monkeypatch):
    import asyncio
    from unittest.mock import AsyncMock

    from fastapi import HTTPException

    from api import generate
    from inference import model_manager

    semaphore = asyncio.Semaphore(1)
    cancelled = asyncio.Event()

    async def pending(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(model_manager, "get_model_manager", lambda: SimpleNamespace(
        _current_provider=SimpleNamespace(value="openai_compat"), set_lora_adapter=lambda _: None))
    monkeypatch.setattr(generate, "db", SimpleNamespace(config={}, loras=[]))
    monkeypatch.setattr(generate, "INPUT_VALIDATOR_AVAILABLE", False)
    monkeypatch.setattr(generate, "response_cache", None)
    monkeypatch.setattr(generate, "circuit_breaker_registry", None)
    monkeypatch.setattr(generate, "get_llm_semaphore", lambda: semaphore)
    monkeypatch.setattr(generate, "_MODEL_INFERENCE_TIMEOUT", 0.01)
    monkeypatch.setattr(generate, "_generate_with_retrieval", pending)
    save = AsyncMock()
    monkeypatch.setattr(generate, "_save_message", save)
    with pytest.raises(HTTPException) as caught:
        await generate._generate_reply_impl(MessageRequest(message="完整测试问题"), record_invocation=False)
    assert caught.value.status_code == 500 and isinstance(caught.value.__cause__, TimeoutError)
    assert cancelled.is_set() and not semaphore.locked()
    save.assert_not_awaited()
