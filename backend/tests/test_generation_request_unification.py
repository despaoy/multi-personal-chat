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
async def test_abstention_fallback_records_model_failure(monkeypatch):
    from api import generate
    from inference import model_manager

    records, counters = [], []
    monkeypatch.setattr(model_manager, "get_model_manager", lambda: SimpleNamespace(
        _current_provider=SimpleNamespace(value="vllm")))
    monkeypatch.setattr(generate, "db", SimpleNamespace(config={}, loras=[]))
    monkeypatch.setattr(generate, "INPUT_VALIDATOR_AVAILABLE", False)
    monkeypatch.setattr(generate, "response_cache", None)
    monkeypatch.setattr(generate, "_vllm_client", object())
    monkeypatch.setattr(generate, "set_consecutive", lambda *args: counters.append(args))

    async def available():
        return True

    async def failed_expression(*args, **kwargs):
        return "暂时无法确定", True, {
            "modelInvoked": True, "abstained": True, "generationError": "RuntimeError",
            "warnings": ["character_abstention_fallback"],
        }

    async def record(*args, **kwargs):
        records.append(kwargs)

    monkeypatch.setattr(generate, "_ensure_vllm", available)
    monkeypatch.setattr(generate, "_generate_with_vllm", failed_expression)
    monkeypatch.setattr(generate, "_record_model_invocation", record)
    result = await generate._generate_reply_impl(MessageRequest(message="问题"), persist_message=False)
    assert result.abstained
    assert records[0]["error_type"] == "RuntimeError"
    assert ("model_failure", False) in counters
    assert ("model_failure", True) not in counters


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError("retrieval unavailable"), TimeoutError("timeout")])
async def test_retrieval_errors_generate_only_character_uncertainty(monkeypatch, failure):
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
    reply, used_rag, meta = await generate._generate_with_vllm(
        MessageRequest(message="人物关系是什么"), None,
        runtime_config={"useKnowledgeBase": True}, model_generate=model,
    )
    assert reply and used_rag and meta["abstained"]
    assert meta["citations"] == []
    assert meta["warnings"] == ["retrieval_unavailable"]
    assert "【本轮证据不足】" in captured["messages"][0]["content"]
    assert "不代表知识库中不存在答案" in captured["messages"][0]["content"]


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
    from api import generate
    from knowledge import intent_detector, rag_helper

    captured = {}

    class Client:
        async def generate(self, **kwargs):
            captured.update(kwargs)
            return "answer"

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
    monkeypatch.setattr(
        rag_helper,
        "get_rag_helper",
        lambda: SimpleNamespace(format_context_results=lambda _: "evidence"),
    )

    await generate._generate_with_vllm(
        MessageRequest(message="question", senderName="琉璃"),
        None,
        runtime_config={"useKnowledgeBase": True, "temperature": 0.7},
    )
    expected = build_generation_request(
        GenerationRequest(
            message="question",
            persona_prompt="persona",
            interlocutor="琉璃",
            retrieval=generate.RetrievalResult(status="ok", evidence="evidence"),
            temperature=0.7,
        )
    )

    assert captured["messages"] == [dict(message) for message in expected.messages]
    assert captured["temperature"] == expected.generation["temperature"] == 0.5
    assert expected.prompt_policy_version
