
from services import chat_answer

from knowledge import chat_retrieval

from knowledge import index_lifecycle
from unittest.mock import AsyncMock
from types import SimpleNamespace

import pytest
from provider_fixtures import cloud_connection

from db.schemas import MessageRequest
from inference.generation_request import GenerationRequest, build_generation_request, generate_character_response

pytestmark = pytest.mark.usefixtures("no_live_context_review")


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "load_failure", "retrieval_failure"])
async def test_character_runtime_owns_domain_and_failure(monkeypatch, outcome):
    from unittest.mock import Mock
    from knowledge.multiscale_rag import runtime
    from test_multiscale_rag_runtime import ranking_score_runtime

    # Exercise the production runtime and its real domain analyzer; only the
    # local index payload and injected I/O failures are synthetic.
    service = runtime.MultiScaleRagRuntime()
    service._service = ranking_score_runtime(0.9)._service
    failure = OSError("synthetic character index failure")
    if outcome == "load_failure":
        service._service = None
        monkeypatch.setattr(service, "_ensure_loaded", Mock(side_effect=failure))
    elif outcome == "retrieval_failure":
        service._service.retrieve = Mock(side_effect=failure)
    generic = Mock(side_effect=AssertionError("Character failure must not switch to generic KB"))
    monkeypatch.setattr(index_lifecycle, "_ensure_vector_index", generic)
    monkeypatch.setattr(runtime, "get_multiscale_rag_service", lambda: service)
    if outcome == "success":
        result = await chat_retrieval.retrieve_chat_knowledge("妃和琉璃是什么关系？", 3, None)
        assert result["domains"] == [service.config.domain_id]
        assert result["results"][0]["content"] == "只能离线验收，禁止联网。"
        assert result["abstained"] is False
    else:
        with pytest.raises(OSError) as raised:
            await chat_retrieval.retrieve_chat_knowledge("妃和琉璃是什么关系？", 3, None)
        assert raised.value is failure
    generic.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["vllm", "openai_compat"])
@pytest.mark.parametrize("failure_kind", ["provider", "reply_validation", "context_authority"])
async def test_generation_failure_is_reported_and_recorded(monkeypatch, provider, failure_kind):
    from api import generate
    from inference import model_manager

    records, counters = [], []
    from unittest.mock import Mock

    switch = Mock(side_effect=AssertionError("A failed request must not switch the configured provider"))
    manager = SimpleNamespace(_current_provider=SimpleNamespace(value=provider),
                              set_lora_adapter=lambda _: None, set_provider=switch)
    connection = cloud_connection("fixture-model")
    connection.name = provider
    manager.get_current_provider = lambda: connection
    monkeypatch.setattr(model_manager, "get_model_manager", lambda: manager)
    monkeypatch.setattr(generate, "db", SimpleNamespace(config={}, loras=[]))
    monkeypatch.setattr(chat_answer, "public_knowledge_db", generate.db)
    monkeypatch.setattr(generate, "response_cache", None)
    monkeypatch.setattr(generate, "get_vllm_client", AsyncMock(return_value=SimpleNamespace(generate=AsyncMock())))
    monkeypatch.setattr(generate, "set_consecutive", lambda *args: counters.append(args))


    from inference.generation_request import ContextAuthorityChangedError, ReplyValidationError
    failure = {
        "provider": RuntimeError("model unavailable"),
        "reply_validation": ReplyValidationError(("unsupported_user_fact",)),
        "context_authority": ContextAuthorityChangedError("Context authority changed again after one regeneration"),
    }[failure_kind]
    save = AsyncMock()
    monkeypatch.setattr(generate, "_save_message", save)

    async def failed_expression(*args, **kwargs):
        raise failure

    async def record(*args, **kwargs):
        records.append(kwargs)

    monkeypatch.setattr(generate, "circuit_breaker_registry", None)
    monkeypatch.setattr(chat_answer, 'generate_chat_answer', failed_expression)
    monkeypatch.setattr(generate, "_record_model_invocation", record)
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as caught:
        await generate._generate_reply_impl(MessageRequest(message="按当前完整资料回答，不使用已撤销版本。"))
    assert caught.value.status_code == (503 if provider == "vllm" or failure_kind != "provider" else 500)
    if failure_kind == "context_authority":
        assert caught.value.detail["stage"] == "context_authority"
        assert caught.value.detail["reason"] == "repeated_authority_change"
    if failure_kind == "reply_validation":
        assert caught.value.detail["stage"] == "reply_validation"
        assert caught.value.detail["violations"] == ["unsupported_user_fact"]
    assert caught.value.__cause__ is failure
    assert len(records) == 1 and records[0]["error_type"] == type(failure).__name__
    save.assert_not_awaited()
    switch.assert_not_called()
    assert ("model_failure", False) in counters
    assert ("model_failure", True) not in counters


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError("retrieval unavailable"), TimeoutError("timeout")])
async def test_retrieval_errors_stop_before_model_generation(monkeypatch, semantic_plan, failure):

    captured = {}

    async def broken_retrieval(*args, **kwargs):
        raise failure

    async def model(**kwargs):
        captured.update(kwargs)
        return "暂时还不能确定。"

    semantic_plan("人物关系是什么", dict(unresolved_source=[0]))
    monkeypatch.setattr(chat_answer, 'retrieve_chat_knowledge', broken_retrieval)
    monkeypatch.setattr(chat_answer, 'get_lora_system_prompt', lambda _: "persona")

    with pytest.raises(chat_answer.ChatRetrievalError) as caught:
        await chat_answer.generate_chat_answer(
            MessageRequest(message="人物关系是什么"), None,
            runtime_config={"useKnowledgeBase": True}, model_generate=model,
        )
    assert caught.value.http_status == 503
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
    from evaluation import character_benchmark_v3

    production_call = {}

    class Client:
        async def generate(self, **kwargs):
            production_call.update(kwargs)
            return "production reply"

    monkeypatch.setattr(chat_answer, 'get_lora_system_prompt', lambda _: "persona")
    await chat_answer.generate_chat_answer(
        MessageRequest(message="question", senderName="琉璃"),
        None,
        runtime_config={
            "useKnowledgeBase": False,
            "temperature": 0.0,
            "maxTokens": 256,
            "topP": 0.9,
        },
        model_generate=Client().generate,
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
async def test_production_rag_uses_shared_grounded_request(monkeypatch, semantic_plan):
    """Transport/building control with complete fiction, not a relevance eval."""
    import re

    from inference import answer_citations
    from inference.answer_citations import prepare_answer_citations
    from inference.context_budget import ReviewContextBudget
    from knowledge import public_question_binding, public_task_evidence
    from knowledge.evidence_packets import document_evidence_packets

    query = "核对虚构夏屿展览的完整条款，保留费用、开放时间和雨天限制。"
    body = "虚构夏屿展览入场费17元，只在周六开放；雨天停止，预约不能免除雨天限制。"
    document = dict(id="fiction-rule", title="虚构展览完整条款", content=body)
    citations = [dict(source_id=document["id"])]
    planning = semantic_plan(query, dict(public_knowledge=[0]))
    monkeypatch.setattr(answer_citations.secrets, "token_hex", lambda _size: "123456abcdef")
    captured = {}

    class Client:
        async def generate(self, **kwargs):
            captured.update(kwargs)
            marker = re.search(r"\[\[cite:[0-9a-f]{12}:S1\]\]", kwargs["messages"][0]["content"]).group(0)
            return "已依据完整虚构条款回答。" + marker

    async def retrieve(question, top_k, filters, **kwargs):
        assert question == query
        return dict(results=[document], citations=citations, confidence=0.9, abstained=False)

    async def binding(*args, **kwargs):
        return None

    async def transmission_control(bundle, *args, **kwargs):
        # Semantic relevance is controlled here; source/negation transmission
        # and the canonical generation builder are the behavior under test.
        assert bundle["results"] == [document] and body.endswith("预约不能免除雨天限制。")
        return bundle

    monkeypatch.setattr(public_question_binding, "resolve_question_binding", binding)
    monkeypatch.setattr(public_task_evidence, "review_public_candidates", transmission_control)
    monkeypatch.setattr(chat_answer, 'retrieve_chat_knowledge', retrieve)
    monkeypatch.setattr(chat_answer, 'get_lora_system_prompt', lambda _: "persona")
    monkeypatch.setattr(chat_answer, 'get_provider_context_budget', lambda: SimpleNamespace(
        review=ReviewContextBudget(65536), window_tokens=65536, evidence_max_chars=0,
    ))
    reply, _, meta = await chat_answer.generate_chat_answer(
        MessageRequest(message=query, senderName="虚构读者"), None,
        runtime_config=dict(useKnowledgeBase=True, temperature=0.7, maxTokens=256),
        model_generate=Client().generate,
    )
    packets = document_evidence_packets([document])
    expected = build_generation_request(GenerationRequest(
        message=query, persona_prompt="persona", interlocutor="虚构读者",
        retrieval=prepare_answer_citations(chat_answer.RetrievalResult(
            status="ok", evidence="\n\n".join(packet["text"] for packet in packets),
            evidence_packets=packets, documents=(document,), citations=tuple(citations), confidence=0.9,
            public_dependency_indices=(0,),
        )), temperature=0.7, max_tokens=256, context_window_tokens=65536, evidence_max_chars=0,
    ))
    assert len(planning) == 1 and planning[0]["query"] == query
    assert reply == "已依据完整虚构条款回答。"
    assert [citation["source_id"] for citation in meta["citations"]] == [document["id"]]
    assert captured["messages"] == [dict(message) for message in expected.messages]
    assert captured["temperature"] == expected.generation["temperature"] == 0.5
    assert body in "\n".join(message["content"] for message in captured["messages"])
    assert expected.prompt_policy_version


@pytest.mark.asyncio
async def test_configuration_read_failure_stops_before_model_or_save(monkeypatch):

    from fastapi import HTTPException

    from api import generate
    from inference import model_manager

    failure = OSError("database disconnected")

    class BrokenDatabase:
        loras = []

        @property
        def config(self):
            raise failure

    connection = cloud_connection("fixture-model")
    monkeypatch.setattr(model_manager, "get_model_manager", lambda: SimpleNamespace(
        _current_provider=SimpleNamespace(value="openai_compat"),
        get_current_provider=lambda: connection))
    model, save = AsyncMock(), AsyncMock()
    monkeypatch.setattr(chat_answer, 'generate_chat_answer', model)
    monkeypatch.setattr(generate, "_save_message", save)
    with pytest.raises(HTTPException) as caught:
        await generate._generate_reply_impl(MessageRequest(message="完整测试问题"), message_db=BrokenDatabase())
    assert caught.value.status_code == 503 and caught.value.__cause__ is failure
    model.assert_not_awaited()
    save.assert_not_awaited()


@pytest.mark.asyncio
async def test_native_timeout_cancels_work_and_releases_capacity_without_breaker(monkeypatch):
    import asyncio

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

    connection = cloud_connection("fixture-model")
    monkeypatch.setattr(model_manager, "get_model_manager", lambda: SimpleNamespace(
        _current_provider=SimpleNamespace(value="openai_compat"), set_lora_adapter=lambda _: None,
        get_current_provider=lambda: connection))
    monkeypatch.setattr(generate, "db", SimpleNamespace(config={}, loras=[]))
    monkeypatch.setattr(chat_answer, "public_knowledge_db", generate.db)
    monkeypatch.setattr(generate, "response_cache", None)
    monkeypatch.setattr(generate, "circuit_breaker_registry", None)
    monkeypatch.setattr(generate, "get_llm_semaphore", lambda: semaphore)
    monkeypatch.setattr(generate, "_MODEL_INFERENCE_TIMEOUT", 0.01)
    monkeypatch.setattr(chat_answer, 'generate_chat_answer', pending)
    save = AsyncMock()
    monkeypatch.setattr(generate, "_save_message", save)
    with pytest.raises(HTTPException) as caught:
        await generate._generate_reply_impl(MessageRequest(message="完整测试问题"), record_invocation=False)
    assert caught.value.status_code == 500 and isinstance(caught.value.__cause__, TimeoutError)
    assert cancelled.is_set() and not semaphore.locked()
    save.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", [None, {"text": "21元"}, " \n "])
async def test_direct_generation_rejects_invalid_model_text_without_retry(reply):
    query = "完整隔离题设：虚构工坊的门票每张7元，买3张，不收其他费用。请回答总价。"
    calls = []

    async def model(**kwargs):
        assert query in kwargs["messages"][-1]["content"]
        calls.append(kwargs)
        return reply

    with pytest.raises(ValueError, match="Model response must be non-empty text"):
        await generate_character_response(
            GenerationRequest(message=query, max_tokens=128, context_window_tokens=8192), model
        )
    assert len(calls) == 1
