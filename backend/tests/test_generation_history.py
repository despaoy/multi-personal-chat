"""Verify multi-turn history flows into the shared generation request."""

from __future__ import annotations

from services import chat_answer


import pytest

from db.schemas import MessageRequest
from inference.request_context import InferenceRequestContext, use_request_context
from inference.token_counting import text_token_counter


@pytest.mark.asyncio
async def test_vllm_generation_receives_request_history(monkeypatch):
    from inference.generation_request import GenerationRequest, GenerationResult, build_generation_request

    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "first-reply"},
    ]
    request = MessageRequest(
        message="second",
        sessionType="private",
        sessionId="room-1",
        platform="qq",
        adapter="nonebot",
        history=history,
    )

    captured = {}

    async def fake_generate(messages, **kwargs):
        return "second-reply"

    async def fake_character_response(req, generate):
        captured["request"] = req
        return GenerationResult(reply="second-reply", plan=build_generation_request(req))

    monkeypatch.setattr(chat_answer, 'generate_character_response', fake_character_response)
    monkeypatch.setattr(chat_answer, 'get_lora_system_prompt', lambda *args, **kwargs: "")
    monkeypatch.setattr(chat_answer, 'retrieve_chat_knowledge', lambda *args, **kwargs: {})
    monkeypatch.setenv("VLLM_MAX_MODEL_LEN", "8192")
    monkeypatch.setenv("RAG_CITATIONS_ENABLED", "true")
    monkeypatch.setenv("CHARACTER_INDEPENDENT_TASKS_ENABLED", "false")

    context = InferenceRequestContext("vllm", None, text_token_counter("vllm", "", ""))
    with use_request_context(context):
        reply, used_rag, rag_meta = await chat_answer.generate_chat_answer(
            request,
            None,
            None,
            runtime_config={"useKnowledgeBase": False},
            enable_rag=False,
            model_generate=fake_generate,
        )

    assert reply == "second-reply"
    assert isinstance(captured["request"], GenerationRequest)
    assert list(captured["request"].history) == history


@pytest.mark.parametrize(
    "history, error_part",
    [
        ([{"role": "system", "content": "bad"}], "role"),
        ([{"role": "user", "content": ""}], "content"),
        ([{"role": "user", "content": "x" * 4001}], "exceeds"),
    ],
)
def test_message_request_rejects_invalid_history(history, error_part):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        MessageRequest(message="hello", history=history)


def test_message_request_rejects_too_many_history_turns():
    from pydantic import ValidationError

    history = [{"role": "user", "content": "x"} for _ in range(41)]
    with pytest.raises(ValidationError):
        MessageRequest(message="hello", history=history)


def test_response_cache_key_covers_history_and_interlocutor(monkeypatch):
    import api.generate as gen

    model_identity = {"provider": "vllm", "status": {"modelName": "test-model"}}
    base = {
        "message": "same question",
        "sessionId": "room-1",
        "platform": "qq",
    }
    a = MessageRequest(**base, history=[{"role": "user", "content": "first-a"}])
    b = MessageRequest(**base, history=[{"role": "user", "content": "first-b"}])
    c = MessageRequest(**base, history=[], userName="alice")
    d = MessageRequest(**base, history=[], userName="bob")

    a_hash, a_key, _ = gen._response_cache_keys(a, "lora", {}, model_identity=model_identity)
    b_hash, b_key, _ = gen._response_cache_keys(b, "lora", {}, model_identity=model_identity)
    c_hash, c_key, _ = gen._response_cache_keys(c, "lora", {}, model_identity=model_identity)
    d_hash, d_key, _ = gen._response_cache_keys(d, "lora", {}, model_identity=model_identity)
    e_hash, e_key, _ = gen._response_cache_keys(a, "lora", {}, enable_rag=False, model_identity=model_identity)
    f_hash, f_key, _ = gen._response_cache_keys(a, "lora", {"topP": 0.1}, model_identity=model_identity)
    g_hash, g_key, _ = gen._response_cache_keys(a, "lora", {"topP": 0.9}, model_identity=model_identity)

    assert (a_hash, a_key) != (b_hash, b_key)
    assert (c_hash, c_key) != (d_hash, d_key)
    assert (a_hash, a_key) != (e_hash, e_key)
    assert (f_hash, f_key) != (g_hash, g_key)
