"""Verify multi-turn history flows into the shared generation request."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from db.schemas import MessageRequest


@pytest.mark.asyncio
async def test_vllm_generation_receives_request_history(monkeypatch):
    import api.generate as gen
    from inference.generation_request import GenerationRequest

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
        return SimpleNamespace(
            reply="second-reply",
            plan=SimpleNamespace(retrieval=SimpleNamespace(has_evidence=False)),
        )

    monkeypatch.setattr(gen, "_vllm_client", SimpleNamespace(generate=fake_generate))
    monkeypatch.setattr(gen, "generate_character_response", fake_character_response)
    monkeypatch.setattr(gen, "_get_system_prompt", lambda *args, **kwargs: "")
    monkeypatch.setattr(gen, "_retrieve_rag_bundle", lambda *args, **kwargs: {})

    reply, used_rag, rag_meta = await gen._generate_with_vllm(
        request,
        None,
        None,
        {"useKnowledgeBase": False},
        enable_rag=False,
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

    monkeypatch.setattr(gen, "get_vllm_served_model_name", lambda: "test-model")
    base = {
        "message": "same question",
        "sessionId": "room-1",
        "platform": "qq",
    }
    a = MessageRequest(**base, history=[{"role": "user", "content": "first-a"}])
    b = MessageRequest(**base, history=[{"role": "user", "content": "first-b"}])
    c = MessageRequest(**base, history=[], userName="alice")
    d = MessageRequest(**base, history=[], userName="bob")

    a_hash, a_key, _ = gen._response_cache_keys(a, "lora", {})
    b_hash, b_key, _ = gen._response_cache_keys(b, "lora", {})
    c_hash, c_key, _ = gen._response_cache_keys(c, "lora", {})
    d_hash, d_key, _ = gen._response_cache_keys(d, "lora", {})
    e_hash, e_key, _ = gen._response_cache_keys(a, "lora", {}, enable_rag=False)
    f_hash, f_key, _ = gen._response_cache_keys(a, "lora", {"topP": 0.1})
    g_hash, g_key, _ = gen._response_cache_keys(a, "lora", {"topP": 0.9})

    assert (a_hash, a_key) != (b_hash, b_key)
    assert (c_hash, c_key) != (d_hash, d_key)
    assert (a_hash, a_key) != (e_hash, e_key)
    assert (f_hash, f_key) != (g_hash, g_key)
