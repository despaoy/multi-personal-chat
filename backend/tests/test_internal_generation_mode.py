"""Internal generation mode must not persist chat rows or model invocations."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from db.schemas import MessageRequest


def _request():
    return MessageRequest(message="tool-json", sessionType="private", sessionId="claw")


def _install_common(monkeypatch, gen):
    monkeypatch.setattr(gen, "INPUT_VALIDATOR_AVAILABLE", False)
    monkeypatch.setattr(gen, "response_cache", None)
    monkeypatch.setattr(gen, "db", SimpleNamespace(config={}, loras=[]))
    monkeypatch.setattr(gen, "circuit_breaker_registry", None)
    saved = []
    recorded = []
    monkeypatch.setattr(gen, "_save_message", _record_save(saved))
    monkeypatch.setattr(gen, "_record_model_invocation", _record_invocation(recorded))
    return saved, recorded


def _record_save(saved):
    async def fake(*args, **kwargs):
        saved.append(args)
    return fake


def _record_invocation(recorded):
    async def fake(*args, **kwargs):
        recorded.append(args)
    return fake


@pytest.mark.asyncio
async def test_internal_mode_success_does_not_persist(monkeypatch):
    import api.generate as gen
    from inference import model_manager as mm

    saved, recorded = _install_common(monkeypatch, gen)

    class FakeManager:
        _current_provider = SimpleNamespace(value="vllm")

    monkeypatch.setattr(mm, "get_model_manager", lambda: FakeManager())
    monkeypatch.setattr(gen, "_ensure_vllm", _async_return(True))
    monkeypatch.setattr(gen, "_vllm_client", object())

    async def fake_vllm(*args, **kwargs):
        return "tool-reply", False, {}

    monkeypatch.setattr(gen, "_generate_with_vllm", fake_vllm)

    result = await gen._generate_reply_impl(
        _request(),
        persist_message=False,
        enable_rag=False,
        record_invocation=False,
    )

    assert result.reply == "tool-reply"
    assert saved == []
    assert recorded == []


@pytest.mark.asyncio
async def test_internal_mode_vllm_failure_does_not_persist(monkeypatch):
    import api.generate as gen
    from inference import model_manager as mm

    saved, recorded = _install_common(monkeypatch, gen)

    class FakeManager:
        _current_provider = SimpleNamespace(value="vllm")

        def set_lora_adapter(self, value):
            pass

        async def async_generate(self, **kwargs):
            return "fallback-reply", 0.01

        def get_status(self):
            return {"currentProvider": "vllm", "providers": {"vllm": {"modelName": "fake"}}}

    monkeypatch.setattr(mm, "get_model_manager", lambda: FakeManager())
    monkeypatch.setattr(gen, "_ensure_vllm", _async_return(True))
    monkeypatch.setattr(gen, "_vllm_client", object())
    monkeypatch.setattr(gen, "get_llm_semaphore", lambda: asyncio.Semaphore(2))

    async def fake_vllm(*args, **kwargs):
        raise RuntimeError("vllm down")

    monkeypatch.setattr(gen, "_generate_with_vllm", fake_vllm)

    result = await gen._generate_reply_impl(
        _request(),
        persist_message=False,
        enable_rag=False,
        record_invocation=False,
    )

    assert result.reply == "fallback-reply"
    assert saved == []
    assert recorded == []


@pytest.mark.asyncio
async def test_internal_mode_model_manager_failure_does_not_record_invocation(monkeypatch):
    import api.generate as gen
    from inference import model_manager as mm

    saved, recorded = _install_common(monkeypatch, gen)

    class FakeManager:
        _current_provider = SimpleNamespace(value="vllm")

        def set_lora_adapter(self, value):
            pass

        async def async_generate(self, **kwargs):
            raise RuntimeError("model manager down")

    monkeypatch.setattr(mm, "get_model_manager", lambda: FakeManager())
    monkeypatch.setattr(gen, "_ensure_vllm", _async_return(True))
    monkeypatch.setattr(gen, "_vllm_client", object())
    monkeypatch.setattr(gen, "get_llm_semaphore", lambda: asyncio.Semaphore(2))

    async def fake_vllm(*args, **kwargs):
        raise RuntimeError("vllm down")

    monkeypatch.setattr(gen, "_generate_with_vllm", fake_vllm)

    with pytest.raises(HTTPException):
        await gen._generate_reply_impl(
            _request(),
            persist_message=False,
            enable_rag=False,
            record_invocation=False,
        )

    assert saved == []
    assert recorded == []


def _async_return(value):
    async def fake(*args, **kwargs):
        return value
    return fake
