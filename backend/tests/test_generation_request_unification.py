from types import SimpleNamespace

import pytest

from db.schemas import MessageRequest
from inference.generation_request import GenerationRequest, build_generation_request


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
