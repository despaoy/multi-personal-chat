from __future__ import annotations

from typing import Any

import pytest

from knowledge.grounded_answer.cache import AnswerCache
from knowledge.grounded_answer.models import AnswerMode, FailureKind
from knowledge.grounded_answer.service import GroundedAnswerService


def _bundle() -> dict[str, Any]:
    return {
        "domains": ["story"],
        "confidence": 0.9,
        "abstained": False,
        "query_analysis": {"entities": ["甲"], "matched_domains": ["story"]},
        "results": [
            {
                "id": "doc-1",
                "document_type": "fact",
                "title": "甲的身份",
                "summary": "甲是学生。",
                "content": "甲是学生。\n证据：甲说自己仍在上学。",
                "domain_id": "story",
                "reality_status": "objective",
                "temporal_scope": "current",
                "content_scope": "main_story",
                "index_version": "v1",
                "rerank_score": 0.9,
                "fused_score": 0.8,
                "vector_score": 0.8,
                "bm25_score": 0.8,
                "source": {
                    "source_path": "story.txt",
                    "line_start": 10,
                    "line_end": 11,
                },
                "metadata": {"story_unit_id": "unit-1"},
            }
        ],
        "citations": [
            {
                "source_id": "doc-1",
                "source_title": "甲的身份",
                "document_type": "fact",
                "source_path": "story.txt",
                "source_line": 10,
                "source_line_end": 11,
                "domain_id": "story",
                "index_version": "v1",
            }
        ],
    }


@pytest.mark.asyncio
async def test_no_domain_abstains_without_invoking_model():
    calls = 0

    async def generate(**_kwargs: Any) -> str:
        nonlocal calls
        calls += 1
        return "不应调用"

    service = GroundedAnswerService(retriever=lambda *_args, **_kwargs: None)
    result = await service.answer("普通问题", generate=generate)

    assert calls == 0
    assert result.answer_mode == AnswerMode.ABSTENTION
    assert result.abstained is True
    assert result.failure_kind == FailureKind.NO_DOMAIN.value


@pytest.mark.asyncio
async def test_retrieval_failure_propagates_and_skips_model():
    def retrieve(*_args: Any, **_kwargs: Any):
        raise OSError("index unavailable")

    async def generate(**_kwargs: Any) -> str:
        raise AssertionError("generation must not run")

    service = GroundedAnswerService(retriever=retrieve)
    with pytest.raises(OSError, match="index unavailable"):
        await service.answer("甲是谁", generate=generate, domain_id="story")


@pytest.mark.asyncio
async def test_missing_citation_raises_instead_of_returning_ungrounded_answer():
    async def generate(**_kwargs: Any) -> str:
        return "甲是学生。"

    service = GroundedAnswerService(
        retriever=lambda *_args, **_kwargs: _bundle(),
        cache=AnswerCache(ttl_seconds=0),
        corrective_enabled=False,
    )
    with pytest.raises(RuntimeError, match="citation validation"):
        await service.answer("甲是谁", generate=generate, domain_id="story")


@pytest.mark.asyncio
async def test_stream_only_emits_validated_answer_and_closes_provider_stream():
    closed = False

    async def generate(**_kwargs: Any) -> str:
        raise AssertionError("stream adapter should be used")

    async def generate_stream(**_kwargs: Any):
        nonlocal closed
        try:
            yield "<think>内部推理</think>甲"
            yield "是学生。[S1]"
        finally:
            closed = True

    service = GroundedAnswerService(
        retriever=lambda *_args, **_kwargs: _bundle(),
        corrective_enabled=False,
    )
    events = [
        event
        async for event in service.answer_stream(
            "甲是谁",
            generate=generate,
            generate_stream=generate_stream,
            domain_id="story",
            model_id="test/model",
        )
    ]

    assert closed is True
    deltas = [event.data["text"] for event in events if event.type == "delta"]
    assert deltas == ["甲是学生。"]
    done = next(event.data for event in events if event.type == "done")
    assert done["answer"] == "甲是学生。"
    assert done["abstained"] is False
    assert done["model_id"] == "test/model"


@pytest.mark.asyncio
async def test_api_generation_adapters_resolve_vllm_only_on_first_generation(monkeypatch):
    from api import ask as ask_api
    from api import generate as generate_api

    resolutions = 0

    class FakeClient:
        async def generate(self, **kwargs: Any):
            if not kwargs.get("stream"):
                return "完成"

            async def chunks():
                yield "流"
                yield "式"

            return chunks()

    async def get_client():
        nonlocal resolutions
        resolutions += 1
        return FakeClient()

    from types import SimpleNamespace

    from inference import model_manager

    provider = SimpleNamespace(get_status=lambda: {"modelName": "local-test-model"})
    manager = SimpleNamespace(_current_provider=SimpleNamespace(value="vllm"), get_current_provider=lambda: provider)
    monkeypatch.setattr(model_manager, "get_model_manager", lambda: manager)
    monkeypatch.setattr(generate_api, "get_vllm_client", get_client)
    generate, generate_stream, _model_id = await ask_api._resolve_generate_adapters()
    assert resolutions == 0

    assert await generate(messages=[], temperature=0.2, max_tokens=64, top_p=0.9) == "完成"
    assert resolutions == 1
    chunks = [chunk async for chunk in generate_stream(messages=[], temperature=0.2, max_tokens=64, top_p=0.9)]
    assert chunks == ["流", "式"]
    assert resolutions == 1


@pytest.mark.parametrize("kind", ["provider", "timeout", "cancelled"])
async def test_generation_error_propagates(kind):
    import asyncio
    error = {"provider": OSError("synthetic provider failure"), "timeout": TimeoutError("synthetic timeout"), "cancelled": asyncio.CancelledError()}[kind]
    async def generate(**kwargs):
        raise error
    service = GroundedAnswerService(retriever=lambda *a, **kw: _bundle(), corrective_enabled=False)
    with pytest.raises(type(error)) as caught:
        await service.answer("甲是谁", generate=generate, domain_id="story")
    assert caught.value is error


@pytest.mark.parametrize("value", [None, {"answer": "invented protocol"}, "", "<think>reasoning only</think>"])
async def test_malformed_model_output_is_an_error(value):
    async def generate(**kwargs):
        return value
    service = GroundedAnswerService(retriever=lambda *a, **kw: _bundle(), corrective_enabled=False)
    with pytest.raises((TypeError, RuntimeError), match="Grounded model"):
        await service.answer("甲是谁", generate=generate, domain_id="story")


async def test_ask_returns_http_error_for_provider_failure(monkeypatch):
    from unittest.mock import AsyncMock

    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from api import ask
    from app.main import global_exception_handler

    generate = AsyncMock(side_effect=OSError("synthetic private detail"))
    service = GroundedAnswerService(retriever=lambda *a, **kw: _bundle(), corrective_enabled=False)
    monkeypatch.setattr(ask, "get_grounded_answer_service", AsyncMock(return_value=service))
    monkeypatch.setattr(ask, "_resolve_generate_adapters", AsyncMock(return_value=(generate, None, "test")))
    app = FastAPI()
    app.include_router(ask.router)
    app.add_exception_handler(Exception, global_exception_handler)
    app.dependency_overrides[ask.get_current_user] = lambda: {"id": 1, "role": "user"}
    async with AsyncClient(transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test") as client:
        result = await client.post("/api/ask", json={"message": "甲是谁", "domainId": "story"})
    assert result.status_code == 500
    assert result.json()["success"] is False
    assert "synthetic private detail" not in result.text
    assert "answer" not in result.json()


@pytest.mark.parametrize("kind", ["provider", "timeout", "cancelled", "invalid_chunk", "empty", "citation"])
async def test_stream_failure_propagates_without_success_events_and_closes(kind):
    import asyncio

    closed = False
    error = OSError("synthetic provider failure")

    async def generate_stream(**kwargs):
        nonlocal closed
        try:
            if kind == "provider":
                yield "unverified partial text"
                raise error
            if kind == "timeout":
                await asyncio.sleep(1)
            if kind == "cancelled":
                raise asyncio.CancelledError
            yield {"invalid_chunk": None, "empty": "<think>only reasoning</think>", "citation": "甲是学生。"}.get(kind, "")
        finally:
            closed = True

    async def generate(**kwargs):
        raise AssertionError("must not fall back to non-stream generation")

    service = GroundedAnswerService(
        retriever=lambda *a, **kw: _bundle(), corrective_enabled=False,
        generation_timeout=0.01 if kind == "timeout" else 10,
    )
    events = []
    expected = {"provider": OSError, "timeout": TimeoutError, "cancelled": asyncio.CancelledError,
                "invalid_chunk": TypeError, "empty": RuntimeError, "citation": RuntimeError}[kind]
    with pytest.raises(expected) as caught:
        async for event in service.answer_stream("甲是谁", generate=generate, generate_stream=generate_stream, domain_id="story"):
            events.append(event.type)
    if kind == "provider":
        assert caught.value is error
    assert closed
    assert events == ["meta"]


@pytest.mark.parametrize("failure", ["retrieval", "provider"])
async def test_ask_stream_ends_with_error_without_done(monkeypatch, failure):
    import json
    from unittest.mock import AsyncMock

    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from api import ask

    def retrieve(*a, **kw):
        if failure == "retrieval":
            raise OSError("synthetic private detail")
        return _bundle()

    generate = AsyncMock(side_effect=OSError("synthetic private detail"))
    service = GroundedAnswerService(retriever=retrieve, corrective_enabled=False)
    monkeypatch.setattr(ask, "get_grounded_answer_service", AsyncMock(return_value=service))
    monkeypatch.setattr(ask, "_resolve_generate_adapters", AsyncMock(return_value=(generate, None, "test")))
    app = FastAPI()
    app.include_router(ask.router)
    app.dependency_overrides[ask.get_current_user] = lambda: {"id": 1, "role": "user"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/ask/stream", json={"message": "甲是谁", "domainId": "story"})
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    assert [event["type"] for event in events] == (["error"] if failure == "retrieval" else ["meta", "error"])
    assert "synthetic private detail" not in response.text
    if failure == "retrieval":
        generate.assert_not_awaited()


async def test_stream_no_domain_remains_normal_abstention():
    async def generate(**kwargs):
        raise AssertionError("no evidence must not invoke model")

    service = GroundedAnswerService(retriever=lambda *a, **kw: None)
    events = [event async for event in service.answer_stream("普通问题", generate=generate)]
    assert [event.type for event in events] == ["meta", "delta", "citations", "done"]
    assert events[-1].data["abstained"] is True


@pytest.mark.parametrize("score", ["invalid", {}, True, float("nan"), float("inf"), float("-inf")])
async def test_rerank_score_contract_rejects_invalid_before_model(score):
    from unittest.mock import AsyncMock

    data = _bundle()
    data["results"][0]["rerank_score"] = score
    data["citations"][0]["rerank_score"] = 0.9
    generate = AsyncMock(side_effect=AssertionError("invalid rank must not invoke model"))
    service = GroundedAnswerService(retriever=lambda *a, **kw: data, corrective_enabled=False)
    with pytest.raises((TypeError, ValueError)):
        await service.answer("甲是谁", generate=generate, domain_id="story")
    generate.assert_not_awaited()


@pytest.mark.parametrize("score,expected", [(None, AnswerMode.GROUNDED_ANSWER), (0.9, AnswerMode.DIRECT_ANSWER),
                                           (1.2, AnswerMode.DIRECT_ANSWER)])
async def test_rerank_score_contract_uses_result_only_and_allows_unranked(score, expected):
    data = _bundle()
    data["results"][0]["rerank_score"] = score
    data["citations"][0]["rerank_score"] = 0.99

    async def generate(**kwargs):
        return "甲是学生。[S1]"

    service = GroundedAnswerService(retriever=lambda *a, **kw: data, corrective_enabled=False)
    result = await service.answer("甲是谁", generate=generate, domain_id="story")
    assert result.answer_mode == expected and not result.abstained


@pytest.mark.parametrize("field", ["results", "citations", "confidence", "abstained"])
async def test_decision_contract_missing_fields_are_errors(field):
    from unittest.mock import AsyncMock

    data = _bundle()
    del data[field]
    generate = AsyncMock()
    service = GroundedAnswerService(retriever=lambda *a, **kw: data, corrective_enabled=False)
    with pytest.raises(KeyError, match=field):
        await service.answer("甲是谁", generate=generate, domain_id="story")
    generate.assert_not_awaited()


@pytest.mark.parametrize("field,value", [
    ("confidence", None), ("confidence", ""), ("confidence", "invalid"), ("confidence", True),
    ("confidence", float("nan")), ("confidence", float("inf")),
    ("abstained", "false"), ("abstained", 0), ("abstained", None),
])
async def test_decision_contract_invalid_values_do_not_become_abstention(field, value):
    from unittest.mock import AsyncMock

    data = _bundle()
    data[field] = value
    generate = AsyncMock()
    service = GroundedAnswerService(retriever=lambda *a, **kw: data, corrective_enabled=False)
    with pytest.raises((TypeError, ValueError)):
        await service.answer("甲是谁", generate=generate, domain_id="story")
    generate.assert_not_awaited()


@pytest.mark.parametrize("confidence,abstained", [(0.0, True), (0.9, False), (1.2, False)])
async def test_decision_contract_normal_low_and_high_confidence(confidence, abstained):
    from unittest.mock import AsyncMock

    data = _bundle()
    data.update(confidence=confidence, abstained=abstained)
    generate = AsyncMock(return_value="甲是学生。[S1]")
    service = GroundedAnswerService(retriever=lambda *a, **kw: data, corrective_enabled=False)
    result = await service.answer("甲是谁", generate=generate, domain_id="story")
    assert result.abstained is abstained
    assert generate.await_count == (0 if abstained else 1)
