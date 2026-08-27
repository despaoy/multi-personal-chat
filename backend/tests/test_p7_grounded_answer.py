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
async def test_retrieval_failure_is_controlled_and_skips_model():
    def retrieve(*_args: Any, **_kwargs: Any):
        raise OSError("index unavailable")

    async def generate(**_kwargs: Any) -> str:
        raise AssertionError("generation must not run")

    service = GroundedAnswerService(retriever=retrieve)
    result = await service.answer("甲是谁", generate=generate, domain_id="story")

    assert result.abstained is True
    assert result.failure_kind == FailureKind.RETRIEVAL_UNAVAILABLE.value
    assert result.model_invoked is False


@pytest.mark.asyncio
async def test_missing_citation_degrades_instead_of_returning_ungrounded_answer():
    async def generate(**_kwargs: Any) -> str:
        return "甲是学生。"

    service = GroundedAnswerService(
        retriever=lambda *_args, **_kwargs: _bundle(),
        cache=AnswerCache(ttl_seconds=0),
        corrective_enabled=False,
    )
    result = await service.answer("甲是谁", generate=generate, domain_id="story")

    assert result.abstained is True
    assert result.failure_kind == FailureKind.INVALID_CITATION.value
    assert result.citations == []


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

    monkeypatch.setattr(generate_api, "get_vllm_client", get_client)
    generate, generate_stream, _model_id = await ask_api._resolve_generate_adapters()
    assert resolutions == 0

    assert await generate(messages=[], temperature=0.2, max_tokens=64, top_p=0.9) == "完成"
    assert resolutions == 1
    chunks = [chunk async for chunk in generate_stream(messages=[], temperature=0.2, max_tokens=64, top_p=0.9)]
    assert chunks == ["流", "式"]
    assert resolutions == 1
