"""Real vLLM transport/circuit and grounded SSE, with isolated HTTP bytes.

Retrieval ranking and HTTP traffic are controlled. No real model or network.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from api import ask
from app import providers
from db.schemas import AskRequest
from inference import model_manager
from inference.vllm_client import VLLMClient
from infra.circuit_breaker import CircuitBreakerRegistry
from knowledge.grounded_answer.service import GroundedAnswerService

pytestmark = pytest.mark.usefixtures("literal_grounded_query")

QUERY = "依据虚构青屿档案馆的完整资料，说明周二开放时间，并保留雨天例外和不开放的日期。"
SOURCE = (
    "虚构青屿档案馆周二09:00至17:00开放，周一不开放。\n"
    "雨天仅关闭室外展区；暴雨则全馆暂停开放，已预约也不能例外入馆。"
)
ANSWER = (
    "周二09:00至17:00开放，周一不开放；雨天仅关闭室外展区，"
    "暴雨则全馆暂停开放，已预约也不能例外入馆。[S1]"
)


def event(delta, finish_reason=None):
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}


def frames(kind):
    role = event({"role": "assistant", "content": ""})
    reasoning = event({"reasoning_content": "合成隐藏推理字段，不得成为回答", "content": None})
    stop = event({}, "stop")
    if kind == "healthy":
        return [
            role, reasoning, event({"content": "<thi"}), event({"content": "nk>合成隐藏内容</th"}),
            event({"content": "ink>" + ANSWER}), stop,
            {"choices": [], "usage": {"prompt_tokens": 35, "completion_tokens": 40, "total_tokens": 75}},
            "[DONE]",
        ]
    if kind == "reasoning_only":
        return [role, reasoning, stop, "[DONE]"]
    prefix = [role, reasoning, event({"content": ANSWER})]
    suffix = {
        "malformed_json": ['{"choices":', stop, "[DONE]"],
        "error_event": [{"error": {"message": "isolated upstream stream error", "type": "server_error"}}, stop, "[DONE]"],
        "eof_without_done": [stop],
        "length": [event({}, "length"), "[DONE]"],
        "invalid_choices": [{"choices": {}}, stop, "[DONE]"],
        "done_without_stop": ["[DONE]"],
    }
    return prefix + suffix[kind]


class TrackedSseBody(httpx.AsyncByteStream):
    def __init__(self, events):
        self.closed = False
        self.content = b"".join(
            ("data: " + (value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)) + "\n\n").encode()
            for value in events
        )

    async def __aiter__(self):
        # Exercise httpx's real line/UTF-8 decoding across arbitrary byte splits.
        for index in range(0, len(self.content), 13):
            yield self.content[index:index + 13]

    async def aclose(self):
        self.closed = True


@asynccontextmanager
async def arranged_vllm_stream(monkeypatch):
    from app import config

    monkeypatch.setattr(config, "circuit_breaker_registry", CircuitBreakerRegistry())
    monkeypatch.setenv("VLLM_MAX_CONCURRENCY", "2")
    rig = SimpleNamespace(events=frames("healthy"), requests=[], bodies=[])

    def respond(request):
        payload = json.loads(request.content)
        assert request.url == "https://vllm.fixture.invalid/v1/chat/completions"
        assert payload["stream"] is True and payload["model"] == "fiction-vllm-model"
        assert payload["messages"] and all(item["role"] in {"system", "user", "assistant"} for item in payload["messages"])
        rig.requests.append(payload)
        body = TrackedSseBody(rig.events)
        rig.bodies.append(body)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=body)

    http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    rig.client = VLLMClient(
        base_urls="https://vllm.fixture.invalid", api_key="synthetic-stream-key",
        model="fiction-vllm-model", timeout=2, max_retries=1,
    )
    # Only this instance's HTTP transport is replaced. generate(), admission,
    # balancing, parsing, filtering, resource cleanup and accounting stay real.
    rig.client._client = http
    try:
        yield rig
    finally:
        await rig.client.close()


@pytest.fixture
async def stream_rig(monkeypatch):
    async with arranged_vllm_stream(monkeypatch) as rig:
        yield rig


async def collect(rig):
    stream = await rig.client.generate(
        messages=[{"role": "system", "content": "按完整给定来源作答，并保留时间、否定和例外。"},
                  {"role": "user", "content": SOURCE + "\n" + QUERY}],
        stream=True, temperature=.2, max_tokens=256, top_p=.9,
    )
    try:
        return [part async for part in stream]
    finally:
        await stream.aclose()


async def assert_outcome(rig, *, success):
    stats = await rig.client.get_stats()
    instance, = stats["instances"]
    circuit = stats["circuit_breaker"]
    assert len(rig.requests) == 1 and len(rig.bodies) == 1 and rig.bodies[0].closed
    assert instance["current_connections"] == 0 and instance["total_requests"] == 1
    assert instance["success_count"] == int(success) and instance["failure_count"] == int(not success)
    assert circuit["total_calls"] == 1
    assert circuit["success_calls"] == int(success) and circuit["failure_calls"] == int(not success)


async def test_valid_role_reasoning_usage_and_stop_done_return_only_complete_visible_text(stream_rig):
    rig = stream_rig
    assert "".join(await collect(rig)) == ANSWER
    await assert_outcome(rig, success=True)


@pytest.mark.parametrize("kind", [
    "malformed_json", "error_event", "eof_without_done", "length", "invalid_choices",
    "done_without_stop", "reasoning_only",
])
async def test_corrupt_or_incomplete_sse_never_records_transport_success(stream_rig, kind):
    rig = stream_rig
    rig.events = frames(kind)
    with pytest.raises(ValueError):
        await collect(rig)
    await assert_outcome(rig, success=False)


async def test_consumer_aclose_releases_upstream_without_counting_partial_text_as_success(stream_rig):
    rig = stream_rig
    stream = await rig.client.generate(
        messages=[{"role": "user", "content": SOURCE + "\n" + QUERY}],
        stream=True, temperature=.2, max_tokens=256, top_p=.9,
    )
    try:
        assert await anext(stream) == ANSWER
        assert len(rig.bodies) == 1 and not rig.bodies[0].closed
    finally:
        await stream.aclose()
    stats = await rig.client.get_stats()
    instance, = stats["instances"]
    assert rig.bodies[0].closed and len(rig.requests) == 1
    assert instance["current_connections"] == 0 and instance["success_count"] == 0
    assert stats["circuit_breaker"]["success_calls"] == 0


def complete_bundle(source_path):
    return {
        "domains": ["fiction-archive"], "confidence": .95, "abstained": False,
        "query_analysis": {"entities": ["虚构青屿档案馆"], "matched_domains": ["fiction-archive"]},
        "results": [{
            "id": "fiction-archive-source", "document_type": "fact", "title": "虚构档案馆完整开放资料",
            "summary": SOURCE, "content": SOURCE, "domain_id": "fiction-archive", "index_version": "fiction-v1",
            "reality_status": "objective", "temporal_scope": "current", "content_scope": "main_story",
            "rerank_score": .95, "fused_score": .95, "vector_score": .95, "bm25_score": .95,
            "source": {"source_path": str(source_path), "line_start": 1, "line_end": 2},
            "metadata": {"story_unit_id": "fiction-archive-unit"},
        }],
        "citations": [{
            "source_id": "fiction-archive-source", "source_title": "虚构档案馆完整开放资料",
            "document_type": "fact", "domain_id": "fiction-archive", "index_version": "fiction-v1",
            "source_path": str(source_path), "source_line": 1, "source_line_end": 2,
        }],
    }


async def test_actual_ask_stream_rejects_corruption_after_fully_cited_body_and_closes_upstream(
    stream_rig, monkeypatch, tmp_path,
):
    from db.database import SQLiteDB

    rig = stream_rig
    rig.events = frames("malformed_json")
    source_path = tmp_path / "fiction-complete-source.txt"
    source_path.write_text(SOURCE, encoding="utf-8")
    retrieved = []

    def retrieve(query, *, domain_id, top_k, filters):
        assert query == QUERY and domain_id == "fiction-archive" and top_k == 1 and filters is None
        assert source_path.read_text(encoding="utf-8") == SOURCE
        retrieved.append(query)
        return complete_bundle(source_path)

    service = GroundedAnswerService(retriever=retrieve)
    provider = SimpleNamespace(name="vllm", get_status=lambda: {"modelName": "fiction-vllm-model"})
    manager = SimpleNamespace(
        _current_provider=SimpleNamespace(value="vllm"), get_current_provider=lambda: provider,
    )
    monkeypatch.setattr(model_manager, "get_model_manager", lambda: manager)
    monkeypatch.setattr(providers, "get_vllm_client", AsyncMock(return_value=rig.client))
    monkeypatch.setattr(ask, "get_grounded_answer_service", AsyncMock(return_value=service))
    request = AskRequest(message=QUERY, domainId="fiction-archive", topK=1, temperature=.2,
                         maxTokens=256, topP=.9, wantCitations=True)
    response = await ask.ask_stream(
        request, SimpleNamespace(is_disconnected=AsyncMock(return_value=False)),
        current_user={"user_id": 901, "username": "synthetic", "role": "user"},
        database=SQLiteDB(tmp_path / "ask-accounting.sqlite"),
    )
    events = []
    async for chunk in response.body_iterator:
        text = chunk.decode() if isinstance(chunk, bytes) else chunk
        assert text.startswith("data: ")
        events.append(json.loads(text[6:]))
    assert [item["type"] for item in events] == ["meta", "error"]
    assert retrieved == [QUERY]
    wire = rig.requests[0]["messages"]
    assert QUERY in wire[-1]["content"] and SOURCE in wire[-1]["content"]
    await assert_outcome(rig, success=False)


@pytest.mark.parametrize("kind", ["healthy", "length", "reasoning_only"])
async def test_nonstream_uses_completed_response_contract_and_visible_content(monkeypatch, kind):
    from app import config

    monkeypatch.setattr(config, "circuit_breaker_registry", CircuitBreakerRegistry())
    content = "<think>合成隐藏推理</think>" + (ANSWER if kind != "reasoning_only" else "")
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        assert payload["stream"] is False and payload["messages"][-1]["content"] == SOURCE + "\n" + QUERY
        requests.append(payload)
        return httpx.Response(200, json={"choices": [{
            "finish_reason": "length" if kind == "length" else "stop", "message": {"content": content},
        }]})

    client = VLLMClient(base_urls="https://vllm.fixture.invalid", api_key="synthetic-test-key", model="fiction-vllm-model", max_retries=1)
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        call = client.generate(messages=[{"role": "user", "content": SOURCE + "\n" + QUERY}])
        if kind == "healthy":
            assert await call == ANSWER
        else:
            with pytest.raises(ValueError):
                await call
        stats = await client.get_stats()
        instance, = stats["instances"]
        success = int(kind == "healthy")
        assert len(requests) == 1 and instance["current_connections"] == 0
        assert instance["success_count"] == success and instance["failure_count"] == 1-success
        assert stats["circuit_breaker"]["success_calls"] == success
        assert stats["circuit_breaker"]["failure_calls"] == 1-success
    finally:
        await client.close()


@pytest.mark.parametrize("streaming", [False, True])
async def test_http_rejection_is_safe_explicit_and_recorded_once(monkeypatch, streaming):
    from app import config
    from inference.openai_protocol import ProviderResponseError

    monkeypatch.setattr(config, "circuit_breaker_registry", CircuitBreakerRegistry())
    calls = []

    def reject(request):
        calls.append(request)
        return httpx.Response(401, text="isolated-private-upstream-detail")

    client = VLLMClient(base_urls="https://vllm.fixture.invalid", api_key="synthetic-test-key", model="fiction-vllm-model", max_retries=1)
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(reject))
    try:
        with pytest.raises(ProviderResponseError) as caught:
            result = await client.generate(messages=[{"role": "user", "content": SOURCE + "\n" + QUERY}], stream=streaming)
            if streaming:
                _ = [part async for part in result]
        assert caught.value.status_code == 401 and "isolated-private" not in str(caught.value)
        stats = await client.get_stats()
        instance, = stats["instances"]
        assert len(calls) == 1 and instance["current_connections"] == 0
        assert instance["success_count"] == 0 and instance["failure_count"] == 1
        assert stats["circuit_breaker"]["success_calls"] == 0 and stats["circuit_breaker"]["failure_calls"] == 1
    finally:
        await client.close()
