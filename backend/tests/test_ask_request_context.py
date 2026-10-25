"""Independent ask source/cache ownership; real SQLite and adapter, isolated HTTP.

Only retrieval ranking and provider output are synthetic. Complete source text
and connection settings are persisted in SQLite and read through the real DB
methods. No semantic-review or token-budget stage is invented for this route.
"""

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from api import ask
from db.database import SQLiteDB
from db.schemas import AskRequest
from inference import model_manager
from inference.request_context import get_request_context
from knowledge.grounded_answer.cache import AnswerCache
from knowledge.grounded_answer.service import GroundedAnswerService

pytestmark = pytest.mark.usefixtures("literal_grounded_query")

SOURCE = "合成原作资料：虚构档案馆周二09:00开门、17:00闭馆；暴雨暂停开放。这是完整给定资料，不涉及真实用户。"
QUESTION = "依据给定档案馆资料说明周二开放时间，并保留暴雨例外。"
ANSWER = "周二09:00开门、17:00闭馆；暴雨暂停开放。"
USER = {"user_id": 1, "username": "synthetic", "role": "user"}


def request():
    return AskRequest(message=QUESTION, domainId="story", topK=1, temperature=.2,
                      maxTokens=128, topP=1.0, wantCitations=True)


def decode_event(chunk):
    text = chunk.decode() if isinstance(chunk, bytes) else chunk
    assert text.startswith("data: ")
    return json.loads(text[6:])


@pytest.fixture
async def ask_rig(tmp_path, monkeypatch):
    # The global chat environment deliberately describes another connection.
    monkeypatch.setenv("MODEL_PROVIDER", "vllm")
    monkeypatch.setenv("OPENAI_COMPAT_MODEL", "unrelated-environment-model")
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://local.invalid/v1")
    database = SQLiteDB(tmp_path / "ask-scope.sqlite")
    database.update_config({
        "openaiCompatBaseUrl": "https://api.deepseek.com",
        "openaiCompatApiKey": "synthetic-first-key",
        "openaiCompatModel": "deepseek-flash",
    })
    document = database.save_knowledge_document({"title": "虚构档案馆完整开放资料", "content": SOURCE}, chunks=[SOURCE])
    source_path = tmp_path / "ask-scope-complete-source.txt"
    source_path.write_text(SOURCE, encoding="utf-8")
    rig = SimpleNamespace(
        database=database, document_id=str(document["id"]), calls=[], retrieval_contexts=[],
        after_retrieve=None, failure=None, hold=False, entered=asyncio.Event(),
        release=asyncio.Event(), transport_cancelled=asyncio.Event(), retrievals=0,
    )

    def config():
        try:
            return dict(database.config)
        finally:
            # Reads may run in the resolver's thread. Close that thread's own
            # SQLite connection, not an unrelated test/worker connection.
            database.close_connection()

    monkeypatch.setattr(model_manager, "_get_db_config", config)
    provider = model_manager.OpenAICompatProvider()
    manager = model_manager.ModelManager.__new__(model_manager.ModelManager)
    manager._current_provider = model_manager.ModelProvider.OPENAI_COMPAT
    manager._providers = {model_manager.ModelProvider.OPENAI_COMPAT: provider}
    monkeypatch.setattr(model_manager, "get_model_manager", lambda: manager)
    rig.snapshots = Mock(wraps=provider.snapshot)
    monkeypatch.setattr(provider, "snapshot", rig.snapshots)
    rig.provider, rig.manager = provider, manager

    def set_config(**values):
        try:
            database.update_config(values)
        finally:
            database.close_connection()

    rig.set_config = set_config

    async def respond(http_request):
        wire = json.loads(http_request.content)
        rig.calls.append({"body": wire, "endpoint": str(http_request.url),
                          "auth": http_request.headers["Authorization"]})
        assert SOURCE in str(wire["messages"])
        assert wire["temperature"] == .2 and wire["max_tokens"] == 128 and wire["top_p"] == 1.0
        rig.entered.set()
        try:
            if rig.hold:
                await rig.release.wait()
            if rig.failure is not None:
                raise rig.failure
            return httpx.Response(200, json={"usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}, "choices": [{"finish_reason": "stop", "message": {
                "role": "assistant", "content": ANSWER + "[S1]",
            }}]})
        except asyncio.CancelledError:
            rig.transport_cancelled.set()
            raise

    transport = httpx.MockTransport(respond)

    @asynccontextmanager
    async def acquire(_provider, timeout):
        async with httpx.AsyncClient(transport=transport, timeout=timeout) as client:
            yield client

    monkeypatch.setattr(model_manager.OpenAICompatProvider, "_acquire_http_client", acquire)

    def retrieve(query, *, domain_id, top_k, filters):
        assert query == QUESTION and domain_id == "story" and top_k == 1 and filters is None
        rig.retrievals += 1
        rig.retrieval_contexts.append(get_request_context())
        try:
            row = database.get_knowledge_document(int(rig.document_id))
            assert row["content"] == source_path.read_text(encoding="utf-8") == SOURCE
            callback, rig.after_retrieve = rig.after_retrieve, None
            if callback is not None:
                callback()
            # Ranking is an explicit retrieval-boundary control. Evidence,
            # attribution, citation binding, prompt and answer handling are real.
            return {
                "domains": ["story"], "confidence": .95, "abstained": False,
                "query_analysis": {"entities": ["虚构档案馆"], "matched_domains": ["story"]},
                "results": [{
                    "id": rig.document_id, "document_type": "fact", "title": row["title"],
                    "summary": SOURCE, "content": row["content"], "domain_id": "story",
                    "reality_status": "objective", "temporal_scope": "current",
                    "content_scope": "main_story", "index_version": "synthetic-source-v1",
                    "rerank_score": .95, "fused_score": .95, "vector_score": .95, "bm25_score": .95,
                    "source": {"source_path": str(source_path), "line_start": 1, "line_end": 1},
                    "metadata": {"story_unit_id": "synthetic-library-unit"},
                }],
                "citations": [{
                    "source_id": rig.document_id, "source_title": row["title"], "document_type": "fact",
                    "source_path": str(source_path), "source_line": 1, "source_line_end": 1,
                    "domain_id": "story", "index_version": "synthetic-source-v1",
                }],
            }
        finally:
            database.close_connection()

    rig.service = GroundedAnswerService(
        retriever=retrieve, cache=AnswerCache(ttl_seconds=60),
        index_version_resolver=lambda domain: "synthetic-source-v1",
    )
    monkeypatch.setattr(ask, "get_grounded_answer_service", AsyncMock(return_value=rig.service))
    rig.http_request = SimpleNamespace(is_disconnected=AsyncMock(return_value=False))
    try:
        yield rig
    finally:
        rig.release.set()
        await transport.aclose()
        database.close_connection()


async def test_answer_configuration_change_during_retrieval_uses_one_snapshot_and_next_request_refreshes(ask_rig):
    rig = ask_rig
    rig.after_retrieve = lambda: rig.set_config(openaiCompatModel="deepseek-v4-pro")
    first = await ask.ask(request(), current_user=USER, database=rig.database)
    assert first.model == "openai_compat/deepseek-flash" and first.answer == ANSWER
    assert first.modelInvoked and not first.cacheHit and len(first.citations) == 1
    assert rig.calls[0]["body"]["model"] == "deepseek-flash"
    assert rig.retrieval_contexts[0].provider.model == "deepseek-flash"
    assert get_request_context() is None

    second = await ask.ask(request(), current_user=USER, database=rig.database)
    assert second.model == "openai_compat/deepseek-v4-pro" and second.answer == ANSWER
    assert second.modelInvoked and not second.cacheHit
    assert rig.calls[1]["body"]["model"] == "deepseek-v4-pro"
    assert rig.retrieval_contexts[1].provider.model == "deepseek-v4-pro"
    rig.set_config(openaiCompatModel="deepseek-flash")
    cached = await ask.ask(request(), current_user=USER, database=rig.database)
    assert cached.cacheHit and not cached.modelInvoked and cached.answer == first.answer
    assert cached.model == first.model and len(rig.calls) == 2 and rig.retrievals == 2
    assert rig.snapshots.call_count == 3 and get_request_context() is None
    assert rig.database.get_knowledge_document(int(rig.document_id))["content"] == SOURCE


async def test_same_model_endpoint_or_credential_change_does_not_reuse_another_source_cache(ask_rig):
    rig = ask_rig
    first = await ask.ask(request(), current_user=USER, database=rig.database)
    rig.set_config(openaiCompatBaseUrl="https://alternate.invalid/v1")
    second = await ask.ask(request(), current_user=USER, database=rig.database)
    rig.set_config(openaiCompatApiKey="synthetic-second-key")
    third = await ask.ask(request(), current_user=USER, database=rig.database)
    assert first.model == second.model == third.model == "openai_compat/deepseek-flash"
    assert all(result.modelInvoked and not result.cacheHit for result in (first, second, third))
    assert len(rig.calls) == 3
    assert [row["endpoint"] for row in rig.calls] == [
        "https://api.deepseek.com/v1/chat/completions",
        "https://alternate.invalid/v1/chat/completions",
        "https://alternate.invalid/v1/chat/completions",
    ]
    assert [row["auth"] for row in rig.calls] == [
        "Bearer synthetic-first-key", "Bearer synthetic-first-key", "Bearer synthetic-second-key",
    ]
    assert "synthetic-" not in first.model + second.model + third.model
    assert get_request_context() is None


async def test_sse_real_iterator_owns_scope_through_retrieval_generation_and_done(ask_rig):
    rig = ask_rig
    rig.after_retrieve = lambda: rig.set_config(openaiCompatModel="deepseek-v4-pro")
    response = await ask.ask_stream(request(), rig.http_request, current_user=USER, database=rig.database)
    # Returning StreamingResponse cannot leave a binding in the endpoint caller.
    assert get_request_context() is None and not rig.calls
    events = [decode_event(chunk) async for chunk in response.body_iterator]
    assert [event["type"] for event in events] == ["meta", "delta", "citations", "done"]
    assert events[1]["text"] == ANSWER
    assert events[-1]["model_id"] == "openai_compat/deepseek-flash"
    assert events[-1]["model_invoked"] is True and events[-1]["abstained"] is False
    assert len(events[2]["citations"]) == 1
    assert rig.calls[0]["body"]["model"] == "deepseek-flash"
    assert rig.retrieval_contexts[0].provider.model == "deepseek-flash"
    assert rig.snapshots.call_count == 1 and get_request_context() is None


async def test_closing_sse_after_meta_restores_scope_without_starting_generation(ask_rig):
    rig = ask_rig
    response = await ask.ask_stream(request(), rig.http_request, current_user=USER, database=rig.database)
    iterator = response.body_iterator
    try:
        assert decode_event(await anext(iterator))["type"] == "meta"
    finally:
        await iterator.aclose()
    assert rig.retrieval_contexts[0] is not None
    assert not rig.calls and rig.service.cache.stats()["entries"] == 0
    assert get_request_context() is None


@pytest.mark.parametrize("timeout", [False, True], ids=["provider-error", "bounded-timeout"])
async def test_nonstream_failure_preserves_original_failure_and_clears_scope_without_cache(ask_rig, timeout):
    rig = ask_rig
    failure = OSError("synthetic-private-provider-detail")
    rig.hold = timeout
    rig.failure = None if timeout else failure
    rig.service.generation_timeout = .01 if timeout else 10
    with pytest.raises(TimeoutError if timeout else OSError) as error:
        await ask.ask(request(), current_user=USER, database=rig.database)
    if timeout:
        assert rig.transport_cancelled.is_set()
    else:
        assert error.value is failure
    assert len(rig.calls) == 1 and rig.service.cache.stats()["entries"] == 0
    assert get_request_context() is None


async def test_external_sse_cancellation_cancels_actual_transport_and_clears_consuming_task_scope(ask_rig):
    rig = ask_rig
    rig.hold = True
    response = await ask.ask_stream(request(), rig.http_request, current_user=USER, database=rig.database)
    events, cleaned = [], []

    async def consume():
        try:
            async for chunk in response.body_iterator:
                events.append(decode_event(chunk))
        finally:
            cleaned.append(get_request_context())

    task = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(rig.entered.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        rig.release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert rig.transport_cancelled.is_set() and len(rig.calls) == 1
    assert [event["type"] for event in events] == ["meta"]
    assert cleaned == [None] and get_request_context() is None
    assert rig.service.cache.stats()["entries"] == 0


async def test_simultaneous_ask_requests_keep_distinct_snapshots_and_reset_their_own_tasks(ask_rig, monkeypatch):
    rig = ask_rig
    original = rig.service._retrieve
    entered = {model: asyncio.Event() for model in ("deepseek-flash", "deepseek-v4-pro")}
    release = asyncio.Event()
    cleaned = []

    async def pause_retrieval(*args, **kwargs):
        context = get_request_context()
        entered[context.provider.model].set()
        await release.wait()
        return await original(*args, **kwargs)

    async def invoke():
        try:
            return await ask.ask(request(), current_user=USER, database=rig.database)
        finally:
            cleaned.append(get_request_context())

    monkeypatch.setattr(rig.service, "_retrieve", pause_retrieval)
    first = asyncio.create_task(invoke())
    second = None
    try:
        await asyncio.wait_for(entered["deepseek-flash"].wait(), 2)
        rig.set_config(openaiCompatModel="deepseek-v4-pro")
        second = asyncio.create_task(invoke())
        await asyncio.wait_for(entered["deepseek-v4-pro"].wait(), 2)
    finally:
        release.set()
        results = await asyncio.gather(first, *([second] if second is not None else []))
    assert [result.model for result in results] == [
        "openai_compat/deepseek-flash", "openai_compat/deepseek-v4-pro",
    ]
    assert all(result.answer == ANSWER and result.modelInvoked for result in results)
    assert {row["body"]["model"] for row in rig.calls} == {"deepseek-flash", "deepseek-v4-pro"}
    assert len(rig.calls) == 2 and rig.snapshots.call_count == 2
    assert len(rig.retrieval_contexts) == 2
    assert rig.retrieval_contexts[0] is not rig.retrieval_contexts[1]
    assert cleaned == [None, None] and get_request_context() is None


async def test_provider_type_change_keeps_current_owner_and_blocks_new_unsupported_request(ask_rig):
    rig = ask_rig
    rig.manager._providers[model_manager.ModelProvider.OLLAMA] = SimpleNamespace(
        name="ollama", get_status=lambda: {"modelName": "synthetic-ollama"},
    )
    rig.after_retrieve = lambda: setattr(rig.manager, "_current_provider", model_manager.ModelProvider.OLLAMA)
    try:
        first = await ask.ask(request(), current_user=USER, database=rig.database)
        assert first.answer == ANSWER and first.model == "openai_compat/deepseek-flash"
        assert first.modelInvoked and not first.cacheHit and len(rig.calls) == 1
        assert rig.calls[0]["body"]["model"] == "deepseek-flash"
        assert rig.retrieval_contexts[0].provider_name == "openai_compat"
        assert get_request_context() is None
        with pytest.raises(ValueError, match="Knowledge answers require"):
            await ask.ask(request(), current_user=USER, database=rig.database)
        assert len(rig.calls) == 1 and rig.retrievals == 2
        assert rig.retrieval_contexts[1].provider_name == "ollama"
        assert get_request_context() is None
    finally:
        rig.manager._current_provider = model_manager.ModelProvider.OPENAI_COMPAT
    cached = await ask.ask(request(), current_user=USER, database=rig.database)
    assert cached.answer == ANSWER and not cached.modelInvoked and cached.cacheHit
    assert cached.model == "openai_compat/deepseek-flash"
    assert len(rig.calls) == 1 and rig.snapshots.call_count == 2
    assert get_request_context() is None
