"""Canonical preparation shares the generation request's usage and identity."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI, HTTPException
from test_compiled_cloud_transport import installed as installed
from test_review_usage_recording import review_usage as review_usage

from api import generate, narrative
from app.runtime import RuntimeContainer
from character.context_builder import MemoryContextBudgetError
from db.schemas import MessageRequest
from inference.provider_context import get_provider_context_budget
from inference.request_context import get_request_context
from inference.usage import has_pending_usage
from infra.resource_pool import HttpClientPool
from services.character_context import CharacterContextService

ROLE = "tsukiyashiro_kisaki"
QUESTION = "本题有三张票，每张两元，没有额外费用。请计算总额。"
pytestmark = pytest.mark.usefixtures("no_live_context_review")


@pytest.fixture
async def canonical_usage(review_usage, monkeypatch):
    rig = review_usage
    monkeypatch.setenv("NARRATIVE_BRANCHES_ENABLED", "true")
    monkeypatch.setenv("OPENAI_COMPAT_CONTEXT_WINDOW_TOKENS", "32768")
    original = CharacterContextService.prepare_turn
    contexts = []
    rig.installed.state.on_request = lambda: contexts.append(get_request_context())
    state = SimpleNamespace(failure=None, preparations=[])

    async def prepare(service, turn, character_id):
        assert get_provider_context_budget().window_tokens == 32768
        await rig.consume()
        if state.failure == "prepare":
            raise ValueError("Synthetic preparation failed after a measured review")
        if state.failure == "budget":
            raise MemoryContextBudgetError("Synthetic complete evidence exceeds its budget")
        if state.failure == "generate":
            rig.installed.state.finish = "length"
        prepared = await original(service, turn, character_id)
        state.preparations.append((service, turn, prepared))
        # A concurrent configuration edit must not change this request's snapshot.
        rig.installed.stored["openaiCompatModel"] = "next-request-model"
        rig.installed.provider._refresh_db_config()
        return prepared

    monkeypatch.setattr(CharacterContextService, "prepare_turn", prepare)
    complete = AsyncMock(return_value=None)
    monkeypatch.setattr(generate, "_complete_character_turn", complete)
    snapshots = Mock(wraps=rig.installed.provider.snapshot)
    monkeypatch.setattr(rig.installed.provider, "snapshot", snapshots)

    async def submit(job, **kwargs):
        return await job()

    pool = HttpClientPool(max_connections=1)
    runtime = SimpleNamespace(check_rate_limits=AsyncMock(), priority_for=lambda *args: 1, submit=submit)
    app = FastAPI()
    app.state.runtime_container = RuntimeContainer(rig.database, lambda: False, runtime, http_client_pool=pool)

    async def run(message=QUESTION):
        return await narrative.canonical_generate(
            narrative.CanonicalMessage(character_id=ROLE, message=message, traceId="synthetic-canonical-usage"),
            SimpleNamespace(app=app),
            {"user_id": 1},
        )

    try:
        yield SimpleNamespace(
            rig=rig, state=state, contexts=contexts, complete=complete, pool=pool, app=app, run=run, snapshots=snapshots
        )
    finally:
        await pool.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "prepare", "budget", "generate"])
async def test_canonical_accounts_all_calls_and_preserves_scope_and_failure(canonical_usage, failure):
    case = canonical_usage
    case.state.failure = failure
    if failure:
        with pytest.raises(HTTPException) as error:
            await case.run()
        assert error.value.status_code == (500 if failure == "generate" else 503)
        assert isinstance(error.value.__cause__, MemoryContextBudgetError if failure == "budget" else ValueError)
        if failure == "budget":
            assert error.value.detail["stage"] == "memory_context"
            assert error.value.detail["reason"] == "selected_evidence_budget_exceeded"
        case.complete.assert_not_awaited()
    else:
        result = await case.run()
        assert result.modelInvoked and result.model == "fixture-model (openai_compat)"
        service, turn, prepared = case.state.preparations[0]
        assert (turn.platform, turn.adapter, turn.sender_id, turn.conversation_id, turn.conversation_type) == (
            "web",
            "narrative:" + ROLE,
            "1",
            "canonical:" + ROLE,
            "private",
        )
        assert turn.message == QUESTION and turn.history == () and turn.received_at is None
        case.complete.assert_awaited_once()
        assert case.complete.await_args.args[0] is prepared
        assert case.complete.await_args.kwargs["character_service"] is service
    calls = 1 if failure in {"prepare", "budget"} else 2
    assert len(case.rig.installed.calls) == calls and case.rig.writer.call_count == 1
    (row,) = case.rig.rows()
    assert (row["promptTokens"], row["completionTokens"], row["totalTokens"]) == (100 * calls, 20 * calls, 120 * calls)
    assert row["usageSource"] == "provider" and row["modelName"] == "fixture-model"
    assert row["errorType"] == ("MemoryContextBudgetError" if failure == "budget" else "ValueError" if failure else "")
    assert case.snapshots.call_count == 1
    assert all(c is case.contexts[0] and c.http_client_pool is case.pool for c in case.contexts)
    assert all(c["body"]["model"] == "fixture-model" for c in case.rig.installed.calls)
    messages = case.rig.database.execute_sql("SELECT * FROM messages")
    assert len(messages) == int(failure is None)
    if not failure:
        assert messages[0]["senderId"] == "1" and messages[0]["adapter"] == "narrative:" + ROLE
        assert messages[0]["branchId"] is None
        replay = await case.run()
        assert replay.reply == result.reply and not replay.modelInvoked
        assert len(case.rig.installed.calls) == calls and case.rig.writer.call_count == 1
        assert case.snapshots.call_count == 1
    assert get_request_context() is None and not has_pending_usage()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["pool", "snapshot"])
async def test_canonical_dependencies_fail_before_paid_work(canonical_usage, failure):
    case = canonical_usage
    error = ValueError("synthetic-private-snapshot-error")
    if failure == "pool":
        case.app.state.runtime_container.http_client_pool = None
    else:
        case.snapshots.side_effect = error
    with pytest.raises(HTTPException) as caught:
        await case.run()
    assert caught.value.status_code == 503
    if failure == "snapshot":
        assert caught.value.__cause__ is error
    assert "synthetic-private" not in str(caught.value.detail)
    assert not case.rig.installed.calls and not case.rig.rows()
    assert case.rig.writer.call_count == 0 and not case.state.preparations
    assert get_request_context() is None and not has_pending_usage()


@pytest.mark.asyncio
async def test_canonical_does_not_enable_pre_generation_memory_erasure(canonical_usage, monkeypatch):
    from character import memory_llm

    monkeypatch.setattr(memory_llm, "is_memory_erasure_request", lambda _message: True)
    erase = AsyncMock(side_effect=AssertionError("Canonical workspace must not execute an interactive erasure"))
    monkeypatch.setattr(CharacterContextService, "prepare_interactive_turn", erase)
    result = await canonical_usage.run("请删除我在此工作区的记忆；此测试只验证已有的操作执行边界。")
    assert result.modelInvoked
    erase.assert_not_awaited()
    canonical_usage.complete.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter", ["narrative", "narrative:" + ROLE])
async def test_generic_generation_cannot_impersonate_workspace_namespace(canonical_usage, adapter):
    case = canonical_usage
    request = MessageRequest(
        message=QUESTION, platform="web", adapter=adapter, senderId="other-owner", characterId=ROLE
    )
    with pytest.raises(HTTPException) as error:
        await generate._generate_reply_impl(request, {"user_id": 1}, message_db=case.rig.database)
    assert error.value.status_code == 422
    assert case.rig.installed.calls == [] and case.rig.rows() == []
    assert case.rig.database.execute_sql("SELECT id FROM messages") == []
