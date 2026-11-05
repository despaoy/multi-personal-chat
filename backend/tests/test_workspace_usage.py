"""Branch preparation and generation belong to one request's usage ledger."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from test_compiled_cloud_transport import installed as installed
from test_review_usage_recording import review_usage as review_usage

from api import generate
from db.schemas import MessageRequest
from inference.request_context import get_request_context
from inference.usage import has_pending_usage
from repositories.narrative import NarrativeRepository
from services import chat_answer
from services.character_context import CharacterContextService

pytestmark = pytest.mark.usefixtures("no_live_context_review")


@pytest.fixture
def branch_usage(review_usage, monkeypatch):
    rig = review_usage
    monkeypatch.setenv("NARRATIVE_BRANCHES_ENABLED", "true")
    repo = NarrativeRepository(rig.database)
    branch = repo.create("1", "tsukiyashiro_kisaki", "完整虚构票价", "本分支有三张票，每张两元，没有额外费用。")
    original = CharacterContextService.prepare_turn
    state = SimpleNamespace(failure=None)

    async def prepare(service, *args, **kwargs):
        await rig.consume()
        if state.failure == "prepare":
            raise HTTPException(503, "Synthetic preparation failure") from ValueError(
                "Synthetic measured review failure"
            )
        if state.failure == "generate":
            rig.installed.state.finish = "length"
        return await original(service, *args, **kwargs)

    monkeypatch.setattr(CharacterContextService, "prepare_turn", prepare)
    request = MessageRequest(
        message="三张票，每张两元且无额外费用，请计算总额。",
        loraName="default",
        branchId=branch["id"],
        traceId="synthetic-branch-usage",
        platform="web",
    )

    async def run(**options):
        return await generate._generate_reply_impl(request, {"user_id": 1}, message_db=rig.database, **options)

    return SimpleNamespace(rig=rig, repo=repo, branch=branch, state=state, run=run)


@pytest.mark.asyncio
@pytest.mark.parametrize("record", [True, False])
async def test_branch_success_accounts_for_preparation_and_generation_once(branch_usage, record):
    case = branch_usage
    result = await case.run(record_invocation=record, invocation_purpose="workspace")
    assert result.modelInvoked and result.branchRevision == 2
    assert len(case.rig.installed.calls) == 2
    rows = case.rig.rows()
    assert len(rows) == case.rig.writer.call_count == int(record)
    if record:
        (row,) = rows
        assert (row["promptTokens"], row["completionTokens"], row["totalTokens"]) == (200, 40, 240)
        assert row["purpose"] == "workspace" and row["usageSource"] == "provider" and row["errorType"] == ""
    assert len(case.repo.history("1", case.branch["id"])) == 1
    # A committed replay performs no additional provider call or accounting write.
    replay = await case.run(record_invocation=record, invocation_purpose="workspace")
    assert not replay.modelInvoked and replay.reply == result.reply
    assert len(case.rig.installed.calls) == 2 and case.rig.writer.call_count == int(record)
    assert get_request_context() is None and not has_pending_usage()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["prepare", "generate"])
async def test_branch_failure_keeps_all_measured_usage_in_one_record(branch_usage, failure):
    case = branch_usage
    case.state.failure = failure
    with pytest.raises(HTTPException) as error:
        await case.run(invocation_purpose="workspace")
    assert error.value.status_code == (503 if failure == "prepare" else 500)
    assert isinstance(error.value.__cause__, ValueError)
    calls = 1 if failure == "prepare" else 2
    assert len(case.rig.installed.calls) == calls and case.rig.writer.call_count == 1
    (row,) = case.rig.rows()
    assert row["totalTokens"] == 120 * calls and row["usageSource"] == "provider"
    assert row["errorType"] == "ValueError" and row["purpose"] == "workspace"
    assert case.repo.get("1", case.branch["id"])["revision"] == 1
    assert case.repo.history("1", case.branch["id"]) == []
    assert get_request_context() is None and not has_pending_usage()


@pytest.mark.asyncio
async def test_branch_usage_write_failure_is_not_retried_or_committed(branch_usage):
    case = branch_usage
    case.rig.writer.side_effect = OSError("synthetic-private-accounting-failure")
    with pytest.raises(HTTPException) as error:
        await case.run()
    assert error.value.detail["stage"] == "model_invocation_record"
    assert error.value.detail["invocationRecorded"] is None
    assert "synthetic-private" not in str(error.value.detail)
    assert len(case.rig.installed.calls) == 2 and case.rig.writer.call_count == 1 and case.rig.rows() == []
    assert case.repo.get("1", case.branch["id"])["revision"] == 1
    assert case.repo.history("1", case.branch["id"]) == []
    assert get_request_context() is None and not has_pending_usage()


@pytest.mark.asyncio
async def test_branch_preserves_explicit_rag_option(branch_usage, monkeypatch):
    answer = AsyncMock(wraps=chat_answer.generate_chat_answer)
    monkeypatch.setattr(chat_answer, "generate_chat_answer", answer)
    result = await branch_usage.run(enable_rag=False)
    assert result.modelInvoked and answer.await_count == 1
    assert answer.await_args.kwargs["enable_rag"] is False
