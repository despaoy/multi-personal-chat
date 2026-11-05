"""Workspace configuration is prepared once after replay lookup, before context."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, HTTPException
from test_lora_provider_selection import install_api

from api import narrative
from app.runtime import RuntimeContainer
from db.database import SQLiteDB
from db.schemas import GenerateResponse, MessageRequest
from infra.resource_pool import HttpClientPool
from repositories.narrative import NarrativeRepository
from services.character_context import CharacterContextService
from services.chat_answer import ChatAnswer

ROLE = "tsukiyashiro_kisaki"
QUESTION = "本题完整虚构数据：三张票，每张两元，请给出总额。"
ANSWER = "三张票，每张两元，总额为六元。"
SETTINGS = {"temperature": 0.3, "maxTokens": 256, "topP": 1.0, "useKnowledgeBase": False}


class WorkspaceDatabase:
    def __init__(self, database):
        self.database = database
        self.settings = dict(SETTINGS)
        self.config_reads = 0

    @property
    def config(self):
        self.config_reads += 1
        return dict(self.settings)

    @property
    def loras(self):
        raise AssertionError("An unselected adapter directory must not be read")

    def __getattr__(self, name):
        return getattr(self.database, name)


@pytest.fixture
async def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("NARRATIVE_BRANCHES_ENABLED", "true")
    monkeypatch.setenv("MEMORY_LLM_ENABLED", "false")
    state = install_api(monkeypatch, "openai_compat")
    state.manager.get_current_provider().base_url = "https://api.deepseek.com"
    state.generation.return_value = ChatAnswer(reply=ANSWER, used_rag=False, model_invoked=True, metadata={})
    database = SQLiteDB(tmp_path / "workspace.sqlite")
    scoped = WorkspaceDatabase(database)
    repository = NarrativeRepository(scoped)
    branch = repository.create("1", ROLE, "虚构票价", "仅在本分支假设三张票且每张两元，原作事实保持原有范围。")
    entry = state.api._generate_reply_impl

    async def controlled_generation(*args, **kwargs):
        # Persistence and inference are independently controlled here. Existing
        # full Web tests retain their real archive, scope and CAS assertions.
        kwargs.update(persist_message=False, record_invocation=False)
        return await entry(*args, **kwargs)

    monkeypatch.setattr(state.api, "_generate_reply_impl", controlled_generation)

    async def submit(job, **kwargs):
        return await job()

    pool = HttpClientPool(max_connections=1)
    app = FastAPI()
    runtime = SimpleNamespace(check_rate_limits=AsyncMock(), priority_for=lambda *args: 1, submit=submit)
    app.state.runtime_container = RuntimeContainer(scoped, lambda: False, runtime, http_client_pool=pool)

    async def call(kind, trace):
        if kind == "branch":
            return await entry(
                MessageRequest(message=QUESTION, branchId=branch["id"], traceId=trace, platform="web"),
                {"user_id": 1},
                message_db=scoped,
                record_invocation=False,
            )
        return await narrative.canonical_generate(
            narrative.CanonicalMessage(character_id=ROLE, message=QUESTION, traceId=trace),
            SimpleNamespace(app=app),
            {"user_id": 1},
        )

    try:
        yield SimpleNamespace(
            state=state,
            db=scoped,
            repo=repository,
            branch=branch,
            entry=entry,
            call=call,
            original_prepare=CharacterContextService.prepare_turn,
        )
    finally:
        state.manager.get_current_provider().async_complete.assert_not_awaited()
        await pool.close()
        database.close_connection()


pytestmark = pytest.mark.usefixtures("no_live_context_review")


@pytest.mark.parametrize("kind", ["branch", "canonical"])
@pytest.mark.parametrize("invalid", [{"useKnowledgeBase": None}, {"maxTokens": 0}], ids=["boolean", "numeric"])
async def test_workspace_invalid_settings_stop_before_context_or_commit(workspace, monkeypatch, kind, invalid):
    rig = workspace
    rig.db.settings.update(invalid)
    original = dict(rig.db.settings)
    prepare = AsyncMock(side_effect=AssertionError("Invalid configuration reached workspace context preparation"))
    monkeypatch.setattr(CharacterContextService, "prepare_turn", prepare)
    with pytest.raises(HTTPException) as error:
        await rig.call(kind, "invalid-setting")
    assert error.value.status_code == 503 and isinstance(error.value.__cause__, ValueError)
    assert rig.db.config_reads == 1 and rig.db.settings == original
    prepare.assert_not_awaited()
    rig.state.generation.assert_not_awaited()
    rig.state.save.assert_not_awaited()
    assert rig.repo.get("1", rig.branch["id"])["revision"] == 1
    assert rig.repo.history("1", rig.branch["id"]) == []


@pytest.mark.parametrize("kind", ["branch", "canonical"])
async def test_workspace_reuses_request_settings_and_next_request_reads_fresh(workspace, monkeypatch, kind):
    rig = workspace
    prepared = []

    async def prepare(service, *args, **kwargs):
        assert rig.db.config_reads == 1
        result = await rig.original_prepare(service, *args, **kwargs)
        prepared.append(result)
        # A settings edit after preparation starts applies to the next request.
        rig.db.settings["useKnowledgeBase"] = None
        return result

    monkeypatch.setattr(CharacterContextService, "prepare_turn", prepare)
    response = await rig.call(kind, "snapshot-settings")
    assert response.reply.startswith(ANSWER) and response.modelInvoked
    assert len(prepared) == 1 and rig.db.config_reads == 1
    rig.state.generation.assert_awaited_once()
    assert rig.state.generation.await_args.kwargs["runtime_config"] == SETTINGS
    with pytest.raises(HTTPException) as error:
        await rig.call(kind, "fresh-settings")
    assert error.value.status_code == 503 and "useKnowledgeBase" in error.value.detail
    assert rig.db.config_reads == 2 and len(prepared) == 1
    rig.state.generation.assert_awaited_once()
    assert len(rig.repo.history("1", rig.branch["id"])) == (1 if kind == "branch" else 0)


@pytest.mark.parametrize("kind", ["branch", "canonical"])
async def test_committed_workspace_reply_replays_without_generation_settings_or_context(workspace, monkeypatch, kind):
    rig = workspace
    trace = "committed-reply"
    if kind == "branch":
        rig.repo.commit_turn(
            rig.branch,
            MessageRequest(message=QUESTION, traceId=trace),
            GenerateResponse(reply=ANSWER, model="stored-fixture", costTime=0),
            [],
            "acquaintance",
        )
    else:
        rig.db.add_message(
            dict(
                platform="web",
                adapter="narrative:" + ROLE,
                senderId="1",
                userId="1",
                sessionId="canonical:" + ROLE,
                conversationId="canonical:" + ROLE,
                conversationType="private",
                characterId=ROLE,
                traceId=trace,
                message=QUESTION,
                reply=ANSWER,
                modelName="stored-fixture",
            )
        )
    rig.db.settings["useKnowledgeBase"] = None
    prepare = AsyncMock(side_effect=AssertionError("Committed replay must not prepare context"))
    monkeypatch.setattr(CharacterContextService, "prepare_turn", prepare)
    response = await rig.call(kind, trace)
    assert response.reply == ANSWER and response.model == "stored-fixture" and not response.modelInvoked
    assert rig.db.config_reads == 0
    prepare.assert_not_awaited()
    rig.state.generation.assert_not_awaited()
    rig.state.save.assert_not_awaited()


@pytest.mark.parametrize("kind", ["branch", "canonical"])
async def test_workspace_mock_provider_stops_before_context(workspace, monkeypatch, kind):
    rig = workspace
    rig.state.manager.get_current_provider().name = "mock"
    rig.state.manager._current_provider.value = "mock"
    prepare = AsyncMock(side_effect=AssertionError("Mock provider reached context preparation"))
    monkeypatch.setattr(CharacterContextService, "prepare_turn", prepare)
    with pytest.raises(HTTPException) as error:
        await rig.call(kind, "mock-provider")
    assert error.value.status_code == 503 and "mock" in error.value.detail
    assert rig.db.config_reads == 0
    prepare.assert_not_awaited()
    rig.state.generation.assert_not_awaited()
    assert rig.repo.history("1", rig.branch["id"]) == []
