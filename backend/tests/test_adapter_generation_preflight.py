"""Selected adapters are resolved once before ordinary or branch context work."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import HTTPException
from test_workspace_generation_config import ANSWER, QUESTION, ROLE
from test_workspace_generation_config import workspace as workspace

from db.schemas import GenerateResponse, MessageRequest
from services.character_context import CharacterContextService

ROW = {"id": "registered-adapter", "name": "controlled-adapter", "status": "active"}
pytestmark = pytest.mark.usefixtures("no_live_context_review")


@pytest.fixture
def adapter_workspace(workspace, monkeypatch):
    rig = workspace
    rig.state.manager.get_current_provider().name = "vllm"
    rig.state.manager._current_provider.value = "vllm"
    inventory = SimpleNamespace(reads=0)

    def rows(database):
        inventory.reads += 1
        return [dict(id="other-adapter", name="unselected-adapter", status="inactive"), dict(ROW)]

    monkeypatch.setattr(type(rig.db), "loras", property(rows))
    monkeypatch.setattr(rig.state.api, "get_lora_character_id", lambda name: ROLE)
    client = SimpleNamespace(list_loras=AsyncMock(return_value=[ROW["name"]]), generate=AsyncMock())
    get_client = AsyncMock(return_value=client)
    monkeypatch.setattr(rig.state.api, "get_vllm_client", get_client)
    return SimpleNamespace(rig=rig, inventory=inventory, client=client, get_client=get_client)


async def call(fixture, kind, trace, *, selection=ROW["name"]):
    rig = fixture.rig
    request = MessageRequest(
        message=QUESTION,
        loraName=selection,
        characterId=ROLE if kind == "ordinary" else None,
        branchId=rig.branch["id"] if kind == "branch" else None,
        traceId=trace,
        platform="web",
        senderId="1",
        userId="1",
        sessionId="synthetic-adapter-session",
        conversationType="private",
    )
    return await rig.entry(
        request,
        {"user_id": 1},
        message_db=rig.db,
        persist_message=kind == "branch",
        record_invocation=False,
    )


@pytest.mark.parametrize("kind", ["ordinary", "branch"])
@pytest.mark.parametrize("inventory", ["absent", "fault", "unavailable"])
async def test_unusable_adapter_stops_before_context_or_commit(adapter_workspace, monkeypatch, kind, inventory):
    fixture = adapter_workspace
    rig = fixture.rig
    failure = httpx.ConnectError("Synthetic adapter inventory is unavailable")
    if inventory == "absent":
        fixture.client.list_loras.return_value = []
    elif inventory == "fault":
        fixture.client.list_loras.side_effect = failure
    else:
        fixture.get_client.return_value = None
    prepare = AsyncMock(side_effect=AssertionError("Unusable adapter reached context preparation"))
    monkeypatch.setattr(CharacterContextService, "prepare_turn", prepare)
    with pytest.raises(HTTPException) as error:
        await call(fixture, kind, "unusable-adapter")
    assert error.value.status_code == (409 if inventory == "absent" else 503)
    if inventory == "fault":
        assert error.value.__cause__ is failure
    assert rig.db.config_reads == 1 and fixture.inventory.reads == 1
    prepare.assert_not_awaited()
    rig.state.generation.assert_not_awaited()
    rig.state.save.assert_not_awaited()
    assert rig.repo.get("1", rig.branch["id"])["revision"] == 1
    assert rig.repo.history("1", rig.branch["id"]) == []


@pytest.mark.parametrize("failure", ["unknown", "unsupported"])
async def test_branch_selection_errors_stop_before_context(adapter_workspace, monkeypatch, failure):
    fixture = adapter_workspace
    rig = fixture.rig
    if failure == "unsupported":
        rig.state.manager.get_current_provider().name = "openai_compat"
        rig.state.manager._current_provider.value = "openai_compat"
    prepare = AsyncMock(side_effect=AssertionError("Invalid selection reached branch preparation"))
    monkeypatch.setattr(CharacterContextService, "prepare_turn", prepare)
    with pytest.raises(HTTPException) as error:
        await call(
            fixture, "branch", "invalid-selection", selection="unknown-adapter" if failure == "unknown" else ROW["name"]
        )
    assert error.value.status_code == 422
    assert rig.db.config_reads == 1 and fixture.inventory.reads == 1
    prepare.assert_not_awaited()
    fixture.get_client.assert_not_awaited()
    rig.state.generation.assert_not_awaited()
    rig.state.save.assert_not_awaited()
    assert rig.repo.history("1", rig.branch["id"]) == []


@pytest.mark.parametrize("kind", ["ordinary", "branch"])
async def test_valid_adapter_inventory_precedes_real_context_and_runs_once(adapter_workspace, monkeypatch, kind):
    fixture = adapter_workspace
    rig = fixture.rig
    prepared = []

    async def prepare(service, *args, **kwargs):
        assert fixture.client.list_loras.await_count == 1
        assert rig.db.config_reads == 1 and fixture.inventory.reads == 1
        result = await rig.original_prepare(service, *args, **kwargs)
        prepared.append(result)
        return result

    monkeypatch.setattr(CharacterContextService, "prepare_turn", prepare)
    response = await call(fixture, kind, "valid-adapter")
    assert response.reply.startswith(ANSWER) and response.modelInvoked
    assert len(prepared) == 1
    fixture.client.list_loras.assert_awaited_once()
    rig.state.generation.assert_awaited_once()
    assert rig.state.generation.await_args.args[1] == ROW["name"]
    assert rig.db.config_reads == 1 and fixture.inventory.reads == 1
    assert len(rig.repo.history("1", rig.branch["id"])) == int(kind == "branch")


async def test_committed_branch_reply_skips_adapter_and_settings_reads(adapter_workspace, monkeypatch):
    fixture = adapter_workspace
    rig = fixture.rig
    rig.repo.commit_turn(
        rig.branch,
        MessageRequest(message=QUESTION, traceId="adapter-replay", loraName=ROW["name"]),
        GenerateResponse(reply=ANSWER, model="stored-adapter-reply", costTime=0),
        [],
        "acquaintance",
    )
    rig.db.settings["useKnowledgeBase"] = None
    fixture.get_client.side_effect = AssertionError("Committed replay must not contact adapter service")
    prepare = AsyncMock(side_effect=AssertionError("Committed replay must not prepare context"))
    monkeypatch.setattr(CharacterContextService, "prepare_turn", prepare)
    response = await call(fixture, "branch", "adapter-replay")
    assert response.reply == ANSWER and response.model == "stored-adapter-reply" and not response.modelInvoked
    assert rig.db.config_reads == 0 and fixture.inventory.reads == 0
    fixture.get_client.assert_not_awaited()
    prepare.assert_not_awaited()
    rig.state.generation.assert_not_awaited()


@pytest.mark.parametrize("kind", ["ordinary", "branch"])
async def test_character_mismatch_stops_before_adapter_service(adapter_workspace, monkeypatch, kind):
    fixture = adapter_workspace
    rig = fixture.rig
    monkeypatch.setattr(rig.state.api, "get_lora_character_id", lambda name: None)
    fixture.get_client.side_effect = AssertionError("Character mismatch must not contact adapter service")
    prepare = AsyncMock(side_effect=AssertionError("Character mismatch must not prepare context"))
    monkeypatch.setattr(CharacterContextService, "prepare_turn", prepare)
    with pytest.raises(HTTPException) as error:
        await call(fixture, kind, "mismatched-character")
    assert error.value.status_code == 422
    fixture.get_client.assert_not_awaited()
    prepare.assert_not_awaited()
    rig.state.generation.assert_not_awaited()
    assert rig.repo.history("1", rig.branch["id"]) == []


@pytest.mark.parametrize("kind", ["ordinary", "branch"])
async def test_explicit_base_model_prepares_without_adapter_inventory(adapter_workspace, monkeypatch, kind):
    fixture = adapter_workspace
    rig = fixture.rig
    prepared = []

    async def prepare(service, *args, **kwargs):
        assert fixture.get_client.await_count == 0
        assert fixture.inventory.reads == 0
        result = await rig.original_prepare(service, *args, **kwargs)
        prepared.append(result)
        return result

    monkeypatch.setattr(CharacterContextService, "prepare_turn", prepare)
    response = await call(fixture, kind, "explicit-base-model", selection="default")
    assert response.reply.startswith(ANSWER) and response.modelInvoked
    assert len(prepared) == 1 and fixture.inventory.reads == 0
    fixture.client.list_loras.assert_not_awaited()
    rig.state.generation.assert_awaited_once()
    assert rig.state.generation.await_args.args[1] is None
