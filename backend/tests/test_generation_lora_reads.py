"""Generation reads only adapter data required by the actual selection."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException

from db.schemas import MessageRequest
from test_lora_provider_selection import install_api

pytestmark = pytest.mark.usefixtures("no_live_context_review")
ROLE = "tsukiyashiro_kisaki"
ROW = {"id": "registered-adapter", "name": "controlled-adapter", "status": "active"}


class Prepared(SimpleNamespace):
    def __bool__(self):
        return False


class OwnedDatabase:
    def __init__(self, *, config_error=None, lora_error=None, rows=()):
        self.config_error = config_error
        self.lora_error = lora_error
        self.rows = list(rows)
        self.config_reads = self.lora_reads = 0

    @property
    def config(self):
        self.config_reads += 1
        if self.config_error:
            raise self.config_error
        return {"topP": 1.0, "useKnowledgeBase": False}

    @property
    def loras(self):
        self.lora_reads += 1
        if self.lora_error:
            raise self.lora_error
        return self.rows


def request(*, selection="default", character=False):
    return MessageRequest(
        message="三张票，每张两元，请计算总额并说明计算过程。",
        loraName=selection,
        characterId=ROLE if character else None,
        platform="web",
        adapter="web-character",
        senderId="synthetic-reader",
        userId="synthetic-reader",
        sessionId="synthetic-adapter-session",
        sourceMessageId="synthetic-adapter-question",
    )


def prepare(state, monkeypatch):
    turn = Prepared(character_id=ROLE)
    state.prepare = AsyncMock(return_value=turn)
    monkeypatch.setattr(state.api, "_prepare_character_turn", state.prepare)
    return turn


def assert_no_generation(state):
    state.prepare.assert_not_awaited()
    state.generation.assert_not_awaited()
    state.manager.get_current_provider().async_complete.assert_not_awaited()
    state.manager.set_provider.assert_not_called()
    state.manager.set_lora_adapter.assert_not_called()
    state.save.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "selection,character,override",
    [("default", False, False), ("", True, False), ("default", True, False), ("", False, True), ("", True, True)],
)
async def test_base_character_and_prepared_turns_do_not_read_loras(monkeypatch, selection, character, override):
    state = install_api(monkeypatch, "openai_compat")
    turn = prepare(state, monkeypatch)
    owned = OwnedDatabase(lora_error=OSError("Unused adapter storage must not be read"))
    monkeypatch.setattr(state.api, "db", OwnedDatabase(config_error=AssertionError("Wrong database")))
    lookup = Mock(side_effect=AssertionError("Base generation must not look up an adapter directory"))
    monkeypatch.setattr(state.api, "get_lora_path_by_id", lookup)
    result = await state.api._generate_reply_impl(
        request(selection=selection, character=character),
        message_db=owned,
        persist_message=False,
        record_invocation=False,
        character_service=SimpleNamespace(),
        prepared_override=turn if override else None,
    )
    assert result.reply == "根据完整当前输入回答。" and result.modelInvoked
    assert owned.config_reads == 1 and owned.lora_reads == 0
    state.generation.assert_awaited_once()
    assert state.generation.await_args.args[1] == "default"
    assert state.generation.await_args.kwargs["message_db"] is owned
    expected = turn if character or override else None
    assert state.generation.await_args.kwargs["prepared_character_turn"] is expected
    assert state.prepare.await_count == int(character and not override)
    state.manager.set_lora_adapter.assert_not_called()
    state.manager.set_provider.assert_not_called()
    lookup.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("selection", ["default", ""])
async def test_required_owned_config_failure_still_stops_generation(monkeypatch, caplog, selection):
    state = install_api(monkeypatch, "openai_compat")
    prepare(state, monkeypatch)
    failure = OSError("synthetic-private-config-error")
    owned = OwnedDatabase(config_error=failure, rows=[ROW])
    with pytest.raises(HTTPException) as caught:
        await state.api._generate_reply_impl(
            request(selection=selection), message_db=owned, persist_message=False, record_invocation=False
        )
    assert caught.value.status_code == 503 and caught.value.__cause__ is failure
    assert "synthetic-private" not in caught.value.detail + caplog.text
    assert_no_generation(state)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "selection,character,override",
    [
        ("", False, False),
        ("controlled-adapter", False, False),
        ("controlled-adapter", True, False),
        ("controlled-adapter", False, True),
    ],
)
async def test_required_lora_read_failure_never_becomes_base_success(
    monkeypatch, caplog, selection, character, override
):
    state = install_api(monkeypatch, "openai_compat")
    turn = prepare(state, monkeypatch)
    failure = OSError("synthetic-private-adapter-error")
    owned = OwnedDatabase(lora_error=failure)
    with pytest.raises(HTTPException) as caught:
        await state.api._generate_reply_impl(
            request(selection=selection, character=character),
            message_db=owned,
            prepared_override=turn if override else None,
            persist_message=False,
            record_invocation=False,
        )
    assert caught.value.status_code == 503 and caught.value.__cause__ is failure
    assert owned.lora_reads == 1
    assert "synthetic-private" not in caught.value.detail + caplog.text
    assert_no_generation(state)


@pytest.mark.asyncio
@pytest.mark.parametrize("character,override", [(False, False), (True, False), (False, True)])
async def test_unknown_explicit_adapter_does_not_select_active_one(monkeypatch, character, override):
    state = install_api(monkeypatch, "transformers_peft")
    turn = prepare(state, monkeypatch)
    owned = OwnedDatabase(rows=[ROW])
    with pytest.raises(HTTPException) as caught:
        await state.api._generate_reply_impl(
            request(selection="unknown", character=character),
            message_db=owned,
            prepared_override=turn if override else None,
            persist_message=False,
            record_invocation=False,
        )
    assert caught.value.status_code == 422 and caught.value.detail == "Specified LoRA does not exist"
    assert owned.lora_reads == 1
    assert_no_generation(state)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "selection,character,override",
    [
        ("", False, False),
        ("controlled-adapter", False, False),
        ("controlled-adapter", True, False),
        ("controlled-adapter", False, True),
    ],
)
async def test_requested_or_active_adapter_is_used_once(monkeypatch, selection, character, override):
    state = install_api(monkeypatch, "transformers_peft")
    turn = prepare(state, monkeypatch)
    owned = OwnedDatabase(rows=[ROW])
    monkeypatch.setattr(state.api, "get_lora_character_id", lambda _: ROLE if character or override else None)
    lookup = Mock(return_value="/synthetic/controlled-adapter")
    monkeypatch.setattr(state.api, "get_lora_path_by_id", lookup)
    result = await state.api._generate_reply_impl(
        request(selection=selection, character=character),
        message_db=owned,
        prepared_override=turn if override else None,
        character_service=SimpleNamespace(),
        persist_message=False,
        record_invocation=False,
    )
    assert result.reply == "根据完整当前输入回答。"
    assert owned.config_reads == owned.lora_reads == 1
    lookup.assert_called_once_with("registered-adapter", loras=[ROW])
    state.manager.set_lora_adapter.assert_called_once_with("/synthetic/controlled-adapter")
    state.generation.assert_awaited_once()
    assert state.generation.await_args.args[1] == "controlled-adapter"
    assert state.prepare.await_count == int(character and not override)
    state.manager.set_provider.assert_not_called()
