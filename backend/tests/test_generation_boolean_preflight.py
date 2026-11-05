"""Invalid retrieval configuration stops before character side effects or paid work."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException
from test_generation_option_preflight import message
from test_generation_option_preflight import rig as rig

pytestmark = pytest.mark.usefixtures("no_live_context_review")


@pytest.mark.parametrize("persist", [False, True], ids=["transient", "persisted"])
@pytest.mark.parametrize("setting", [None, "synthetic-private-invalid"], ids=["null", "invalid-text"])
async def test_invalid_knowledge_flag_stops_before_slot_character_preparation_or_generation(
    rig, monkeypatch, caplog, setting, persist
):
    from services import turn_completion

    original = {"temperature": 0.3, "maxTokens": 256, "topP": 1.0, "useKnowledgeBase": setting}
    owned = SimpleNamespace(config=dict(original), loras=[])
    reserve = Mock(side_effect=AssertionError("Invalid settings reserved a completion slot"))
    monkeypatch.setattr(turn_completion, "get_turn_completion_runtime", reserve)
    with pytest.raises(HTTPException) as error:
        await rig.api._generate_reply_impl(
            message(),
            persist_message=persist,
            record_invocation=False,
            character_service=SimpleNamespace(),
            message_db=owned,
        )
    assert error.value.status_code == 503 and isinstance(error.value.__cause__, ValueError)
    assert "useKnowledgeBase" in error.value.detail
    assert "synthetic-private" not in error.value.detail + str(error.value.__cause__) + caplog.text
    rig.prepare.assert_not_awaited()
    rig.generation.assert_not_awaited()
    rig.manager.get_current_provider().async_complete.assert_not_awaited()
    rig.manager.set_provider.assert_not_called()
    rig.save.assert_not_awaited()
    reserve.assert_not_called()
    assert owned.config == original


@pytest.mark.parametrize(
    "setting", [False, "false", True, "true"], ids=["disabled", "disabled-text", "enabled", "enabled-text"]
)
async def test_valid_knowledge_flag_reaches_generation_without_rewriting_stored_settings(rig, setting):
    original = {"temperature": 0.3, "maxTokens": 256, "topP": 1.0, "useKnowledgeBase": setting}
    owned = SimpleNamespace(config=dict(original), loras=[])
    response = await rig.api._generate_reply_impl(
        message(character=False),
        persist_message=False,
        record_invocation=False,
        message_db=owned,
    )
    assert response.reply == "根据完整当前输入回答。"
    rig.generation.assert_awaited_once()
    assert rig.generation.await_args.kwargs["runtime_config"] == original
    rig.prepare.assert_not_awaited()
    rig.save.assert_not_awaited()
    rig.manager.set_provider.assert_not_called()
    assert owned.config == original
