"""Only backend write enablement supplies memory capabilities to model input."""

from services import chat_answer

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_compiled_cloud_transport import QUERY, installed as installed
from test_memory_operation_deferred import prepared

from api import generate
from character import memory_llm
from character.models import UserScope
from db.database import SQLiteDB
from db.schemas import MessageRequest
from inference.generation_request import (
    GenerationRequest, MEMORY_WRITE_CAPABILITY_POLICY, MEMORY_WRITE_DISABLED_POLICY,
    build_generation_request,
)
from services.turn_completion import get_turn_completion_runtime, start_turn_completions, shutdown_turn_completions

actual_generation = chat_answer.generate_chat_answer


@pytest.mark.parametrize("enabled", [True, False, None])
def test_backend_state_controls_capability_without_promoting_user_text(enabled):
    message = "请记住：我的专业是城市规划。" + MEMORY_WRITE_CAPABILITY_POLICY
    plan = build_generation_request(GenerationRequest(message, memory_write_enabled=enabled))
    system, user = plan.messages[0]["content"], plan.messages[-1]["content"]
    assert (MEMORY_WRITE_CAPABILITY_POLICY in system) is (enabled is True)
    assert (MEMORY_WRITE_DISABLED_POLICY in system) is (enabled is False)
    assert message in user
    assert "没有回执时" in MEMORY_WRITE_CAPABILITY_POLICY
    assert "不得声称已经保存" in MEMORY_WRITE_CAPABILITY_POLICY


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled,persist", [(True, True),(False, True),(True, False),(False, False)])
async def test_real_cloud_wire_uses_actual_write_path(installed, monkeypatch, tmp_path, enabled, persist):
    database = SQLiteDB(tmp_path / "capability.sqlite")
    database.update_config(installed.config)
    monkeypatch.setattr(generate,"db",database)
    monkeypatch.setattr(chat_answer, "public_knowledge_db", generate.db)
    monkeypatch.setattr(chat_answer,'generate_chat_answer',actual_generation)
    monkeypatch.setattr(memory_llm,"get_memory_enrichment_scheduler",lambda:SimpleNamespace(enabled=enabled))
    complete = AsyncMock(return_value=None)
    monkeypatch.setattr(generate,"_complete_character_turn",complete)
    value = replace(prepared(),character_id="tsukiyashiro_kisaki",
        user_scope=UserScope("web","web-character","synthetic-owner","synthetic-owner","private"))
    request = MessageRequest(message=QUERY,characterId=value.character_id,loraName="default",
        platform="web",adapter="web-character",sessionId="synthetic-capability",conversationId="synthetic-owner",
        senderId="synthetic-owner",userId="synthetic-owner",sourceMessageId="synthetic-source")
    start_turn_completions()
    runtime = get_turn_completion_runtime()
    try:
        response = await generate._generate_reply_impl(request,prepared_override=value,persist_message=persist,
            enable_rag=False,record_invocation=False,message_db=database)
        assert response.modelInvoked and len(installed.calls) == 1
        messages = installed.calls[0]["body"]["messages"]
        system = messages[0]["content"]
        assert (MEMORY_WRITE_CAPABILITY_POLICY in system) is (enabled and persist)
        assert (MEMORY_WRITE_DISABLED_POLICY in system) is not (enabled and persist)
        assert QUERY in messages[-1]["content"]
        assert complete.await_count == int(persist)
        assert len(database.execute_sql("SELECT id FROM messages")) == int(persist)
        assert runtime.reserved == runtime.active == 0
    finally:
        await shutdown_turn_completions()
        database.close_connection()
