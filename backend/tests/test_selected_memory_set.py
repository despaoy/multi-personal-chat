"""Necessary, semantically selected memories must fit as one set.

SQLite sources/claims, scoped recall, semantic decision parsing, service,
compiler and API error conversion are real. Only semantic labels and the final
answer are synthetic. No network or paid model calls.
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from html import unescape
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from provider_fixtures import cloud_connection
from repositories.character_memory import DatabaseCharacterMemoryRepository
from repositories.messages import DatabaseMessageRepository
from services.character_context import CharacterContextService

from api import generate
from character import context_builder, profile_registry
from character.evidence_selector import ContextualEvidenceSelector
from character.memory_service import CharacterMemoryService
from character.models import MemoryItem, UserScope
from db.database import SQLiteDB
from db.schemas import MessageRequest
from inference import model_manager
from inference.context_budget import ReviewContextBudget
from infra.circuit_breaker import CircuitBreakerRegistry

pytestmark = pytest.mark.usefixtures("no_live_context_review")
ROLE = "fiction-selected-set"
OWNER = "fiction-selected-owner"
QUERY = "完整核对已保存的两份虚构排练记录：逐份保留时间、所有编号边界、末尾否定与例外。只读，不把引述当成本人经历。"


def full_source(index):
    return (
        f"虚构记录{index}：林澈说周四14:00至15:00排练，尚未报名。"
        + "".join(f"第{number:02}条仅离线核对编号{index}-{number:02}，未获书面确认不得执行；" for number in range(1, 63))
        + "末尾例外：仅当天不下雨且收到书面确认才参加；预约不豁免雨天限制，也不证明已经完成。"
    )


async def arrange_selected_memory(tmp_path, monkeypatch, *, cap=6000):
    from character.memory_mentions import mention_query

    # This is a complete evidence-selection task, not the separate provenance
    # display endpoint. No additional semantic mention reviewer is involved.
    assert mention_query(QUERY) is None
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / "fixture.json").write_text(json.dumps({
        "character_id": ROLE, "display_name": "虚构完整证据核对角色",
        "identity": "只读取当前可见且授权的记录", "values": ["完整保留引述、否定与例外"],
    }, ensure_ascii=False), encoding="utf-8")
    registry = profile_registry.CharacterProfileRegistry(profiles)
    registry.load_profiles()
    monkeypatch.setattr(profile_registry, "_default_registry", registry)
    database = SQLiteDB(tmp_path / "selected-set.sqlite")
    database.update_config(dict(useKnowledgeBase=False, temperature=.2, maxTokens=4096, topP=1.0))
    repository = DatabaseCharacterMemoryRepository(database)
    scope = UserScope("web", "selected-set", OWNER, OWNER, "private")
    stamp = datetime.now(timezone.utc) - timedelta(days=1)
    bodies = tuple(full_source(index) for index in (1, 2))
    identities = []
    for index, body in enumerate(bodies, 1):
        source_id = f"fiction-complete-source-{index}"
        assert await repository.capture_source(ROLE, scope, source_message_id=source_id, body=body, observed_at=stamp) == "recorded"
        stored = await repository.append_claim(
            ROLE, scope, MemoryItem("", "user_fact", f"虚构记录{index}全文引述：" + body, .9),
            memory_key=f"preference_rehearsal_{index}", confidence=.99, evidence=(body,),
            source_message_id=source_id, observed_at=stamp.isoformat(),
            metadata={"content_semantics": "quoted_source", "speaker_role": "user", "described_subject": "not_resolved"},
        )
        identities.append(str(stored["id"]))
    rig = SimpleNamespace(
        database=database, repository=repository, scope=scope, bodies=bodies,
        identities=tuple(identities), reviews=[], answers=[], prepared=[], registry=CircuitBreakerRegistry(),
    )

    async def all_necessary(messages):
        payload = json.loads(messages[-1]["content"])
        assert payload["query"] == QUERY and set(payload["required_ids"]) == set(rig.identities)
        assert len(payload["candidates"]) == 2
        for body in rig.bodies:
            candidate, = [candidate for candidate in payload["candidates"] if candidate["evidence"] == [body]]
            assert body in candidate["content"] and candidate["content_complete"] is True
            assert body.endswith("预约不豁免雨天限制，也不证明已经完成。")
        rig.reviews.append(payload)
        return json.dumps({"decisions": [{"id": identity, "label": "use"} for identity in payload["required_ids"]]})

    async def answer(messages, **params):
        wire = unescape("\n".join(message["content"] for message in messages))
        assert QUERY in wire and all(body in wire for body in rig.bodies)
        rig.answers.append(wire)
        return "\n".join(rig.bodies), 0.0

    for name, value in {
        "OPENAI_COMPAT_CONTEXT_WINDOW_TOKENS": "65536", "CAHM_SEMANTIC_MEMORY_ENABLED": "false",
        "CONTEXTUAL_DECISION_POLICY_ENABLED": "false", "DYNAMIC_CONTEXT_SEMANTIC_REVIEW_ENABLED": "false",
        "MEMORY_SOURCE_RECALL_ENABLED": "false", "CHARACTER_INDEPENDENT_TASKS_ENABLED": "false",
    }.items():
        monkeypatch.setenv(name, value)
    connection = cloud_connection("fiction-selected-model", completion=answer)
    manager = SimpleNamespace(_current_provider=model_manager.ModelProvider.OPENAI_COMPAT, get_current_provider=lambda: connection)
    monkeypatch.setattr(model_manager, "get_model_manager", lambda: manager)
    monkeypatch.setattr(generate, "db", database)
    monkeypatch.setattr(generate, "response_cache", None)
    monkeypatch.setattr(generate, "circuit_breaker_registry", rig.registry)
    monkeypatch.setattr(generate, "get_llm_semaphore", lambda: asyncio.Semaphore(4))
    monkeypatch.setattr(generate, "INPUT_VALIDATOR_AVAILABLE", False)
    monkeypatch.setattr(generate, "_get_system_prompt", lambda _: "仅核对两份完整虚构引文，保留所有否定和例外。")
    service = CharacterContextService(
        registry, repository, DatabaseMessageRepository(database),
        memory_service=CharacterMemoryService(repository, semantic_enabled=False),
        memory_selector=ContextualEvidenceSelector(all_necessary, context_budget=ReviewContextBudget(65536)),
        source_recall_enabled=False, reference_max_chars=cap,
    )
    actual_prepare = service.prepare_turn

    async def capture_prepared(*args, **kwargs):
        prepared = await actual_prepare(*args, **kwargs)
        rig.prepared.append(prepared)
        return prepared

    service.prepare_turn = capture_prepared
    rig.service = service
    return rig


async def generate_turn(rig):
    return await generate._generate_reply_impl(
        MessageRequest(
            message=QUERY, characterId=ROLE, loraName="default", platform=rig.scope.platform,
            adapter=rig.scope.adapter, senderId=OWNER, userId=OWNER, conversationId=OWNER,
            sessionId="fiction-selected-session", conversationType="private", sourceMessageId="fiction-read-only",
        ),
        character_service=rig.service, message_db=rig.database,
        enable_rag=False, persist_message=False, record_invocation=False,
    )


async def cleanup(rig):
    await rig.registry.clear()
    rig.database.close_connection()


@pytest.mark.asyncio
async def test_selected_complete_set_over_budget_errors_before_answer_model(tmp_path, monkeypatch):
    rig = await arrange_selected_memory(tmp_path, monkeypatch)
    try:
        with pytest.raises(HTTPException) as caught:
            await generate_turn(rig)
        assert caught.value.status_code == 503
        assert caught.value.detail["reason"] == "selected_evidence_budget_exceeded"
        assert isinstance(caught.value.__cause__, ValueError)
        assert "budget" in str(caught.value.__cause__).lower()
        assert len(rig.reviews) == 1 and not rig.prepared and not rig.answers
        assert rig.database.execute_sql("SELECT * FROM messages") == []
        assert rig.database.execute_sql("SELECT * FROM model_invocations") == []
        assert len(await rig.repository.list_memory_records(ROLE, rig.scope, limit=None)) == 2
    finally:
        await cleanup(rig)


@pytest.mark.asyncio
async def test_fitting_complete_set_reaches_answer_model_with_both_full_sources(tmp_path, monkeypatch):
    rig = await arrange_selected_memory(tmp_path, monkeypatch, cap=20000)
    try:
        result = await generate_turn(rig)
        assert result.reply == "\n".join(rig.bodies) and result.modelInvoked is True
        assert len(rig.reviews) == len(rig.prepared) == len(rig.answers) == 1
        prepared = rig.prepared[0]
        assert set(prepared.compiled.used_memory_ids) == set(rig.identities)
        assert prepared.memory_budget["budget_skipped"] == 0
        assert 6000 < prepared.memory_budget["used_chars"] < prepared.memory_budget["budget_chars"] == 20000
        assert prepared.memory_budget["used_chars"] == len(prepared.compiled.reference_context)
        assert all(body in prepared.compiled.reference_context and body in rig.answers[0] for body in rig.bodies)
    finally:
        await cleanup(rig)


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [TypeError, ValueError])
async def test_selected_packet_render_failure_remains_api_failure(tmp_path, monkeypatch, error_type):
    rig = await arrange_selected_memory(tmp_path, monkeypatch, cap=20000)
    failure = error_type("synthetic evidence serialization failure")

    def fail_render(_item):
        raise failure

    monkeypatch.setattr(context_builder, "_complete_memory_evidence_packet", fail_render)
    try:
        with pytest.raises(HTTPException) as caught:
            await generate_turn(rig)
        assert caught.value.status_code == 503 and caught.value.__cause__ is failure
        assert len(rig.reviews) == 1 and not rig.prepared and not rig.answers
        assert rig.database.execute_sql("SELECT * FROM messages") == []
        assert rig.database.execute_sql("SELECT * FROM model_invocations") == []
    finally:
        await cleanup(rig)


def test_complete_mode_still_filters_revoked_lifecycle_before_capacity():
    revoked = MemoryItem("revoked", "user_fact", "已撤销引文" * 5000, status="erased", evidence=("不可使用",))
    retained = MemoryItem("retained", "user_fact", "虚构记录：周四14:00排练，尚未报名；下雨取消。")
    text, identities = context_builder.compile_reference_context((revoked, retained), complete_evidence=True, max_chars=2000)
    assert identities == ("retained",) and "已撤销引文" not in text and retained.content in text
