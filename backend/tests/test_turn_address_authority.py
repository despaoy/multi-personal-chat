"""Semantic address evidence, default settings and completion authority."""

import asyncio
import os
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from character.models import build_user_scope
from character.models import RelationshipState
from db.database import SQLiteDB
from repositories.character_memory import DatabaseCharacterMemoryRepository
from services.character_context import CharacterContextService, TurnInput


@pytest.fixture(scope="module", params=["sqlite", "pg"])
def database(request, tmp_path_factory):
    if request.param == "sqlite":
        db = SQLiteDB(tmp_path_factory.mktemp("turn-address") / "records.sqlite")
    else:
        url = os.environ["STAGE35_ADDRESS_PG_URL"]
        assert "/stage3_stage35_address_guards?" in url and "port=25433" in url
        with pytest.MonkeyPatch.context() as env:
            env.setenv("DATABASE_URL", url)
            from db.pg_database import PgDatabase, SyncPgAdapter
        db = SyncPgAdapter(PgDatabase(url))
    yield db
    (db.close if request.param == "pg" else db.close_connection)()


def source(database):
    user = "address-" + uuid.uuid4().hex
    fields = ("tsukiyashiro_kisaki", "qq", "unit-address", user, "private", user)
    source_id = "source-" + uuid.uuid4().hex
    stamp = datetime.now(timezone.utc)
    database.upsert_character_relationship(*fields, "familiar", "知禾", "手动关系摘要", interaction_count=11)
    assert (
        database.capture_memory_source(
            *fields,
            source_message_id=source_id,
            body="请叫我柚汀。这个称呼只用于对话，不是更改我的姓名。",
            observed_at=stamp,
        )
        == "recorded"
    )
    return fields, source_id, stamp


@pytest.mark.asyncio
async def test_complete_turn_does_not_write_rejected_source_address(database, monkeypatch):
    from character import memory_llm

    fields, sid, stamp = source(database)
    database.clear_character_memories(*fields)
    repo = DatabaseCharacterMemoryRepository(database)
    scope = build_user_scope(fields[1], fields[2], fields[3], fields[5], fields[4])
    service = CharacterContextService.__new__(CharacterContextService)
    service._memory_repo = repo
    scheduler = SimpleNamespace(
        enabled=True,
        status=SimpleNamespace(last_outcome="write_gate"),
        schedule=lambda **_: pytest.fail("Stale source must not enqueue semantic work"),
    )
    monkeypatch.setattr(memory_llm, "get_memory_enrichment_scheduler", lambda: scheduler)
    prepared = SimpleNamespace(
        character_id=fields[0],
        user_scope=scope,
        received_at=stamp,
        memory_operation_receipt=None,
        compiled=SimpleNamespace(used_memory_ids=()),
        history=(),
        relationship=RelationshipState(stage="familiar", preferred_address="知禾"),
    )
    turn = TurnInput(
        message="请叫我柚汀。这个称呼只用于对话，不是更改我的姓名。",
        platform=scope.platform,
        adapter=scope.adapter,
        sender_id=scope.sender_id,
        conversation_id=scope.conversation_id,
        conversation_type="private",
        received_at=stamp,
    )
    result = await service.complete_turn(prepared, turn, source_message_id=sid)
    assert result.source_capture == "stale" and not result.memory_enrichment_scheduled
    assert database.get_character_relationship(*fields)["preferred_address"] == "知禾"


@pytest.mark.asyncio
async def test_source_rejection_overrides_earlier_semantic_success(database, monkeypatch):
    import json

    from db.schemas import MessageRequest
    from services import delivery_memory as dm
    from services.character_context import _TurnOutcome
    from services.turn_completion import TurnCompletionRuntime

    fields, sid, stamp = source(database)
    scope = build_user_scope(fields[1], fields[2], fields[3], fields[5], fields[4])
    prepared = SimpleNamespace(
        character_id=fields[0],
        user_scope=scope,
        received_at=stamp,
        memory_operation_receipt=None,
        compiled=SimpleNamespace(used_memory_ids=()),
        history=(),
        relationship=RelationshipState(),
    )
    msg = MessageRequest(
        message="请叫我柚汀。",
        platform=scope.platform,
        adapter=scope.adapter,
        senderId=scope.sender_id,
        userId=scope.sender_id,
        conversationId=scope.conversation_id,
        sessionId=scope.conversation_id,
        sourceMessageId=sid,
    )
    stored = dict(
        reply=dict(replyText="原始回复"),
        context=dict(
            request=msg.model_dump(),
            message_saved=True,
            character_id=fields[0],
            completion_snapshot=dm.freeze_completion(prepared),
        ),
    )
    key, owner = uuid.uuid4().hex * 2, uuid.uuid4().hex
    assert database.integration_receipt("claim", key=key, owner=owner, now=0, expires_at=0)
    assert database.integration_receipt("finish", key=key, owner=owner, status="generated", response=json.dumps(stored))
    initial = database.integration_receipt("get", key=key)
    pending, _ = dm.delivery_envelope(stored)
    assert database.integration_receipt(
        "delivery_memory", key=key, owner=owner, expected=initial["response"], response=pending, expires_at=0
    )
    runtime = TurnCompletionRuntime(capacity=1)
    monkeypatch.setattr(dm, "get_turn_completion_runtime", lambda: runtime)
    monkeypatch.setattr(dm, "POLL_SECONDS", 0.01)

    async def complete(*_, memory_receipt, **kwargs):
        memory_receipt.set_result(dict(status="saved", source_capture="recorded", persisted=1))
        return _TurnOutcome(source_capture="stale", memory_enrichment_scheduled=True)

    worker = dm.ensure_delivery_memory_worker(database, SimpleNamespace(complete_turn=complete))
    try:
        await worker.wait_idle(timeout=3)
        row = database.integration_receipt("get", key=key)
        assert row["status"] == "delivered" and json.loads(row["response"])["memory_completion_state"] == "blocked"
    finally:
        await dm.shutdown_delivery_memory()
        await runtime.shutdown(timeout=1)



@pytest.mark.asyncio
@pytest.mark.parametrize("operation_status", ["erased", "failed"])
async def test_explicit_completion_never_reinterprets_erasure_as_address(database, monkeypatch, operation_status):
    import json
    from api import generate
    from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig, is_memory_erasure_request
    from db.schemas import MessageRequest
    from repositories.character_memory import relationship_from_record

    fields, old_source, _ = source(database)
    role, platform, adapter, sender, kind, conversation = fields
    scope = build_user_scope(platform, adapter, sender, conversation, kind)
    repository = DatabaseCharacterMemoryRepository(database)
    message = "请从长期记忆中删除我让你叫我柚汀的那条原话记录。"
    assert is_memory_erasure_request(message)
    received = datetime.now(timezone.utc)
    current_source = "operation-" + uuid.uuid4().hex

    class Completion:
        calls = 0

        async def complete(self, messages):
            self.calls += 1
            payload = json.loads(messages[1]["content"])
            assert old_source in {row["source_id"] for row in payload["source_erasure_candidates"]}
            return (
                json.dumps({"memories": [], "erase_source_ids": [old_source]})
                if operation_status == "erased"
                else "{invalid"
            )

        async def close(self):
            pass

    completion = Completion()
    scheduler = MemoryEnrichmentScheduler(config=MemoryLlmConfig(True, "unit-only", "unit-only"), completion=completion)
    try:
        receipt = await scheduler.schedule_and_wait(
            repository=repository,
            character_id=role,
            user_scope=scope,
            message=message,
            rule_hints=(),
            source_message_id=current_source,
            observed_at=received,
        )
        assert receipt["status"] == operation_status
        relationship = relationship_from_record(await repository.get_relationship_record(role, scope))
        before = database.get_character_relationship(*fields)
        prepared = SimpleNamespace(
            character_id=role,
            user_scope=scope,
            received_at=received,
            memory_operation_receipt=receipt,
            relationship=relationship,
            history=(),
            compiled=SimpleNamespace(used_memory_ids=()),
        )
        service = CharacterContextService(object(), repository, object(), source_recall_enabled=False)
        monkeypatch.setattr(
            "character.memory_llm.get_memory_enrichment_scheduler",
            lambda: pytest.fail("An executed operation must not initialize another writer"),
        )
        outcomes = []
        complete_turn = service.complete_turn

        async def complete(*args, **kwargs):
            outcome = await complete_turn(*args, **kwargs)
            outcomes.append(outcome)
            return outcome

        monkeypatch.setattr(service, "complete_turn", complete)
        request = MessageRequest(
            message=message,
            characterId=role,
            platform=platform,
            adapter=adapter,
            senderId=sender,
            conversationId=conversation,
            sessionId=conversation,
            conversationType=kind,
            sourceMessageId=current_source,
        )
        warning = await generate._complete_character_turn(prepared, request, character_service=service)
        after = database.get_character_relationship(*fields)
        assert after["preferred_address"] == before["preferred_address"] == "知禾"
        assert after["relationship_stage"] == before["relationship_stage"] == "familiar"
        assert after["summary"] == before["summary"] and after["interaction_count"] == before["interaction_count"] + 1
        assert len(outcomes) == 1 and outcomes[0].memory_enrichment_status == receipt["status"]
        assert outcomes[0].source_capture == receipt.get("source_capture", "")
        assert (
            outcomes[0].stage == relationship.stage and outcomes[0].preferred_address == relationship.preferred_address
        )
        assert not outcomes[0].failed_writes and not outcomes[0].memory_enrichment_scheduled
        assert (warning is None) if operation_status == "erased" else ("失败" in warning and "称呼" not in warning)
        rows = await repository.list_sources(role, scope)
        assert [row["source_message_id"] for row in rows] == ([] if operation_status == "erased" else [old_source])
        assert completion.calls == 1
    finally:
        await scheduler.shutdown()



@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["ordinary", "conditional", "excluded"])
async def test_semantic_address_evidence_never_overwrites_default_setting(database, monkeypatch, case):
    import json
    from character import memory_llm
    from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig
    from character.memory_service import CharacterMemoryService
    from character.profile_registry import get_default_profile_registry
    from repositories.character_memory import relationship_from_record
    from repositories.messages import DatabaseMessageRepository

    cases = {
        "ordinary": (
            "请叫我柚汀。这个称呼只用于对话，不是更改我的姓名。",
            [("alias", "柚汀", "请叫我柚汀。这个称呼只用于对话，不是更改我的姓名。", {})],
        ),
        "conditional": (
            "周末请叫我杉岚，工作日请叫我知禾。",
            [
                ("alias", "杉岚", "周末请叫我杉岚", {"time": "周末"}),
                ("alias", "知禾", "工作日请叫我知禾。", {"time": "工作日"}),
            ],
        ),
        "excluded": (
            "杉岚不是我的别名，是朋友的别名，请不要叫我杉岚。",
            [("alias_exclusion", "杉岚", "杉岚不是我的别名，是朋友的别名，请不要叫我杉岚。", {})],
        ),
    }
    message, expected = cases[case]
    fields, _, _ = source(database)
    role, platform, adapter, sender, kind, conversation = fields
    scope = build_user_scope(platform, adapter, sender, conversation, kind)
    repo = DatabaseCharacterMemoryRepository(database)
    before = database.get_character_relationship(*fields)
    response = json.dumps(
        {
            "memories": [
                dict(
                    kind=k,
                    value=v,
                    evidence=e,
                    qualifiers=q,
                    content="",
                    operation="ADD",
                    confidence=0.99,
                    attributed_to="user",
                    scope_level="conversation",
                )
                for k, v, e, q in expected
            ]
        },
        ensure_ascii=False,
    )

    class Completion:
        calls = 0

        async def complete(self, messages):
            self.calls += 1
            assert json.loads(messages[1]["content"])["current_user_message"] == message
            return response

        async def close(self):
            pass

    completion = Completion()
    scheduler = MemoryEnrichmentScheduler(
        config=MemoryLlmConfig(True, "unit-only", "unit-only"),
        completion=completion,
        embedding_provider=SimpleNamespace(embed_texts=lambda texts: [[1.0, 0.0] for _ in texts]),
    )
    monkeypatch.setattr(memory_llm, "get_memory_enrichment_scheduler", lambda: scheduler)
    service = CharacterContextService(
        get_default_profile_registry(), repo, DatabaseMessageRepository(database), source_recall_enabled=False
    )
    turn = TurnInput(message, platform, adapter, sender, conversation, kind)
    prepared = await service.prepare_turn(turn, role)
    receipt = asyncio.get_running_loop().create_future()
    sid = "semantic-address-" + uuid.uuid4().hex
    try:
        outcome = await service.complete_turn(prepared, turn, source_message_id=sid, memory_receipt=receipt)
        result = await asyncio.wait_for(asyncio.shield(receipt), 5)
        assert result["status"] == "saved" and result["persisted"] == len(expected)
        assert completion.calls == 1 and outcome.memory_enrichment_scheduled
        rows = await repo.list_memory_records(role, scope)
        assert len(rows) == len(expected)
        by_evidence = {tuple(row["evidence"]): row for row in rows}
        assert len(by_evidence) == len(expected)
        for kind, value, evidence, qualifiers in expected:
            row = by_evidence[(evidence,)]
            assert value in row["content"] and row["evidence"] == [evidence]
            assert row["metadata"]["qualifiers"] == qualifiers and row["metadata"]["fact_kind"] == kind
            assert row["memory_key"] != "user_name"
        sources = await repo.list_sources(role, scope, source_message_ids=(sid,))
        assert len(sources) == 1 and sources[0]["body"] == message
        packets, _ = await CharacterMemoryService(repo, semantic_enabled=False).load_relevant_memories(
            role, scope, "请根据记录说明我的称呼及其条件", for_contextual_selection=True
        )
        assert len(packets) == len(expected) and all(message in packet.evidence for packet in packets)
        after = database.get_character_relationship(*fields)
        for name in ["preferred_address", "relationship_stage", "summary", "created_at"]:
            assert after[name] == before[name]
        assert after["interaction_count"] == before["interaction_count"] + 1
        assert outcome.preferred_address == relationship_from_record(before).preferred_address
    finally:
        await scheduler.shutdown()
