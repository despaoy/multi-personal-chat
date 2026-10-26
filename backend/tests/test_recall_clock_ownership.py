"""Ordinary recall uses the turn receipt, including across New Year.

Real SQLite sources/claims, profile loading, repository reads, memory service,
CharacterContextService and final reference compilation. No model is used.
Only memory_service's processing clock is fixed. The receipt is an instance of
that datetime subclass too, preserving its actual isinstance validation.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from repositories.character_memory import DatabaseCharacterMemoryRepository
from repositories.messages import DatabaseMessageRepository
from services.character_context import CharacterContextService, TurnInput

from character import memory_service as memory_module
from character.context_builder import compile_reference_context
from character.memory_service import CharacterMemoryService
from character.models import MemoryItem, UserScope
from character.profile_registry import CharacterProfileRegistry
from db.database import SQLiteDB


class ProcessingDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        value = cls(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
        return value.astimezone(tz) if tz is not None else value.replace(tzinfo=None)


RECEIVED = ProcessingDateTime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
PROCESSING = ProcessingDateTime.now(timezone.utc)
ROLE = "fiction-recall-clock"
SCOPE = UserScope("web", "recall-clock", "fiction-owner", "fiction-owner", "private")
QUERY = "去年我的居住地是什么？"
BODIES = {
    2024: "2024年3月8日，我本人陈榆住在青栎镇，只有租约有效时才住在那里，未购房；朋友林澈没有同住。",
    2025: "2025年4月9日，我本人陈榆住在白榆镇，只有公司提供宿舍时才住在那里，未买房；朋友林澈没有同住。",
}


async def arrange_recall_clock(workdir, patch):
    from inference import review_client
    from knowledge.retrieval_core import embedding

    root = Path(workdir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "recall-clock.sqlite"
    if path.exists():
        raise ValueError("recall clock fixture requires a fresh directory")
    for flag in ("CONTEXTUAL_MEMORY_SELECTION_ENABLED", "CONTEXTUAL_DECISION_POLICY_ENABLED",
                 "DYNAMIC_CONTEXT_SEMANTIC_REVIEW_ENABLED", "MEMORY_SOURCE_RECALL_ENABLED"):
        patch.setenv(flag, "false")
    models = AsyncMock(side_effect=AssertionError("recall clock test must never resolve a model"))
    embeddings = Mock(side_effect=AssertionError("recall clock test must never construct an embedding model"))
    patch.setattr(review_client, "get_context_review_client", models)
    patch.setattr(embedding, "get_default_embedding_provider", embeddings)
    patch.setattr(memory_module, "datetime", ProcessingDateTime)

    database = SQLiteDB(path)
    connections = {}
    actual_connection = database._get_connection

    def connection():
        value = actual_connection()
        connections[id(value)] = value
        return value

    patch.setattr(database, "_get_connection", connection)
    repo = DatabaseCharacterMemoryRepository(database)
    ids = {}
    for year, body in BODIES.items():
        stamp = ProcessingDateTime(year, 6, 1, tzinfo=timezone.utc)
        sid = f"fiction-residence-{year}"
        assert await repo.capture_source(ROLE, SCOPE, source_message_id=sid,
                                        body=body, observed_at=stamp) == "recorded"
        row = await repo.append_claim(ROLE, SCOPE,
            MemoryItem("", "user_fact", "用户自述：" + body, .9),
            memory_key="user_residence", status="archived", evidence=(body,),
            source_message_id=sid, observed_at=stamp.isoformat(), confidence=.99,
            valid_from=ProcessingDateTime(year, 1, 1, tzinfo=timezone.utc).isoformat(),
            valid_to=ProcessingDateTime(year + 1, 1, 1, tzinfo=timezone.utc).isoformat(),
            metadata={"origin": "isolated_historical_clock_fixture"})
        assert row["persisted"] is True
        ids[year] = str(row["id"])
    stored = await repo.list_memory_records(ROLE, SCOPE, limit=None, include_inactive=True)
    assert len(stored) == 2 and {row["memory_key"] for row in stored} == {"user_residence"}
    assert {row["status"] for row in stored} == {"archived"}
    assert {row["evidence"][0] for row in stored} == set(BODIES.values())

    profiles = root / "profiles"
    profiles.mkdir()
    (profiles / "clock.json").write_text(json.dumps({
        "character_id": ROLE, "display_name": "虚构时钟核对角色",
        "identity": "仅依据所问年度的完整历史记录回答，不推断条件已满足。",
        "boundaries": ["保留人物、日期、条件和否定，不推断购房或共同居住。"],
    }, ensure_ascii=False), encoding="utf-8")
    registry = CharacterProfileRegistry(profiles)
    assert registry.load_profiles() == 1
    memory = CharacterMemoryService(repo, semantic_enabled=False, gate_enabled=False,
        evidence_enabled=True, version_filter_enabled=True, query_expansion_enabled=False)
    context = CharacterContextService(registry, repo, DatabaseMessageRepository(database),
        memory_service=memory, source_recall_enabled=False, reference_max_chars=6000)
    assert context._memory_selector is None and context._contextual_policy is None
    assert context._semantic_estimator is None

    # Positive controls establish that the real time filter distinguishes both
    # adjacent years; the regression cannot pass merely because one row is absent.
    for clock, expected_year in ((RECEIVED, 2024), (PROCESSING, 2025)):
        selected, total, trace = await memory.recall_with_diagnostics(
            ROLE, SCOPE, QUERY, reference_time=clock)
        assert total == 2 and tuple(item.memory_id for item in selected) == (ids[expected_year],)
        assert selected[0].evidence == (BODIES[expected_year],) and selected[0].historical
        assert trace["selected_count"] == 1

    readings = []
    actual_window = memory_module._historical_query_window

    def observe_window(query, reference):
        window = actual_window(query, reference)
        readings.append(dict(reference=reference.isoformat(),
            start=window.start.isoformat() if window is not None and window.start else None,
            end=window.end.isoformat() if window is not None and window.end else None))
        return window

    patch.setattr(memory_module, "_historical_query_window", observe_window)
    turn = TurnInput(QUERY, SCOPE.platform, SCOPE.adapter, SCOPE.sender_id,
        SCOPE.conversation_id, SCOPE.conversation_type, received_at=RECEIVED)
    return SimpleNamespace(database=database, connections=connections, repo=repo, memory=memory,
        context=context, turn=turn, ids=ids, readings=readings, models=models, embeddings=embeddings)


def close_recall_clock(rig):
    main = rig.database._get_connection()
    for connection in tuple(rig.connections.values()):
        if connection is not main:
            connection.close()
    rig.database.close_connection()


@pytest.fixture
async def clock_rig(tmp_path, monkeypatch):
    rig = await arrange_recall_clock(tmp_path, monkeypatch)
    try:
        yield rig
    finally:
        close_recall_clock(rig)


def assert_received_selection(rig, selected, reference):
    assert tuple(item.memory_id for item in selected) == (rig.ids[2024],)
    item, = selected
    assert item.historical and item.status == "archived"
    assert item.evidence == (BODIES[2024],) and item.source_message_ids == ("fiction-residence-2024",)
    assert item.storage_versions and item.source_record_versions
    assert item.valid_from == "2024-01-01T00:00:00+00:00"
    assert item.valid_to == "2025-01-01T00:00:00+00:00"
    assert BODIES[2024] in reference and BODIES[2025] not in reference
    assert all(fragment in reference for fragment in (
        "2024年3月8日", "我本人陈榆", "只有租约有效时才住在那里", "未购房", "朋友林澈没有同住"))
    assert rig.readings and {row["reference"] for row in rig.readings} == {RECEIVED.isoformat()}
    assert {row["start"] for row in rig.readings} == {"2024-01-01T00:00:00+00:00"}
    rig.models.assert_not_awaited()
    rig.embeddings.assert_not_called()


@pytest.mark.asyncio
async def test_ordinary_prepare_compiles_receipt_year_across_processing_new_year(clock_rig):
    rig = clock_rig
    prepared = await rig.context.prepare_turn(rig.turn, ROLE)
    assert prepared.received_at == RECEIVED
    assert prepared.history == () and prepared.memory_candidates == 2
    assert prepared.memory_selection_status == "disabled"
    assert prepared.compiled.used_memory_ids == (rig.ids[2024],)
    assert prepared.compiled.memory_status == "available"
    assert "虚构时钟核对角色" in prepared.compiled.profile_context
    assert BODIES[2024] not in prepared.compiled.profile_context
    assert_received_selection(rig, prepared.compiled.memory_packets, prepared.compiled.reference_context)


@pytest.mark.asyncio
async def test_ordinary_candidate_helper_honors_explicit_receipt_and_compiles_correct_source(clock_rig):
    rig = clock_rig
    selected, total, trace = await rig.context._load_memory_candidates(
        ROLE, SCOPE, QUERY, reference_time=RECEIVED)
    assert total == 2 and trace["selected_count"] == 1
    reference, used = compile_reference_context(selected, max_chars=6000)
    assert used == (rig.ids[2024],)
    assert_received_selection(rig, selected, reference)


async def baseline_preflight(workdir):
    """Never auto-run: return safe actual read/selection observations.

    Setup proves both years are selectable with explicit real-service clocks.
    The baseline then normally reports processing-year selection; a repair
    reports receipt-year selection. No regression assertion is relaxed.
    """
    with pytest.MonkeyPatch.context() as patch:
        rig = await arrange_recall_clock(workdir, patch)
        try:
            prepared = await rig.context.prepare_turn(rig.turn, ROLE)
            used = set(prepared.compiled.used_memory_ids)
            return dict(received_at=RECEIVED.isoformat(), processing_at=PROCESSING.isoformat(),
                read_times=sorted({row["reference"] for row in rig.readings}),
                actual_window_starts=sorted({row["start"] for row in rig.readings if row["start"]}),
                selected_years=[year for year, identity in rig.ids.items() if identity in used],
                prepared_receipt_preserved=prepared.received_at == RECEIVED,
                received_source_complete=BODIES[2024] in prepared.compiled.reference_context,
                processing_source_complete=BODIES[2025] in prepared.compiled.reference_context,
                source_record_count=len(await rig.repo.list_sources(ROLE, SCOPE)), model_calls=0)
        finally:
            close_recall_clock(rig)



@pytest.mark.asyncio
@pytest.mark.parametrize("supplied_clock", [None, RECEIVED], ids=["capture_once", "preserve_receipt"])
async def test_interactive_operation_keeps_one_clock_and_reloads_revoked_history(clock_rig, monkeypatch, supplied_clock):
    from dataclasses import replace

    from character.memory_llm import is_memory_erasure_request
    from services import character_context

    rig = clock_rig
    clock = [RECEIVED]

    class TurnDateTime(ProcessingDateTime):
        @classmethod
        def now(cls, tz=None):
            return clock[0].astimezone(tz) if tz is not None else clock[0].replace(tzinfo=None)

    monkeypatch.setattr(character_context, "datetime", TurnDateTime)
    phone_body = "我本人陈榆保存的联系电话是010-55551234，只有工作日可联系；不是朋友林澈的号码。"
    phone_source = "fiction-phone-to-delete"
    assert await rig.repo.capture_source(ROLE, SCOPE, source_message_id=phone_source,
        body=phone_body, observed_at=RECEIVED) == "recorded"
    phone = await rig.repo.append_claim(ROLE, SCOPE, MemoryItem("", "user_fact", phone_body, .9),
        memory_key="user_contact_phone", source_message_id=phone_source, evidence=(phone_body,),
        observed_at=RECEIVED.isoformat(), confidence=.99)
    assert phone["persisted"] is True
    rig.database.add_message(dict(platform=SCOPE.platform, adapter=SCOPE.adapter,
        senderId=SCOPE.sender_id, characterId=ROLE, conversationType="private", conversationId=SCOPE.conversation_id,
        sourceMessageId=phone_source, message=phone_body, reply="已记录这项合成资料。", createdAt=RECEIVED.isoformat()))
    histories, observed = [], []
    load_history = rig.context._load_history

    async def history(*args):
        result = await load_history(*args)
        histories.append(result)
        return result

    async def operation(**kwargs):
        observed.append(kwargs["observed_at"])
        assert kwargs["history"][0]["content"] == phone_body
        erased = await rig.repo.erase_memory(ROLE, SCOPE, memory_id=int(phone["id"]))
        assert erased == 1
        assert await rig.repo.source_windows(ROLE, SCOPE, source_message_ids=(phone_source,), radius=1) == []
        clock[0] = PROCESSING
        return dict(status="erased", persisted=erased, source_erased=1)

    monkeypatch.setattr(rig.context, "_load_history", history)
    worker = SimpleNamespace(schedule_and_wait=AsyncMock(side_effect=operation))
    monkeypatch.setattr("character.memory_llm.get_memory_enrichment_scheduler", lambda: worker)
    query = "请从记忆里删除我的电话号码，保留其他资料。" + QUERY
    assert is_memory_erasure_request(query)
    turn = replace(rig.turn, message=query, received_at=supplied_clock)
    prepared = await rig.context.prepare_interactive_turn(turn, ROLE, source_message_id="delete-phone-on-new-year")
    assert prepared.received_at == RECEIVED and observed == [RECEIVED]
    assert turn.received_at is supplied_clock
    worker.schedule_and_wait.assert_awaited_once()
    assert len(histories) == 2 and histories[0][0]["content"] == phone_body and histories[1] == []
    assert prepared.history == () and prepared.memory_operation_receipt["persisted"] == 1
    assert prepared.compiled.used_memory_ids == (rig.ids[2024],)
    assert_received_selection(rig, prepared.compiled.memory_packets, prepared.compiled.reference_context)
    remaining = await rig.repo.list_memory_records(ROLE, SCOPE, limit=None, include_inactive=True)
    assert {str(row["id"]) for row in remaining} == set(rig.ids.values())
    assert phone_body not in prepared.compiled.reference_context
