"""Scoped controls for fresh, exact, batched pre-send claim and source reads.

The 201-source boundary uses a complete controlled repository, not a model.
No remote provider or semantic quality evaluation is claimed.
"""

import hashlib
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from character.context_builder import compile_reference_context
from character.memory_read_authority import record_version, source_record_version, source_version
from character.models import CompiledCharacterContext, MemoryItem, UserScope
from db.memory_source import source_identity, source_scope
from inference.generation_request import GenerationRequest, generate_character_response
from inference.structured_context_authority import revalidate_private_memories
from tests.test_structured_memory_authority import structured_request

ROLE = "fiction-role"


async def bound_request(tmp_path):
    request, _prepared, repo, db, scope, target, public = await structured_request(tmp_path)
    packets = []
    for packet in request.character_context.memory_packets:
        pairs = tuple((int(key), sid) for key, sid in packet.source_record_pairs)
        rows = await repo.linked_source_receipts(ROLE, scope, claim_sources=pairs)
        assert len(rows) == len(set(pairs))
        packets.append(replace(packet, source_versions=tuple(
            (str(row["memory_id"]), row["source_message_id"], source_version(row)) for row in rows
        )))
    first = packets[0]
    packets[0] = replace(first, source_record_pairs=first.source_record_pairs + first.source_record_pairs[:1],
                         source_versions=first.source_versions + first.source_versions[:1])
    request = replace(request, character_context=replace(request.character_context, memory_packets=tuple(packets)))
    for name in ("linked_source_revisions", "linked_source_receipts"):
        setattr(repo, name, AsyncMock(wraps=getattr(repo, name)))
    repo.linked_sources = AsyncMock(side_effect=AssertionError("Broad source reads are not authorized"))
    return request, repo, db, scope, target, public


async def test_source_reads_are_deduplicated_and_fresh_on_each_send(tmp_path):
    request, repo, db, scope, target, public = await bound_request(tmp_path)
    try:
        assert await revalidate_private_memories(request, repo, ROLE, scope) is request
        expected = {(int(key), sid) for p in request.character_context.memory_packets for key, sid in p.source_record_pairs}
        for reader in (repo.linked_source_revisions, repo.linked_source_receipts):
            reader.assert_awaited_once()
            args = reader.await_args
            assert args.args == (ROLE, scope)
            assert set(args.kwargs["claim_sources"]) == expected
            assert len(args.kwargs["claim_sources"]) == len(expected)

        identity = source_identity(source_scope(
            ROLE, scope.platform, scope.adapter, scope.sender_id, scope.conversation_type, scope.conversation_id
        ), target.source_message_ids[0])
        row = await repo.get_memory_record(int(target.memory_id), ROLE, scope)
        connection = db._get_connection()
        cursor = connection.execute(
            "UPDATE memory_sources SET body=NULL,state='revoked' WHERE source_key=? AND state='recorded'",
            (identity["source_key"],),
        )
        assert cursor.rowcount == 1
        connection.commit()
        assert record_version(await repo.get_memory_record(int(target.memory_id), ROLE, scope)) == record_version(row)
        updated = await revalidate_private_memories(request, repo, ROLE, scope)
        assert len(updated.character_context.memory_packets) == 1
        assert target.memory_id not in updated.character_context.used_memory_ids
        assert "豆浆" in updated.character_context.reference_context
        assert target.content not in updated.character_context.reference_context
        assert updated.retrieval.evidence == public
        assert repo.linked_source_revisions.await_count == repo.linked_source_receipts.await_count == 2
        repo.linked_sources.assert_not_awaited()
    finally:
        db.close_connection()


@pytest.mark.parametrize("reader_name,corruption", [
    ("linked_source_revisions", "duplicate"),
    ("linked_source_receipts", "duplicate"),
    ("linked_source_revisions", "unrequested"),
])
async def test_malformed_source_read_stops_before_model(tmp_path, reader_name, corruption):
    request, repo, db, scope, _target, _public = await bound_request(tmp_path)
    original = getattr(repo, reader_name)

    async def corrupt(*args, **kwargs):
        rows = await original(*args, **kwargs)
        extra = rows[0] if corruption == "duplicate" else dict(rows[0], memory_id=999999)
        return [*rows, extra]

    setattr(repo, reader_name, corrupt)
    request = replace(request, private_context_revalidator=lambda value: revalidate_private_memories(value, repo, ROLE, scope))
    model = AsyncMock(return_value="不应生成")
    try:
        with pytest.raises(RuntimeError, match="duplicate or unrequested"):
            await generate_character_response(request, model)
        model.assert_not_awaited()
        repo.linked_sources.assert_not_awaited()
    finally:
        db.close_connection()


def many_source_request():
    scope = UserScope("web", "batch-control", "fiction-owner", "fiction-owner", "private")
    claim = {"id": 1, "content": "完整虚构来源集证明书页登记收费13元。"}
    rows = tuple(dict(memory_id=1, source_message_id=f"fiction-source-{i:03}",
                      body=f"虚构完整来源{i}：书页登记收费13元。", observed_at="2026-10-01T00:00:00+00:00")
                 for i in range(201))
    revisions = tuple(dict(row, body_sha256=hashlib.sha256(row["body"].encode()).hexdigest()) for row in rows)
    packet = MemoryItem("1", "user_fact", claim["content"], evidence=tuple(row["body"] for row in rows),
                        source_message_ids=tuple(row["source_message_id"] for row in rows),
                        storage_versions=(("1", record_version(claim)),),
                        source_record_pairs=tuple(("1", row["source_message_id"]) for row in rows),
                        source_record_versions=tuple(("1", row["source_message_id"], source_record_version(row))
                                                     for row in revisions),
                        source_versions=tuple(("1", row["source_message_id"], source_version(row)) for row in rows))
    reference, ids = compile_reference_context((packet,), complete_evidence=True, max_chars=100000)
    context = CompiledCharacterContext("虚构角色", "", reference, memory_packets=(packet,), used_memory_ids=ids)
    request = GenerationRequest(message="登记收费是多少？只根据完整虚构来源回答。", character_context=context)

    def reader_for(records):
        async def read(role, actual_scope, *, claim_sources):
            assert role == ROLE and actual_scope == scope
            assert 0 < len(claim_sources) <= 200
            pairs = set(claim_sources)
            return [row for row in records if (row["memory_id"], row["source_message_id"]) in pairs]
        return AsyncMock(side_effect=read)

    repo = SimpleNamespace(list_memory_records=AsyncMock(return_value=[claim]),
                           linked_source_revisions=reader_for(revisions), linked_source_receipts=reader_for(rows))
    return request, repo, scope


async def test_more_than_200_pairs_are_all_read_in_bounded_batches():
    request, repo, scope = many_source_request()
    assert await revalidate_private_memories(request, repo, ROLE, scope) is request
    for reader in (repo.linked_source_revisions, repo.linked_source_receipts):
        batches = [call.kwargs["claim_sources"] for call in reader.await_args_list]
        assert [len(batch) for batch in batches] == [200, 1]
        assert len(set(pair for batch in batches for pair in batch)) == 201
    repo.list_memory_records.assert_awaited_once_with(ROLE, scope, limit=None, include_inactive=True, memory_ids=(1,))


@pytest.mark.parametrize("reader_name", ["linked_source_revisions", "linked_source_receipts"])
async def test_failed_second_batch_is_not_cached_or_sent_to_model(reader_name):
    request, repo, scope = many_source_request()
    reader = getattr(repo, reader_name)
    original = reader.side_effect

    async def fail_second(*args, **kwargs):
        if reader.await_count == 2:
            raise OSError("Controlled second batch unavailable")
        return await original(*args, **kwargs)

    reader.side_effect = fail_second
    request = replace(request, private_context_revalidator=lambda value: revalidate_private_memories(value, repo, ROLE, scope))
    model = AsyncMock(return_value="不应生成")
    with pytest.raises(OSError, match="second batch unavailable"):
        await generate_character_response(request, model)
    model.assert_not_awaited()
    assert reader.await_count == 2
    if reader_name == "linked_source_revisions":
        repo.linked_source_receipts.assert_not_awaited()
    reader.side_effect = original
    repo.linked_source_revisions.reset_mock()
    repo.linked_source_receipts.reset_mock()
    assert await revalidate_private_memories(request, repo, ROLE, scope) is request
    for reader in (repo.linked_source_revisions, repo.linked_source_receipts):
        assert [len(call.kwargs["claim_sources"]) for call in reader.await_args_list] == [200, 1]


def many_claim_request():
    request, repo, scope = many_source_request()
    rows = [dict(id=index, content=f"完整虚构档案{index}：借期14天，不得自动续借，周末不受理。")
            for index in range(1, 202)]
    packets = tuple(MemoryItem(str(row["id"]), "user_fact", row["content"], evidence=(row["content"],),
        storage_versions=((str(row["id"]), record_version(row)),)) for row in rows)
    packets = (replace(packets[0], storage_versions=packets[0].storage_versions + (("001", record_version(rows[0])),)),
               *packets[1:])
    reference, ids = compile_reference_context(packets, complete_evidence=True, max_chars=1000000)
    request = replace(request, character_context=replace(request.character_context,
        memory_packets=packets, used_memory_ids=ids, reference_context=reference))

    async def read(role, actual_scope, *, limit, include_inactive, memory_ids):
        assert role == ROLE and actual_scope == scope and limit is None and include_inactive
        assert 0 < len(memory_ids) <= 200
        return [row for row in rows if row["id"] in memory_ids]

    repo.list_memory_records = AsyncMock(side_effect=read)
    repo.get_memory_record = AsyncMock(side_effect=AssertionError("Claims must be read in exact batches"))
    return request, repo, scope, rows


async def test_claim_batches_deduplicate_ids_and_recheck_fresh_versions():
    request, repo, scope, rows = many_claim_request()
    assert await revalidate_private_memories(request, repo, ROLE, scope) is request
    batches = [call.kwargs["memory_ids"] for call in repo.list_memory_records.await_args_list]
    assert [len(batch) for batch in batches] == [200, 1]
    assert len({key for batch in batches for key in batch}) == 201
    original = rows[0]["content"]
    rows[0]["content"] = "未经本轮审核的新内容"
    refreshed = await revalidate_private_memories(request, repo, ROLE, scope, max_chars=1000000)
    assert len(refreshed.character_context.memory_packets) == 200
    assert original not in refreshed.character_context.reference_context
    assert "未经本轮审核" not in refreshed.character_context.reference_context
    assert rows[-1]["content"] in refreshed.character_context.reference_context
    assert repo.list_memory_records.await_count == 4
    repo.get_memory_record.assert_not_awaited()


@pytest.mark.parametrize("corruption", ["duplicate", "unrequested"])
async def test_malformed_claim_batch_stops_before_model(corruption):
    request, repo, scope, _rows = many_claim_request()
    read = repo.list_memory_records.side_effect

    async def corrupt(*args, **kwargs):
        rows = await read(*args, **kwargs)
        return [*rows, rows[0] if corruption == "duplicate" else dict(rows[0], id=999999)]

    repo.list_memory_records.side_effect = corrupt
    request = replace(request, private_context_revalidator=lambda value: revalidate_private_memories(value, repo, ROLE, scope))
    model = AsyncMock()
    with pytest.raises(RuntimeError, match="duplicate or unrequested"):
        await generate_character_response(request, model)
    model.assert_not_awaited()


async def test_failed_second_claim_batch_preserves_error_and_can_recover():
    request, repo, scope, _rows = many_claim_request()
    read = repo.list_memory_records.side_effect
    failure = OSError("Controlled second claim batch unavailable")

    async def fail_second(*args, **kwargs):
        if repo.list_memory_records.await_count == 2:
            raise failure
        return await read(*args, **kwargs)

    repo.list_memory_records.side_effect = fail_second
    request = replace(request, private_context_revalidator=lambda value: revalidate_private_memories(value, repo, ROLE, scope))
    model = AsyncMock()
    with pytest.raises(OSError) as caught:
        await generate_character_response(request, model)
    assert caught.value is failure
    model.assert_not_awaited()
    repo.list_memory_records.side_effect = read
    repo.list_memory_records.reset_mock()
    assert await revalidate_private_memories(request, repo, ROLE, scope) is request
    assert [len(call.kwargs["memory_ids"]) for call in repo.list_memory_records.await_args_list] == [200, 1]


async def test_revoked_observation_removes_source_history_without_erasing_shared_status_label(tmp_path):
    from datetime import datetime, timedelta, timezone
    from character.memory_service import CharacterMemoryService
    from character.temporal_provenance import model_temporal_provenance
    from db.database import SQLiteDB
    from repositories.character_memory import DatabaseCharacterMemoryRepository

    database = SQLiteDB(tmp_path / "observation-label.sqlite")
    repository = DatabaseCharacterMemoryRepository(database)
    scope = UserScope("test", "observation-label", "owner", "room", "private")
    observed = datetime.now(timezone.utc) - timedelta(days=1)
    bodies = ("我这周在南宁出差，下周返回。", "我周末才用芷露这个昵称，平日不用。")
    try:
        for index, body in enumerate(bodies):
            await repository.capture_source(ROLE, scope, source_message_id=str(index), body=body, observed_at=observed)
            await repository.append_claim(ROLE, scope, MemoryItem("", "user_fact", "用户提到：" + body, .9),
                memory_key="fact_observation_" + str(index), evidence=(body,), source_message_id=str(index),
                observed_at=observed.isoformat(), metadata={"fact_kind": "other_user_fact",
                    "temporal_provenance": model_temporal_provenance(evidence=body, observed_at=observed)})
        packets, _ = await CharacterMemoryService(repository, semantic_enabled=False).load_relevant_memories(
            ROLE, scope, "查看原话", for_contextual_selection=True)
        assert len(packets) == 2 and all(packet.temporal_mode == "observation" for packet in packets)
        label = packets[0].content
        assert label == packets[1].content
        retained = next(packet for packet in packets if bodies[1] in packet.evidence)
        reference, used = compile_reference_context(packets, complete_evidence=True)
        history = ({"role": "user", "content": "查看独立显示说明"}, {"role": "assistant", "content": label},
                   {"role": "user", "content": bodies[0]}, {"role": "assistant", "content": "已记录出差说明"},
                   {"role": "user", "content": bodies[1]}, {"role": "assistant", "content": "已记录称呼说明"})
        conversation = "独立显示说明：" + label
        branch = "独立分支说明：" + label
        context = CompiledCharacterContext("", "", reference, used, memory_packets=packets, memory_status="available",
            conversation_reference_context=conversation, branch_context=branch)
        request = GenerationRequest(message="继续查看原话", history=history, character_context=context)
        identity = source_identity(source_scope(ROLE, scope.platform, scope.adapter, scope.sender_id,
            scope.conversation_type, scope.conversation_id), "0")
        connection = database._get_connection()
        cursor = connection.execute("UPDATE memory_sources SET body=NULL,state='revoked' WHERE source_key=? AND state='recorded'",
                                    (identity["source_key"],))
        assert cursor.rowcount == 1
        connection.commit()
        assert len(await repository.list_memory_records(ROLE, scope)) == 2
        updated = await revalidate_private_memories(request, repository, ROLE, scope)
        assert updated.character_context.used_memory_ids == (retained.memory_id,)
        assert bodies[0] not in updated.character_context.reference_context
        assert bodies[1] in updated.character_context.reference_context
        assert updated.history == history[:2] + history[4:]
        assert updated.character_context.conversation_reference_context == conversation
        assert updated.character_context.branch_context == branch
    finally:
        database.close_connection()
