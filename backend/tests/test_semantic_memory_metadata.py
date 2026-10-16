"""Known semantic provenance failures cannot become ordinary asserted facts."""

import json
from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from character.memory_read_authority import MemoryRecordError
from character.memory_service import CharacterMemoryService
from character.models import MemoryItem
from character.temporal_provenance import model_temporal_provenance
from inference.generation_request import generate_character_response
from tests.test_structured_memory_authority import structured_request


@pytest.mark.parametrize("mutation,value,field", [
    ("observed_at", "synthetic-private-invalid-clock", "observed_at"),
    ("observed_at", "2026-10-01T01:00:00", "observed_at"),
    ("envelope", None, "metadata_json.temporal_provenance"),
    ("version", 2, "metadata_json.temporal_provenance.version"),
    ("version", True, "metadata_json.temporal_provenance.version"),
    ("producer", "synthetic-private-unknown", "metadata_json.temporal_provenance.producer"),
    ("validity_authority", "verified", "metadata_json.temporal_provenance.validity_authority"),
])
async def test_corrupt_semantic_metadata_stops_reads_and_generation(tmp_path, mutation, value, field):
    request, _prepared, repo, db, scope, target, _public = await structured_request(tmp_path, projected=True)
    connection = db._get_connection()
    model = AsyncMock(return_value="不应生成")
    try:
        if mutation == "observed_at":
            column, encoded = "observed_at", value
        else:
            row = await repo.get_memory_record(int(target.memory_id), "fiction-role", scope)
            metadata = dict(row["metadata"])
            if mutation == "envelope":
                metadata["temporal_provenance"] = value
            else:
                metadata["temporal_provenance"] = dict(metadata["temporal_provenance"], **{mutation: value})
            column, encoded = "metadata_json", json.dumps(metadata)
        connection.execute(f"UPDATE character_memories SET {column}=? WHERE id=?", (encoded, int(target.memory_id)))
        connection.commit()
        for read in (
            lambda: repo.get_memory_record(int(target.memory_id), "fiction-role", scope),
            lambda: CharacterMemoryService(repo, semantic_enabled=False).load_relevant_memories(
                "fiction-role", scope, request.message
            ),
            lambda: generate_character_response(request, model),
        ):
            with pytest.raises(ValueError, match=field) as caught:
                await read()
            assert caught.value.field == field
            assert "synthetic-private" not in str(caught.value)
        model.assert_not_awaited()
        assert connection.execute(
            f"SELECT {column} FROM character_memories WHERE id=?", (int(target.memory_id),)
        ).fetchone()[0] == encoded
    finally:
        db.close_connection()


@pytest.mark.parametrize("mutation,value,field", [
    ("clock", None, "observed_at"),
    ("clock", "synthetic-private-invalid-clock", "observed_at"),
    ("version", 2, "metadata_json.temporal_provenance.version"),
])
async def test_invalid_semantic_write_is_rejected_before_persistence(tmp_path, mutation, value, field):
    _request, _prepared, repo, db, scope, target, _public = await structured_request(tmp_path)
    try:
        row = await repo.get_memory_record(int(target.memory_id), "fiction-role", scope)
        before = await repo.list_memory_records("fiction-role", scope)
        metadata = dict(temporal_provenance=model_temporal_provenance(
            evidence=target.evidence[0], observed_at=datetime.fromisoformat(row["observed_at"])
        ))
        clock = row["observed_at"]
        if mutation == "clock":
            clock = value
        else:
            metadata["temporal_provenance"][mutation] = value
        with pytest.raises(MemoryRecordError, match=field):
            await repo.append_claim(
                "fiction-role", scope, MemoryItem("", "user_fact", "合成不应持久化的摘要"),
                memory_key="rejected_semantic_metadata", evidence=target.evidence,
                source_message_id=target.source_message_ids[0], source_message_ids=target.source_message_ids,
                observed_at=clock, metadata=metadata,
            )
        assert await repo.list_memory_records("fiction-role", scope) == before
    finally:
        db.close_connection()


async def test_valid_semantic_write_keeps_original_source_and_clock(tmp_path):
    _request, _prepared, repo, db, scope, target, _public = await structured_request(tmp_path)
    try:
        row = await repo.get_memory_record(int(target.memory_id), "fiction-role", scope)
        metadata = dict(temporal_provenance=model_temporal_provenance(
            evidence=target.evidence[0], observed_at=datetime.fromisoformat(row["observed_at"])
        ))
        saved = await repo.append_claim(
            "fiction-role", scope, MemoryItem("", "user_fact", "语义提供者摘要，归属尚需核验", .9),
            memory_key="semantic_metadata_control", evidence=target.evidence,
            source_message_id=target.source_message_ids[0], source_message_ids=target.source_message_ids,
            observed_at=row["observed_at"], metadata=metadata,
        )
        assert saved["persisted"] is True and saved["observed_at"] == row["observed_at"]
        assert saved["evidence"] == list(target.evidence)
        assert saved["metadata"]["temporal_provenance"] == metadata["temporal_provenance"]
        assert await repo.linked_source_receipts("fiction-role", scope, claim_sources=tuple(
            (saved["id"], source_id) for source_id in target.source_message_ids
        ))
    finally:
        db.close_connection()
