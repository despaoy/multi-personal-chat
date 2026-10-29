"""Query time belongs to semantic selection, after authorized version recall."""

import json
from datetime import datetime, timezone

import numpy as np
import pytest

from character.context_builder import compile_reference_context
from character.evidence_selector import ContextualEvidenceSelector
from character.memory_extractor import extract_memories
from character.memory_service import CharacterMemoryService
from character.models import UserScope
from character.rule_memory_writer import write_rule_memory
from character.source_memory import SourceMemoryService
from db.database import SQLiteDB
from repositories.character_memory import DatabaseCharacterMemoryRepository


@pytest.fixture
async def versioned_residence(tmp_path):
    database = SQLiteDB(tmp_path / "temporal.sqlite")
    repository = DatabaseCharacterMemoryRepository(database)
    owner = UserScope("web", "test", "synthetic-user", "synthetic-user", "private")
    for sid, body, stamp in [
        ("old", "我现在住在宁德。", datetime(2024, 3, 1, 1, tzinfo=timezone.utc)),
        ("current", "更正，我现在住在淮安。", datetime(2026, 10, 2, 1, tzinfo=timezone.utc)),
    ]:
        assert (
            await repository.capture_source("kisaki", owner, source_message_id=sid, body=body, observed_at=stamp)
            == "recorded"
        )
        (item,) = extract_memories(body, reference_time=stamp)
        assert await write_rule_memory(repository, "kisaki", owner, item, sid, observed_at=stamp)
    yield repository, owner
    database.close_connection()


@pytest.mark.parametrize(
    "query,expected",
    [
        ("我的现居地是什么？", {"current"}),
        ("2024年我的居住地是什么？", {"old"}),
        ("我不是在问2024年的情况，我问的是现居地是什么？", {"current"}),
        ("2024年我的居住地是什么？我的现居地是什么？", {"old", "current"}),
    ],
)
async def test_semantic_selection_receives_needed_time_versions_and_preserves_final_evidence(
    versioned_residence, query, expected
):
    repository, owner = versioned_residence
    before = await repository.list_memory_records("kisaki", owner, limit=None, include_inactive=True)
    expected_ids = {str(row["id"]) for row in before if set(row["source_message_ids"]) & expected}

    class ExplicitEmbeddings:
        def embed_texts(self, texts, **kwargs):
            return np.asarray([[1.0, 0.0, 0.0] for _ in texts], dtype=np.float32)

    items, total, trace = await CharacterMemoryService(
        repository,
        semantic_enabled=True,
        embedding_provider=ExplicitEmbeddings(),
    ).recall_with_diagnostics(
        "kisaki",
        owner,
        query,
        for_contextual_selection=True,
        reference_time=datetime(2026, 10, 30, 2, tzinfo=timezone.utc),
    )

    async def explicit_model_decisions(messages):
        payload = json.loads(messages[-1]["content"])
        assert payload["query"] == query
        assert payload["required_ids"] == [item.memory_id for item in items]
        return json.dumps(
            {
                "decisions": [
                    {"id": identifier, "label": "use" if identifier in expected_ids else "stale"}
                    for identifier in payload["required_ids"]
                ]
            }
        )

    selected = await ContextualEvidenceSelector(explicit_model_decisions).select(
        query,
        items,
        reference_time=datetime(2026, 10, 30, 2, tzinfo=timezone.utc),
    )
    assert {item.memory_id for item in selected.memories} == expected_ids
    assert {sid for item in selected.memories for sid in item.source_message_ids} == expected
    assert {item.source_message_ids[0]: item.historical for item in selected.memories} == {
        sid: sid == "old" for sid in expected
    }
    reference, used = compile_reference_context(selected.memories, complete_evidence=True)
    assert set(used) == expected_ids and reference
    sources = await SourceMemoryService(repository).recall("kisaki", owner, query, memories=selected.memories)
    assert set(sources.diagnostics["selected_ids"]) == expected
    if expected == {"current"}:
        assert "淮安" in reference and "宁德" not in reference
    assert trace["semantic_status"] == "available" and total >= len(items)
    assert before == await repository.list_memory_records("kisaki", owner, limit=None, include_inactive=True)


async def select_known_ids(query, candidates, expected_ids, *, reference_time):
    """An explicit protocol fixture; expected IDs come from the test's facts."""

    async def explicit_model_decisions(messages):
        payload = json.loads(messages[-1]["content"])
        assert payload["query"] == query
        assert payload["reference_time"] == reference_time.isoformat()
        assert payload["required_ids"] == [item.memory_id for item in candidates]
        return json.dumps(
            {
                "decisions": [
                    {"id": identifier, "label": "use" if identifier in expected_ids else "stale"}
                    for identifier in payload["required_ids"]
                ]
            }
        )

    outcome = await ContextualEvidenceSelector(explicit_model_decisions).select(
        query, candidates, reference_time=reference_time
    )
    return outcome.memories


async def test_current_correction_is_reviewed_without_rewriting_stored_versions(versioned_residence):
    repository, owner = versioned_residence
    before = await repository.list_memory_records("kisaki", owner, limit=None, include_inactive=True)
    query = "更正，我现在住在安顺，不再住在淮安。我现在住在哪里？"
    stamp = datetime(2026, 10, 30, 2, tzinfo=timezone.utc)
    candidates, _ = await CharacterMemoryService(repository, semantic_enabled=False).load_relevant_memories(
        "kisaki", owner, query, for_contextual_selection=True, reference_time=stamp
    )
    assert {item.memory_id for item in candidates} == {str(row["id"]) for row in before}
    selected = await select_known_ids(query, candidates, set(), reference_time=stamp)
    assert selected == ()
    reference, used = compile_reference_context(selected, complete_evidence=True)
    assert reference == "" and used == ()
    assert before == await repository.list_memory_records("kisaki", owner, limit=None, include_inactive=True)
