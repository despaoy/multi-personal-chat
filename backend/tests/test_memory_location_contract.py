"""Semantic writes must use the same location fields as rule writes and reads."""

import json

import pytest

from character.memory_llm import parse_llm_proposals
from character.memory_query import plan_memory_query


def proposal(source, value, **changes):
    raw = dict(kind="location", value=value, evidence=source, confidence=.96, operation="ADD")
    raw.update(changes)
    return json.dumps({"memories": [raw]}, ensure_ascii=False)


@pytest.mark.parametrize("source,value,key,query", [
    ("我来自泉州。", "泉州", "user_origin", "我的家乡是哪儿？"),
    ("我现在住在大理。", "大理", "user_residence", "我住在哪里？"),
    ("我住在青岛。", "青岛", "user_residence", "我的住址是什么？"),
])
def test_explicit_locations_are_retrievable_by_their_actual_field(source, value, key, query):
    parsed, = parse_llm_proposals(proposal(source, value), source_message=source)
    assert parsed.memory.memory_key == key
    plan = plan_memory_query(query)
    assert plan.matched_fields({"memory_key": parsed.memory.memory_key})


@pytest.mark.parametrize("target_key,source,value,accepted", [
    ("user_residence", "我现在住在无锡。", "无锡", True),
    ("user_origin", "我来自襄阳。", "襄阳", True),
    ("user_origin", "我现在住在无锡。", "无锡", False),
    ("user_residence", "我来自襄阳。", "襄阳", False),
])
def test_location_update_respects_predicate_not_only_kind(target_key, source, value, accepted):
    parsed = parse_llm_proposals(proposal(source, value, operation="SUPERSEDE", target_memory_id="7",
                                         target_memory_key=target_key), source_message=source,
                                existing_memories=({"id": "7", "memory_key": target_key,
                                                    "content": "用户旧地理信息", "status": "active"},))
    assert bool(parsed) is accepted


def test_same_place_origin_and_residence_is_not_arbitrarily_assigned():
    source = "我来自绍兴，我现在住在绍兴。"
    parsed, = parse_llm_proposals(proposal(source, "绍兴"), source_message=source)
    assert parsed.memory.memory_key == "user_location"


@pytest.mark.parametrize("value,key", [("延安", "user_origin"), ("北海", "user_residence")])
def test_two_predicates_in_one_evidence_are_matched_by_exact_value(value, key):
    source = "我来自延安，我现在住在北海。"
    parsed, = parse_llm_proposals(proposal(source, value), source_message=source)
    assert parsed.memory.memory_key == key


def test_legacy_location_target_is_not_silently_migrated():
    source = "我现在住在威海。"
    parsed, = parse_llm_proposals(
        proposal(source, "威海", operation="SUPERSEDE", target_memory_id="7", target_memory_key="user_location"),
        source_message=source, existing_memories=({"id": "7", "memory_key": "user_location",
                                                   "content": "用户说自己来自或居住在洛阳"},))
    assert parsed.memory.memory_key == "user_location"


@pytest.mark.asyncio
async def test_semantic_write_sqlite_recall_and_closed_read_agree(tmp_path):
    from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig
    from character.memory_service import CharacterMemoryService
    from character.models import CompiledCharacterContext, UserScope
    from db.database import SQLiteDB
    from inference.memory_response import read_memory_fields, render_memory_response
    from repositories.character_memory import DatabaseCharacterMemoryRepository

    statements = [("我来自赣州。", "赣州"), ("我现在住在宜昌。", "宜昌")]

    class Completion:
        def __init__(self):
            self.index = 0

        async def complete(self, messages):
            source, value = statements[self.index]
            self.index += 1
            return proposal(source, value)

        async def close(self):
            pass

    class Embedding:
        def embed_texts(self, texts):
            return [[1.0, 0.0] for _ in texts]

    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / "locations.db"))
    scope = UserScope("test", "test", "user", "user", "private")
    scheduler = MemoryEnrichmentScheduler(
        config=MemoryLlmConfig(enabled=True, base_url="http://unused", model="stub"),
        completion=Completion(), embedding_provider=Embedding())
    for index, (source, _) in enumerate(statements):
        assert scheduler.schedule(repository=repo, character_id="role", user_scope=scope,
                                  message=source, rule_hints=[], source_message_id=f"source-{index}")
    await scheduler.shutdown(timeout=3)
    assert scheduler.status.saved == 2
    service = CharacterMemoryService(repo, semantic_enabled=False)
    query = "我来自哪里，目前住哪里？"
    items, _, trace = await service.recall_with_diagnostics("role", scope, query)
    assert trace["covered_fields"] == ["origin", "residence"]
    compiled = CompiledCharacterContext("", "", "", tuple(item.memory_id for item in items),
                                        memory_packets=items, memory_status="available")
    assert [(read.field, read.value) for read in read_memory_fields(query, compiled)] == [
        ("origin", "赣州"), ("residence", "宜昌")]
    assert render_memory_response(query, compiled) == "我这里记着的是：你来自赣州，现在住在宜昌。"
