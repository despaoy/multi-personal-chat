"""Frozen real-provider protocol failures and source-preserving admission."""
import json
from pathlib import Path

import pytest

from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig, MemoryResponseError, parse_llm_proposals
from character.models import CompiledCharacterContext, UserScope
from character.semantic_state_estimator import SemanticStateEstimator
from character.situation_analyzer import SituationAnalyzer
from inference.generation_request import GenerationRequest, build_generation_request


def fixture(name):
    return json.loads((Path(__file__).parent / "fixtures" / name).read_text(encoding="utf-8"))

def merge(changes=None, targets=None):
    data = fixture("deepseek_merge_admission.json")
    candidate = dict(data["writer_output"]["memories"][0])
    candidate.update(changes or {})
    return parse_llm_proposals(json.dumps({"memories": [candidate]}, ensure_ascii=False),
                              source_message=data["source_message"],
                              existing_memories=tuple(data["existing_memories"] if targets is None else targets),
                              history=tuple(data["history"]))

def test_frozen_paraphrase_errors_and_literal_coexist_keeps_source_and_target():
    with pytest.raises(MemoryResponseError) as caught:
        merge()
    assert caught.value.field == "value"
    data = fixture("deepseek_merge_admission.json")
    proposal, = merge(dict(operation="COEXIST", value="芝麻", content="", evidence=data["source_message"]))
    assert proposal.operation == "COEXIST" and proposal.fact_kind == "other_user_fact"
    assert proposal.target_memory_id == "13"
    assert proposal.memory.memory_key == data["existing_memories"][0]["memory_key"]
    assert proposal.evidence == data["source_message"]
    assert "麦芽" in proposal.memory.content and "芝麻" in proposal.memory.content


def test_unverified_free_form_summary_is_error_instead_of_quote_recovery():
    with pytest.raises(MemoryResponseError) as caught:
        merge({"value": "养了三只猫：麦芽、芝麻、雪饼", "content": "用户又养了雪饼，共三只猫"})
    assert caught.value.field == "value"


@pytest.mark.parametrize("changes,field", [
    ({"target_memory_id": "999"}, "target_memory_id"), ({"target_memory_key": "user_name"}, "target_memory_key"),
    ({"confidence": .1}, None), ({"attributed_to": "assistant"}, "attributed_to"),
    ({"evidence": "我新养的猫叫雪饼。"}, "evidence"),
])
def test_recorded_candidate_preserves_specific_admission_boundaries(changes, field):
    if field:
        with pytest.raises(MemoryResponseError) as caught:
            merge(changes)
        assert caught.value.field == field
    else:
        assert not merge(changes)


def test_missing_target_is_protocol_failure_not_new_paraphrased_fact():
    with pytest.raises(MemoryResponseError) as caught:
        merge(targets=())
    assert caught.value.field == "target_memory_id"


@pytest.mark.asyncio
async def test_sqlite_keeps_both_records_when_original_source_recall_is_disabled(tmp_path):
    from legacy_memory_fixtures import seed_legacy_memory
    from repositories.character_memory import DatabaseCharacterMemoryRepository

    from character.context_builder import compile_reference_context
    from character.memory_service import CharacterMemoryService
    from db.database import SQLiteDB
    data = fixture("deepseek_merge_admission.json")
    database = SQLiteDB(tmp_path / "merge.sqlite")
    repo = DatabaseCharacterMemoryRepository(database)
    scope = UserScope("test", "test", "user", "private", "private")
    old = data["existing_memories"][0]
    from character.memory_extractor import ExtractedMemory

    seed_legacy_memory(repo, "role", scope,
        ExtractedMemory("user_fact", old["memory_key"], old["content"], .6),
        memory_key=old["memory_key"], source_message_id="original-source")
    records = await repo.list_memory_records("role", scope, limit=None, include_inactive=True)
    target = next(record for record in records if record["status"] == "active")
    candidate = dict(data["writer_output"]["memories"][0], operation="COEXIST", value="芝麻", content="",
                     evidence=data["source_message"], target_memory_id=str(target["id"]))
    class Completion:
        async def complete(self, messages):
            return json.dumps({"memories": [candidate]}, ensure_ascii=False)
        async def close(self):
            pass
    class Embedding:
        def embed_texts(self, texts):
            return [[1.0, 0.0] for _ in texts]
    scheduler = MemoryEnrichmentScheduler(
        config=MemoryLlmConfig(enabled=True, base_url="http://unused", model="fixture", idle_seconds=0),
        completion=Completion(), embedding_provider=Embedding())
    assert scheduler.schedule(repository=repo, character_id="role", user_scope=scope,
                              message=data["source_message"], rule_hints=[], source_message_id="additional-source")
    assert await scheduler.flush_memory(timeout=5)
    await scheduler.shutdown(timeout=2)
    assert scheduler.status.saved == 1
    records = await repo.list_memory_records("role", scope, limit=None, include_inactive=True)
    assert len([r for r in records if r["status"] == "active"]) == 2
    added = next(r for r in records if r["relation_type"] == "COEXIST")
    assert added["metadata"]["fact_kind"] == "other_user_fact"
    assert added["evidence"] == [data["source_message"]]
    service = CharacterMemoryService(repo, semantic_enabled=False)
    packets, _, _ = await service.recall_with_diagnostics("role", scope, "我养了几只猫，分别叫什么？")
    reference = compile_reference_context(packets, complete_evidence=True)[0]
    assert "麦芽" in reference and "芝麻" in reference
    assert next(item for item in packets if item.memory_id == str(added["id"])).temporal_mode == "observation"

@pytest.mark.asyncio
@pytest.mark.parametrize("language", ["json", ""])
async def test_exact_code_fence_keeps_valid_semantic_state(language):
    data = fixture("deepseek_semantic_fence.json")
    raw = data["review_output"].replace("```json", "```" + language, 1)
    async def reviewer(messages):
        return raw
    original = SituationAnalyzer().estimate(data["source_message"])
    outcome = await SemanticStateEstimator(reviewer, review_mode="all_non_safety").refine_with_diagnostics(data["source_message"], data["history"], original)
    assert outcome.status == "applied"
    assert outcome.state.primary_situation == "emotional"

@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["prefix", "suffix", "language", "invalid_enum", "invalid_json"])
async def test_fence_normalization_does_not_accept_unsupported_content(change):
    data = fixture("deepseek_semantic_fence.json")
    raw = data["review_output"]
    raw = {"prefix": "explanation\n" + raw, "suffix": raw + "\nextra", "language": raw.replace("```json", "```python"),
           "invalid_enum": raw.replace('"emotional"', '"made_up"'), "invalid_json": "```json\n{bad}\n```"}[change]
    async def reviewer(messages):
        return raw
    original = SituationAnalyzer().estimate(data["source_message"])
    with pytest.raises(ValueError):
        await SemanticStateEstimator(reviewer, review_mode="all_non_safety").refine_with_diagnostics(data["source_message"], data["history"], original)

@pytest.mark.parametrize("status", ["available", "no_match", "retrieval_error"])
def test_partial_visibility_does_not_prove_never_saved_or_never_told(status):
    context = CompiledCharacterContext("", "", "some unrelated saved memory", memory_status=status)
    plan = build_generation_request(GenerationRequest(message="你还保存着我的猫名字吗？我的大学专业是什么？",
                                                       character_context=context))
    prompt = plan.messages[0]["content"]
    assert "从未保存" in prompt and "从未说过" in prompt
    assert "不能" in prompt


@pytest.mark.parametrize("value,literal", [("养了一只名叫荔枝的猫", "荔枝"), ("养了两只猫：荔枝和乌梅", ["荔枝", "乌梅"])])
def test_generic_paraphrase_errors_and_explicit_literals_keep_full_source(value, literal):
    source = "我养了一只猫，名叫荔枝。现在又养了一只猫叫乌梅，两只猫都在。"
    candidate = dict(kind="other_user_fact", value=value, evidence=source,
                     confidence=.98, operation="ADD", attributed_to="user")
    with pytest.raises(MemoryResponseError) as caught:
        parse_llm_proposals(json.dumps({"memories": [candidate]}, ensure_ascii=False), source_message=source)
    assert caught.value.field == "value"
    # Explicit corrected protocol example, not a fresh model result or repair.
    candidate["value"] = literal
    proposal, = parse_llm_proposals(json.dumps({"memories": [candidate]}, ensure_ascii=False), source_message=source)
    assert proposal.fact_kind == "other_user_fact" and proposal.operation == "ADD"
    assert proposal.evidence == source and proposal.memory.memory_key == "fact_" + (literal if isinstance(literal, str) else "".join(literal))
    assert "荔枝" in proposal.memory.content and "乌梅" in proposal.memory.content


@pytest.mark.asyncio
async def test_erase_latest_claim_removes_older_key_versions_and_preserves_other_facts(tmp_path):
    from datetime import datetime, timezone
    from types import SimpleNamespace

    from repositories.character_memory import DatabaseCharacterMemoryRepository

    from db.database import SQLiteDB
    database = SQLiteDB(tmp_path / "erasure.sqlite")
    repo = DatabaseCharacterMemoryRepository(database)
    scope = UserScope("test", "test", "user", "room", "private")
    fields = dict(character_id="role", platform="test", adapter="test", sender_id="user",
                  conversation_type="private", conversation_id="room")
    other = {**fields, "sender_id": "other"}
    first_source = "我的猫叫豆沙。我的大学专业是海洋工程。"
    next_source = "我又养了一只猫叫糯米，豆沙仍然在。"
    for sid, body in [("original", first_source), ("addition", next_source)]:
        database.capture_memory_source(**fields, source_message_id=sid, body=body,
                                       observed_at=datetime.now(timezone.utc))
    old = database.append_character_memory_claim(**fields, memory_type="user_fact",
        memory_key="fact_pet", content="用户的猫叫豆沙", source_message_id="original",
        evidence_json=json.dumps(["我的猫叫豆沙"]))
    latest = database.append_character_memory_claim(**fields, memory_type="user_fact",
        memory_key="fact_pet", content="用户的猫叫豆沙、糯米", source_message_id="addition",
        relation_type="MERGE", parent_memory_id=old["id"], supersedes_memory_id=old["id"],
        evidence_json=json.dumps([next_source]))
    major = database.append_character_memory_claim(**fields, memory_type="user_fact",
        memory_key="user_major", content="用户的大学专业是海洋工程", source_message_id="original",
        evidence_json=json.dumps(["我的大学专业是海洋工程"]))
    foreign = database.append_character_memory_claim(**other, memory_type="user_fact",
        memory_key="fact_pet", content="另一个用户的猫叫绿豆")
    source = "请忘掉我的猫名字和新增猫的记录，保留大学专业。"
    output = json.dumps({"memories": [dict(kind="other_user_fact", operation="ERASE",
        value="", evidence=source, confidence=.99, target_memory_id=str(latest["id"]),
        target_memory_key="fact_pet", attributed_to="user")]}, ensure_ascii=False)
    class Completion:
        async def complete(self, messages):
            return output
        async def close(self):
            pass
    scheduler = MemoryEnrichmentScheduler(
        config=MemoryLlmConfig(enabled=True, base_url="http://test", model="frozen", idle_seconds=0),
        completion=Completion(),
        embedding_provider=SimpleNamespace(embed_texts=lambda texts: [[1., 0.] for _ in texts]))
    try:
        assert scheduler.schedule(repository=repo, character_id="role", user_scope=scope,
            message=source, rule_hints=[], source_message_id="erase")
        assert await scheduler.flush_memory(timeout=5)
        claims = await repo.list_memory_records("role", scope, limit=None, include_inactive=True)
        assert [record["id"] for record in claims] == [major["id"]]
        assert scheduler.status.erased == 1
        from character.memory_service import CharacterMemoryService

        memories, _ = await CharacterMemoryService(repo, semantic_enabled=False).load_relevant_memories(
            "role", scope, "我的大学专业是什么？", for_contextual_selection=True)
        assert len(memories) == 1 and "海洋工程" in memories[0].content
        assert all("豆沙" not in item.content and "糯米" not in item.content for item in memories)
        remaining_foreign = database.list_character_memory_claims(**other, include_inactive=True)
        assert [record["id"] for record in remaining_foreign] == [foreign["id"]]
        stored = database._get_connection().execute("SELECT body, state FROM memory_sources").fetchall()
        assert all(row["body"] is None and row["state"] == "revoked" for row in stored)
        links = database._get_connection().execute("SELECT memory_id FROM memory_source_links").fetchall()
        assert {row[0] for row in links} == {major["id"]}
        assert not database._get_connection().execute("SELECT * FROM memory_source_terms").fetchall()
    finally:
        await scheduler.shutdown(timeout=1)


def test_unknown_model_qualifier_is_error_without_quote_recovery():
    with pytest.raises(ValueError, match="qualifiers"):
        merge({"value": "芝麻", "content": "", "qualifiers": {"unknown": "x"}})
