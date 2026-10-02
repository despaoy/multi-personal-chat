import asyncio
import copy
import json
from pathlib import Path

import pytest

from character.erasure_authority import partial_erasure_plan
from character.memory_llm import (
    MemoryEnrichmentScheduler,
    MemoryLlmConfig,
    _search_existing_memories,
    build_memory_llm_messages,
    parse_llm_proposals,
)
from character.profile_registry import CharacterProfileRegistry
from character.quoted_erasure_authority import masked_quotes, partial_source_packet
from db.database import SQLiteDB
from repositories.character_memory import DatabaseCharacterMemoryRepository
from repositories.messages import DatabaseMessageRepository
from services.character_context import CharacterContextService, TurnInput

FIXTURE = json.loads((Path(__file__).parent / "fixtures/deepseek_memory_fragment_lifecycle_cases.json").read_text())
SOURCE = FIXTURE["cases"][-1]["message"]


def records():
    return copy.deepcopy(FIXTURE["actual_prior_records"])


def response(target="8", evidence=None):
    row = next(x for x in records() if str(x["id"]) == target)
    return json.dumps(
        dict(
            memories=[
                dict(
                    kind="shared_event",
                    value=row["memory_key"],
                    content="",
                    evidence=evidence or FIXTURE["affirmative_erase_clause"],
                    confidence=0.99,
                    operation="ERASE",
                    target_memory_id=target,
                    target_memory_key=row["memory_key"],
                    attributed_to="user",
                )
            ]
        )
    )


def test_actual_prior_fragments_identify_delete_and_keep_from_original_source_prefix():
    plan = partial_erasure_plan(SOURCE, records())
    assert plan.valid and not plan.unresolved_protection and plan.allowed_ids == ("8",) and plan.protected_ids == ("6",)
    parsed = parse_llm_proposals(response(), source_message=SOURCE, existing_memories=records())
    assert (
        len(parsed) == 1
        and parsed[0].target_memory_id == "8"
        and parsed[0].protected_memory_keys == (FIXTURE["retained_memory_key"],)
    )
    assert not parse_llm_proposals(response("6"), source_message=SOURCE, existing_memories=records())


@pytest.mark.parametrize("opening,closing", [("“", "”"), ("‘", "’"), ("「", "」"), ("『", "』"), ('"', '"')])
def test_supported_quote_pairs_keep_current_literal_erasure_evidence_and_original_anchor(opening, closing):
    message = SOURCE.replace("“" + FIXTURE["erase_anchor"] + "”", opening + FIXTURE["erase_anchor"] + closing)
    assert partial_erasure_plan(message, records()).allowed_ids == ("8",)
    masked, quotes = masked_quotes(message)
    assert FIXTURE["erase_anchor"] not in masked and quotes[0][2] == FIXTURE["erase_anchor"]


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.pop("metadata_json"),
        lambda r: r.update(observed_at="2026-10-01T23:44:31"),
        lambda r: r.update(memory_type="user_fact"),
        lambda r: r.update(source_message_id="different"),
        lambda r: r.update(evidence_json='["clipped"]'),
        lambda r: r.update(status="retracted"),
    ],
)
def test_unproved_or_changed_fragment_provenance_does_not_authorize_a_target(change):
    current = records()
    later = next(x for x in current if x["id"] == 8)
    change(later)
    assert partial_source_packet(later) is None
    assert partial_erasure_plan(SOURCE, current).allowed_ids == ()
    assert not parse_llm_proposals(response(), source_message=SOURCE, existing_memories=current)


@pytest.mark.parametrize(
    "anchor", ["我本人", "海庭鹤林工坊", "我本人今天再次核对了不存在的课程。", FIXTURE["erase_anchor"][:-1]]
)
def test_short_partial_or_unknown_prefix_does_not_supply_erasure_authority(anchor):
    message = SOURCE.replace(FIXTURE["erase_anchor"], anchor)
    assert partial_erasure_plan(message, records()).allowed_ids == ()


class NoEmbedding:
    def embed_texts(self, *args):
        raise AssertionError("Literal destructive source binding cannot use ranking")


def test_all_visible_duplicate_prefixes_remain_ambiguous_before_top_k():
    current = records()
    for identity in range(20, 32):
        row = copy.deepcopy(current[1])
        row["id"] = identity
        row["memory_key"] = "source_fragment_" + str(identity)
        current.append(row)
    assert _search_existing_memories(tuple(current), SOURCE, (), (), NoEmbedding()) == ()
    assert partial_erasure_plan(SOURCE, current).unresolved_protection


def test_actual_fragment_authority_is_selected_before_unrelated_ranking_and_writer_receives_evidence():
    current = records()
    selected = _search_existing_memories(tuple(current), SOURCE, (), (), NoEmbedding())
    assert {x["id"] for x in selected} == {6, 8}
    wire = build_memory_llm_messages(SOURCE, (), (), selected, 10000, 0.8)
    payload = json.loads(wire[-1]["content"])
    assert payload["partial_erasure_authorization"]["allowed_erase_memory_ids"] == ["8"]
    assert payload["current_user_message"] == SOURCE
    assert {row["memory_key"] for row in payload["existing_memories"]} == {row["memory_key"] for row in current}
    for row in payload["existing_memories"]:
        original = next(x for x in current if str(x["id"]) == row["memory_id"])
        packet = row["source_observation"]
        assert (
            packet["evidence"] == json.loads(original["evidence_json"])
            and packet["observed_at"] == original["observed_at"]
        )
        assert packet["complete_original_source"] is False and packet["described_subject"] == "not_resolved"
        assert "深蓝色" not in json.dumps(packet, ensure_ascii=False)


class Writer:
    def __init__(self):
        self.calls = []

    async def complete(self, messages):
        self.calls.append(messages)
        payload = json.loads(messages[-1]["content"])
        target = next(x for x in payload["existing_memories"] if x["memory_key"] == FIXTURE["erased_memory_key"])
        raw = json.loads(response())
        raw["memories"][0]["target_memory_id"] = target["memory_id"]
        return json.dumps(raw)

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_actual_completion_routes_quoted_erasure_to_writer_and_preserves_original_course(tmp_path, monkeypatch):
    import character.memory_llm as memory_llm

    database = SQLiteDB(tmp_path / "lifecycle.sqlite")
    repo = DatabaseCharacterMemoryRepository(database)
    # Actual saved prior rows are imported as isolated fixture data, with their
    # genuine clocks/fragments. No source body or fake new receipt is invented.
    fields = ("tsukiyashiro_kisaki", "web", "lifecycle-test", "owner", "private", "owner")
    try:
        for row in records():
            database.append_character_memory_claim(
                *fields,
                row["memory_type"],
                row["memory_key"],
                row["content"],
                source_message_id=row["source_message_id"],
                source_message_ids_json=row["source_message_ids_json"],
                evidence_json=row["evidence_json"],
                metadata_json=row["metadata_json"],
                observed_at=row["observed_at"],
            )
        registry = CharacterProfileRegistry()
        registry.load_profiles()
        service = CharacterContextService(registry, repo, DatabaseMessageRepository(database))
        writer = Writer()
        scheduler = MemoryEnrichmentScheduler(
            config=MemoryLlmConfig(True, "unused", "unused"), completion=writer, embedding_provider=NoEmbedding()
        )
        monkeypatch.setattr(memory_llm, "get_memory_enrichment_scheduler", lambda: scheduler)
        turn = TurnInput(SOURCE, "web", "lifecycle-test", "owner", "owner", "private")
        prepared = await service.prepare_turn(turn, "tsukiyashiro_kisaki")
        receipt = asyncio.get_running_loop().create_future()
        outcome = await service.complete_turn(
            prepared, turn, "收到。", source_message_id="new-lifecycle-erase", memory_receipt=receipt
        )
        assert await scheduler.flush_memory(timeout=4)
        assert outcome.memory_enrichment_scheduled and len(writer.calls) == 1
        assert (
            receipt.result()["status"] == "erased"
            and receipt.result()["accepted"] == receipt.result()["persisted"] == 1
        )
        current = await repo.list_memory_records(prepared.character_id, prepared.user_scope)
        assert len(current) == 1 and current[0]["memory_key"] == FIXTURE["retained_memory_key"]
        assert current[0]["evidence"] == json.loads(records()[0]["evidence_json"])
        assert not await repo.list_sources(prepared.character_id, prepared.user_scope)
    finally:
        if "scheduler" in locals():
            await scheduler.shutdown(timeout=4)
        database.close_connection()


@pytest.mark.parametrize("prefix", ["如果", "朋友说："])
def test_source_prefix_selector_is_not_authorized_by_hypothesis_or_third_party(prefix):
    assert not parse_llm_proposals(response(), source_message=prefix + SOURCE, existing_memories=records())


def test_unknown_retained_object_keeps_literal_delete_selector_unresolved():
    message = SOURCE.replace(
        "请保留我本人最初收到的海庭鹤林工坊纸版压印课预约确认记录", "请保留我本人不存在的其他课程记录"
    )
    assert partial_erasure_plan(message, records()).unresolved_protection
    assert not parse_llm_proposals(response(), source_message=message, existing_memories=records())


def test_model_cannot_redirect_a_valid_id_using_its_truncated_key():
    raw = json.loads(response())
    raw["memories"][0]["target_memory_key"] = FIXTURE["erased_memory_key"][:60]
    assert not parse_llm_proposals(json.dumps(raw), source_message=SOURCE, existing_memories=records())


def test_nested_quoted_delete_instruction_is_not_current_authorization():
    message = (
        "我只是引用一条历史请求：“"
        + FIXTURE["affirmative_erase_clause"].replace("“", "「").replace("”", "」")
        + "”请保留海庭鹤林工坊纸版压印课预约确认。"
    )
    assert not parse_llm_proposals(response(), source_message=message, existing_memories=records())


@pytest.mark.parametrize("opening,closing", [("“", "”"), ("‘", "’"), ("「", "」"), ("『", "』"), ('"', '"')])
def test_all_supported_outer_quotes_do_not_authorize_a_historical_erasure(opening, closing):
    from character.memory_llm import is_memory_erasure_request

    message = "这只是我本人以前的一条请求：" + opening + FIXTURE["affirmative_erase_clause"] + closing
    assert not is_memory_erasure_request(message)
    assert partial_erasure_plan(message, records()).valid is False
    assert not parse_llm_proposals(response(), source_message=message, existing_memories=records())
