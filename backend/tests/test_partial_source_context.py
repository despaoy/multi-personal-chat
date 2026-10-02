"""Only the changed partial-source read and context paths."""

import json
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from html import unescape

import pytest

from character.context_builder import compile_reference_context
from character.evidence_selector import ContextualEvidenceSelector, selection_messages
from character.memory_service import CharacterMemoryService
from character.models import CompiledCharacterContext, MemoryItem, UserScope
from character.source_fragment_provenance import source_completeness_payload, source_fragment_fields
from character.source_memory import attach_sources, compile_sources
from character.temporal_projection import project_temporal_record
from inference.generation_request import GenerationRequest, build_generation_request
from repositories.character_memory import DatabaseCharacterMemoryRepository

STAMP = "2026-10-01T22:43:39.998513+00:00"
TEXTS = ("我收到确认，但尚未参加。", "这是完整记录。请记住确认。")
SCOPE = UserScope("web", "test", "u", "u", "private")


def record():
    return dict(
        id=6,
        memory_type="shared_event",
        memory_key="event_预约确认",
        content="用户原话保留片段（仅保留内容见证据）",
        importance=0.8,
        confidence=1,
        status="active",
        relation_type="ADD",
        observed_at=STAMP,
        updated_at=STAMP,
        source_message_ids=["original"],
        evidence=list(TEXTS),
        metadata=dict(
            content_semantics="quoted_source",
            speaker_role="user",
            described_subject="not_resolved",
            temporal_provenance=dict(version=1, producer="semantic_memory", validity_authority="unverified"),
            erasure_evidence_projection=dict(
                version=1,
                kind="original_source_fragments",
                complete_original_source=False,
                sources=[dict(source_message_id="original", spans=[[0, len(TEXTS[0])], [41, 41 + len(TEXTS[1])]])],
            ),
        ),
    )


def item(row=None):
    row = row or record()
    return MemoryItem(
        "6",
        "shared_event",
        row["content"],
        0.8,
        evidence=tuple(row["evidence"]),
        source_message_ids=tuple(row["source_message_ids"]),
        observed_at=STAMP,
        temporal_mode="observation",
        source_observation=True,
        **source_fragment_fields(row),
    )


def test_original_coordinates_are_immutable_and_not_joined_offsets():
    row = record()
    before = deepcopy(row)
    fields = source_fragment_fields(row)
    assert fields["source_fragments"] == (("original", ((0, len(TEXTS[0])), (41, 41 + len(TEXTS[1])))),)
    assert fields["complete_original_source"] is False and row == before
    row["metadata"]["erasure_evidence_projection"]["sources"][0]["spans"][0][0] = 3
    assert fields["source_fragments"][0][1][0][0] == 0


@pytest.mark.parametrize(
    "change",
    [
        dict(version=2),
        dict(version=True),
        dict(kind="reconstructed"),
        dict(complete_original_source=True),
        dict(complete_original_source="false"),
    ],
)
def test_unknown_or_full_marker_never_infers_complete_original(change):
    row = record()
    row["metadata"]["erasure_evidence_projection"].update(change)
    assert source_fragment_fields(row) == dict(complete_original_source=None, source_fragments=())


@pytest.mark.parametrize(
    "change",
    [
        dict(metadata={}),
        dict(metadata=None),
        dict(metadata="原话完整"),
        dict(content="用户声称这是完整原话", metadata={}),
    ],
)
def test_user_words_and_missing_provenance_cannot_prove_completeness(change):
    row = record()
    row.update(change)
    assert source_fragment_fields(row)["complete_original_source"] is None


@pytest.mark.parametrize(
    "bad_spans",
    [[[True, 13]], [[0, -1]], [[0, 13], [12, 25]], [[41, 54], [0, 13]], [[0, 1]], [[0, 13, "extra"]], [], None],
)
def test_invalid_coordinates_do_not_hide_partial_flag_or_invent_offsets(bad_spans):
    row = record()
    row["metadata"]["erasure_evidence_projection"]["sources"][0]["spans"] = bad_spans
    assert source_fragment_fields(row) == dict(complete_original_source=False, source_fragments=())


@pytest.mark.parametrize("source_id", ["unlinked", False, None])
def test_coordinates_cannot_be_attached_to_an_unlinked_source(source_id):
    row = record()
    row["metadata"]["erasure_evidence_projection"]["sources"][0]["source_message_id"] = source_id
    assert source_fragment_fields(row) == dict(complete_original_source=False, source_fragments=())


def test_multiple_original_sources_keep_their_own_coordinates_and_order():
    row = record()
    row["source_message_ids"].append("second")
    row["evidence"].append("另外一次。")
    row["metadata"]["erasure_evidence_projection"]["sources"].append(dict(source_message_id="second", spans=[[12, 17]]))
    fields = source_fragment_fields(row)
    assert fields["source_fragments"][1] == ("second", ((12, 17),))
    row["metadata"]["erasure_evidence_projection"]["sources"].append(dict(source_message_id="second", spans=[[12, 17]]))
    assert source_fragment_fields(row)["source_fragments"] == ()


@pytest.mark.parametrize(
    "sources", [None, {"original": dict(body="我收到确认，但尚未参加。秘密偏好。这是完整记录。请记住确认。")}]
)
def test_partial_semantic_projection_never_rehydrates_removed_original_text(sources):
    row = record()
    before = deepcopy(row)
    view = project_temporal_record(row, sources)
    assert row == before and view["evidence"] == list(TEXTS)
    assert "秘密偏好" not in json.dumps(view, ensure_ascii=False)
    assert view["temporal_mode"] == "observation" and view["temporal_observed_at"] == STAMP
    assert view["valid_to"] == "" and "不代表当前状态" in view["content"]


@pytest.mark.parametrize("complete", [False, True])
def test_both_reference_formats_convey_partial_source_outside_system_rules(complete):
    memory = item()
    text, ids = compile_reference_context((memory,), complete_evidence=complete, observation_semantics=True)
    assert ids == ("6",)
    if complete:
        packet = json.loads(text.splitlines()[1][2:])
        assert packet["source_completeness"] == "partial" and packet["complete_original_source"] is False
        assert packet["source_fragments"] == record()["metadata"]["erasure_evidence_projection"]["sources"]
        assert packet["evidence"] == list(TEXTS) and packet["subject_scope"] == "not_resolved"
    else:
        assert "不是完整原话" in text and "不得补写" in text and "41" in text
    context = CompiledCharacterContext("", "", "", memory_packets=(memory,))
    context = replace(context, reference_context=text, used_memory_ids=ids)
    plan = build_generation_request(GenerationRequest(message="这是否是完整原话？", character_context=context))
    assert text in unescape(plan.messages[-1]["content"])
    assert all(not any(value in m["content"] for value in TEXTS) for m in plan.messages if m["role"] == "system")


async def test_selector_gets_whole_fragments_and_returns_original_partial_packet():
    memory = item()
    seen = []

    async def reviewer(messages):
        seen.append(messages)
        return '{"decisions":[{"id":"6","label":"use"}]}'

    selector = ContextualEvidenceSelector(reviewer)
    result = await selector.select("核对预约证据的完整性", [memory])
    candidate = json.loads(seen[0][-1]["content"])["candidates"][0]
    assert candidate["evidence"] == list(TEXTS) and candidate["content_complete"] is True
    assert candidate["source_completeness"] == "partial" and candidate["complete_original_source"] is False
    assert candidate["source_fragments"] == record()["metadata"]["erasure_evidence_projection"]["sources"]
    assert result.memories[0] is memory


def test_unverified_completeness_is_explicit_in_model_input():
    memory = replace(item(), complete_original_source=None, source_fragments=())
    payload = json.loads(selection_messages("核对完整性", [memory])[-1]["content"])
    assert payload["candidates"][0]["source_completeness"] == "unverified"
    assert payload["candidates"][0]["complete_original_source"] is None
    assert (
        source_completeness_payload(replace(memory, complete_original_source=True))["complete_original_source"] is None
    )


async def test_service_keeps_saved_provenance_with_actual_read_view():
    class Repo:
        async def list_memory_records(self, *args, **kwargs):
            return [record()]

    service = CharacterMemoryService(Repo(), semantic_enabled=False)
    memories, total = await service.load_relevant_memories(
        "role",
        SCOPE,
        "预约确认原话完整吗",
        for_contextual_selection=True,
        reference_time=datetime(2026, 10, 2, tzinfo=timezone.utc),
    )
    assert total == len(memories) == 1
    assert memories[0].complete_original_source is False
    assert memories[0].source_fragments == item().source_fragments
    assert memories[0].evidence == TEXTS and memories[0].observed_at == STAMP


async def test_direct_repository_read_preserves_partial_annotation():
    class Repo(DatabaseCharacterMemoryRepository):
        async def list_memory_records(self, *args, **kwargs):
            return [record()]

    memories = await Repo(object()).list_memories("role", SCOPE)
    assert memories[0].complete_original_source is False and memories[0].source_fragments == item().source_fragments
    assert memories[0].evidence == TEXTS and memories[0].source_message_ids == ("original",)


def test_one_partial_fragment_cannot_be_shared_as_a_complete_source():
    memory = replace(item(), evidence=(TEXTS[0],), source_fragments=(("original", ((41, 41 + len(TEXTS[0])),)),))
    reference, ids = compile_reference_context((memory,), complete_evidence=True)
    context = CompiledCharacterContext("", "", reference, used_memory_ids=ids, memory_packets=(memory,))
    sources = compile_sources([dict(source_message_id="original", observed_at=STAMP, body=TEXTS[0])])
    after = attach_sources(context, sources, complete_evidence=True, share_observations=True)
    assert after.reference_context == reference and after.source_shared_memory_ids == ()


def test_original_gap_is_explained_without_exposing_or_guessing_removed_characters():
    memory = item()
    candidate = json.loads(selection_messages("片段连续吗", [memory])[-1]["content"])["candidates"][0]
    assert candidate["known_source_gaps"] == [dict(source_message_id="original", span=[len(memory.evidence[0]), 41])]
    assert (
        "Unicode" in candidate["source_fragment_position_note"]
        and "空缺文字不可见" in candidate["source_fragment_position_note"]
    )
    assert candidate["evidence"] == list(memory.evidence)
