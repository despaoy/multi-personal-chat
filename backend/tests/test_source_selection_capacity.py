"""Serving-budget source admission preserves complete ranked source candidates."""

import json
from datetime import datetime, timezone

import pytest

from character.models import UserScope
from character.source_memory import SourceMemoryService, compile_sources, select_sources

SCOPE = UserScope("web", "source-capacity", "owner", "owner", "private")


def rows(count=6):
    return [
        dict(
            source_message_id=f"source-{i}",
            observed_at=datetime.now(timezone.utc).isoformat(),
            body=f"DL752-R{i}: 完整交接记录第{i}段，数值为{i + 19}。限制条件：末尾条件{i}不能省略。",
        )
        for i in range(count)
    ]


class Repository:
    def __init__(self, candidates, *, removed=(), linked=(), contextual=()):
        self.candidates = candidates
        self.removed = set(removed)
        self.linked = linked
        self.contextual = contextual
        self.read_batches = []
        self.window_batches = []

    async def linked_sources(self, *a, **kw):
        return self.linked

    async def search_sources(self, *a, query, limit):
        return self.contextual if query == "loaded-history" else self.candidates

    async def list_sources(self, *a, source_message_ids):
        self.read_batches.append(source_message_ids)
        assert len(source_message_ids) <= 100
        by_id = {r["source_message_id"]: r for r in [*self.candidates, *self.linked, *self.contextual]}
        return [by_id[sid] for sid in source_message_ids if sid not in self.removed]

    async def source_windows(self, *a, source_message_ids, radius):
        self.window_batches.append(source_message_ids)
        assert len(source_message_ids) <= 4 and radius == 1
        by_id = {r["source_message_id"]: r for r in self.candidates}
        return [dict(anchor_id=sid, rows=[by_id[sid]]) for sid in source_message_ids]


async def test_cloud_serving_budget_retains_all_six_original_sources():
    original = rows()
    repo = Repository(original)
    result = await SourceMemoryService(repo, max_chars=32768, defer_budget=True).recall(
        "role", SCOPE, "DL752 全部六段交接记录"
    )
    packet = json.loads(result.context)
    assert {(r["source_id"], r["text"]) for r in packet["records"]} == {
        (r["source_message_id"], r["body"]) for r in original
    }
    assert result.diagnostics["selection_omitted"] == 0 and not result.candidate_context
    assert len(repo.read_batches) == 1 and len(repo.read_batches[0]) == 6
    assert packet["current_validity"] == packet["described_subject"] == "not_resolved"


async def test_legacy_fixed_source_count_and_explicit_compiler_caps_remain():
    original = rows()
    result = await SourceMemoryService(Repository(original), max_chars=32768).recall("role", SCOPE, "DL752")
    assert len(json.loads(result.context)["records"]) == 4 and result.diagnostics["selection_omitted"] == 2
    assert len(json.loads(compile_sources(original, max_chars=32768).context)["records"]) == 4
    assert len(json.loads(compile_sources(original, max_chars=32768, max_items=None).context)["records"]) == 6
    assert compile_sources(original, max_chars=32768, max_items=0).context == ""


async def test_deferred_whole_packet_keeps_six_and_never_uses_smaller_substitute():
    original = rows()
    repo = Repository(original)
    result = await SourceMemoryService(repo, max_chars=300, defer_budget=True).recall("role", SCOPE, "DL752")
    assert result.context == "" and result.diagnostics["status"] == "budget_omitted"
    packet = json.loads(result.candidate_context)
    assert len(packet["records"]) == 6
    assert {r["text"] for r in packet["records"]} == {r["body"] for r in original}


async def test_fresh_revocation_still_excludes_selected_source():
    original = rows()
    repo = Repository(original, removed={"source-0", "source-5"})
    result = await SourceMemoryService(repo, max_chars=32768, defer_budget=True).recall("role", SCOPE, "DL752")
    assert {r["source_id"] for r in json.loads(result.context)["records"]} == {
        "source-1",
        "source-2",
        "source-3",
        "source-4",
    }
    assert result.diagnostics["fresh_recheck_omitted"] == 2


async def test_cloud_source_windows_use_original_four_anchor_batches():
    original = rows(9)
    repo = Repository(original)
    result = await SourceMemoryService(repo, max_chars=32768, window_radius=1, defer_budget=True).recall(
        "role", SCOPE, "DL752"
    )
    assert len(json.loads(result.context)["records"]) == 9
    assert [len(x) for x in repo.window_batches] == [4, 4, 1]
    assert result.diagnostics["window_semantic_relation"] == "not_inferred"


async def test_union_over_read_limit_preserves_all_sources_in_bounded_fresh_batches():
    original = rows(100)
    current = rows(32)
    contextual = rows(32)
    current = [r | {"source_message_id": f"current-{i}"} for i, r in enumerate(current)]
    contextual = [r | {"source_message_id": f"history-{i}"} for i, r in enumerate(contextual)]
    repo = Repository(current, linked=original, contextual=contextual)
    result = await SourceMemoryService(repo, max_chars=100000, defer_budget=True).recall(
        "role", SCOPE, "DL752", retrieval_context="loaded-history"
    )
    assert len(json.loads(result.context)["records"]) == 164
    assert [len(x) for x in repo.read_batches] == [100, 64]
    assert result.diagnostics["selection_omitted"] == result.diagnostics["fresh_recheck_omitted"] == 0


def test_ranking_keeps_each_lane_leader_and_object_identity_with_no_item_cap():
    original = rows(6)
    selected, count = select_sources(original[:2], original[2:4], contextual=original[4:], limit=None)
    assert count == len(selected) == 6
    assert [x["source_message_id"] for x in selected[:3]] == ["source-0", "source-2", "source-4"]
    assert all(any(actual is expected for expected in original) for actual in selected)


@pytest.mark.parametrize("window,expected_count", [(65536, 6), (4096, 0)])
async def test_whole_six_source_deferred_packet_obeys_actual_final_window(window, expected_count):
    import re
    from html import unescape

    from character.models import CompiledCharacterContext
    from character.source_memory import attach_sources
    from inference.generation_request import GenerationRequest, build_generation_request

    original = [r | {"body": r["body"] + "完整限定资料。" * 330 + f"最后限定编号{i}。"} for i, r in enumerate(rows())]
    result = await SourceMemoryService(Repository(original), max_chars=300, defer_budget=True).recall(
        "role", SCOPE, "DL752"
    )
    assert not result.context and len(json.loads(result.candidate_context)["records"]) == 6
    ctx = attach_sources(
        CompiledCharacterContext(
            "角色", "", "", memory_status="no_match", memory_field_presence=(("workplace", False),)
        ),
        result,
    )
    plan = build_generation_request(
        GenerationRequest(
            message="逐字列出所有提供的原话", character_context=ctx, context_window_tokens=window, max_tokens=1024
        )
    )
    match = re.search(
        r"<dialogue_evidence[^>]*>\n(.*?)\n</dialogue_evidence>", unescape(plan.messages[-1]["content"]), re.S
    )
    if expected_count:
        records = json.loads(match[1])["records"]
        assert len(records) == expected_count
        assert {r["text"] for r in records} == {r["body"] for r in original}
    else:
        assert match is None and plan.character_context.memory_source_status == "budget_omitted"
        assert plan.character_context.memory_field_presence == (("workplace", None),)
    assert plan.character_context.memory_packets == ()
    assert all(r["body"] not in m["content"] for r in original for m in plan.messages if m["role"] == "system")


async def test_actual_sqlite_cloud_windows_cover_six_anchors_and_exclude_other_owner(tmp_path):
    from datetime import timedelta

    from db.database import SQLiteDB
    from repositories.character_memory import DatabaseCharacterMemoryRepository

    db = SQLiteDB(tmp_path / "sources.sqlite")
    repo = DatabaseCharacterMemoryRepository(db)
    original = rows()
    now = datetime.now(timezone.utc) - timedelta(minutes=1)
    for i, r in enumerate(original):
        assert (
            await repo.capture_source(
                "role",
                SCOPE,
                source_message_id=r["source_message_id"],
                body=r["body"],
                observed_at=now + timedelta(seconds=i),
            )
            == "recorded"
        )
    foreign = UserScope("web", "source-capacity", "other", "other", "private")
    assert (
        await repo.capture_source(
            "role", foreign, source_message_id="foreign", body="DL752: 其他用户的限制内容不能注入。", observed_at=now
        )
        == "recorded"
    )
    result = await SourceMemoryService(repo, max_chars=32768, window_radius=1, defer_budget=True).recall(
        "role", SCOPE, "DL752 全部交接记录"
    )
    records = json.loads(result.context)["records"]
    assert len(records) == 6 and {r["text"] for r in records} == {r["body"] for r in original}
    assert len(result.diagnostics["anchor_ids"]) == 6 and result.diagnostics["selection_omitted"] == 0
    assert result.diagnostics["window_semantic_relation"] == "not_inferred"
    assert await repo.list_memory_records("role", SCOPE) == []
