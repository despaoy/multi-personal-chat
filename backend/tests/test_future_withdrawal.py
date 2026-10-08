"""Temporary synthetic unit DBs are not paid native evaluation memory seeds."""

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig, parse_llm_proposals
from character.models import MemoryItem, UserScope
from character.temporal_projection import project_temporal_record
from db.database import SQLiteDB
from repositories.character_memory import DatabaseCharacterMemoryRepository

NOW = datetime(2026, 10, 3, 15, tzinfo=timezone.utc)
SCOPE = UserScope("web", "web-character", "unit-owner", "unit-owner", "private")
FUTURE_RETRACT = "从2027年2月1日起，我撤回此前的测试绿茶偏好。这是未来撤回安排。"
FUTURE_ERASE = "从2027年2月1日起，我要求从记忆中删除测试绿茶偏好。这是指定未来日期的遗忘请求。"


async def writer(tmp_path, operation, source, with_qualifier=True):
    db = SQLiteDB(tmp_path / "withdrawal-unit.db")
    repo = DatabaseCharacterMemoryRepository(db)
    old = await repo.append_claim(
        "unit-character",
        SCOPE,
        MemoryItem(memory_id="", memory_type="user_fact", content="用户喜欢测试绿茶", importance=0.6),
        memory_key="preference_测试绿茶",
        observed_at="2026-10-02T12:00:00+00:00",
        evidence=("我喜欢测试绿茶",),
    )
    old.pop("persisted", None)
    source_id = "unit-withdrawal-source"
    await repo.capture_source("unit-character", SCOPE, source_message_id=source_id, body=source, observed_at=NOW)
    raw = dict(
        kind="like",
        value="测试绿茶",
        content="",
        evidence=source,
        confidence=0.95,
        operation=operation,
        target_memory_id=str(old["id"]),
        target_memory_key=old["memory_key"],
        attributed_to="user",
        qualifiers={"time": "从2027年2月1日起"} if with_qualifier else {},
        valid_to="2027-02-01T00:00:00+08:00",
        observed_at="2020-01-01T00:00:00+00:00",
    )
    proposals = parse_llm_proposals(
        json.dumps(dict(memories=[raw]), ensure_ascii=False), source_message=source, existing_memories=(old,)
    )
    assert len(proposals) == 1
    scheduler = MemoryEnrichmentScheduler(config=MemoryLlmConfig(True, "unit-only", "unit-only"), completion=None)
    job = SimpleNamespace(
        repository=repo,
        character_id="unit-character",
        user_scope=SCOPE,
        observed_at=NOW,
        source_message_id=source_id,
        write_mode="hot",
        feedback_target_ids=(),
    )
    return repo, old, scheduler, job, proposals[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation,source,qualifier",
    [
        ("RETRACT", FUTURE_RETRACT, True),
        ("RETRACT", FUTURE_RETRACT, False),
        ("ERASE", FUTURE_ERASE, True),
        ("ERASE", FUTURE_ERASE, False),
    ],
)
async def test_future_withdrawal_does_not_retract_or_erase_present_claim(tmp_path, operation, source, qualifier):
    repo, old, scheduler, job, proposal = await writer(tmp_path, operation, source, qualifier)
    assert await scheduler._persist_proposal(job, proposal) == "saved"
    rows = await repo.list_memory_records("unit-character", SCOPE, limit=None, include_inactive=True)
    assert next((r for r in rows if r["id"] == old["id"]), None) == old, (
        "future withdrawal must preserve the active present claim"
    )
    new = next(r for r in rows if r["id"] != old["id"])
    assert (
        new["relation_type"] == "COEXIST"
        and new["parent_memory_id"] == old["id"]
        and new["supersedes_memory_id"] is None
    )
    assert new["metadata"]["deferred_mutation"]["original_operation"] == operation
    assert new["metadata"]["deferred_mutation"]["source_observed_at"] == NOW.isoformat()
    sources = await repo.linked_source_receipts(
        "unit-character", SCOPE, claim_sources=((new["id"], job.source_message_id),)
    )
    view = project_temporal_record(new, {r["source_message_id"]: r for r in sources})
    assert view["temporal_mode"] == "observation" and source in view["content"]
    assert scheduler.status.erased == 0 and scheduler.status.saved == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source",
    [
        "我现在撤回此前的测试绿茶偏好。",
        "我撤回此前那个从2027年2月1日起的测试绿茶偏好安排。",
    ],
)
async def test_present_retraction_is_not_postponed_by_model_endpoints_or_a_quoted_future_arrangement(tmp_path, source):
    repo, old, scheduler, job, proposal = await writer(tmp_path, "RETRACT", source, False)
    assert await scheduler._persist_proposal(job, proposal) == "saved"
    rows = await repo.list_memory_records("unit-character", SCOPE, limit=None, include_inactive=True)
    assert next(r for r in rows if r["id"] == old["id"])["status"] == "retracted"
    assert all("deferred_mutation" not in r["metadata"] for r in rows)


@pytest.mark.asyncio
async def test_present_erasure_is_not_postponed_by_a_model_guessed_future_endpoint(tmp_path):
    repo, old, scheduler, job, proposal = await writer(tmp_path, "ERASE", "我要求从记忆中删除测试绿茶偏好。", False)
    assert await scheduler._persist_proposal(job, proposal) == "erased"
    assert not await repo.list_memory_records("unit-character", SCOPE, limit=None, include_inactive=True)
    assert scheduler.status.erased == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("operation,source", [("RETRACT", FUTURE_RETRACT), ("ERASE", FUTURE_ERASE)])
async def test_legacy_adapter_cannot_execute_future_withdrawal_now(tmp_path, operation, source):
    _, _, scheduler, job, proposal = await writer(tmp_path, operation, source)

    class Legacy:
        async def delete_memory(self, *args, **kwargs):
            pytest.fail("future withdrawal cannot delete current claim")

        async def erase_memory(self, *args, **kwargs):
            pytest.fail("future erasure cannot run at observation time")

        async def add_or_update_memory(self, *args, **kwargs):
            pytest.fail("legacy overwrite cannot retain both observations")

    job.repository = Legacy()
    with pytest.raises(AttributeError, match="append_claim"):
        await scheduler._persist_proposal(job, proposal)
