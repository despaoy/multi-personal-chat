"""Synthetic unit databases below are not native paid evaluation seeds."""

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from character.deferred_memory_mutation import deferred_source_start
from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig, parse_llm_proposals
from character.models import MemoryItem, UserScope
from db.database import SQLiteDB
from repositories.character_memory import DatabaseCharacterMemoryRepository

NOW = datetime(2026, 10, 3, 12, tzinfo=timezone.utc)
SCOPE = UserScope("web", "web-character", "unit-owner", "unit-owner", "private")
FUTURE = "从2027年1月1日起，我喜欢测试红茶。在此日期之前保持现有偏好不变。"


@pytest.mark.parametrize(
    "evidence,future",
    [
        ("从2027年1月1日起，我喜欢测试红茶。", True),
        ("记住：我从明天起喜欢测试红茶。", True),
        ("从2026-10-03T21:00:00+08:00起，我喜欢测试红茶。", True),
        ("从今天起，我喜欢测试红茶。", False),
        ("从昨天起，我喜欢测试红茶。", False),
        ("从明天起，他喜欢测试红茶。", False),
        ("从2027年2月30日起，我喜欢测试红茶。", False),
        ("我喜欢测试红茶，活动记录日期是2027年1月1日。", False),
    ],
)
def test_future_guard_uses_self_start_role_and_source_clock(evidence, future):
    assert (deferred_source_start(evidence, observed_at=NOW) is not None) == future


@pytest.mark.parametrize(
    "evidence,future",
    [
        ("我喜欢测试红茶，但从明天起，我只有预约确认才饮用测试红茶。", True),
        ("我喜欢测试红茶，不过我从明天起只有预约确认才饮用测试红茶。", True),
        ("我喜欢测试红茶，但从明天起，他只有预约确认才饮用测试红茶。", False),
    ],
)
def test_contrast_keeps_the_temporal_subject_and_condition_bound(evidence, future):
    assert (deferred_source_start(evidence, observed_at=NOW) is not None) == future


async def setup_writer(tmp_path, evidence, relation="SUPERSEDE", proposed_from=""):
    db = SQLiteDB(tmp_path / "future-unit.db")
    repo = DatabaseCharacterMemoryRepository(db)
    old = await repo.append_claim(
        "unit-character",
        SCOPE,
        MemoryItem(memory_id="", memory_type="user_fact", content="用户喜欢测试红茶", importance=0.6),
        memory_key="preference_测试红茶",
        observed_at="2026-10-02T12:00:00+00:00",
        evidence=("我喜欢测试红茶",),
    )
    old.pop("persisted", None)
    raw = dict(
        kind="like",
        value="测试红茶",
        content="用户原话：" + evidence,
        evidence=evidence,
        confidence=0.95,
        operation=relation,
        target_memory_id=str(old["id"]),
        target_memory_key=old["memory_key"],
        attributed_to="user",
        valid_from=proposed_from,
        qualifiers={"time": "从2027年1月1日起"} if evidence == FUTURE else {},
    )
    proposals = parse_llm_proposals(
        json.dumps(dict(memories=[raw]), ensure_ascii=False), source_message=evidence, existing_memories=(old,)
    )
    assert len(proposals) == 1
    scheduler = MemoryEnrichmentScheduler(config=MemoryLlmConfig(True, "unit-only", "unit-only"), completion=None)
    job = SimpleNamespace(
        repository=repo,
        character_id="unit-character",
        user_scope=SCOPE,
        observed_at=NOW,
        source_message_id=None,
        write_mode="hot",
        feedback_target_ids=(),
    )
    return db, repo, old, scheduler, job, proposals[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("relation", ["SUPERSEDE", "MERGE"])
async def test_future_relation_keeps_present_target_and_preserves_new_observation(tmp_path, relation):
    _, repo, old, scheduler, job, proposal = await setup_writer(tmp_path, FUTURE, relation)
    assert await scheduler._persist_proposal(job, proposal) == "saved"
    rows = await repo.list_memory_records("unit-character", SCOPE, limit=None, include_inactive=True)
    before = next(r for r in rows if r["id"] == old["id"])
    assert before == old, "future source must not close the currently active target at observation time"
    successor = next(r for r in rows if r["id"] != old["id"])
    assert successor["relation_type"] == "COEXIST" and successor["parent_memory_id"] == old["id"]
    assert successor["supersedes_memory_id"] is None and successor["evidence"] == [FUTURE]
    provenance = successor["metadata"]["deferred_mutation"]
    assert provenance["original_operation"] == relation and provenance["expression"] == "2027年1月1日"
    assert provenance["source_observed_at"] == NOW.isoformat()
    assert proposal.operation == relation and proposal.valid_from == ""


@pytest.mark.asyncio
async def test_present_source_still_replaces_target_without_trusting_model_future_endpoint(tmp_path):
    _, repo, old, scheduler, job, proposal = await setup_writer(
        tmp_path, "我喜欢测试红茶。", proposed_from="2027-01-01T00:00:00+08:00"
    )
    assert await scheduler._persist_proposal(job, proposal) == "saved"
    rows = await repo.list_memory_records("unit-character", SCOPE, limit=None, include_inactive=True)
    assert next(r for r in rows if r["id"] == old["id"])["status"] == "superseded"
    successor = next(r for r in rows if r["id"] != old["id"])
    assert successor["relation_type"] == "SUPERSEDE" and "deferred_mutation" not in successor["metadata"]


@pytest.mark.asyncio
async def test_current_assertion_in_same_job_is_not_dated_by_other_future_evidence(tmp_path):
    _, repo, old, scheduler, job, proposal = await setup_writer(tmp_path, "我喜欢测试红茶。")
    job.message = FUTURE + "我喜欢测试红茶。"
    assert await scheduler._persist_proposal(job, proposal) == "saved"
    rows = await repo.list_memory_records("unit-character", SCOPE, limit=None, include_inactive=True)
    assert next(r for r in rows if r["id"] == old["id"])["status"] == "superseded"


@pytest.mark.asyncio
async def test_legacy_writer_cannot_overwrite_target_when_it_cannot_store_coexisting_version(tmp_path):
    _, _, _, scheduler, job, proposal = await setup_writer(tmp_path, FUTURE)

    class Legacy:
        async def add_or_update_memory(self, *args, **kwargs):
            pytest.fail("legacy overwrite cannot preserve the present target and future source together")

    job.repository = Legacy()
    assert await scheduler._persist_proposal(job, proposal) == "skipped"
