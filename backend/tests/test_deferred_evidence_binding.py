"""Complete synthetic unit sources; these are not native paid memory seeds."""

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig, parse_llm_proposals
from character.models import MemoryItem, UserScope
from character.temporal_projection import project_temporal_record
from db.database import SQLiteDB
from repositories.character_memory import DatabaseCharacterMemoryRepository

NOW = datetime(2026, 10, 3, 16, tzinfo=timezone.utc)
SCOPE = UserScope("web", "unit-binding", "unit-owner", "unit-owner", "private")
RETRACT = "我撤回此前的测试紫茶偏好。"
LIKE = "我喜欢测试紫茶且只有预约确认通过才饮用测试紫茶。"


async def setup_writer(tmp_path, operation, source, evidence):
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / "evidence-binding.sqlite"))
    old = await repo.append_claim(
        "unit-character",
        SCOPE,
        MemoryItem(memory_id="", memory_type="user_fact", content="用户喜欢测试紫茶", importance=0.6),
        memory_key="preference_测试紫茶",
        observed_at="2026-10-02T12:00:00+00:00",
        evidence=("我喜欢测试紫茶。",),
    )
    old.pop("persisted", None)
    source_id = "binding-source"
    await repo.capture_source("unit-character", SCOPE, source_message_id=source_id, body=source, observed_at=NOW)
    raw = dict(
        kind="like",
        value="测试紫茶",
        content="",
        evidence=evidence,
        confidence=0.95,
        operation=operation,
        target_memory_id=str(old["id"]),
        target_memory_key=old["memory_key"],
        attributed_to="user",
        qualifiers={"condition": "预约确认通过"} if operation in {"SUPERSEDE", "MERGE"} else {},
        observed_at="2000-01-01T00:00:00+00:00",
    )
    proposals = parse_llm_proposals(
        json.dumps(dict(memories=[raw]), ensure_ascii=False), source_message=source, existing_memories=(old,)
    )
    assert len(proposals) == 1 and proposals[0].operation == operation
    job = SimpleNamespace(
        repository=repo,
        character_id="unit-character",
        user_scope=SCOPE,
        message=source,
        observed_at=NOW,
        source_message_id=source_id,
        write_mode="hot",
        feedback_target_ids=(),
    )
    scheduler = MemoryEnrichmentScheduler(config=MemoryLlmConfig(True, "unit-only", "unit-only"), completion=None)
    return repo, old, scheduler, job, proposals[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation,evidence", [("RETRACT", RETRACT), ("SUPERSEDE", LIKE), ("MERGE", LIKE)])
async def test_omitted_literal_start_cannot_retire_the_present_target(tmp_path, operation, evidence):
    source = "从2027年3月1日起，" + evidence + "该日期之前当前偏好继续有效。"
    repo, old, scheduler, job, proposal = await setup_writer(tmp_path, operation, source, evidence)
    assert "2027" not in proposal.evidence and proposal.evidence in source
    assert await scheduler._persist_proposal(job, proposal) == "saved"
    rows = await repo.list_memory_records("unit-character", SCOPE, include_inactive=True, limit=None)
    assert next(r for r in rows if r["id"] == old["id"]) == old
    new = next(r for r in rows if r["id"] != old["id"])
    assert (
        new["relation_type"] == "COEXIST"
        and new["parent_memory_id"] == old["id"]
        and new["supersedes_memory_id"] is None
    )
    assert new["metadata"]["deferred_mutation"]["expression"] == "2027年3月1日"
    assert new["metadata"]["deferred_mutation"]["source_observed_at"] == NOW.isoformat()
    assert new["evidence"] == [evidence]
    sources = await repo.linked_source_receipts(
        "unit-character", SCOPE, claim_sources=((new["id"], job.source_message_id),)
    )
    view = project_temporal_record(new, {r["source_message_id"]: r for r in sources})
    assert view["temporal_mode"] == "observation" and source in view["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source",
    [
        "从2027年3月1日起，我喜欢测试红茶，" + RETRACT,
        "从2027年3月1日起，我喜欢测试红茶。" + RETRACT,
        "我撤回此前那个从2027年3月1日起的测试紫茶偏好安排。",
    ],
)
async def test_other_future_clause_does_not_postpone_current_retraction(tmp_path, source):
    evidence = source if source.startswith("我撤回此前那个") else RETRACT
    repo, old, scheduler, job, proposal = await setup_writer(tmp_path, "RETRACT", source, evidence)
    assert await scheduler._persist_proposal(job, proposal) == "saved"
    rows = await repo.list_memory_records("unit-character", SCOPE, include_inactive=True, limit=None)
    assert next(r for r in rows if r["id"] == old["id"])["status"] == "retracted"
    assert all("deferred_mutation" not in r["metadata"] for r in rows)


@pytest.mark.asyncio
async def test_repeated_evidence_does_not_guess_future_over_explicit_current_statement(tmp_path):
    source = "从2027年3月1日起，" + LIKE + "今天" + LIKE
    repo, old, scheduler, job, proposal = await setup_writer(tmp_path, "SUPERSEDE", source, LIKE)
    assert await scheduler._persist_proposal(job, proposal) == "saved"
    rows = await repo.list_memory_records("unit-character", SCOPE, include_inactive=True, limit=None)
    assert next(r for r in rows if r["id"] == old["id"])["status"] == "superseded"
    assert all("deferred_mutation" not in r["metadata"] for r in rows)


@pytest.mark.parametrize(
    "source,evidence,future",
    [
        ("从2027年3月1日起，我撤回偏好。", "我撤回偏好。", True),
        ("从明天起我撤回偏好。", "我撤回偏好。", True),
        ("我从2027年3月1日起撤回偏好。", "撤回偏好。", True),
        ("从2027-03-01T00:00:00+08:00起，我撤回偏好。", "我撤回偏好。", True),
        ("从2027年3月1日起，我撤回“紫茶”偏好。", "我撤回“紫茶”偏好。", True),
        ("从2027年3月1日起，他撤回偏好。", "他撤回偏好。", False),
        ("从今天起，我撤回偏好。", "我撤回偏好。", False),
        ("从未知日期起，我撤回偏好。", "我撤回偏好。", False),
        ("“从2027年3月1日起，我撤回偏好。”", "我撤回偏好。", False),
        ("从2027年3月1日起，我喜欢红茶，我撤回偏好。", "我撤回偏好。", False),
        ("从2027年3月1日起，我撤回偏好。今天我撤回偏好。", "我撤回偏好。", False),
        ("从2027年3月1日起，我撤回偏好。", "我撤回别的偏好。", False),
        ("从2027年3月1日起，我撤回“紫茶偏好。", "我撤回“紫茶偏好。", False),
    ],
)
def test_binding_needs_unique_literal_evidence_inside_its_own_future_self_clause(source, evidence, future):
    from character.deferred_memory_mutation import deferred_evidence_start

    assert (deferred_evidence_start(evidence, source, observed_at=NOW) is not None) == future


@pytest.mark.asyncio
async def test_legacy_adapter_cannot_overwrite_after_date_omission(tmp_path):
    source = "从2027年3月1日起，" + RETRACT
    _, _, scheduler, job, proposal = await setup_writer(tmp_path, "RETRACT", source, RETRACT)

    class Legacy:
        async def delete_memory(self, *args, **kwargs):
            pytest.fail("omitted date cannot authorize current deletion")

    job.repository = Legacy()
    assert await scheduler._persist_proposal(job, proposal) == "skipped"


@pytest.mark.asyncio
async def test_future_erasure_short_quote_remains_deferred_among_other_current_erasure(tmp_path):
    evidence = "我要求从记忆中删除测试紫茶偏好。"
    source = "从2027年3月1日起，" + evidence + "现在我要求从记忆中删除测试红茶偏好。"
    repo, old, scheduler, job, proposal = await setup_writer(tmp_path, "ERASE", source, evidence)
    assert await scheduler._persist_proposal(job, proposal) == "saved"
    rows = await repo.list_memory_records("unit-character", SCOPE, include_inactive=True, limit=None)
    assert next(r for r in rows if r["id"] == old["id"]) == old
    new = next(r for r in rows if r["id"] != old["id"])
    assert new["relation_type"] == "COEXIST" and new["metadata"]["deferred_mutation"]["original_operation"] == "ERASE"


@pytest.mark.asyncio
async def test_current_erasure_short_quote_does_not_borrow_another_future_preference_date(tmp_path):
    evidence = "我要求从记忆中删除测试紫茶偏好。"
    source = "从2027年3月1日起，我喜欢测试红茶。" + evidence
    repo, _, scheduler, job, proposal = await setup_writer(tmp_path, "ERASE", source, evidence)
    assert await scheduler._persist_proposal(job, proposal) == "erased"
    assert not await repo.list_memory_records("unit-character", SCOPE, include_inactive=True, limit=None)
