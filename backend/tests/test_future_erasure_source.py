"""Synthetic unit repositories, independent of native evaluation seeds."""

import json
from dataclasses import replace
from datetime import timedelta

import pytest
from test_future_withdrawal import FUTURE_ERASE, NOW, SCOPE, writer

from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig
from db.database import SQLiteDB
from repositories.character_memory import DatabaseCharacterMemoryRepository


@pytest.mark.asyncio
@pytest.mark.parametrize("response", ['{"memories":[]}', "invalid JSON", '{"memories":[],"erase_source_ids":["old"]}'])
async def test_future_only_erasure_preserves_full_request_and_old_source(tmp_path, response):
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / "future-source.sqlite"))
    await repo.capture_source(
        "unit-character", SCOPE, source_message_id="old", body="我喜欢测试绿茶。", observed_at=NOW - timedelta(days=1)
    )

    class Completion:
        async def complete(self, messages):
            self.payload = json.loads(messages[1]["content"])
            return response

        async def close(self):
            pass

    completion = Completion()
    scheduler = MemoryEnrichmentScheduler(config=MemoryLlmConfig(True, "unit-only", "unit-only"), completion=completion)
    try:
        receipt = await scheduler.schedule_and_wait(
            repository=repo,
            character_id="unit-character",
            user_scope=SCOPE,
            message=FUTURE_ERASE,
            rule_hints=(),
            history=(),
            source_message_id="future-request",
            observed_at=NOW,
        )
        sources = {r["source_message_id"]: r for r in await repo.list_sources("unit-character", SCOPE)}
        assert sources["future-request"]["body"] == FUTURE_ERASE
        assert sources["old"]["body"] == "我喜欢测试绿茶。"
        assert receipt["source_capture"] == "recorded"
        assert receipt["source_erasure_policy"] == "deferred_until_source_start"
        assert not completion.payload.get("source_erasure_candidates")
        if response == '{"memories":[]}':
            assert receipt['status'] == 'no_change'
        else:
            assert receipt['status'] == 'failed' and receipt['stage'] == 'proposal_validation'
        assert scheduler.status.erased == 0
    finally:
        await scheduler.shutdown(timeout=2)


@pytest.mark.asyncio
async def test_future_whole_request_cannot_be_erased_by_evidence_omitting_its_date(tmp_path):
    repo, old, scheduler, job, proposal = await writer(tmp_path, "ERASE", FUTURE_ERASE, False)
    job.message = FUTURE_ERASE
    proposal = replace(proposal, evidence="我要求从记忆中删除测试绿茶偏好。")
    assert await scheduler._persist_proposal(job, proposal) == "saved"
    rows = await repo.list_memory_records("unit-character", SCOPE, include_inactive=True, limit=None)
    assert next(r for r in rows if r["id"] == old["id"]) == old
    assert (
        next(r for r in rows if r["id"] != old["id"])["metadata"]["deferred_mutation"]["original_operation"] == "ERASE"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "我要求从记忆中删除测试绿茶偏好。",
        "从2027年2月1日起，我要求从记忆中删除测试绿茶偏好，现在我要求从记忆中删除测试红茶偏好。",
        "从2027年2月1日起，我要求从记忆中删除测试绿茶偏好。现在我要求从记忆中删除测试红茶偏好。",
        "我要求从记忆中删除此前那个从2027年2月1日起的测试绿茶偏好安排。",
    ],
)
async def test_immediate_or_independent_current_source_erasure_is_not_postponed(tmp_path, message):
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / "immediate-source.sqlite"))
    for source, body in [("old", "我喜欢测试红茶。"), ("other", "我喜欢测试绿茶。")]:
        await repo.capture_source(
            "unit-character", SCOPE, source_message_id=source, body=body, observed_at=NOW - timedelta(days=1)
        )

    class Completion:
        async def complete(self, messages):
            payload = json.loads(messages[1]["content"])
            assert any(r["source_id"] == "old" for r in payload["source_erasure_candidates"])
            return '{"memories":[],"erase_source_ids":["old"]}'

        async def close(self):
            pass

    scheduler = MemoryEnrichmentScheduler(
        config=MemoryLlmConfig(True, "unit-only", "unit-only"), completion=Completion()
    )
    try:
        receipt = await scheduler.schedule_and_wait(
            repository=repo,
            character_id="unit-character",
            user_scope=SCOPE,
            message=message,
            rule_hints=(),
            history=(),
            source_message_id="current-request",
            observed_at=NOW,
        )
        assert receipt["source_capture"] == "erase_request" and receipt["source_erased"] == 1
        sources = {r["source_message_id"]: r for r in await repo.list_sources("unit-character", SCOPE)}
        assert set(sources) == {"other"}
    finally:
        await scheduler.shutdown(timeout=2)


@pytest.mark.parametrize(
    "message, expected",
    [
        (FUTURE_ERASE, True),
        ("从2027年2月1日起，我要求从记忆中删除测试绿茶偏好。", True),
        ("我从2027年2月1日起要求从记忆中删除测试绿茶偏好。", True),
        ("从明天起，我要求从记忆中删除测试绿茶偏好。", True),
        ("从不明日期起，我要求从记忆中删除测试绿茶偏好。", False),
        ("“从2027年2月1日起，我要求从记忆中删除绿茶偏好。”我现在要求从记忆中删除红茶偏好。", False),
        ("从2027年2月1日起，我要求从记忆中删除绿茶偏好，另外我要求从记忆中删除红茶偏好。", False),
    ],
)
def test_whole_request_future_authority_requires_every_erasure_clause_to_start_later(message, expected):
    from character.memory_llm import _future_only_erasure_request

    assert _future_only_erasure_request(message, observed_at=NOW) is expected
