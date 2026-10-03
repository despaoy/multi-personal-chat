"""Whitespace variants retain full synthetic sources in independent unit DBs."""

import json

import pytest
from test_deferred_evidence_binding import LIKE, NOW, RETRACT, SCOPE, setup_writer

from character.deferred_memory_mutation import deferred_evidence_start
from character.temporal_projection import project_temporal_record


@pytest.mark.asyncio
@pytest.mark.parametrize("operation,evidence", [("RETRACT", RETRACT), ("SUPERSEDE", LIKE), ("MERGE", LIKE)])
async def test_admitted_whitespace_normalization_cannot_remove_future_authority(tmp_path, operation, evidence):
    formatted = evidence.replace("撤回", "撤回\t  ").replace("喜欢", "喜欢\u3000  ")
    source = "从2027年4月1日起，\n" + formatted + "\n该日期之前当前偏好继续有效。"
    repo, old, scheduler, job, proposal = await setup_writer(tmp_path, operation, source, evidence)
    assert proposal.evidence not in source
    assert await scheduler._persist_proposal(job, proposal) == "saved"
    rows = await repo.list_memory_records("unit-character", SCOPE, include_inactive=True, limit=None)
    assert next(r for r in rows if r["id"] == old["id"]) == old
    new = next(r for r in rows if r["id"] != old["id"])
    assert (
        new["relation_type"] == "COEXIST"
        and new["parent_memory_id"] == old["id"]
        and new["supersedes_memory_id"] is None
    )
    assert new["metadata"]["deferred_mutation"]["expression"] == "2027年4月1日"
    sources = await repo.linked_source_receipts(
        "unit-character", SCOPE, claim_sources=((new["id"], job.source_message_id),)
    )
    assert sources[0]["body"] == source
    view = project_temporal_record(new, {r["source_message_id"]: r for r in sources})
    quotes = json.loads(view["content"].split("：", 1)[1])
    assert view["temporal_mode"] == "observation" and source in quotes and source in view["evidence"]


@pytest.mark.parametrize(
    "source,evidence,future",
    [
        ("从2027年4月1日起，\n我撤回\t  紫茶偏好。", "我撤回紫茶偏好。", True),
        ("从2027年4月1日起，我撤回紫茶偏好。", "我 撤回 紫茶偏好。", True),
        ("从2027 年 4 月 1 日起，我撤回\u3000紫茶偏好。", "我撤回紫茶偏好。", True),
        ("从2027年4月1日起，\r\n我撤回\u00a0紫茶偏好。", "我撤回紫茶偏好。", True),
        ("从2027年4月1日起，我撤回\t紫茶偏好。今天我撤回紫茶偏好。", "我撤回紫茶偏好。", False),
        ("从2027年4月1日起，我喜欢红茶，今天我撤回\t紫茶偏好。", "我撤回紫茶偏好。", False),
        ("从2027年4月1日起，我喜欢红茶\n我撤回\t紫茶偏好。", "我撤回紫茶偏好。", False),
        ("“从2027年4月1日起，我撤回\t紫茶偏好。”", "我撤回紫茶偏好。", False),
        ("从2027年4月1日起，他撤回\t紫茶偏好。", "他撤回紫茶偏好。", False),
        ("从2027年4月1日起，我撤回\n紫茶偏好。", "我撤回紫茶偏好。", False),
        ("从未知日期起，我撤回\t紫茶偏好。", "我撤回紫茶偏好。", False),
        ("从2027年4月1日起，我撤回紫茶偏好。", " \t\n ", False),
    ],
)
def test_original_positions_quote_masks_and_hard_line_boundaries_survive_whitespace(source, evidence, future):
    assert (deferred_evidence_start(evidence, source, observed_at=NOW) is not None) == future


def test_formatted_date_receipt_keeps_literal_original_expression():
    source = "从2027 年\t4 月 1 日起，我撤回\t紫茶偏好。"
    result = deferred_evidence_start("我撤回紫茶偏好。", source, observed_at=NOW)
    assert result.text == "2027 年\t4 月 1 日" and result.text in source
    assert result.observed_at == NOW
