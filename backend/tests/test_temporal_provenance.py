"""Temporal proposal provenance must not manufacture semantic authority."""

import json
from datetime import datetime, timezone

import pytest

from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig, parse_llm_proposals
from character.models import UserScope
from character.temporal_provenance import model_temporal_provenance
from db.database import SQLiteDB
from repositories.character_memory import DatabaseCharacterMemoryRepository

ANCHOR = datetime(2026, 12, 31, 16, 30, tzinfo=timezone.utc)


@pytest.mark.parametrize(("start", "end", "shape"), [
    ("", "", "unspecified"),
    ("2026-01-01", "", "open"),
    ("", "2026-01-01", "open"),
    ("2026-01-01", "2027-01-01", "coarse_or_unresolved"),
    ("2026-01-01T00:00:00Z", "2027-01-01T00:00:00Z", "ordered"),
    ("2026-01-01T08:00:00+08:00", "2026-01-01T00:00:00Z", "zero_width"),
    ("2027-01-01T00:00:00Z", "2026-01-01T00:00:00Z", "reversed"),
    ("2026-01-01T00:00:00", "2027-01-01T00:00:00", "coarse_or_unresolved"),
])
def test_endpoint_shape_is_not_authority(start, end, shape):
    result = model_temporal_provenance(evidence=f"我在{start}开始，到{end}结束。",
                                       observed_at=ANCHOR, proposed_from=start, proposed_to=end)
    assert result["interval_shape"] == shape
    assert result["validity_authority"] == ("unverified" if start or end else "unspecified")
    for bound in result["proposed_bounds"].values():
        assert bound["authority"] == "model_proposal"
        assert bound["role_verified"] is False
        assert bound["literal_in_evidence"] is True
    assert "valid_from" not in result and "valid_to" not in result


def test_date_precision_survives_without_midnight_claim():
    result = model_temporal_provenance(evidence="我今年刚升大三", observed_at=ANCHOR,
                                       proposed_from="2025-09-01", proposed_to="2026-07-01")
    start = result["proposed_bounds"]["start"]
    assert start["text"] == "2025-09-01"
    assert start["precision"] == "day"
    assert start["kind"] == "calendar_period"
    assert start["literal_in_evidence"] is False
    assert start["lower"] == "2025-09-01T00:00:00+08:00"
    assert start["upper"] == "2025-09-02T00:00:00+08:00"


def test_source_expression_has_precision_but_no_invented_start_or_end_role():
    result = model_temporal_provenance(evidence="我下周参加面试", observed_at=ANCHOR,
                                       time_expression="下周")
    assert result["expression"] == {
        "text": "下周", "role": "unspecified", "authority": "source_span",
        "kind": "calendar_period", "precision": "week",
        "lower": "2027-01-04T00:00:00+08:00", "upper": "2027-01-11T00:00:00+08:00",
    }
    assert result["proposed_bounds"] == {}


@pytest.mark.parametrize("expression", ["明天", "下个月", "2026-12-31"])
def test_model_cannot_create_source_span(expression):
    result = model_temporal_provenance(evidence="我下周参加面试", observed_at=ANCHOR,
                                       time_expression=expression)
    assert result["expression"] is None


def test_fuzzy_source_expression_is_retained_not_guessed():
    result = model_temporal_provenance(evidence="我过阵子参加面试", observed_at=ANCHOR,
                                       time_expression="过阵子")
    assert result["expression"]["text"] == "过阵子"
    assert result["expression"]["kind"] == "unresolved"
    assert "lower" not in result["expression"]


@pytest.mark.parametrize("clock", [datetime(2026, 1, 1), datetime(2026, 1, 1, tzinfo=None)])
def test_requires_trusted_aware_clock(clock):
    with pytest.raises(ValueError, match="timezone"):
        model_temporal_provenance(evidence="今年", observed_at=clock)


@pytest.mark.parametrize("location", ["top", "aliases", "qualifiers"])
def test_parser_keeps_raw_precision_across_legacy_date_locations(location):
    entry = {"kind": "study_stage", "value": "大三", "evidence": "今年刚升大三", "confidence": .96}
    dates = {"valid_from": "2025-09-01", "valid_to": "2026-07-01"}
    if location == "top":
        entry.update(dates)
    elif location == "aliases":
        entry.update(valid_at=dates["valid_from"], invalid_at=dates["valid_to"])
    else:
        entry["qualifiers"] = dates
    proposals = parse_llm_proposals(json.dumps({"memories": [entry]}), source_message=entry["evidence"])
    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal.proposed_valid_from == "2025-09-01"
    assert proposal.proposed_valid_to == "2026-07-01"
    assert "T" in proposal.valid_from  # Legacy execution unchanged in this migration step.


@pytest.mark.asyncio
async def test_scheduler_records_authority_and_does_not_accept_model_provenance(tmp_path):
    class Completion:
        async def complete(self, messages):
            return json.dumps({"memories": [{
                "kind": "study_stage", "value": "大三", "evidence": "今年刚升大三",
                "confidence": .96, "valid_from": "2025-09-01", "valid_to": "2026-07-01",
                "observed_at": "2099-01-01",
                "temporal_provenance": {"validity_authority": "verified", "producer": "rule"},
            }]})

        async def close(self):
            pass

    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / "provenance.db"))
    scope = UserScope("qq", "test", "user", "user", "private")
    scheduler = MemoryEnrichmentScheduler(
        config=MemoryLlmConfig(enabled=True, base_url="http://unused", model="stub"), completion=Completion())
    assert scheduler.schedule(repository=repo, character_id="test", user_scope=scope,
                              message="今年刚升大三", rule_hints=[], source_message_id="source",
                              observed_at=ANCHOR)
    await scheduler.shutdown(timeout=3)
    rows = await repo.list_memory_records("test", scope)
    assert len(rows) == 1
    row = rows[0]
    provenance = row["metadata"]["temporal_provenance"]
    assert provenance["producer"] == "semantic_memory"
    assert provenance["validity_authority"] == "unverified"
    assert provenance["observed_at"] == ANCHOR.isoformat() == row["observed_at"]
    assert provenance["proposed_bounds"]["start"]["text"] == "2025-09-01"
    assert row["evidence"] == ["今年刚升大三"]
    assert row["source_message_ids"] == ["source"]
