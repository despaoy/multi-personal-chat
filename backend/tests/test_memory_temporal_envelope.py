"""Equivalent temporal envelope layouts must keep grounded semantic qualifiers."""

import json
from copy import deepcopy
from datetime import datetime, timezone

import pytest

from character.memory_llm import parse_llm_proposals
from character.temporal_projection import project_temporal_record
from character.temporal_provenance import model_temporal_provenance

SOURCE = "下个月我会搬到宝鸡，现在还没有搬。"


def candidate(**extra):
    result = dict(kind="location", value="宝鸡", content="用户现在住在宝鸡", evidence=SOURCE,
                  confidence=.95, operation="ADD", attributed_to="user")
    result.update(extra)
    return result


def parse(entry):
    return parse_llm_proposals(json.dumps({"memories": [entry]}, ensure_ascii=False), source_message=SOURCE)


@pytest.mark.parametrize("field", ["valid_from", "valid_at"])
@pytest.mark.parametrize("clock", ["2099-01-01", "invalid-clock", ""])
def test_known_misplaced_fields_preserve_validity_and_real_qualifier(field, clock):
    entry = candidate(qualifiers={field: "2026-10-01", "observed_at": clock, "time": "下个月"})
    before = deepcopy(entry)
    proposal, = parse(entry)
    assert proposal.proposed_valid_from == "2026-10-01"
    assert proposal.valid_from == "2026-10-01T00:00:00+00:00"
    assert proposal.observed_at == ""
    assert proposal.qualifiers == (("time", "下个月"),)
    assert entry == before


def test_top_level_and_nested_envelopes_are_equivalent():
    dates = dict(valid_from="2026-10-01", valid_to="2026-11-01", observed_at="2099-01-01")
    a, = parse(candidate(**dates, qualifiers={"time": "下个月"}))
    b, = parse(candidate(qualifiers={**dates, "time": "下个月"}))
    assert a == b


@pytest.mark.parametrize("qualifier", [{"unknown_key": "下个月"}, {"time": "明年"}, {"condition": "如果下雨"}])
def test_real_qualifiers_are_still_validated(qualifier):
    assert parse(candidate(qualifiers={"valid_from": "2026-10-01", "observed_at": "bad", **qualifier})) == []


@pytest.mark.parametrize("dates", [dict(valid_from="not-a-date"),
                                   dict(valid_from="2026-11-01", valid_to="2026-10-01")])
def test_date_validation_is_not_bypassed(dates):
    with pytest.raises(ValueError, match="valid_from"):
        parse(candidate(qualifiers={"observed_at": "bad", **dates}))


def test_existing_top_level_precedence_is_preserved():
    proposal, = parse(candidate(valid_from="2026-10-01", qualifiers={"valid_from": "2026-11-01"}))
    assert proposal.proposed_valid_from == "2026-10-01"


@pytest.mark.parametrize("nested", [True, False])
def test_empty_canonical_placeholders_do_not_hide_nonempty_legacy_aliases(nested):
    dates = dict(valid_from="", valid_to="", valid_at="2026-10-01", invalid_at="2026-11-01")
    proposal, = parse(candidate(qualifiers=dates) if nested else candidate(**dates))
    assert proposal.proposed_valid_from == "2026-10-01"
    assert proposal.proposed_valid_to == "2026-11-01"


def test_schema_repair_does_not_promote_false_present_summary():
    proposal, = parse(candidate(qualifiers={"valid_from": "2026-10-01T00:00:00+00:00",
                                           "valid_to": "", "observed_at": "2099-01-01"}))
    observed = datetime(2026, 9, 27, tzinfo=timezone.utc)
    view = project_temporal_record(dict(
        id="m", memory_type=proposal.memory.memory_type, memory_key=proposal.memory.memory_key,
        content=proposal.memory.content, status="active", observed_at=observed.isoformat(),
        evidence=[proposal.evidence], source_message_ids=["source"], valid_from=proposal.valid_from,
        metadata={"qualifiers": dict(proposal.qualifiers), "temporal_provenance": model_temporal_provenance(
            evidence=proposal.evidence, observed_at=observed, proposed_from=proposal.proposed_valid_from)}))
    assert view["temporal_mode"] == "observation"
    assert SOURCE in view["content"]
    assert "用户现在住在宝鸡" not in view["content"]
