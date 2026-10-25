"""One model wire format; historical persisted-date readers remain separate."""

import json
from copy import deepcopy
from datetime import datetime, timezone

import pytest

from character.memory_llm import MemoryResponseError, parse_llm_proposals
from character.temporal_projection import project_temporal_record
from character.temporal_provenance import model_temporal_provenance

SOURCE = "下个月我会搬到宝鸡，现在还没有搬。"


def candidate(**extra):
    return dict(kind="location", value="宝鸡", content="用户现在住在宝鸡", evidence=SOURCE,
                confidence=.95, operation="ADD", attributed_to="user", **extra)


def parse(entry):
    return parse_llm_proposals(json.dumps({"memories": [entry]}, ensure_ascii=False), source_message=SOURCE)


def test_canonical_dates_preserve_precision_qualifiers_and_input():
    entry = candidate(valid_from="2026-10-01", valid_to="2026-11-01", observed_at="2099-01-01",
                      qualifiers={"time": "下个月"})
    before = deepcopy(entry)
    proposal, = parse(entry)
    assert proposal.proposed_valid_from == "2026-10-01"
    assert proposal.proposed_valid_to == "2026-11-01"
    assert proposal.valid_from == "2026-10-01T00:00:00+00:00"
    assert proposal.qualifiers == (("time", "下个月"),)
    assert entry == before


@pytest.mark.parametrize("field", ["valid_at", "invalid_at"])
def test_legacy_model_alias_is_protocol_error_even_with_empty_canonical_dates(field):
    with pytest.raises(MemoryResponseError) as caught:
        parse(candidate(valid_from="", valid_to="", **{field: "2026-10-01"}))
    assert caught.value.field == "memories[]"


@pytest.mark.parametrize("field", ["valid_from", "valid_to", "valid_at", "invalid_at", "observed_at"])
def test_temporal_fields_in_qualifiers_are_not_repaired(field):
    with pytest.raises(MemoryResponseError) as caught:
        parse(candidate(qualifiers={field: "2026-10-01", "time": "下个月"}))
    assert caught.value.field == "qualifiers"


@pytest.mark.parametrize("qualifier", [{"unknown_key": "下个月"}, {"time": "明年"}, {"condition": "如果下雨"}])
def test_unsupported_or_ungrounded_qualifier_is_explicit_error(qualifier):
    with pytest.raises(MemoryResponseError) as caught:
        parse(candidate(qualifiers=qualifier))
    assert caught.value.field == "qualifiers"


@pytest.mark.parametrize("dates", [dict(valid_from="not-a-date"),
                                   dict(valid_from="2026-11-01", valid_to="2026-10-01")])
def test_date_validation_is_not_bypassed(dates):
    with pytest.raises(ValueError, match="valid_from"):
        parse(candidate(**dates))


def test_canonical_dates_do_not_promote_false_present_summary():
    proposal, = parse(candidate(valid_from="2026-10-01T00:00:00+00:00", valid_to="", observed_at="2099-01-01"))
    observed = datetime(2026, 9, 27, tzinfo=timezone.utc)
    view = project_temporal_record(dict(
        id="m", memory_type=proposal.memory.memory_type, memory_key=proposal.memory.memory_key,
        content=proposal.memory.content, status="active", observed_at=observed.isoformat(),
        evidence=[proposal.evidence], source_message_ids=["source"], valid_from=proposal.valid_from,
        metadata={"qualifiers": dict(proposal.qualifiers), "temporal_provenance": model_temporal_provenance(
            evidence=proposal.evidence, observed_at=observed, proposed_from=proposal.proposed_valid_from)}))
    assert view["temporal_mode"] == "observation"
    assert SOURCE in view["evidence"] and SOURCE not in view["content"]
    assert "用户现在住在宝鸡" not in view["content"]
