import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from evaluation.temporal_memory_live_replay import load_cases, trace_json_default


def test_full_preparation_trace_serializes_receipt_without_hiding_unknown_types():
    receipt = datetime(2026, 9, 27, tzinfo=timezone.utc)
    assert json.loads(json.dumps({"received_at": receipt}, default=trace_json_default)) == {
        "received_at": "2026-09-27T00:00:00+00:00"}
    with pytest.raises(TypeError):
        json.dumps({"unknown": object()}, default=trace_json_default)


def test_independent_boundary_fixture_has_explicit_actor_and_revision_cases():
    cases = load_cases(Path(__file__).resolve().parents[1] / "evaluation/fixtures/source_boundaries_20260927.json")
    assert len(cases) == 7
    assert sum(len(case["questions"]) for case in cases) == 9
    assert {"friend_and_self", "fictional_quote", "explicit_cancel", "implicit_cancel", "conditional_plan",
            "speaker_role", "preference_change"} == {case["id"] for case in cases}


@pytest.mark.parametrize("case_id", ["../elsewhere", "", "a/b", "UPPER"])
def test_fixture_ids_cannot_escape_isolated_database_directory(tmp_path, case_id):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps([dict(id=case_id, turns=["input"], questions=["question"], criteria="review")]))
    with pytest.raises(ValueError):
        load_cases(path)
