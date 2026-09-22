import json

import pytest

from evaluation.planned_evidence_review import PlannedEvidenceReviewer, parse_plan


@pytest.mark.parametrize(
    "raw",
    [
        "{}",
        "[]",
        '{"task_kind":[],"target_scope":"general","time_reference":"现在"}',
        '{"task_kind":"standalone_task","target_scope":"general","time_reference":""}',
    ],
)
def test_bad_plan_fails_closed(raw):
    with pytest.raises(ValueError):
        parse_plan(raw)


async def test_first_stage_cannot_see_candidates_and_plan_stays_untrusted():
    calls = []
    plan = {"task_kind": "personal_recall", "target_scope": "character_side", "time_reference": "现在"}

    async def reviewer(messages):
        calls.append(messages)
        data = json.loads(messages[1]["content"])
        if len(calls) == 1:
            assert set(data) == {"query", "history", "character"}
            assert "CANDIDATE_MARKER" not in messages[1]["content"]
            return json.dumps(plan)
        assert data["tentative_query_interpretation"] == plan
        assert "CANDIDATE_MARKER" in messages[1]["content"]
        assert "tentative_query_interpretation" not in messages[0]["content"]
        return '{"decisions":[{"id":"a","label":"wrong_subject"}]}'

    payload = {
        "query": "你喜欢什么？",
        "history": [],
        "character": {},
        "candidates": [{"id": "a", "content": "CANDIDATE_MARKER"}],
    }
    result = await PlannedEvidenceReviewer(reviewer)(
        [{"role": "system", "content": "old"}, {"role": "user", "content": json.dumps(payload)}]
    )
    assert len(calls) == 2
    assert json.loads(result)["decisions"][0]["label"] == "wrong_subject"


async def test_invalid_plan_prevents_second_call():
    calls = []

    async def reviewer(messages):
        calls.append(messages)
        return "{}"

    payload = {"query": "q", "history": [], "character": {}}
    with pytest.raises(ValueError):
        await PlannedEvidenceReviewer(reviewer)(
            [{"role": "system", "content": "old"}, {"role": "user", "content": json.dumps(payload)}]
        )
    assert len(calls) == 1
