import json

import pytest

from character.evidence_selector import ContextualEvidenceSelector, InputBudgetError, selection_messages
from character.models import MemoryItem
from evaluation.temporal_metadata_view import TemporalMetadataReviewer, transform_messages


def memory(**kwargs):
    return MemoryItem("old", "user_fact", "用户在南京", evidence=("原始证据",), **kwargs)


@pytest.mark.parametrize("view", ["status_neutral", "interval_prose"])
def test_preserves_all_original_evidence_and_times_without_gold(view):
    item = memory(status="superseded", historical=True, valid_from="2023-01-01", valid_to="2025-01-01")
    original = selection_messages("2024年呢？", [item])
    before = json.loads(original[1]["content"])
    result = transform_messages(original, view)
    payload = json.loads(result[1]["content"])
    assert original[0] == result[0]
    assert json.loads(original[1]["content"]) == before
    candidate = payload["candidates"][0]
    assert "status" not in candidate
    for key, value in before["candidates"][0].items():
        if key != "status":
            assert candidate[key] == value
    assert payload["required_ids"] == before["required_ids"]
    assert payload["query"] == "2024年呢？"
    assert "gold" not in payload
    assert ("time_scope_explanation" in candidate) == (view == "interval_prose")


@pytest.mark.parametrize(
    "status,historical", [("erased", True), ("retracted", True), ("pending", False), ("superseded", False)]
)
def test_cannot_hide_ineligible_states(status, historical):
    with pytest.raises(ValueError):
        transform_messages(selection_messages("q", [memory(status=status, historical=historical)]), "status_neutral")


def test_no_silent_budget_truncation():
    messages = selection_messages("q", [memory()])
    payload = json.loads(messages[1]["content"])
    payload["candidates"][0]["evidence"] = ["x" * 18000]
    messages[1]["content"] = json.dumps(payload)
    with pytest.raises(InputBudgetError):
        transform_messages(messages, "interval_prose")


async def test_does_not_force_a_correct_label_or_rewrite_original_object():
    item = memory(status="superseded", historical=True)

    async def reviewer(messages):
        assert "status" not in json.loads(messages[1]["content"])["candidates"][0]
        return '{"decisions":[{"id":"old","label":"stale"}]}'

    selector = ContextualEvidenceSelector(TemporalMetadataReviewer(reviewer, "status_neutral"))
    result = await selector.select("以前呢", [item])
    assert result.memories == ()
    assert result.decisions == (("old", "stale"),)
    assert item.status == "superseded"
