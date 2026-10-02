"""Task metadata preserves original evidence and does not force classifier labels."""

import json

import pytest

from character.evidence_selector import ContextualEvidenceSelector, selection_messages
from character.models import MemoryItem

QUERY = "2026年9月我的居住地是什么？请告诉我，我的现居地是什么？"


def candidates():
    return (
        MemoryItem(
            "old",
            "user_fact",
            "用户说自己居住在德阳",
            memory_key="user_residence",
            evidence=("我现在住在德阳。",),
            status="superseded",
            historical=True,
            valid_from="2026-09-30T01:00:00+00:00",
            valid_to="2026-10-01T01:00:00+00:00",
        ),
        MemoryItem(
            "new",
            "user_fact",
            "用户说自己居住在晋中",
            memory_key="user_residence",
            evidence=("更正，我现在住在晋中。",),
            status="active",
            historical=False,
            valid_from="2026-10-01T01:00:00+00:00",
        ),
    )


def test_selector_receives_all_independent_task_times_and_unmodified_evidence():
    items = candidates()
    messages = selection_messages(QUERY, items)
    data = json.loads(messages[1]["content"])
    assert data["query"] == QUERY
    assert data["query_tasks"] == [
        {
            "index": 0,
            "query": "2026年9月我的居住地是什么",
            "fields": ["residence"],
            "time_expression": "2026年9月",
            "time_mode": "historical",
        },
        {
            "index": 1,
            "query": "请告诉我，我的现居地是什么",
            "fields": ["residence"],
            "time_expression": "",
            "time_mode": "current",
        },
    ]
    for row, item in zip(data["candidates"], items, strict=True):
        assert row["memory_key"] == "user_residence" and row["evidence"] == list(item.evidence)
        assert row["valid_from"] == item.valid_from and row["valid_to"] == item.valid_to
        assert row["historical"] == item.historical and row["status"] == item.status
    assert all(item.content not in messages[0]["content"] for item in items)


@pytest.mark.parametrize(
    "query",
    [
        "2026年9月我的室友住哪里？我的现居地是什么？",
        "2026年9月。我的现居地是什么？",
    ],
)
def test_unknown_or_other_owner_query_is_not_partially_annotated(query):
    data = json.loads(selection_messages(query, candidates())[1]["content"])
    assert "query_tasks" not in data
    assert data["query"] == query


async def test_time_tasks_do_not_force_use_or_rewrite_original_objects():
    items = candidates()

    async def reviewer(messages):
        assert json.loads(messages[1]["content"])["query_tasks"][0]["time_mode"] == "historical"
        return json.dumps({"decisions": [{"id": "old", "label": "stale"}, {"id": "new", "label": "use"}]})

    result = await ContextualEvidenceSelector(reviewer).select(QUERY, items)
    assert result.status == "selected" and result.memories == (items[1],)
    assert result.memories[0] is items[1]
    assert result.decisions == (("old", "stale"), ("new", "use"))
