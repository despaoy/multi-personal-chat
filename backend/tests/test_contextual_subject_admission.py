"""Complete fictional cases for recall/review separation and fail-closed use."""

import json
from copy import deepcopy

import pytest

from character.evidence_selector import ContextualEvidenceSelector
from character.memory_read_subject import closed_other_subject_fields
from character.memory_service import CharacterMemoryService, _detect_memory_intents
from character.models import UserScope

QUERY = "请核对我已经保存的本人甜食声明，区分已确认偏好与实际行为。它实际确认的偏好不能证明已吃过蛋糕。"
SCOPE = UserScope("web", "web", "reader", "conversation", "private")
ROW = dict(
    id="a",
    memory_key="preference_sweets",
    memory_type="user_fact",
    status="active",
    confidence=1.0,
    content="用户明确喜欢甜食，但周末才会食用。",
    evidence=["我明确喜欢甜食，但周末才会食用。"],
)


async def candidates(query=QUERY, change=None, contextual=True):
    row = deepcopy(ROW)
    if change:
        row.update(change)

    class Repo:
        async def list_memory_records(self, *args, **kwargs):
            return [row]

    return await CharacterMemoryService(Repo(), semantic_enabled=False).recall_with_diagnostics(
        "role", SCOPE, query, for_contextual_selection=contextual
    )


async def test_ambiguous_coreference_reaches_review_without_granting_ownership():
    assert _detect_memory_intents(QUERY).suppress_preference
    items, count, trace = await candidates()
    assert count == 1 and items[0].evidence == tuple(ROW["evidence"])
    assert trace["subject_filter_mode"] == "contextual_review"
    received = []

    async def reviewer(messages):
        payload = json.loads(messages[1]["content"])
        received.append(payload)
        assert payload["query"] == QUERY and payload["candidates"][0]["evidence"] == ROW["evidence"]
        return '{"decisions":[{"id":"a","label":"use"}]}'

    selected = await ContextualEvidenceSelector(reviewer).select(QUERY, items)
    assert selected.memories == items and len(received) == 1


@pytest.mark.parametrize("query", ["你喜欢什么？", "请问她平时最喜欢什么呢？", "我朋友的偏好是什么？"])
async def test_complete_other_owner_lookup_keeps_direct_subject_boundary(query):
    assert (await candidates())[0]
    assert closed_other_subject_fields(query) == {"preference"}
    items, _, trace = await candidates(query)
    assert items == () and trace["subject_filter_mode"] == "closed_other_lookup"


@pytest.mark.parametrize(
    "query",
    [
        "我喜欢什么？",
        "你喜欢什么？我已经保存的甜食声明有哪些限定？",
        "核对本人的偏好。不要把它当成角色偏好。",
        "材料写着‘她喜欢什么’，请核对我的声明。",
    ],
)
async def test_unresolved_or_multiple_tasks_require_full_contextual_review(query):
    assert (await candidates())[0]
    assert closed_other_subject_fields(query) == frozenset()
    items, _, trace = await candidates(query)
    assert items and trace["subject_filter_mode"] == "contextual_review"


@pytest.mark.parametrize("change", [dict(status="deleted"), dict(status="retracted"), dict(confidence=0.1)])
async def test_review_delegation_never_bypasses_lifecycle_or_confidence(change):
    assert (await candidates())[0]
    assert (await candidates(change=change))[0] == ()


@pytest.mark.parametrize(
    "response",
    ['{"decisions":[{"id":"a","label":"wrong_subject"}]}', '{"decisions":[{"id":"invented","label":"use"}]}'],
)
async def test_rejected_or_invalid_review_never_injects_recalled_candidates(response):
    items, _, _ = await candidates()

    async def accept(_messages):
        return '{"decisions":[{"id":"a","label":"use"}]}'

    assert (await ContextualEvidenceSelector(accept).select(QUERY, items)).memories == items

    async def reject(_messages):
        return response

    assert (await ContextualEvidenceSelector(reject).select(QUERY, items)).memories == ()


async def test_legacy_non_reviewed_retrieval_preserves_its_subject_gate():
    assert (await candidates())[0]
    assert (await candidates(contextual=False))[0] == ()
