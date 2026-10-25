"""Complete fictional cases for recall/review separation and fail-closed use."""

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from character.evidence_selector import ContextualEvidenceSelector
from character.memory_service import CharacterMemoryService, _detect_memory_intents
from character.models import CharacterProfile, UserScope
from repositories.character_memory import decode_legacy_memory_lifecycle
from services.character_context import CharacterContextService, TurnInput

QUERY = "请核对我已经保存的本人甜食声明，区分已确认偏好与实际行为。它实际确认的偏好不能证明已吃过蛋糕。"
SCOPE = UserScope("web", "web", "reader", "conversation", "private")
ROW = dict({"source_message_ids": []},
    id="a",
    memory_key="preference_sweets",
    memory_type="user_fact",
    status="active",
    confidence=1.0,
    content="用户明确喜欢甜食，但周末才会食用。",
    evidence=["我明确喜欢甜食，但周末才会食用。"],
)


class _Repository:
    def __init__(self, change=None):
        self.row = deepcopy(ROW)
        self.row.update(change or {})
        self.row.update(decode_legacy_memory_lifecycle(self.row))

    async def list_memory_records(self, *args, **kwargs):
        return [self.row]

    async def get_relationship_record(self, *args):
        return {}

    async def list_relationship_notes(self, *args):
        return []


async def candidates(query=QUERY, change=None, contextual=True):
    return await CharacterMemoryService(_Repository(change), semantic_enabled=False).recall_with_diagnostics(
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
async def test_other_owner_lookup_reaches_review_but_cannot_use_rejected_evidence(query):
    items, _, trace = await candidates(query)
    assert items and items[0].evidence == tuple(ROW["evidence"])
    assert trace["subject_filter_mode"] == "contextual_review"

    async def reviewer(messages):
        payload = json.loads(messages[1]["content"])
        assert payload["query"] == query and payload["candidates"][0]["evidence"] == ROW["evidence"]
        return '{"decisions":[{"id":"a","label":"wrong_subject"}]}'

    outcome = await ContextualEvidenceSelector(reviewer).select(query, items)
    assert outcome.memories == () and outcome.decisions == (("a", "wrong_subject"),)


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
    items, _, trace = await candidates(query)
    assert items and trace["subject_filter_mode"] == "contextual_review"


@pytest.mark.parametrize("change", [dict(status="deleted"), dict(status="retracted"), dict(confidence=0.1)])
async def test_review_delegation_never_bypasses_lifecycle_or_confidence(change):
    assert (await candidates())[0]
    assert (await candidates(change=change))[0] == ()


@pytest.mark.parametrize(
    "response,error",
    [('{"decisions":[{"id":"a","label":"wrong_subject"}]}', None),
     ('{"decisions":[{"id":"invented","label":"use"}]}', "invalid_candidate_id")],
)
async def test_rejected_or_invalid_review_never_injects_recalled_candidates(response, error):
    items, _, _ = await candidates()

    async def accept(_messages):
        return '{"decisions":[{"id":"a","label":"use"}]}'

    assert (await ContextualEvidenceSelector(accept).select(QUERY, items)).memories == items

    async def reject(_messages):
        return response

    if error:
        with pytest.raises(ValueError, match=error):
            await ContextualEvidenceSelector(reject).select(QUERY, items)
    else:
        assert (await ContextualEvidenceSelector(reject).select(QUERY, items)).memories == ()


async def test_legacy_non_reviewed_retrieval_preserves_its_subject_gate():
    assert (await candidates())[0]
    assert (await candidates(contextual=False))[0] == ()



@pytest.mark.parametrize("questionnaire", [False, True])
async def test_full_history_controls_subject_selection_before_reference_compilation(questionnaire):
    history = (
        {"role": "user", "content": "接下来填写我本人的饮食问卷，每道题中的‘你’指填表人，也就是我。请依据保存的我的偏好填写，并保留条件。"},
        {"role": "assistant", "content": "明白，会按你的资料填写，并保留限制条件。"},
    ) if questionnaire else ()
    repository = _Repository()
    reviewed = []

    async def reviewer(messages):
        payload = json.loads(messages[1]["content"])
        assert payload["query"] == "你喜欢什么？" and payload["history"] == list(history)
        assert payload["candidates"][0]["evidence"] == ROW["evidence"]
        reviewed.append(payload)
        return json.dumps({"decisions": [{"id": "a", "label": "use" if questionnaire else "wrong_subject"}]})

    messages = SimpleNamespace(list_recent_conversation_history=AsyncMock(return_value=[]))
    service = CharacterContextService(
        SimpleNamespace(get_profile=lambda _: CharacterProfile("role", "隔离问卷助手")),
        repository, messages, memory_service=CharacterMemoryService(repository, semantic_enabled=False),
        memory_selector=ContextualEvidenceSelector(reviewer), source_recall_enabled=False,
    )
    prepared = await service.prepare_turn(TurnInput(
        "你喜欢什么？", "web", "web", "reader", "conversation", "private", history=history,
    ), "role")
    assert len(reviewed) == prepared.memory_selection_candidate_count == 1
    assert prepared.history == history
    assert prepared.compiled.used_memory_ids == (("a",) if questionnaire else ())
    reference = prepared.compiled.reference_context
    assert (ROW["evidence"][0] in reference) is questionnaire
    assert messages.list_recent_conversation_history.await_count == int(not questionnaire)
