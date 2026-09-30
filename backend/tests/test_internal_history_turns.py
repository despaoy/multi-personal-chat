"""Internal decisions must not see answers stripped of their user premise."""

import json
from copy import deepcopy

import pytest

from character.evidence_selector import InputBudgetError, _history_view, selection_messages
from character.models import MemoryItem


def _history():
    rows = [{'role': 'user', 'content': '这是虚构故事，不是我的经历，请用第二人称改写。'},
            {'role': 'assistant', 'content': '你养了一只猫。'}]
    for index in range(5):
        rows.extend([{'role': 'user', 'content': f'新话题{index}'},
                     {'role': 'assistant', 'content': f'回答{index}'}])
    rows.append({'role': 'user', 'content': '继续聊现在的话题。'})
    return rows


def test_selector_count_boundary_keeps_complete_user_turns():
    rows = _history()
    before = deepcopy(rows)
    payload = json.loads(selection_messages('继续', [MemoryItem('1', 'user_fact', '记忆')], history=rows)[1]['content'])
    assert payload['history'] == rows[2:]
    assert rows == before


def test_assistant_chunks_never_outlive_cut_user_premise():
    rows = [{'role': 'user', 'content': '限定条件'}] + [
        {'role': 'assistant', 'content': f'分段{i}'} for i in range(12)]
    with pytest.raises(InputBudgetError):
        _history_view(rows)


async def test_decision_policy_uses_same_complete_turn_boundary():
    from character.contextual_policy import ContextualDecisionPolicy
    from character.models import CharacterProfile, DecisionPlan, InteractionState, RelationshipState

    seen = []

    async def reviewer(messages):
        seen.append(json.loads(messages[1]['content'])['history'])
        return '{"strategy_ids":["respond_directly"]}'

    outcome = await ContextualDecisionPolicy(reviewer).refine(
        DecisionPlan(strategy_ids=('respond_directly',)), query='继续', history=_history(),
        profile=CharacterProfile('char', '角色'), relationship=RelationshipState(),
        interaction=InteractionState(), has_relevant_memory=False)
    assert outcome.status == 'applied'
    assert seen == [_history()[2:]]


def test_fitting_and_user_only_histories_remain_unchanged():
    for rows in ([], _history()[2:], [{'role': 'assistant', 'content': '旧系统记录'}]):
        assert _history_view(rows) == rows
    rows = [{'role': 'user', 'content': str(index)} for index in range(15)]
    assert _history_view(rows) == rows[-12:]
