from copy import deepcopy

import pytest

from inference.generation_request import CONTEXT_SAFETY_MARGIN_TOKENS, _trim_history_to_budget


def trim(history, budget):
    return _trim_history_to_budget(history, fixed_messages=(), max_output_tokens=0,
                                   context_window_tokens=CONTEXT_SAFETY_MARGIN_TOKENS + budget)


def test_budget_does_not_leave_answer_without_its_user_turn():
    history = [{'role': 'user', 'content': '请把这段虚构经历改写得简短一些。' * 50},
               {'role': 'assistant', 'content': '你住在北京。'}]
    assert trim(history, 20) == []


def test_keep_newest_complete_turn_without_reaching_back_across_gap():
    history = [{'role': 'user', 'content': '很旧的消息'}, {'role': 'assistant', 'content': '很旧的回答'},
               {'role': 'user', 'content': '很长的限定条件' * 100}, {'role': 'assistant', 'content': '中间回复'},
               {'role': 'user', 'content': '新问题'}, {'role': 'assistant', 'content': '新回答'}]
    assert trim(history, 24) == history[-2:]


def test_multiple_assistant_messages_stay_with_same_user_turn():
    history = [{'role': 'user', 'content': '用户问题'}, {'role': 'assistant', 'content': '第一段'},
               {'role': 'assistant', 'content': '第二段'}]
    assert trim(history, 22) == history
    assert trim(history, 21) == []


def test_user_only_view_remains_independent_messages():
    history = [{'role': 'user', 'content': '旧问题'}, {'role': 'user', 'content': '新问题'}]
    assert trim(history, 7) == history[-1:]


@pytest.mark.parametrize('history', [[], [{'role': 'assistant', 'content': '旧系统的单条历史'}],
    [{'role': 'user', 'content': '问题'}, {'role': 'assistant', 'content': '回答'}]])
def test_full_fitting_history_unchanged_and_input_not_mutated(history):
    before = deepcopy(history)
    assert trim(history, 1000) == history
    assert history == before
    assert trim(history, 0) == []


def test_real_window_generation_plan_has_no_orphan_answer():
    from inference.generation_request import GenerationRequest, build_generation_request

    history = ({'role': 'user', 'content': '这是一段虚构材料，不是我的经历。' * 450},
               {'role': 'assistant', 'content': '你住在北京。'})
    plan = build_generation_request(GenerationRequest(
        message='我住在哪里？', history=history, context_window_tokens=8192,
        max_tokens=2048, apply_prompt_policy=False))
    assert all(m['role'] != 'assistant' for m in plan.messages)
    assert '你住在北京' not in str(plan.messages)
    assert '我住在哪里' in plan.messages[-1]['content']


def test_all_budget_boundaries_preserve_suffix_turns_and_reservation():
    from inference.generation_request import _estimated_tokens

    history = [{'role': role, 'content': str(i) + '中文 mixed text' * (i + 1)}
               for i, role in enumerate(['user', 'assistant', 'assistant', 'user', 'user', 'assistant'])]
    for budget in range(240):
        kept = trim(history, budget)
        assert sum(_estimated_tokens(m['content']) + 4 for m in kept) <= budget
        if kept:
            assert kept == history[-len(kept):]
            assert kept[0]['role'] == 'user'
        fixed = [{'role': 'system', 'content': '固定角色设定'}]
        cost = _estimated_tokens(fixed[0]['content']) + 4
        assert _trim_history_to_budget(
            history, fixed_messages=fixed, max_output_tokens=200,
            context_window_tokens=CONTEXT_SAFETY_MARGIN_TOKENS + 200 + cost + budget) == kept
