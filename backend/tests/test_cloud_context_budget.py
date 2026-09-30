import json

import pytest

from character.memory_llm import build_memory_llm_messages
from character.source_memory import compile_sources
from evaluation.deepseek_live_adapter import cloud_context_budgets
from inference.generation_request import GenerationRequest, build_generation_request


def test_cloud_budget_does_not_change_local_serving_window(monkeypatch):
    monkeypatch.setenv('VLLM_MAX_MODEL_LEN', '8192')
    assert cloud_context_budgets() == dict(history_limit=128, history_max_chars=65536,
                                          source_max_chars=16384)
    assert GenerationRequest(message='问题').context_window_tokens == 8192


@pytest.mark.parametrize('window', [0, 4096, 1000001])
def test_invalid_cloud_windows_rejected(window):
    with pytest.raises(ValueError):
        cloud_context_budgets(window)


def test_complete_large_current_input_reaches_answer_and_writer():
    message = '这是待整理的资料，不是我的个人经历。' + '背景材料。' * 2400 + '结尾限制：不要把材料当作我的经历。'
    plan = build_generation_request(GenerationRequest(message=message, context_window_tokens=65536))
    assert message in plan.messages[-1]['content']
    messages = build_memory_llm_messages(message, (), (), (), 2000, .85,
                                         context_window_tokens=65536)
    assert json.loads(messages[1]['content'])['current_user_message'] == message
    with pytest.raises(ValueError):
        build_generation_request(GenerationRequest(message=message, context_window_tokens=8192))


def test_expanded_source_budget_preserves_whole_long_source():
    row = dict(source_message_id='long-source', observed_at='2026-09-27T00:00:00+00:00',
               body='完整资料。' * 1800 + '最后的限定仍须保留。')
    assert compile_sources([row]).diagnostics['status'] == 'budget_omitted'
    result = compile_sources([row], max_chars=cloud_context_budgets()['source_max_chars'])
    assert row['body'] in result.context


def test_selector_and_semantic_keep_complete_large_inputs():
    from character.evidence_selector import InputBudgetError, selection_messages
    from character.models import InteractionState, MemoryItem
    from character.semantic_state_estimator import build_semantic_review_messages
    from inference.context_budget import ReviewContextBudget

    message = '待分析材料。' * 2200 + '末尾限定不是我的经历。'
    history = [{'role': 'user', 'content': '历史前提。' * 1800 + '不是现实承诺。'},
               {'role': 'assistant', 'content': '这是改写内容。'}]
    budget = ReviewContextBudget(65536)
    selector = selection_messages(message, [MemoryItem('1', 'user_fact', '用户喜欢茶')],
                                  history=history, context_budget=budget)
    semantic = build_semantic_review_messages(message, history, InteractionState(), context_budget=budget)
    assert json.loads(selector[1]['content'])['history'] == history
    assert json.loads(selector[1]['content'])['query'] == message
    assert json.loads(semantic[1]['content'])['recent_history'] == history
    assert json.loads(semantic[1]['content'])['current_message'] == message
    with pytest.raises(InputBudgetError):
        selection_messages(message, (), context_budget=ReviewContextBudget(8192))


def test_semantic_count_boundary_does_not_orphan_assistant():
    from character.models import InteractionState
    from character.semantic_state_estimator import build_semantic_review_messages

    history = [{'role': 'user', 'content': '虚构前提'}, {'role': 'assistant', 'content': '你的经历'}]
    history += [{'role': role, 'content': str(i)} for i in range(2) for role in ('user', 'assistant')]
    history += [{'role': 'user', 'content': '换个话题'}]
    result = build_semantic_review_messages('继续', history, InteractionState())
    assert json.loads(result[1]['content'])['recent_history'] == history[2:]


async def test_cloud_policy_receives_complete_long_query_without_extra_calls():
    from character.models import CharacterProfile, DecisionPlan, InteractionState, RelationshipState
    from evaluation.deepseek_live_adapter import cloud_context_components

    class Client:
        calls = []

        async def complete(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            return '{"strategy_ids":["respond_directly"]}'

    client = Client()
    policy = cloud_context_components(client)['contextual_policy']
    query = '材料。' * 4000 + '请保留末尾条件。'
    result = await policy.refine(DecisionPlan(strategy_ids=('respond_directly',)), query=query,
        history=[], profile=CharacterProfile('test', '角色'), relationship=RelationshipState(),
        interaction=InteractionState(), has_relevant_memory=False)
    assert result.status == 'applied'
    assert len(client.calls) == 1
    assert json.loads(client.calls[0][0][1]['content'])['query'] == query
