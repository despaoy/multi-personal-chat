from dataclasses import replace
from types import SimpleNamespace

import pytest

from inference.generation_request import GenerationRequest, RetrievalResult, build_generation_request


def identity_request(name='林远'):
    query = f'{name}是谁？'
    return GenerationRequest(
        message=query, persona_prompt='固定角色设定', apply_prompt_policy=False, max_tokens=100,
        history=({'role': 'user', 'content': '以后请简短回答。'},
                 {'role': 'assistant', 'content': '之前未经证实的身份回答。'}),
        retrieval=RetrievalResult(status='ok', evidence='人物的来源证据。',
                                  identity_task={'query': query, 'subject': name}))


@pytest.mark.parametrize('name', ['林远', '安宁', 'Alex'])
def test_only_assistant_input_view_changes_and_original_history_is_preserved(name):
    request = identity_request(name)
    plan = build_generation_request(request)
    assert plan.history_policy == 'independent_identity'
    assert plan.excluded_assistant_messages == 1
    assert not any(m['role'] == 'assistant' for m in plan.messages)
    assert plan.messages[1] == dict(request.history[0])
    assert len(request.history) == 2
    assert plan.messages[0]['content'] == request.persona_prompt
    assert plan.retrieval.evidence in plan.messages[-1]['content']


@pytest.mark.parametrize('message', ['她是谁？', '继续', '你刚才说了什么？',
                                      '林远是谁？再总结你的回答。', '假设林远是医生，他是谁？'])
def test_expanded_or_different_current_query_cannot_inherit_task(message):
    plan = build_generation_request(replace(identity_request(), message=message))
    assert plan.history_policy == 'conversation'
    assert any(m['role'] == 'assistant' for m in plan.messages)


@pytest.mark.parametrize('changes', [dict(identity_task={}), dict(identity_task=None),
                                    dict(identity_task={'query': '林远是谁？', 'subject': []}),
                                    dict(status='character_abstention'), dict(evidence=''),
                                    dict(source_lookup=True)])
def test_unknown_legacy_empty_and_source_paths_keep_history(changes):
    request = identity_request()
    plan = build_generation_request(replace(request, retrieval=replace(request.retrieval, **changes)))
    assert plan.history_policy == 'conversation'
    assert plan.excluded_assistant_messages == 0
    assert any(m['role'] == 'assistant' for m in plan.messages)


def test_counterfactual_branch_keeps_conversation_dependencies():
    request = replace(identity_request(), character_context=SimpleNamespace(
        branch_context='在这个分支，身份已经改变。', reference_context=''))
    plan = build_generation_request(request)
    assert plan.history_policy == 'conversation'
    assert any(m['role'] == 'assistant' for m in plan.messages)


def test_classified_identity_with_controls_keeps_full_current_user_message():
    message = '我只想聊聊自己的感受，林远是谁？'
    original = identity_request()
    request = replace(original, message=message, retrieval=replace(original.retrieval,
        identity_task={'query': message, 'subject': '林远'}))
    plan = build_generation_request(request)
    assert plan.history_policy == 'independent_identity'
    assert message in plan.messages[-1]['content']
    assert request.history == original.history


def test_packet_budget_preserves_contract_but_no_evidence_disables_it():
    request = identity_request()
    packet = {'text': '实际原文', 'document_ids': ['one'], 'kind': 'evidence'}
    retrieval = replace(request.retrieval, evidence_packets=(packet,))
    plan = build_generation_request(replace(request, retrieval=retrieval))
    assert plan.history_policy == 'independent_identity'
    assert plan.excluded_assistant_messages == 1
    assert plan.retrieval.identity_task == retrieval.identity_task
    huge = {**packet, 'text': '长' * 9000}
    plan = build_generation_request(replace(request, retrieval=replace(retrieval, evidence_packets=(huge,))))
    assert plan.retrieval.reason == 'evidence_budget_exhausted'
    assert plan.history_policy == 'conversation'
    assert any(m['role'] == 'assistant' for m in plan.messages)


@pytest.mark.asyncio
@pytest.mark.parametrize('query_bound', [True, False])
async def test_production_adapter_preserves_query_binding(monkeypatch, query_bound):
    from api import generate
    from db.schemas import MessageRequest
    from knowledge import intent_detector

    observed = []

    async def retrieve(*args):
        return dict(retrieval_strategy='multi_scale_character', confidence=.9, abstained=False,
                    context_text='已检索的身份来源', results=[], citations=[],
                    identity_task={'query': '林远是谁？' if query_bound else '另一问题', 'subject': '林远'})

    async def model(**kwargs):
        observed.append(kwargs['messages'])
        return '身份回答'

    monkeypatch.setattr(intent_detector, 'needs_rag', lambda _: (True, 'identity', None))
    monkeypatch.setattr(generate, '_retrieve_rag_bundle', retrieve)
    monkeypatch.setattr(generate, '_get_system_prompt', lambda _: 'persona')
    await generate._generate_with_vllm(
        MessageRequest(message='林远是谁？', history=list(identity_request().history)), None,
        runtime_config={'useKnowledgeBase': True}, model_generate=model)
    assert len(observed) == 1
    assert any(m['role'] == 'assistant' for m in observed[0]) is not query_bound
