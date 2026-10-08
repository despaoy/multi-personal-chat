"""Legacy strict-mode retry contract; normal chat no longer blocks this heuristic."""
from dataclasses import replace

import pytest

from character.models import CompiledCharacterContext
from character.output_guard import ReplyGuard
from inference.generation_request import (
    GenerationRequest,
    ReplyValidationError,
    RetrievalResult,
    generate_character_response,
)


def request(**kwargs):
    return GenerationRequest(message='画插画的是谁？', max_tokens=200,
        reply_guard_mode='strict',
        character_context=CompiledCharacterContext('', '', '', memory_status='no_match'),
        reply_guard=ReplyGuard(forbid_unsupported_user_fact=True), **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize('memory_status', ['no_match', 'not_checked'])
@pytest.mark.parametrize('channel', ['episodic_reference_context', 'conversation_reference_context',
                                    'reference_context', 'history'])
async def test_source_presence_does_not_by_itself_expand_retry_policy(channel, memory_status):
    req = request()
    req = replace(req, character_context=replace(req.character_context, memory_status=memory_status))
    if channel == 'history':
        req = replace(req, history=({'role': 'user', 'content': '我最近在画插画。'},))
    else:
        req = replace(req, character_context=replace(req.character_context, **{channel: '我最近在画插画。'}))
    calls = []

    async def model(**kwargs):
        calls.append(kwargs)
        return '你一直喜欢画插画。' if len(calls) == 1 else '画插画的是你。'

    result = await generate_character_response(req, model)
    assert len(calls) == 2
    assert result.guard_retried and not result.guard_fallback
    assert result.reply == '画插画的是你。'
    assert calls[0]['messages'][-1] == calls[1]['messages'][-1]
    assert '你一直喜欢画插画。' not in str(calls[1]['messages'])


@pytest.mark.asyncio
@pytest.mark.parametrize('evidence', ['none', 'rag_only', 'assistant_only', 'trimmed_user_history'])
async def test_no_admitted_user_evidence_rejects_failed_correction(evidence):
    req = request()
    if evidence == 'rag_only':
        req = replace(req, retrieval=RetrievalResult(status='ok', evidence='原作人物的朋友关系'))
    elif evidence == 'assistant_only':
        req = replace(req, history=({'role': 'assistant', 'content': '你画插画。'},))
    elif evidence == 'trimmed_user_history':
        req = replace(req, history=({'role': 'user', 'content': '我在画插画。' * 2000},), context_window_tokens=1800)
    calls = []

    async def model(**kwargs):
        calls.append(kwargs)
        return '你一直喜欢画插画。'

    with pytest.raises(ReplyValidationError) as caught:
        await generate_character_response(req, model)
    assert len(calls) == 2
    assert caught.value.violations == ('unsupported_user_fact',)



@pytest.mark.asyncio
async def test_source_presence_never_licenses_unrelated_fact_or_unbounded_retries():
    req = request()
    req = replace(req, character_context=replace(req.character_context,
        memory_status='not_checked', episodic_reference_context='我在画插画。'))
    calls = []

    async def model(**kwargs):
        calls.append(kwargs)
        return '你一直喜欢咖啡。'

    with pytest.raises(ReplyValidationError) as caught:
        await generate_character_response(req, model)
    assert len(calls) == 2
    assert caught.value.violations == ('unsupported_user_fact',)
    assert '咖啡' not in str(caught.value)
