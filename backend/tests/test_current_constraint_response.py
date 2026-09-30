from dataclasses import replace
from types import SimpleNamespace

import pytest

from character.models import CompiledCharacterContext
from inference.current_constraint_response import render_current_constraint_response
from inference.generation_request import GenerationRequest, generate_character_response


def context():
    return CompiledCharacterContext('', '', '', memory_status='no_match')


@pytest.mark.parametrize('action,condition', [('预约会议', '负责人批准'),
    ('去游泳', '天气合适'), ('启动仪器', '取得许可')])
@pytest.mark.asyncio
async def test_complete_correction_ignores_obsolete_assistant_answers(action, condition):
    message = f'更正一下，我只有{condition}才{action}。'
    async def forbidden(**kwargs):
        pytest.fail('An explicit correction must not be reinterpreted by the model')

    result = await generate_character_response(GenerationRequest(message=message,
        character_context=context(), history=({'role': 'assistant', 'content': '你必须满足旧条件甲和旧条件乙。'},)), forbidden)
    assert condition in result.reply and action in result.reply and '旧条件' not in result.reply
    assert result.response_mode == 'current_constraint' and not result.model_invoked
    assert not result.response_citations
    assert '保存' not in result.reply and '记得' not in result.reply


@pytest.mark.parametrize('message', [
    '我只有周末才去游泳。', '补充，我只有周末才去游泳。',
    '更正一下，我只有周末才去游泳。顺便介绍游泳技巧。',
    '更正一下，我只有周末才去游泳请介绍游泳技巧',
    '更正一下，我只有周末才去游泳并告诉我泳池在哪里',
    '假设更正一下，我只有周末才去游泳。',
    '不要保存：更正一下，我只有周末才去游泳。',
    '“更正一下，我只有周末才去游泳。”',
    '更正一下，我只有周末才去游泳吗？',
    '更正一下，我只有完成工作才自杀。',
])
def test_unknown_mixed_nonasserted_and_risky_corrections_keep_normal_path(message):
    assert render_current_constraint_response(message, context()) is None


def test_branch_and_missing_context_do_not_use_current_shortcut():
    query = '更正一下，我只有周末才去游泳。'
    assert render_current_constraint_response(query, None) is None
    assert render_current_constraint_response(query, replace(context(), branch_context='fiction')) is None


@pytest.mark.asyncio
async def test_current_statement_contract_bypasses_rag_but_does_not_report_storage(monkeypatch):
    from api import generate
    from db.schemas import MessageRequest

    async def forbidden(*args, **kwargs):
        pytest.fail('No RAG or model call required')

    monkeypatch.setattr(generate, '_retrieve_rag_bundle', forbidden)
    monkeypatch.setattr(generate, '_get_system_prompt', lambda _: 'persona')
    reply, rag, meta = await generate._generate_with_vllm(
        MessageRequest(message='更正一下，我只有负责人批准才预约会议。'), None,
        prepared_character_turn=SimpleNamespace(compiled=context(), history=()),
        runtime_config={'useKnowledgeBase': True}, model_generate=forbidden)
    assert not rag and meta['answerMode'] == 'current_constraint' and meta['modelInvoked'] is False
    assert not meta['abstained'] and not meta['citations'] and '保存' not in reply
