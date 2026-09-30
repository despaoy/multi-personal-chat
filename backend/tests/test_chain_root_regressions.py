"""Regressions from the real-model effect evaluation, without model calls."""
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from character.memory_extractor import extract_memories
from character.models import CharacterProfile, CompiledCharacterContext, DecisionPlan, InteractionState
from character.output_guard import ReplyGuard, build_reply_guard, retryable_violations, validate_reply
from db.schemas import MessageRequest
from inference.generation_request import GenerationRequest, _estimated_tokens, build_generation_request


def test_protagonist_is_not_master_servant_relation():
    from knowledge.multiscale_rag.service import choose_card_types
    from knowledge.retrieval_core.query import QueryAnalyzer
    query = '月社妃是哪本魔法之书的主人公？'
    analysis = QueryAnalyzer([]).analyze(query)
    assert not analysis.relation_type_preferences
    assert 'fact' in choose_card_types(analysis, query)
    relation = QueryAnalyzer([]).analyze('她的主人是谁？')
    assert relation.relation_type_preferences


def test_recalling_a_fact_is_not_a_bid_for_affection():
    from character.situation_analyzer import SituationAnalyzer
    state = SituationAnalyzer().estimate('你记得我喜欢什么饮料吗？')
    acts = {s.signal_id: s.score for s in state.user_acts}
    assert acts.get('information_request', 0) >= .5
    assert acts.get('affiliation_bid', 0) < .5
    social = SituationAnalyzer().estimate('你还记得我吗？')
    assert any(s.signal_id == 'affiliation_bid' for s in social.user_acts)


@pytest.mark.parametrize('history', [[], [{'role': 'user', 'content': '假如我喜欢咖啡呢？'}]])
def test_recall_question_or_hypothesis_does_not_disable_fact_guard(history):
    guard = build_reply_guard(CharacterProfile('test', 'Test'), '你记得我喜欢什么饮料吗？',
                              history, InteractionState(), DecisionPlan())
    assert 'unsupported_user_fact' in validate_reply('我记得你喜欢喝柠檬茶。', guard)


def test_no_match_is_explicit_and_does_not_claim_user_never_said_it():
    context = CompiledCharacterContext('', '', '', memory_status='no_match')
    plan = build_generation_request(GenerationRequest(message='我喜欢喝什么？', character_context=context))
    assert '没有找到' in plan.messages[0]['content']
    assert '不代表用户从未说过' in plan.messages[0]['content']


@pytest.mark.asyncio
async def test_missing_evidence_fabrication_closes_without_second_model_call():
    from unittest.mock import AsyncMock

    from inference.generation_request import generate_character_response
    model = AsyncMock(return_value='我记得你喜欢喝柠檬茶。')
    result = await generate_character_response(GenerationRequest(
        message='我喜欢喝什么？', character_context=CompiledCharacterContext('', '', '', memory_status='no_match'),
        reply_guard=ReplyGuard(forbid_unsupported_user_fact=True), reply_guard_mode='strict'), model)
    model.assert_awaited_once()
    assert result.guard_fallback == 'unsupported_user_fact'
    assert not result.guard_retried and '柠檬茶' not in result.reply
    assert '刚才' not in result.reply and '没告诉' not in result.reply


@pytest.mark.parametrize('reply', ['我不知道你喜欢什么饮料。', '我并不记得你喜欢喝什么。', '你喜欢喝什么？'])
def test_honest_uncertainty_is_not_a_fabricated_preference(reply):
    assert 'unsupported_user_fact' not in validate_reply(reply, ReplyGuard(forbid_unsupported_user_fact=True))


def test_uncertainty_prefix_does_not_license_later_fabrication():
    guard = ReplyGuard(forbid_unsupported_user_fact=True)
    assert 'unsupported_user_fact' in validate_reply('我不知道你喜欢什么，但你肯定喜欢咖啡。', guard)
    assert 'unsupported_user_fact' in validate_reply('我不知道为什么你喜欢咖啡。', guard)


def test_natural_preference_preserves_condition():
    items = extract_memories('咖啡我挺爱喝的，不过过了下午三点就不碰了。')
    assert len(items) == 1 and items[0].memory_key == 'preference_咖啡'
    assert '下午三点' in items[0].content
    assert items[0].qualifiers


@pytest.mark.parametrize('text', [
    '我喜欢咖啡。小说里的角色说：“我讨厌咖啡。”',
    '我喜欢咖啡，你喜欢什么？',
])
def test_separate_fiction_or_question_does_not_discard_self_assertion(text):
    items = extract_memories(text)
    assert len(items) == 1 and items[0].polarity == 'like'


@pytest.mark.parametrize('text', ['假如咖啡我挺爱喝的。', '“咖啡我挺爱喝的。”', '不要记住，咖啡我挺爱喝的。'])
def test_new_word_order_keeps_write_gates(text):
    assert not extract_memories(text)


def test_serving_budget_is_configurable_and_bounds_history(monkeypatch):
    monkeypatch.setenv('VLLM_MAX_MODEL_LEN', '4096')
    request = GenerationRequest(message='继续', max_tokens=512, history=tuple(
        {'role': 'user', 'content': '旧' * 600} for _ in range(20)))
    plan = build_generation_request(request)
    assert request.context_window_tokens == 4096
    assert sum(_estimated_tokens(m['content']) + 4 for m in plan.messages) + 512 + 512 <= 4096
    with pytest.raises(ValueError, match='context budget'):
        build_generation_request(GenerationRequest(message='长' * 5000))


def test_lightweight_advice_is_soft_but_explicit_boundary_still_blocks():
    reply = '你可以休息一下。'
    soft = ReplyGuard(forbid_unprompted_advice=True)
    assert not retryable_violations(reply, soft, validate_reply(reply, soft))
    assert retryable_violations(reply, soft, validate_reply(reply, soft), strict=True)
    hard = ReplyGuard(forbid_advice=True)
    assert retryable_violations(reply, hard, validate_reply(reply, hard))


@pytest.mark.asyncio
async def test_character_entry_does_not_require_or_auto_select_lora(monkeypatch):
    from unittest.mock import AsyncMock

    import api.generate as gen
    from inference import model_manager as mm

    monkeypatch.setattr(mm, 'get_model_manager', lambda: SimpleNamespace(_current_provider=SimpleNamespace(value='vllm')))
    monkeypatch.setattr(gen, 'INPUT_VALIDATOR_AVAILABLE', False)
    monkeypatch.setattr(gen, 'response_cache', None)
    monkeypatch.setattr(gen, '_ensure_vllm', AsyncMock(return_value=True))
    monkeypatch.setattr(gen, '_vllm_client', object())
    prepare = AsyncMock(return_value=SimpleNamespace(character_id='tsukiyashiro_kisaki'))
    monkeypatch.setattr(gen, '_prepare_character_turn', prepare)
    generate = AsyncMock(return_value=('reply', False, {}))
    monkeypatch.setattr(gen, '_generate_with_vllm', generate)
    database = SimpleNamespace(config={}, loras=[{'name': 'hutao', 'status': 'active'}])
    req = MessageRequest(message='你好', characterId='tsukiyashiro_kisaki', userId='alice', sessionId='a')
    await gen._generate_reply_impl(req, message_db=database, persist_message=False, record_invocation=False)
    prepare.assert_awaited_once()
    assert generate.call_args.args[1] is None
    req.loraName = 'hutao'
    with pytest.raises(HTTPException) as exc:
        await gen._generate_reply_impl(req, message_db=database, persist_message=False, record_invocation=False)
    assert exc.value.status_code == 422


@pytest.mark.asyncio
async def test_public_character_scope_belongs_to_login():
    from unittest.mock import AsyncMock

    import api.generate as gen
    service = SimpleNamespace(generate_queued=AsyncMock(return_value='ok'))
    req = MessageRequest(message='你好', characterId='tsukiyashiro_kisaki', senderId='victim', userId='victim', sessionId='test')
    await gen.generate_reply(req, {'id': 'alice'}, service)
    assert req.senderId == req.userId == 'alice'
    assert req.platform == 'web' and req.adapter == 'web-character'
