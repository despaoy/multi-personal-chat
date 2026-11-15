import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from character.library import CharacterSettings
from character.memory_llm import MemoryLlmConfig
from db.database import SQLiteDB
from repositories.character_chat import CharacterChatRepository
from services.character_chat_memory import enrich_state
from services.character_chat_policy import LocalOnlyPolicyError, local_endpoint_allowed, require_local_provider
from services.character_chat_relationship import enrich_relationship
from services.character_chat_summary import enrich_summary


@pytest.mark.parametrize('url', ['https://api.deepseek.com', 'http://127.0.0.1.evil.test', 'file:///tmp/model', 'https://u:p@localhost', 'http://10.0.0.1:8000'])
def test_unapproved_endpoints_are_not_local(monkeypatch, url):
    monkeypatch.delenv('LOCAL_MODEL_ALLOWED_ORIGINS', raising=False)
    assert not local_endpoint_allowed(url)


def test_local_policy_checks_origin_not_hostname_substring(monkeypatch):
    monkeypatch.setenv('LOCAL_MODEL_ALLOWED_ORIGINS', 'http://192.168.166.171:8001')
    assert local_endpoint_allowed('http://127.0.0.1:1234/v1')
    assert local_endpoint_allowed('http://[::1]:1234/v1')
    assert local_endpoint_allowed('http://192.168.166.171:8001/v1')
    assert not local_endpoint_allowed('http://192.168.166.171:8002/v1')
    context = SimpleNamespace(provider_name='openai_compat', provider=SimpleNamespace(base_url='https://api.deepseek.com'))
    with pytest.raises(LocalOnlyPolicyError):
        require_local_provider({'network_policy': 'local_only'}, context)
    require_local_provider({'network_policy': 'cloud_allowed'}, context)


def test_local_policy_survives_inheritance_and_shared_memory(tmp_path):
    db = SQLiteDB(tmp_path / 'policy.db')
    try:
        repo = CharacterChatRepository(db)
        char = repo.characters.create('owner', CharacterSettings(name='旅人', persona='谨慎', greeting='你好'))
        source = repo.create('owner', char['id'], '私密', network_policy='local_only')
        source = repo.manage_memory('owner', source['id'], source['revision'], content='喜欢茶')
        turn = repo.commit_turn(source, 'one', '你好', '你好', 'test')
        source = repo.get('owner', source['id'])
        for turn_id in (None, turn['id']):
            child = repo.create('owner', char['id'], '继承', source_session_id=source['id'],
                                source_revision=source['revision'], source_turn_id=turn_id)
            assert child['network_policy'] == 'local_only'
        with pytest.raises(ValueError, match='仅本地'):
            repo.create('owner', char['id'], '不可降级', source_session_id=source['id'],
                        source_revision=source['revision'], network_policy='cloud_allowed')
        repo.share_memory('owner', source['id'], source['state']['memories'][0]['id'], source['revision'])
        shared = repo.shared_memories('owner')[0]
        child = repo.create('owner', char['id'], '共享', shared_memory_ids=[shared['id']])
        assert child['network_policy'] == 'local_only'
        with pytest.raises(ValueError, match='仅本地'):
            repo.create('owner', char['id'], '共享不可降级', shared_memory_ids=[shared['id']], network_policy='cloud_allowed')
    finally:
        db.close_connection()


def test_all_auxiliary_models_skip_cloud_without_sending(monkeypatch):
    monkeypatch.setenv('RELATIONSHIP_LLM_ENABLED', 'true')
    monkeypatch.setenv('SUMMARY_LLM_ENABLED', 'true')
    monkeypatch.delenv('LOCAL_MODEL_ALLOWED_ORIGINS', raising=False)
    session = {'id': 'private', 'mode': 'initial', 'network_policy': 'local_only',
               'state': {'memories': [], 'relationship': {'stage': 'stranger', 'summary': ''}}}
    config = MemoryLlmConfig(True, 'https://api.deepseek.com', 'test')
    writer = SimpleNamespace(complete=AsyncMock(side_effect=AssertionError('cloud must not be invoked')))
    history = [{'id': str(i), 'message': f'我喜欢喝茶{i}', 'reply': '知道了'} for i in range(12)]
    states = [
        asyncio.run(enrich_state('我喜欢喝茶', session, 'test', config=config, completion=writer)),
        asyncio.run(enrich_relationship('我喜欢喝茶', '知道了', session, 'test', config=config, completion=writer)),
        asyncio.run(enrich_summary(session, history, 'test', config=config, completion=writer)),
    ]
    for kind, state in zip(('memory', 'relationship', 'summary'), states, strict=True):
        assert state[kind + '_processing'] == 'skipped'
        assert state[kind + '_policy_reason'] == 'local_only'
    writer.complete.assert_not_called()


def test_main_completion_rejects_cloud_before_generation(monkeypatch):
    from inference import model_manager, request_context
    from services.character_chat import complete_with_provider

    provider = SimpleNamespace(base_url='https://api.deepseek.com', async_complete=AsyncMock())
    context = SimpleNamespace(provider_name='openai_compat', provider=provider)
    monkeypatch.setattr(model_manager, 'get_model_manager', lambda **kwargs: object())
    monkeypatch.setattr(request_context, 'resolve_request_context', AsyncMock(return_value=context))
    with pytest.raises(LocalOnlyPolicyError):
        asyncio.run(complete_with_provider(None, {'network_policy': 'local_only'}, 'test',
                                          database=None, http_client_pool=None))
    provider.async_complete.assert_not_called()
