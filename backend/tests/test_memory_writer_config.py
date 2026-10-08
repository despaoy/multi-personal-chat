from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from character.memory_llm import MemoryLlmConfig


def test_explicit_empty_environment_is_disabled_and_isolated(monkeypatch):
    monkeypatch.setenv('MEMORY_LLM_ENABLED', 'true')
    config = MemoryLlmConfig.from_env({})
    assert not config.enabled and config.base_url == '' and config.model == ''
    assert config.queue_size == 64 and config.timeout_seconds == 30


@pytest.mark.parametrize('missing', ['MEMORY_LLM_BASE_URL', 'MEMORY_LLM_MODEL'])
def test_enabled_writer_requires_complete_model_configuration(missing):
    env = dict(MEMORY_LLM_ENABLED='true', MEMORY_LLM_BASE_URL='http://writer.invalid', MEMORY_LLM_MODEL='fixture')
    del env[missing]
    with pytest.raises(ValueError, match='Enabled memory writer requires'):
        MemoryLlmConfig.from_env(env)


def test_documented_inheritance_and_explicit_precedence():
    env = dict(MEMORY_LLM_ENABLED='true', VLLM_BASE_URLS='http://first.invalid,http://second.invalid',
               VLLM_BASE_URL='http://single.invalid', VLLM_SERVED_MODEL_NAME='served', VLLM_MODEL='path',
               VLLM_API_KEY='fixture-shared-key', VLLM_MAX_MODEL_LEN='4096')
    inherited = MemoryLlmConfig.from_env(env)
    assert (inherited.base_url, inherited.model, inherited.api_key, inherited.context_window_tokens) == (
        'http://first.invalid', 'served', 'fixture-shared-key', 4096)
    env.update(MEMORY_LLM_BASE_URL='http://writer.invalid', MEMORY_LLM_MODEL='writer',
               MEMORY_LLM_API_KEY='fixture-writer-key', MEMORY_LLM_CONTEXT_WINDOW_TOKENS='8192')
    explicit = MemoryLlmConfig.from_env(env)
    assert (explicit.base_url, explicit.model, explicit.api_key, explicit.context_window_tokens) == (
        'http://writer.invalid', 'writer', 'fixture-writer-key', 8192)


@pytest.mark.parametrize('value', ['', 'enabled', 'tru', '2'])
def test_invalid_boolean_cannot_disable_the_writer(value):
    with pytest.raises(ValueError, match='MEMORY_LLM_ENABLED'):
        MemoryLlmConfig.from_env({'MEMORY_LLM_ENABLED': value})


@pytest.mark.parametrize(('key', 'value'), [
    ('TIMEOUT', '0'), ('TIMEOUT', 'nan'), ('TIMEOUT', 'inf'),
    ('QUEUE_SIZE', '0'), ('QUEUE_SIZE', '1.5'), ('MAX_INPUT_CHARS', '255'),
    ('CONTEXT_WINDOW_TOKENS', '-1'), ('CONFIDENCE_THRESHOLD', '-0.1'),
    ('CONFIDENCE_THRESHOLD', '1.1'), ('CONFIDENCE_THRESHOLD', 'nan'),
    ('IDLE_SECONDS', '-1'), ('IDLE_SECONDS', 'inf'), ('BATCH_SIZE', '0'),
])
def test_invalid_numeric_configuration_is_rejected_without_clamping(key, value):
    key = 'MEMORY_LLM_' + key
    with pytest.raises(ValueError, match=key):
        MemoryLlmConfig.from_env({key: value})


def test_numeric_boundaries_remain_exact_and_errors_do_not_echo_values():
    config = MemoryLlmConfig.from_env(dict(MEMORY_LLM_TIMEOUT='1', MEMORY_LLM_QUEUE_SIZE='1',
        MEMORY_LLM_MAX_INPUT_CHARS='256', MEMORY_LLM_CONTEXT_WINDOW_TOKENS='0',
        MEMORY_LLM_CONFIDENCE_THRESHOLD='0', MEMORY_LLM_IDLE_SECONDS='0', MEMORY_LLM_BATCH_SIZE='1'))
    assert (config.timeout_seconds, config.queue_size, config.max_input_chars, config.context_window_tokens,
            config.confidence_threshold, config.idle_seconds, config.batch_size) == (1, 1, 256, 0, 0, 0, 1)
    with pytest.raises(ValueError) as error:
        MemoryLlmConfig.from_env({'MEMORY_LLM_TIMEOUT': 'private-accidental-value'})
    assert 'MEMORY_LLM_TIMEOUT' in str(error.value) and 'private-accidental-value' not in str(error.value)


@pytest.mark.asyncio
async def test_invalid_writer_configuration_stops_startup_before_database(monkeypatch):
    from app import main

    runtime = SimpleNamespace(db=object(), is_pg_mode=lambda: False,
                              startup_env={'MEMORY_LLM_ENABLED': 'true'})
    initialize = Mock()
    monkeypatch.setattr(main, 'get_runtime_container', lambda app: runtime)
    monkeypatch.setattr(main, 'validate_or_raise_for_startup', lambda env: None)
    monkeypatch.setattr(main, '_initialize_database', initialize)
    with pytest.raises(ValueError, match='Enabled memory writer requires'):
        async with main.lifespan(SimpleNamespace()):
            pytest.fail('invalid writer configuration admitted startup')
    initialize.assert_not_called()
