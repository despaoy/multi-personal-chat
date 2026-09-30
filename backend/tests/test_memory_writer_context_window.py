import json

import pytest

from character.memory_llm import MemoryLlmConfig, build_memory_llm_messages


def build(message, history=(), window=8192):
    return build_memory_llm_messages(message, (), tuple(history), (), 2000, .85,
                                     context_window_tokens=window)


def test_complete_long_message_uses_spare_context_instead_of_fixed_prefix_cap():
    message = '我的专业是档案学。' + '整理资料。' * 430 + '最后的限定也必须保留。'
    messages = build(message)
    assert json.loads(messages[1]['content'])['current_user_message'] == message


def test_long_previous_turn_does_not_block_short_current_fact_when_whole_request_fits():
    previous = '这段小说不是我的经历。' + '上下文。' * 540 + '不是我的住址。'
    messages = build('我的专业是统计学。', [{'role': 'user', 'content': previous}])
    assert json.loads(messages[1]['content'])['recent_history'][0]['content'] == previous


def test_full_serialized_prompt_output_and_margin_all_count_toward_window():
    from inference.generation_request import CONTEXT_SAFETY_MARGIN_TOKENS, _estimated_tokens

    messages = build('我的专业是统计学。')
    required = sum(_estimated_tokens(m['content']) + 4 for m in messages) + 768 + CONTEXT_SAFETY_MARGIN_TOKENS
    assert build('我的专业是统计学。', window=required)
    with pytest.raises(ValueError, match='budget'):
        build('我的专业是统计学。', window=required - 1)


def test_runtime_uses_serving_window_and_allows_explicit_legacy_mode(monkeypatch):
    monkeypatch.delenv('MEMORY_LLM_CONTEXT_WINDOW_TOKENS', raising=False)
    monkeypatch.setenv('VLLM_MAX_MODEL_LEN', '8192')
    assert MemoryLlmConfig.from_env().context_window_tokens == 8192
    monkeypatch.setenv('MEMORY_LLM_CONTEXT_WINDOW_TOKENS', '4096')
    assert MemoryLlmConfig.from_env().context_window_tokens == 4096
    monkeypatch.setenv('MEMORY_LLM_CONTEXT_WINDOW_TOKENS', '0')
    assert MemoryLlmConfig.from_env().context_window_tokens == 0
