import json

import pytest

from inference.reasoning_text import ReasoningTextFilter, strip_reasoning_text


@pytest.mark.parametrize('raw,expected', [
    ('普通回复 < 3', '普通回复 < 3'), ('保留尾部<', '保留尾部<'),
    ('<think>隐藏过程</think>答复', '答复'),
    ('前言<think>隐藏过程</think>答复', '前言答复'),
    ('<THINK>隐藏\n过程</THINK>答复', '答复'),
    ('<think>未完成的推理', ''), ('前言<think>未完成的推理', '前言'),
    ('<think>内层<think>秘密</think>仍隐藏</think>答复', '答复'),
    ('<think>一</think>答复<think>二</think>结束', '答复结束'),
])
def test_every_two_chunk_boundary_and_single_character_stream(raw, expected):
    for index in range(len(raw) + 1):
        stream = ReasoningTextFilter()
        assert stream.feed(raw[:index]) + stream.feed(raw[index:]) + stream.finish() == expected
    stream = ReasoningTextFilter()
    assert ''.join(stream.feed(char) for char in raw) + stream.finish() == expected
    assert strip_reasoning_text(raw) == expected


def test_hidden_text_is_discarded_not_buffered():
    stream = ReasoningTextFilter()
    assert stream.feed('<think>' + '隐' * 100000 + '</thi') == ''
    assert len(stream._pending) <= 7
    assert stream.feed('nk>可见') == '可见'


def test_nonstream_template_supplied_opener():
    assert strip_reasoning_text('隐藏过程</think>最终回复') == '最终回复'


@pytest.mark.asyncio
async def test_nonstream_adapter_does_not_return_truncated_reasoning(monkeypatch):
    from inference.vllm_client import VLLMClient

    class Response:
        status_code = 200
        def json(self):
            return {'choices': [{'finish_reason': 'length', 'message': {'content': '<think>未完成隐藏内容'}}]}
    class Client:
        async def post(self, *args, **kwargs):
            return Response()
    client = VLLMClient(base_urls='http://model:8001', model='qwen3-8b-instruct-awq')
    async def fake_client():
        return Client()
    monkeypatch.setattr(client, '_ensure_client', fake_client)
    assert await client._generate_non_stream([], None, 0.7, 8, 0.9, 1.0, 0.0, True) == ''
    assert client._instances[0].current_connections == 0


@pytest.mark.asyncio
@pytest.mark.parametrize('parts,expected', [
    (['<thi', 'nk>隐藏', '</th', 'ink>', '正常回答'], '正常回答'),
    (['<think>', '未完成隐藏内容'], ''),
    (['原样', '正常文本<'], '原样正常文本<'),
])
async def test_actual_stream_adapter_filters_before_yield(monkeypatch, parts, expected):
    from inference.vllm_client import VLLMClient

    class Response:
        status_code = 200
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def aiter_lines(self):
            yield 'data: ' + json.dumps({'choices': [{'delta': {'reasoning_content': '原生推理字段'}}]})
            for part in parts:
                yield 'data: ' + json.dumps({'choices': [{'delta': {'content': part}}]})
            yield 'data: [DONE]'

    class Client:
        def stream(self, *args, **kwargs):
            return Response()

    client = VLLMClient(base_urls='http://model:8001', model='qwen3-8b-instruct-awq')
    async def fake_client():
        return Client()
    monkeypatch.setattr(client, '_ensure_client', fake_client)
    chunks = [part async for part in client._generate_stream([], None, 0.7, 2048, 0.9, 1.0, 0.0, False)]
    assert ''.join(chunks) == expected
    assert all('隐藏' not in part and '原生推理字段' not in part for part in chunks)
    assert client._instances[0].current_connections == 0
