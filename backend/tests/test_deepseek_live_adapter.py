import httpx
import pytest

from evaluation.deepseek_live_adapter import DeepSeekEvaluationClient, DeepSeekRecordedWriter


def test_replay_answer_budget_is_cloud_specific_and_explicitly_overridable():
    from evaluation.deepseek_live_adapter import replay_answer_budget

    assert replay_answer_budget(cloud=True) == 1024
    assert replay_answer_budget(cloud=False) == 256
    assert replay_answer_budget(cloud=True, requested=2048) == 2048
    assert replay_answer_budget(cloud=False, requested=512) == 512


@pytest.mark.parametrize('value', [0, 31, 8193, True, 256.5])
def test_replay_answer_budget_rejects_invalid_values(value):
    from evaluation.deepseek_live_adapter import replay_answer_budget

    with pytest.raises(ValueError):
        replay_answer_budget(cloud=True, requested=value)


async def test_answer_and_writer_use_cloud_without_local_parameters_or_secret_in_trace():
    requests = []

    def respond(request):
        import json
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=dict(choices=[dict(finish_reason='stop', message=dict(content='回答'))]))

    client = DeepSeekEvaluationClient('test-credential', 'test-model')
    await client.close()
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    writer = DeepSeekRecordedWriter(client)
    try:
        await client.generate(messages=[dict(role='user', content='问题')], lora_name=None,
            max_tokens=256, temperature=.2, top_p=.9, frequency_penalty=0, repetition_penalty=1)
        await writer.complete([dict(role='user', content='资料')])
        assert [r['purpose'] for r in client.calls] == ['answer', 'writer']
        assert requests[1]['max_tokens'] == 768 and requests[1]['temperature'] == 0
        assert all('chat_template_kwargs' not in r and 'repetition_penalty' not in r for r in requests)
        assert 'test-credential' not in str(client.calls) and len(writer.calls) == 1
        with pytest.raises(ValueError):
            await client.generate(lora_name='adapter')
    finally:
        await client.close()


async def test_all_optional_reviewers_use_named_cloud_routes():
    from evaluation.deepseek_live_adapter import cloud_context_components

    class Client:
        def __init__(self):
            self.calls = []

        async def complete(self, messages, **kwargs):
            self.calls.append(kwargs)
            return '{}'

    client = Client()
    components = cloud_context_components(client)
    messages = [{'role': 'user', 'content': '合成测试'}]
    await components['semantic_estimator']._reviewer(messages)
    await components['memory_selector'].reviewer(messages)
    await components['contextual_policy'].reviewer(messages)
    assert [c['purpose'] for c in client.calls] == ['semantic_review', 'memory_selection', 'contextual_policy']
    assert [c['max_tokens'] for c in client.calls] == [768, 2048, 160]
    assert all(c['temperature'] == 0 for c in client.calls)


async def test_cancelled_reviewer_is_not_missing_from_trace():
    import asyncio

    async def cancelled(request):
        raise asyncio.CancelledError()

    client = DeepSeekEvaluationClient('test-secret', 'test-model')
    await client.close()
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(cancelled))
    try:
        with pytest.raises(asyncio.CancelledError):
            await client.complete([], purpose='semantic_review')
        assert client.calls[0]['status'] == 'cancelled_or_timed_out'
        assert 'test-secret' not in str(client.calls)
    finally:
        await client.close()
