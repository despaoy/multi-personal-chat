import json

import httpx
import pytest

from character.memory_llm import MemoryLlmConfig, OpenAICompatibleMemoryCompletion
from inference.openai_protocol import completed_chat_content, nonthinking_parameters


@pytest.mark.parametrize(
    "base_url,deepseek",
    [
        ("https://api.deepseek.com", True),
        ("https://api.deepseek.com/v1", True),
        ("https://API.DEEPSEEK.COM/v1", True),
        ("http://localhost:8001", False),
        ("https://api.deepseek.com.example.invalid/v1", False),
        ("https://api.deepseek.com@example.invalid/v1", False),
    ],
)
def test_nonthinking_switch_matches_actual_provider_host(base_url, deepseek):
    expected = {"thinking": {"type": "disabled"}} if deepseek else {"chat_template_kwargs": {"enable_thinking": False}}
    assert nonthinking_parameters(base_url) == expected


@pytest.mark.parametrize(
    "reason,content",
    [
        ("length", '{"memories":[]}'),
        ("content_filter", "filtered"),
        ("tool_calls", "unusable"),
        (None, "unconfirmed"),
        ("stop", None),
        ("stop", " "),
        ("stop", 42),
    ],
)
def test_incomplete_or_empty_responses_cannot_be_used_as_completed_content(reason, content):
    payload = {"choices": [{"finish_reason": reason, "message": {"content": content, "reasoning_content": "hidden"}}]}
    with pytest.raises(ValueError):
        completed_chat_content(payload)


@pytest.mark.parametrize(
    "payload",
    [None, [], {}, {"choices": []}, {"choices": [None]}, {"choices": [{"finish_reason": "stop", "message": None}]}],
)
def test_malformed_response_cannot_enter_generation_or_memory(payload):
    with pytest.raises(ValueError):
        completed_chat_content(payload)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "base_url,deepseek", [("https://api.deepseek.com", True), ("http://local.test:8001/v1", False)]
)
async def test_native_memory_http_sends_the_correct_switch_and_rejects_even_parseable_truncation(base_url, deepseek):
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        reason = "stop" if len(requests) == 1 else "length"
        return httpx.Response(
            200, json={"choices": [{"finish_reason": reason, "message": {"content": '{"memories":[]}'}}]}
        )

    completion = OpenAICompatibleMemoryCompletion(
        MemoryLlmConfig(
            enabled=True, base_url=base_url, model="large-configured-model", api_key="private-test-credential"
        )
    )
    completion._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    messages = [{"role": "user", "content": "请记住我叫岚舟，老家宣城，现住丽水，专业环境工程。"}]
    try:
        assert await completion.complete(messages) == '{"memories":[]}'
        with pytest.raises(ValueError, match="finish normally"):
            await completion.complete(messages)
        assert all(request["messages"] == messages for request in requests)
        assert all(request["model"] == "large-configured-model" for request in requests)
        switch = "thinking" if deepseek else "chat_template_kwargs"
        other = "chat_template_kwargs" if deepseek else "thinking"
        assert all(
            request[switch] == nonthinking_parameters(base_url)[switch] and other not in request for request in requests
        )
    finally:
        await completion.close()


@pytest.mark.parametrize(
    "base_url", ["https://api.openai.com", "http://localhost:8001", "https://api.deepseek.com.example.invalid"]
)
def test_generic_chat_provider_does_not_receive_local_template_extensions(base_url):
    assert nonthinking_parameters(base_url, local_template=False) == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("base_url,deepseek", [("https://api.deepseek.com", True), ("https://generic.test", False)])
async def test_native_answer_uses_complete_content_and_provider_specific_thinking_switch(
    monkeypatch, base_url, deepseek
):
    from contextlib import asynccontextmanager

    from inference import model_manager

    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        reason = "stop" if len(requests) == 1 else "length"
        return httpx.Response(200, json={"choices": [{"finish_reason": reason, "message": {"content": "完整回答"}}]})

    monkeypatch.setattr(model_manager, "_get_db_config", lambda: {"maxTokens": 1024, "temperature": 0.2})
    provider = model_manager.OpenAICompatProvider()
    provider.base_url = base_url
    provider.api_key = "private-test-credential"
    provider.model = "configured-large-model"
    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))

    @asynccontextmanager
    async def acquire(timeout):
        yield client

    monkeypatch.setattr(provider, "_acquire_http_client", acquire)
    history = [
        {"role": "system", "content": "完整人物上下文"},
        {"role": "user", "content": "我叫岚舟，老家宣城，现住丽水，专业环境工程。"},
    ]
    try:
        answer, _ = await provider.async_generate("请逐项回答", session_history=history, max_tokens_override=733)
        assert answer == "完整回答"
        with pytest.raises(ValueError, match="finish normally"):
            await provider.async_generate("请逐项回答", session_history=history, max_tokens_override=733)
        assert all(
            request["max_tokens"] == 733
            and request["messages"] == history + [{"role": "user", "content": "请逐项回答"}]
            for request in requests
        )
        assert all("chat_template_kwargs" not in request for request in requests)
        assert all((request.get("thinking") == {"type": "disabled"}) == deepseek for request in requests)
    finally:
        await client.aclose()
