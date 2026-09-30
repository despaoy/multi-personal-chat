from types import SimpleNamespace

import pytest


async def invoke_default_reviewer(kind, monkeypatch):
    messages = [
        {"role": "system", "content": "仅判断完整输入并返回 JSON。"},
        {"role": "user", "content": "我叫岚舟，老家宣城，目前住在丽水，大学专业环境工程。这四项均为本人资料。"},
    ]
    if kind == "semantic":
        from character.semantic_review_adapter import create_default_semantic_reviewer

        monkeypatch.setenv("DYNAMIC_CONTEXT_SEMANTIC_REVIEW_ENABLED", "true")
        reviewer = create_default_semantic_reviewer()
    elif kind == "selection":
        from character.evidence_selector import create_evidence_selector

        monkeypatch.setenv("CONTEXTUAL_MEMORY_SELECTION_ENABLED", "true")
        reviewer = create_evidence_selector().reviewer
    else:
        from character.contextual_policy import create_contextual_policy

        monkeypatch.setenv("CONTEXTUAL_DECISION_POLICY_ENABLED", "true")
        reviewer = create_contextual_policy().reviewer
    return await reviewer(messages)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,budget", [("semantic", 768), ("selection", 2048), ("policy", 160)])
async def test_native_reviewers_follow_selected_cloud_without_reentering_reply_generation(monkeypatch, kind, budget):
    from inference import model_manager, vllm_client

    calls = []

    class Cloud:
        async def async_complete(self, messages, *, temperature, max_tokens):
            calls.append(dict(messages=messages, temperature=temperature, max_tokens=max_tokens))
            return "云端复核结果", 0.01

        async def async_generate(self, *args, **kwargs):
            raise AssertionError("review must not rebuild a reply prompt")

    cloud = Cloud()
    manager = SimpleNamespace(
        _current_provider=SimpleNamespace(value="openai_compat"), get_current_provider=lambda: cloud
    )
    monkeypatch.setattr(model_manager, "get_model_manager", lambda: manager)

    async def wrong_local_provider():
        raise AssertionError("cloud selection must not access vLLM")

    monkeypatch.setattr(vllm_client, "get_vllm_client", wrong_local_provider)
    assert await invoke_default_reviewer(kind, monkeypatch) == "云端复核结果"
    assert len(calls) == 1 and calls[0]["temperature"] == 0
    assert calls[0]["max_tokens"] == budget
    assert len(calls[0]["messages"]) == 2 and "环境工程" in calls[0]["messages"][1]["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,budget", [("semantic", 768), ("selection", 2048), ("policy", 160)])
async def test_vllm_selection_keeps_shared_low_level_client(monkeypatch, kind, budget):
    from inference import model_manager, vllm_client

    calls = []

    class Local:
        async def generate(self, **kwargs):
            calls.append(kwargs)
            return "本地复核结果"

    manager = SimpleNamespace(_current_provider=SimpleNamespace(value="vllm"))
    monkeypatch.setattr(model_manager, "get_model_manager", lambda: manager)

    async def selected_local():
        return Local()

    monkeypatch.setattr(vllm_client, "get_vllm_client", selected_local)
    assert await invoke_default_reviewer(kind, monkeypatch) == "本地复核结果"
    assert calls[0]["max_tokens"] == budget and calls[0]["lora_name"] is None
    assert calls[0]["temperature"] == 0 and calls[0]["enable_thinking"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["semantic", "selection", "policy"])
async def test_unsupported_selected_provider_does_not_silently_switch_review_model(monkeypatch, kind):
    from inference import model_manager, vllm_client

    manager = SimpleNamespace(_current_provider=SimpleNamespace(value="ollama"))
    monkeypatch.setattr(model_manager, "get_model_manager", lambda: manager)

    async def wrong_local_provider():
        raise AssertionError("unsupported selection must not access vLLM")

    monkeypatch.setattr(vllm_client, "get_vllm_client", wrong_local_provider)
    with pytest.raises(RuntimeError, match="unsupported"):
        await invoke_default_reviewer(kind, monkeypatch)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "base,path",
    [
        ("https://api.deepseek.com", "/v1/chat/completions"),
        ("https://api.deepseek.com/v1", "/v1/chat/completions"),
        ("https://generic.test/v1/", "/v1/chat/completions"),
    ],
)
async def test_native_lowlevel_completion_keeps_complete_messages_and_explicit_reviewer_sampling(
    monkeypatch, base, path
):
    import json
    from contextlib import asynccontextmanager

    import httpx

    from inference import model_manager

    captured = []

    def respond(request):
        captured.append((request.url.path, json.loads(request.content)))
        return httpx.Response(
            200,
            json={"choices": [{"finish_reason": "stop", "message": {"content": '{"strategy_ids":["stay_present"]}'}}]},
        )

    monkeypatch.setattr(
        model_manager,
        "_get_db_config",
        lambda: {
            "temperature": 0.9,
            "maxTokens": 999,
            "openaiCompatBaseUrl": base,
            "openaiCompatApiKey": "private-test-key",
            "openaiCompatModel": "configured-large-model",
        },
    )
    provider = model_manager.OpenAICompatProvider()
    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))

    @asynccontextmanager
    async def acquire(timeout):
        yield client

    monkeypatch.setattr(provider, "_acquire_http_client", acquire)
    messages = [
        {"role": "system", "content": "仅从允许的策略中选择。"},
        {"role": "user", "content": "木刻板坏了，只听吐槽、不建议也不追问；这项边界仍然有效。"},
    ]
    try:
        result, _ = await provider.async_complete(messages, temperature=0, max_tokens=160)
        assert result == '{"strategy_ids":["stay_present"]}'
        assert captured[0][0] == path
        body = captured[0][1]
        assert body["messages"] == messages and len(body["messages"]) == 2
        assert body["temperature"] == 0 and body["max_tokens"] == 160
        assert body["model"] == "configured-large-model"
        assert "chat_template_kwargs" not in body
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("override", [{"lora_name": "adapter"}, {"stream": True}, {"enable_thinking": True}])
async def test_cloud_review_client_cannot_enable_reply_modes(override):
    from inference.review_client import OpenAICompatibleReviewClient

    class NeverCalled:
        async def async_complete(self, *args, **kwargs):
            raise AssertionError("invalid reviewer mode must not access provider")

    arguments = dict(lora_name=None, temperature=0, max_tokens=768, stream=False, enable_thinking=False)
    arguments.update(override)
    with pytest.raises(ValueError):
        await OpenAICompatibleReviewClient(NeverCalled()).generate(
            [{"role": "user", "content": "完整输入"}], **arguments
        )
