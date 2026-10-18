"""One configuration owner; invalid explicit values cannot select defaults."""

import os

import pytest

from inference import vllm_client


@pytest.fixture(autouse=True)
def isolated_vllm_environment(monkeypatch):
    for name in list(os.environ):
        if name.startswith("VLLM_"):
            monkeypatch.delenv(name)


@pytest.mark.parametrize("urls", [
    " https://one.fixture.invalid/ , https://two.fixture.invalid/ ",
    [" https://one.fixture.invalid/ ", "https://two.fixture.invalid/"],
])
def test_url_inputs_preserve_all_configured_instances(urls):
    client = vllm_client.VLLMClient(base_urls=urls, model="fiction-model")
    assert [item.base_url for item in client._instances] == ["https://one.fixture.invalid", "https://two.fixture.invalid"]


@pytest.mark.parametrize("urls", ["", [], "https://one.fixture.invalid,,https://two.fixture.invalid"])
def test_explicit_empty_urls_do_not_select_environment(monkeypatch, urls):
    monkeypatch.setenv("VLLM_BASE_URL", "https://other.fixture.invalid")
    with pytest.raises(ValueError, match="VLLM_BASE_URLS"):
        vllm_client.VLLMClient(base_urls=urls, model="fiction-model")


@pytest.mark.parametrize("env,arguments,setting", [
    ({}, {"timeout": 0}, "VLLM_TIMEOUT"),
    ({}, {"timeout": float("nan")}, "VLLM_TIMEOUT"),
    ({}, {"max_retries": 0}, "VLLM_MAX_RETRIES"),
    ({"VLLM_MAX_RETRIES": "-1"}, {}, "VLLM_MAX_RETRIES"),
    ({"VLLM_MAX_CONCURRENCY": "0"}, {}, "VLLM_MAX_CONCURRENCY"),
])
def test_invalid_numeric_configuration_fails_before_client_creation(monkeypatch, env, arguments, setting):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match=setting):
        vllm_client.VLLMClient(model="fiction-model", **arguments)


@pytest.mark.asyncio
async def test_singleton_and_constructor_share_configuration(monkeypatch):
    monkeypatch.setattr(vllm_client, "_shared_client", None)
    monkeypatch.setenv("VLLM_BASE_URLS", "https://one.fixture.invalid,https://two.fixture.invalid")
    monkeypatch.setenv("VLLM_TIMEOUT", "2.5")
    monkeypatch.setenv("VLLM_MAX_RETRIES", "1")
    monkeypatch.setenv("VLLM_MAX_CONCURRENCY", "2")
    try:
        client = await vllm_client.get_vllm_client()
        assert client is await vllm_client.get_vllm_client()
        assert client._base_urls == ["https://one.fixture.invalid", "https://two.fixture.invalid"]
        assert (client._timeout, client._max_retries, client._max_concurrency) == (2.5, 1, 2)
    finally:
        await vllm_client.close_shared_vllm_client()


def test_explicit_values_take_precedence_including_empty_api_key(monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "synthetic-environment-key")
    monkeypatch.setenv("VLLM_TIMEOUT", "99")
    monkeypatch.setenv("VLLM_MAX_RETRIES", "4")
    client = vllm_client.VLLMClient(model="fiction-model", timeout=2.5, max_retries=1, api_key="")
    assert (client._timeout, client._max_retries) == (2.5, 1)
    assert "Authorization" not in client._build_headers()
