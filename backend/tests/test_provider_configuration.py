"""Provider selection and configuration failure contracts; no cloud calls."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException


@pytest.mark.parametrize("environment,env_provider,stored,expected", [
    ("production", "openai_compat", "mock", "openai_compat"),
    ("production", "", "ollama", "ollama"),
    ("development", "mock", "vllm", "mock"),
    ("test", "", "mock", "mock"),
    ("production", "mock", "vllm", None),
    ("production", "", "mock", None),
    ("development", "mistyped-provider", "vllm", None),
    ("development", "", "mistyped-provider", None),
    ("development", "", None, None),
])
def test_provider_selection_is_explicit(tmp_path, monkeypatch, caplog, environment, env_provider, stored, expected):
    from inference import model_manager

    monkeypatch.setenv("ENVIRONMENT", environment)
    monkeypatch.setenv("MODEL_PROVIDER", env_provider)
    monkeypatch.setattr(model_manager, "_get_db_config", lambda: {"modelProvider": stored})
    if expected is None:
        with pytest.raises(ValueError):
            model_manager.ModelManager(base_dir=tmp_path)
        assert not (tmp_path / "models").exists()
        return
    manager = model_manager.ModelManager(base_dir=tmp_path)
    try:
        assert manager._current_provider.value == expected
        if expected == "mock":
            assert "Explicit mock provider" in caplog.text
    finally:
        manager.shutdown()


def test_runtime_mock_switch_is_rejected_without_changing_provider(tmp_path, monkeypatch):
    from inference import model_manager

    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("MODEL_PROVIDER", "openai_compat")
    monkeypatch.setattr(model_manager, "_get_db_config", lambda: {})
    manager = model_manager.ModelManager(base_dir=tmp_path)
    try:
        with pytest.raises(ValueError, match="mock"):
            manager.set_provider(model_manager.ModelProvider.MOCK)
        assert manager._current_provider == model_manager.ModelProvider.OPENAI_COMPAT
    finally:
        manager.shutdown()


def test_database_configuration_failure_is_not_empty_defaults(monkeypatch):
    from cache import config_cache
    from db import adapter
    from inference import model_manager

    failure = OSError("isolated configuration read failure")

    class Database:
        @property
        def config(self):
            raise failure

    monkeypatch.setattr(config_cache, "get_cached_config", lambda: None)
    monkeypatch.setattr(adapter, "db", Database())
    with pytest.raises(OSError) as caught:
        model_manager._get_db_config()
    assert caught.value is failure


def test_database_configuration_is_converted_once_and_cached(monkeypatch):
    from cache import config_cache
    from db import adapter
    from inference import model_manager

    cached = []
    monkeypatch.setattr(config_cache, "get_cached_config", lambda: None)
    monkeypatch.setattr(config_cache, "set_cached_config", cached.append)
    monkeypatch.setattr(adapter, "db", SimpleNamespace(config={"maxTokens": "733", "useKnowledgeBase": "false"}))
    result = model_manager._get_db_config()
    assert result == {"maxTokens": 733, "useKnowledgeBase": False} and cached == [result]


def test_valid_ttl_cache_is_not_mutated_by_consumer(monkeypatch):
    from cache import config_cache
    from inference import model_manager

    cached = {"modelProvider": "openai_compat"}
    monkeypatch.setattr(config_cache, "get_cached_config", lambda: cached)
    result = model_manager._get_db_config()
    result["modelProvider"] = "mock"
    assert cached == {"modelProvider": "openai_compat"}


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [{}, {"provider": "mistyped-provider"}, [], {"provider": "mock"}])
async def test_provider_api_rejects_missing_invalid_or_production_mock(monkeypatch, body):
    from api import config
    from inference import model_manager

    switches = []

    def switch(provider):
        model_manager.validate_provider(provider)
        switches.append(provider)
        return True

    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setattr(model_manager, "get_model_manager", lambda: SimpleNamespace(set_provider=switch))
    request = SimpleNamespace(json=AsyncMock(return_value=body))
    with pytest.raises(HTTPException) as caught:
        await config.set_model_provider(request, current_user={"isAdmin": True})
    assert caught.value.status_code == 400 and not switches
