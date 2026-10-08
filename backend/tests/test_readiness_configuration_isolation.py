from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.main import create_app, readiness_check
from app.runtime import RuntimeContainer


@pytest.mark.asyncio
async def test_readiness_dependency_and_cache_are_isolated_per_application(monkeypatch):
    calls = [0, 0]

    def healthy():
        calls[0] += 1
        return True

    def unavailable():
        calls[1] += 1
        return False

    def application(check):
        return create_app(RuntimeContainer(
            db=SimpleNamespace(execute_sql=lambda query: [{"ok": 1}]),
            is_pg_mode=lambda: False,
            model_check=check,
            startup_env={"SECURITY_MIDDLEWARE_ENABLED": "false", "MODEL_PROVIDER": "openai_compat", "VLLM_ENABLED": "false"},
        ))

    first, second = application(healthy), application(unavailable)
    monkeypatch.setenv("MODEL_PROVIDER", "vllm")
    monkeypatch.setenv("VLLM_ENABLED", "true")
    try:
        assert (await readiness_check(SimpleNamespace(app=first)))["deps"]["model"] is True
        with pytest.raises(HTTPException) as caught:
            await readiness_check(SimpleNamespace(app=second))
        assert caught.value.status_code == 503
        assert caught.value.detail["deps"] == {"database": True, "model": False}
        assert (await readiness_check(SimpleNamespace(app=first)))["status"] == "ready"
        assert calls == [1, 1]
    finally:
        first.state.readiness_probe.shutdown()
        second.state.readiness_probe.shutdown()
