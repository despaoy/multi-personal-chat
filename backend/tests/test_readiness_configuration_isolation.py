from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from app.main import create_app, readiness_check
from app.runtime import RuntimeContainer


def _application(startup_env: dict[str, str]):
    database = SimpleNamespace(execute_sql=lambda _query: [{"ok": 1}])
    container = RuntimeContainer(
        db=database,
        is_pg_mode=lambda: False,
        inference_runtime=None,
        startup_env={
            "SECURITY_MIDDLEWARE_ENABLED": "false",
            **startup_env,
        },
    )
    return create_app(container)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "required_environment",
    [
        {"MODEL_PROVIDER": "vllm", "VLLM_ENABLED": "false"},
        {"MODEL_PROVIDER": "transformers", "VLLM_ENABLED": "true"},
    ],
)
async def test_readiness_model_requirement_is_isolated_per_application(
    monkeypatch,
    required_environment: dict[str, str],
) -> None:
    vector_db = ModuleType("knowledge.vector_db")
    vector_db.get_vector_db = lambda: object()
    monkeypatch.setitem(sys.modules, "knowledge.vector_db", vector_db)

    from api import generate

    model_checks: list[str] = []

    class HealthyClient:
        async def health_check(self):
            model_checks.append("health")
            return {"summary": {"healthy": 1}}

    async def get_vllm_client():
        model_checks.append("client")
        return HealthyClient()

    monkeypatch.setattr(generate, "get_vllm_client", get_vllm_client)
    monkeypatch.setenv("MODEL_PROVIDER", "vllm")
    monkeypatch.setenv("VLLM_ENABLED", "true")

    model_not_required = _application(
        {
            "MODEL_PROVIDER": "transformers",
            "VLLM_ENABLED": "false",
        }
    )
    model_required = _application(required_environment)

    first_result = await readiness_check(SimpleNamespace(app=model_not_required))
    second_result = await readiness_check(SimpleNamespace(app=model_required))

    assert first_result["deps"]["model"] is True
    assert second_result["deps"]["model"] is True
    assert model_checks == ["client", "health"]
