from __future__ import annotations

import asyncio
import sys
import time
from types import ModuleType, SimpleNamespace

import pytest

from app.main import create_app, readiness_check
from app.runtime import RuntimeContainer


@pytest.mark.asyncio
async def test_app_readiness_collapses_concurrent_calls_without_loading_rag(
    monkeypatch,
):
    calls = {"database": 0, "rag": 0}

    class Database:
        def execute_sql(self, query):
            assert query == "SELECT 1"
            calls["database"] += 1
            time.sleep(0.03)
            return [{"ok": 1}]

    def fail_if_rag_is_loaded():
        calls["rag"] += 1
        raise AssertionError("readiness must not initialize optional RAG")

    vector_db = ModuleType("knowledge.vector_db")
    vector_db.get_vector_db = fail_if_rag_is_loaded
    monkeypatch.setitem(sys.modules, "knowledge.vector_db", vector_db)
    app = create_app(
        RuntimeContainer(
            db=Database(),
            is_pg_mode=lambda: False,
            inference_runtime=None,
            startup_env={
                "SECURITY_MIDDLEWARE_ENABLED": "false",
                "MODEL_PROVIDER": "transformers",
                "VLLM_ENABLED": "false",
            },
        )
    )

    try:
        results = await asyncio.gather(
            *(readiness_check(SimpleNamespace(app=app)) for _ in range(50))
        )
    finally:
        app.state.readiness_probe.shutdown()

    assert all(result["status"] == "ready" for result in results)
    assert calls == {"database": 1, "rag": 0}
