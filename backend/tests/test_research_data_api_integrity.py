"""Regression tests for research-data APIs and dynamic LoRA discovery."""

from __future__ import annotations

import asyncio
import json
import threading

import pytest
from fastapi import HTTPException
from pydantic import ValidationError


class _ResearchDatabase:
    def __init__(self) -> None:
        self.thread_ids: list[int] = []
        self.router_config: str | None = None

    def execute_sql(self, query, params=None):
        self.thread_ids.append(threading.get_ident())
        if "COUNT(*)" in query:
            return [{"cnt": 0}]
        if "SELECT value FROM config" in query:
            return [{"value": self.router_config}] if self.router_config is not None else []
        if "INSERT INTO config" in query:
            self.router_config = params["val"]
            return 1
        if query.lstrip().upper().startswith(("DELETE", "UPDATE")):
            return 0
        return []

    def execute_sql_insert(self, query, params=None):
        self.thread_ids.append(threading.get_ident())
        return {"lastrowid": 1, "rowcount": 1}


def test_preference_queries_are_bounded_offloaded_and_preserve_not_found(monkeypatch) -> None:
    from api import preferences

    database = _ResearchDatabase()
    monkeypatch.setattr(preferences, "db", database)
    caller_thread = threading.get_ident()

    listed = asyncio.run(
        preferences.list_preferences(
            limit=-100,
            offset=-50,
            current_user={"role": "admin"},
        )
    )
    assert listed["limit"] == 1
    assert listed["offset"] == 0
    assert database.thread_ids
    assert all(thread_id != caller_thread for thread_id in database.thread_ids)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            preferences.delete_preference(
                "missing",
                current_user={"role": "admin"},
            )
        )
    assert exc_info.value.status_code == 404


def test_retrieval_question_delete_preserves_not_found(monkeypatch) -> None:
    from api import retrieval_eval

    monkeypatch.setattr(retrieval_eval, "db", _ResearchDatabase())

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            retrieval_eval.delete_question(
                "missing",
                current_user={"role": "admin"},
            )
        )
    assert exc_info.value.status_code == 404


def test_router_config_updates_are_serialized_without_lost_fields(monkeypatch) -> None:
    from api import router as router_api
    from db.schemas import RouterConfigUpdate

    database = _ResearchDatabase()
    monkeypatch.setattr(router_api, "db", database)

    async def update_both():
        return await asyncio.gather(
            router_api.update_router_config(
                RouterConfigUpdate(enabled=True),
                current_user={"role": "admin"},
            ),
            router_api.update_router_config(
                RouterConfigUpdate(default_adapter="kisaki"),
                current_user={"role": "admin"},
            ),
        )

    asyncio.run(update_both())
    persisted = json.loads(database.router_config or "{}")

    assert persisted["enabled"] is True
    assert persisted["default_adapter"] == "kisaki"


def test_research_request_schemas_reject_unbounded_values() -> None:
    from db.schemas import (
        PreferenceExportRequest,
        RouterConfigUpdate,
        SampleFromHistoryRequest,
    )

    with pytest.raises(ValidationError):
        PreferenceExportRequest(limit=0)
    with pytest.raises(ValidationError):
        SampleFromHistoryRequest(limit=10_000)
    with pytest.raises(ValidationError):
        RouterConfigUpdate(rag_confidence_threshold=1.1)
    with pytest.raises(ValidationError):
        RouterConfigUpdate(persona_keywords={"kisaki": ["x"] * 101})


def test_lora_directory_map_refreshes_after_runtime_upload(tmp_path, monkeypatch) -> None:
    from db import database as database_module

    original_map = dict(database_module.LORA_DIR_MAP)
    monkeypatch.setattr(database_module, "LORA_ROOT", tmp_path)
    try:
        first = tmp_path / "first"
        first.mkdir()
        (first / "adapter_config.json").write_text("{}", encoding="utf-8")
        assert database_module.refresh_lora_dir_map() == {"first": str(first)}

        second = tmp_path / "second" / "final"
        second.mkdir(parents=True)
        (second / "adapter_config.json").write_text("{}", encoding="utf-8")
        refreshed = database_module.refresh_lora_dir_map()

        assert refreshed == {
            "first": str(first),
            "second": str(second.parent),
        }
    finally:
        database_module.LORA_DIR_MAP.clear()
        database_module.LORA_DIR_MAP.update(original_map)