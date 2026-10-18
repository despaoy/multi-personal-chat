"""Style catalogs are complete published snapshots, with no empty failure cache."""

import builtins
import json
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from api import training as training_api
from app.dependencies import get_current_admin
from training import preprocessor


@pytest.mark.asyncio
@pytest.mark.parametrize("variant", ["configured", "missing", "empty", "broken", "partial", "wrong_root", "read_error"])
async def test_style_catalog_publishes_only_complete_loads(monkeypatch, tmp_path, variant):
    data = tmp_path / "data"
    data.mkdir()
    path = data / "user_styles.json"
    valid = {"fiction": {"name": "虚构语气配置", "description": "合成完整配置", "filler_words": ["嗯"], "formality": .3}}
    value = {} if variant == "empty" else [valid] if variant == "wrong_root" else dict(valid)
    if variant == "partial":
        value["invalid"] = {"description": "缺少必要名称"}
    if variant != "missing":
        path.write_text("{broken" if variant == "broken" else json.dumps(value), encoding="utf-8")
    original_class, original_open = preprocessor.DatasetPreprocessor, builtins.open
    monkeypatch.setattr(preprocessor, "_dataset_preprocessor", None)
    monkeypatch.setattr(preprocessor, "DatasetPreprocessor", lambda: original_class(base_dir=tmp_path))
    if variant == "read_error":
        def deny_read(file, *args, **kwargs):
            if Path(file) == path and args and args[0] == "r":
                raise PermissionError("synthetic-style-read-error")
            return original_open(file, *args, **kwargs)
        monkeypatch.setattr(builtins, "open", deny_read)
    app = FastAPI()
    app.include_router(training_api.router)
    app.dependency_overrides[get_current_admin] = lambda: {"role": "admin"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/training/styles")
        if variant in {"configured", "missing", "empty"}:
            assert response.status_code == 200 and response.json()["success"] is True
            assert [item["name"] for item in response.json()["styles"]] == (["fiction"] if variant == "configured" else [])
            if variant == "configured":
                instance = preprocessor.get_dataset_preprocessor()
                style = instance.get_style("fiction")
                assert style.name == "虚构语气配置" and style.description == "合成完整配置"
                assert style.filler_words == ["嗯"] and style.sentence_enders == [] and style.formality == .3
                instance.list_styles().clear()
                assert list(instance.list_styles()) == ["fiction"]
        else:
            assert response.status_code == 500 and "success" not in response.json()
            assert preprocessor._dataset_preprocessor is None
            if variant == "broken":
                path.write_text(json.dumps(valid), encoding="utf-8")
                repaired = await client.get("/api/training/styles")
                assert repaired.status_code == 200 and [item["name"] for item in repaired.json()["styles"]] == ["fiction"]
