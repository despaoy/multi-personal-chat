"""Dataset imports report actual parsing/copy outcomes with isolated files."""

import json
import shutil
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from api import training as training_api
from app.dependencies import get_current_admin
from training import preprocessor


@pytest.mark.asyncio
@pytest.mark.parametrize("variant", ["metadata", "raw", "corrupt_metadata", "corrupt_raw", "outside", "traversal", "missing", "copy_error"])
async def test_import_reports_confirmed_outcome(monkeypatch, tmp_path, variant):
    allowed = tmp_path / "allowed"
    source = allowed / "source"
    source.mkdir(parents=True)
    target = tmp_path / "datasets"
    target.mkdir()
    raw = [{"instruction": "3张票，每张2元，尚未付款，下雨取消。", "input": "", "output": "共6元，尚未付款，下雨取消。"}]
    (source / "train.json").write_text(json.dumps(raw), encoding="utf-8")
    if variant in {"metadata", "corrupt_metadata"}:
        (source / "dataset_info.json").write_text("{broken" if variant == "corrupt_metadata" else json.dumps({"stats": {"total": 1, "train": 1, "eval": 0}}), encoding="utf-8")
    if variant == "corrupt_raw":
        (source / "train.json").write_text("{broken", encoding="utf-8")
    source_path = str(source)
    if variant == "outside":
        outside = tmp_path / "outside"
        outside.mkdir()
        source_path = str(outside)
    elif variant == "traversal":
        source_path = str(allowed / ".." / "allowed" / "source")
    elif variant == "missing":
        source_path = str(allowed / "missing")
    elif variant == "copy_error":
        def fail_copy(*args, **kwargs):
            raise OSError("synthetic-copy-failure")
        monkeypatch.setattr(shutil, "copytree", fail_copy)
    monkeypatch.setenv("MULTIPERSONAL_LAB_ROOT", str(allowed))
    monkeypatch.delenv("QQCHAT_LAB_ROOT", raising=False)
    monkeypatch.setattr(preprocessor, "get_dataset_preprocessor", lambda: SimpleNamespace(data_dir=target))
    app = FastAPI()
    app.include_router(training_api.router)
    app.dependency_overrides[get_current_admin] = lambda: {"role": "admin"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/training/datasets/scan/import", json={"source_path": source_path, "dataset_name": "verified"})
    if variant in {"metadata", "raw"}:
        assert response.status_code == 200 and response.json()["success"] is True
        assert response.json()["stats"] == {"total": 1, "train": 1, "eval": 0}
        assert (target / "verified" / "train.json").read_bytes() == (source / "train.json").read_bytes()
    else:
        assert response.status_code == (404 if variant == "missing" else 500 if variant == "copy_error" else 400)
        assert list(target.iterdir()) == [] and "success" not in response.json()
    assert source.is_dir()
