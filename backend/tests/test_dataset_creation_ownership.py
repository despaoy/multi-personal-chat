"""Dataset names have one owner and failed writes leave no new dataset."""

import copy
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from api import training as training_api
from app.dependencies import get_current_admin, get_request_database
from db.database import SQLiteDB
from training import preprocessor


def rows(label):
    return [
        {
            "instruction": f"完整问题{index}？",
            "input": "尚未付款，下雨取消。",
            "output": f"{label}回答{index}，尚未付款，下雨取消。",
        }
        for index in range(10)
    ]


def test_concurrent_creation_has_one_owner_and_complete_original_rows(monkeypatch, tmp_path):
    processor = preprocessor.DatasetPreprocessor(base_dir=tmp_path)
    destination = processor.data_dir / "same-name"
    barrier = threading.Barrier(2)
    original_mkdir = Path.mkdir

    def enter_directory(path, *args, **kwargs):
        if path == destination:
            barrier.wait(timeout=5)
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", enter_directory)

    def create(label):
        try:
            processor.prepare_training_data(rows(label), output_name="same-name")
        except FileExistsError:
            return "conflict", label
        return "created", label

    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(create, ["first", "second"]))
    assert sorted(status for status, _ in results) == ["conflict", "created"]
    winner = next(label for status, label in results if status == "created")
    stored = [
        *json.loads((destination / "train.json").read_text()),
        *json.loads((destination / "eval.json").read_text()),
    ]
    assert sorted(json.dumps(row, sort_keys=True) for row in stored) == sorted(
        json.dumps(row, sort_keys=True) for row in rows(winner)
    )
    assert json.loads((destination / "dataset_info.json").read_text())["stats"] == {"total": 10, "train": 9, "eval": 1}


def test_write_failure_preserves_error_and_same_name_can_be_created_after_repair(monkeypatch, tmp_path):
    processor = preprocessor.DatasetPreprocessor(base_dir=tmp_path)
    source = rows("complete")
    original = copy.deepcopy(source)
    original_dump = preprocessor.json.dump
    failure = OSError("synthetic-dataset-write-failure")

    def dump(value, stream, **kwargs):
        if Path(stream.name).name == "eval.json":
            raise failure
        return original_dump(value, stream, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(preprocessor.json, "dump", dump)
        with pytest.raises(OSError) as caught:
            processor.prepare_training_data(source, output_name="repaired")
    assert caught.value is failure and not (processor.data_dir / "repaired").exists()
    destination, stats = processor.prepare_training_data(source, output_name="repaired")
    assert source == original and stats["total_samples"] == 10
    assert {path.name for path in destination.iterdir()} == {"train.json", "eval.json", "dataset_info.json"}


@pytest.mark.parametrize("entry", ["upload", "saved"])
async def test_creation_api_reports_conflict_keeps_old_bytes_and_accepts_new_name(monkeypatch, tmp_path, entry):
    processor = preprocessor.DatasetPreprocessor(base_dir=tmp_path)
    monkeypatch.setattr(preprocessor, "get_dataset_preprocessor", lambda: processor)
    database = SQLiteDB(tmp_path / "saved.sqlite")
    app = FastAPI()
    app.include_router(training_api.router)
    app.dependency_overrides[get_current_admin] = lambda: {"role": "admin", "username": "synthetic-owner"}
    app.dependency_overrides[get_request_database] = lambda: database
    source = rows("original")

    def dialogues(records):
        return [
            {
                "conversations": [
                    {"role": "user", "content": row["instruction"] + row["input"]},
                    {"role": "assistant", "content": row["output"]},
                ]
            }
            for row in records
        ]

    saved_id = database.execute_sql_insert(
        "INSERT INTO saved_dialogues (name,character_desc,dialogue_count,dialogues_json,created_at,updated_at) "
        "VALUES ('synthetic','complete',10,:dialogues,'2026-10-30','2026-10-30')",
        {"dialogues": json.dumps(dialogues(source))},
    )["lastrowid"]
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
        ) as client:

            async def create(name, records):
                if entry == "upload":
                    return await client.post("/api/training/datasets", json={"dataset_name": name, "data": records})
                database.execute_sql(
                    "UPDATE saved_dialogues SET dialogues_json=:dialogues WHERE id=:id",
                    {"dialogues": json.dumps(dialogues(records)), "id": saved_id},
                )
                return await client.post(
                    f"/api/training/saved-dialogues/{saved_id}/create-dataset", params={"dataset_name": name}
                )

            first = await create("existing", source)
            assert first.status_code == 200 and first.json()["success"] is True
            destination = processor.data_dir / "existing"
            before = {path.name: path.read_bytes() for path in destination.iterdir()}
            conflict = await create("existing", rows("replacement"))
            assert conflict.status_code == 409 and "已存在" in conflict.json()["detail"]
            assert not conflict.json().get("success")
            assert {path.name: path.read_bytes() for path in destination.iterdir()} == before
            repaired = await create("new-name", rows("replacement"))
            assert repaired.status_code == 200 and repaired.json()["success"] is True
            assert {path.name: path.read_bytes() for path in destination.iterdir()} == before
            assert {path.name for path in processor.data_dir.iterdir()} == {"existing", "new-name"}
    finally:
        database.close_connection()
