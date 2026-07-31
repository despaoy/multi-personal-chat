from __future__ import annotations

import asyncio
import io
import itertools
import threading
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.responses import FileResponse

from api import training as training_api
from training import preprocessor as preprocessor_module


def _prepare_export(monkeypatch, tmp_path: Path, *, slots: int = 1):
    data_root = tmp_path / "datasets"
    dataset_dir = data_root / "demo"
    (dataset_dir / "nested").mkdir(parents=True)
    (dataset_dir / "train.json").write_text('{"sample": 1}', encoding="utf-8")
    (dataset_dir / "nested" / "meta.txt").write_text("metadata", encoding="utf-8")

    monkeypatch.setattr(
        preprocessor_module,
        "get_dataset_preprocessor",
        lambda: SimpleNamespace(data_dir=data_root),
    )
    semaphore = asyncio.BoundedSemaphore(slots)
    monkeypatch.setattr(training_api, "_dataset_export_slots", semaphore)
    monkeypatch.setattr(
        training_api, "_dataset_export_slots_loop", asyncio.get_running_loop()
    )
    monkeypatch.setattr(training_api, "_DATASET_EXPORT_ACQUIRE_TIMEOUT", 0.02)

    export_root = tmp_path / "exports"
    export_root.mkdir()
    sequence = itertools.count()
    created_paths: list[Path] = []

    def create_temp_path() -> Path:
        path = export_root / f"archive-{next(sequence)}.zip"
        path.touch(exist_ok=False)
        created_paths.append(path)
        return path

    monkeypatch.setattr(training_api, "_create_export_temp_path", create_temp_path)
    return dataset_dir, semaphore, created_paths


async def _send_response(response: FileResponse, *, fail_on_body: bool = False):
    messages: list[dict] = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict):
        if fail_on_body and message["type"] == "http.response.body":
            raise RuntimeError("client disconnected")
        messages.append(message)

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/api/training/datasets/demo/export",
        "raw_path": b"/api/training/datasets/demo/export",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "extensions": {},
    }
    await response(scope, receive, send)
    return messages


@pytest.mark.asyncio
async def test_export_compresses_off_loop_and_streams_temp_file(monkeypatch, tmp_path):
    _, semaphore, created_paths = _prepare_export(monkeypatch, tmp_path)
    event_loop_thread = threading.get_ident()
    worker_threads: list[int] = []
    original_writer = training_api._write_dataset_archive

    def recording_writer(dataset_dir: Path, archive_path: Path):
        worker_threads.append(threading.get_ident())
        return original_writer(dataset_dir, archive_path)

    monkeypatch.setattr(training_api, "_write_dataset_archive", recording_writer)

    response = await training_api.export_dataset("demo", current_user={"username": "tester"})

    assert isinstance(response, FileResponse)
    assert worker_threads and worker_threads[0] != event_loop_thread
    archive_path = Path(response.path)
    assert archive_path.exists()
    assert response.headers["cache-control"] == "private, no-store"
    assert semaphore._value == 0

    messages = await _send_response(response)
    body = b"".join(
        message.get("body", b"")
        for message in messages
        if message["type"] == "http.response.body"
    )
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        assert sorted(archive.namelist()) == ["nested/meta.txt", "train.json"]

    assert not archive_path.exists()
    assert all(not path.exists() for path in created_paths)
    assert semaphore._value == 1


@pytest.mark.asyncio
async def test_export_admission_is_bounded_until_response_finishes(monkeypatch, tmp_path):
    _, semaphore, _ = _prepare_export(monkeypatch, tmp_path)

    first_response = await training_api.export_dataset(
        "demo", current_user={"username": "tester"}
    )
    assert semaphore._value == 0

    with pytest.raises(HTTPException) as exc_info:
        await training_api.export_dataset("demo", current_user={"username": "tester"})
    assert exc_info.value.status_code == 503
    assert exc_info.value.headers == {"Retry-After": "1"}

    await _send_response(first_response)
    assert semaphore._value == 1

    next_response = await training_api.export_dataset(
        "demo", current_user={"username": "tester"}
    )
    await _send_response(next_response)
    assert semaphore._value == 1


@pytest.mark.asyncio
async def test_export_cleans_temp_file_when_client_send_fails(monkeypatch, tmp_path):
    _, semaphore, _ = _prepare_export(monkeypatch, tmp_path)
    response = await training_api.export_dataset(
        "demo", current_user={"username": "tester"}
    )
    archive_path = Path(response.path)

    with pytest.raises(RuntimeError, match="client disconnected"):
        await _send_response(response, fail_on_body=True)

    assert not archive_path.exists()
    assert semaphore._value == 1


@pytest.mark.asyncio
async def test_export_cancellation_waits_for_writer_then_cleans(monkeypatch, tmp_path):
    _, semaphore, created_paths = _prepare_export(monkeypatch, tmp_path)
    writer_started = threading.Event()
    allow_writer_to_finish = threading.Event()
    original_writer = training_api._write_dataset_archive

    def delayed_writer(dataset_dir: Path, archive_path: Path):
        writer_started.set()
        assert allow_writer_to_finish.wait(timeout=2)
        return original_writer(dataset_dir, archive_path)

    monkeypatch.setattr(training_api, "_write_dataset_archive", delayed_writer)

    export_task = asyncio.create_task(
        training_api.export_dataset("demo", current_user={"username": "tester"})
    )
    assert await asyncio.to_thread(writer_started.wait, 1)
    export_task.cancel()
    allow_writer_to_finish.set()

    with pytest.raises(asyncio.CancelledError):
        await export_task

    assert created_paths
    assert all(not path.exists() for path in created_paths)
    assert semaphore._value == 1


@pytest.mark.asyncio
async def test_export_rejects_symlink_entries_without_leaking_partial_archive(
    monkeypatch, tmp_path
):
    dataset_dir, semaphore, created_paths = _prepare_export(monkeypatch, tmp_path)
    link = dataset_dir / "escape.txt"
    link.write_text("simulated symlink", encoding="utf-8")
    original_is_symlink = Path.is_symlink

    def fake_is_symlink(path: Path) -> bool:
        if path == link:
            return True
        return original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", fake_is_symlink)

    with pytest.raises(HTTPException) as exc_info:
        await training_api.export_dataset("demo", current_user={"username": "tester"})

    assert exc_info.value.status_code == 500
    assert created_paths
    assert all(not path.exists() for path in created_paths)
    assert semaphore._value == 1



@pytest.mark.asyncio
async def test_export_owns_temp_path_before_cancellation_can_run(
    monkeypatch,
    tmp_path,
):
    _, semaphore, created_paths = _prepare_export(monkeypatch, tmp_path)
    original_create = training_api._create_export_temp_path
    export_task_holder: dict[str, asyncio.Task] = {}

    def create_then_cancel() -> Path:
        path = original_create()
        asyncio.get_running_loop().call_soon(export_task_holder["task"].cancel)
        return path

    monkeypatch.setattr(training_api, "_create_export_temp_path", create_then_cancel)
    export_task = asyncio.create_task(
        training_api.export_dataset("demo", current_user={"username": "tester"})
    )
    export_task_holder["task"] = export_task

    with pytest.raises(asyncio.CancelledError):
        await export_task

    assert created_paths
    assert all(not path.exists() for path in created_paths)
    assert semaphore._value == 1


@pytest.mark.asyncio
async def test_export_double_cancel_keeps_slot_until_writer_finishes(
    monkeypatch,
    tmp_path,
):
    _, semaphore, created_paths = _prepare_export(monkeypatch, tmp_path)
    writer_started = threading.Event()
    allow_writer_to_finish = threading.Event()
    original_writer = training_api._write_dataset_archive

    def delayed_writer(dataset_dir: Path, archive_path: Path):
        writer_started.set()
        assert allow_writer_to_finish.wait(timeout=2)
        return original_writer(dataset_dir, archive_path)

    monkeypatch.setattr(training_api, "_write_dataset_archive", delayed_writer)
    export_task = asyncio.create_task(
        training_api.export_dataset("demo", current_user={"username": "tester"})
    )
    assert await asyncio.to_thread(writer_started.wait, 1)

    export_task.cancel()
    await asyncio.sleep(0)
    export_task.cancel()
    await asyncio.sleep(0)

    assert export_task.done() is False
    assert semaphore._value == 0
    assert any(path.exists() for path in created_paths)

    allow_writer_to_finish.set()
    with pytest.raises(asyncio.CancelledError):
        await export_task

    assert all(not path.exists() for path in created_paths)
    assert semaphore._value == 1


def test_export_semaphore_restarts_on_a_fresh_event_loop(monkeypatch):
    monkeypatch.setattr(training_api, "_dataset_export_slots", None)
    monkeypatch.setattr(training_api, "_dataset_export_slots_loop", None)

    async def get_slots():
        semaphore = training_api._get_dataset_export_slots()
        async with semaphore:
            pass
        return semaphore

    first = asyncio.run(get_slots())
    second = asyncio.run(get_slots())

    assert second is not first