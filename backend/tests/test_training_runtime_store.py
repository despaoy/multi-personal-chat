"""Live training state and durable history must report real storage outcomes."""

import asyncio
import json
import threading
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from db.database import SQLiteDB
from training import task_manager, trainer as trainer_module
from training.preprocessor import DatasetPreprocessor


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    monkeypatch.setenv("LORA_PATH", str(tmp_path / "loras"))
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    dataset, _ = DatasetPreprocessor(base_dir=tmp_path).prepare_training_data(
        [
            {"instruction": f"完整问题{index}？", "input": "尚未付款，下雨取消。", "output": "尚未付款，下雨取消。"}
            for index in range(10)
        ],
        output_name="training-input",
    )
    database = SQLiteDB(tmp_path / "tasks.sqlite")
    database.save_training_task(
        "history",
        {
            "lora_name": "historical",
            "status": "completed",
            "progress": 100,
            "config": {},
            "created_at": "2026-01-01",
            "updated_at": "2026-01-01",
        },
    )
    manager = task_manager.SimpleLoRATrainer(base_dir=tmp_path, db=database)
    yield manager, database, dataset, {"base_model_path": str(model), "fp16": False, "load_in_4bit": False}
    database.close_connection()


async def test_failed_admission_owns_no_runtime_job_and_repaired_store_accepts_request(runtime, monkeypatch):
    manager, database, dataset, config = runtime
    failure = OSError("synthetic-admission-store-failure")
    worker = AsyncMock()
    existing = set(manager.tasks)
    monkeypatch.setattr(manager, "_run_training", worker)

    def fail(*args):
        raise failure

    with monkeypatch.context() as patch:
        patch.setattr(database, "save_training_task", fail)
        with pytest.raises(OSError) as caught:
            await manager.start_training("admitted", dataset, config)
    assert caught.value is failure
    assert set(manager.tasks) == existing and manager._cancel_events == {} and manager._runner_tasks == {}
    worker.assert_not_called()
    assert [row["task_id"] for row in database.get_all_training_tasks()] == ["history"]
    task_id = await manager.start_training("admitted", dataset, config)
    await asyncio.gather(*list(manager._runner_tasks.values()))
    assert database.get_training_task(task_id)["status"] == "pending"
    assert (await manager.get_task_status(task_id))["task_id"] == task_id
    worker.assert_awaited_once()


async def test_live_progress_and_failed_save_are_visible_without_losing_artifacts(runtime, monkeypatch):
    manager, database, dataset, config = runtime
    database.update_training_task("history", {"error_message": "explicit-history-correction"})
    assert (await manager.get_task_status("history"))["error_message"] == "explicit-history-correction"
    started, release = threading.Event(), threading.Event()
    produced = []

    class ControlledTrainer:
        def __init__(self, configuration):
            self.configuration = configuration

        def train(self):
            self.progress_fn(60, 6, 10)
            started.set()
            assert release.wait(timeout=5)
            output = Path(self.configuration.output_dir)
            output.mkdir(exist_ok=True)
            (output / "synthetic-artifact.txt").write_text("explicit test artifact; no real model training")
            produced.append(output)
            return output

    monkeypatch.setattr(trainer_module, "LoRATrainer", ControlledTrainer)
    save = database.save_training_task
    failure = OSError("synthetic-completion-store-failure")

    def fail_terminal(task_id, value):
        if value["status"] in {"completed", "failed"}:
            raise failure
        return save(task_id, value)

    monkeypatch.setattr(database, "save_training_task", fail_terminal)
    task_id = await manager.start_training("produced", dataset, config)
    runner = manager._runner_tasks[task_id]
    try:
        assert await asyncio.to_thread(started.wait, 2)
        listed = {row["task_id"]: row for row in await manager.get_all_tasks()}
        assert listed[task_id]["progress"] == 60 and listed[task_id]["current_step"] == 6
        assert database.get_training_task(task_id)["progress"] == 10
    finally:
        release.set()
        outcomes = await asyncio.gather(runner, return_exceptions=True)
    assert outcomes == [failure]
    task = await manager.get_task_status(task_id)
    listed = {row["task_id"]: row for row in await manager.get_all_tasks()}
    assert task["status"] == listed[task_id]["status"] == "failed"
    assert task["error_message"] == str(failure) and listed[task_id]["error_message"] == str(failure)
    assert Path(task["output_dir"]) == produced[0] and (produced[0] / "synthetic-artifact.txt").is_file()
    assert database.get_training_task(task_id)["status"] == "training"
    monkeypatch.setattr(database, "save_training_task", save)
    manager._persist_task(task_id)
    assert database.get_training_task(task_id)["status"] == "failed"
    recovered = await manager.start_training("recovered", dataset, config)
    await asyncio.gather(*list(manager._runner_tasks.values()))
    assert database.get_training_task(recovered)["status"] == "completed"
    assert (await manager.get_task_status(recovered))["progress"] == 100
    assert len(produced) == 2


async def test_shutdown_joins_owned_worker_and_releases_singleton_when_save_fails(runtime, monkeypatch):
    manager, database, dataset, config = runtime
    started, ended = asyncio.Event(), asyncio.Event()

    async def controlled_worker(task_id, *args):
        started.set()
        while not manager._cancel_events[task_id].is_set():
            await asyncio.sleep(0)
        ended.set()

    monkeypatch.setattr(manager, "_run_training", controlled_worker)
    monkeypatch.setattr(task_manager, "_simple_lora_trainer", manager)
    task_id = await manager.start_training("stopped", dataset, config)
    await started.wait()
    save = database.save_training_task
    failure = OSError("synthetic-shutdown-store-failure")

    def fail_interrupt(task_id, value):
        if value["status"] == "interrupted":
            raise failure
        return save(task_id, value)

    monkeypatch.setattr(database, "save_training_task", fail_interrupt)
    with pytest.raises(OSError) as caught:
        await asyncio.wait_for(task_manager.shutdown_simple_lora_trainer(), timeout=1)
    assert caught.value is failure and ended.is_set()
    assert manager._runner_tasks == {} and task_manager._simple_lora_trainer is None
    assert (await manager.get_task_status(task_id))["status"] == "interrupted"
    monkeypatch.setattr(database, "save_training_task", save)
    restored = task_manager.get_simple_lora_trainer(db=database)
    assert restored is not manager and database.get_training_task(task_id)["status"] == "interrupted"
    assert (await restored.get_task_status("history"))["status"] == "completed"


async def test_cancelled_shutdown_keeps_live_owner_until_the_same_worker_finishes(runtime, monkeypatch):
    manager, database, dataset, config = runtime
    started, release = asyncio.Event(), asyncio.Event()

    async def controlled_worker(*args):
        started.set()
        await release.wait()

    monkeypatch.setattr(manager, "_run_training", controlled_worker)
    monkeypatch.setattr(task_manager, "_simple_lora_trainer", manager)
    task_id = await manager.start_training("held", dataset, config)
    runner = manager._runner_tasks[task_id]
    await started.wait()
    stopping = asyncio.create_task(task_manager.shutdown_simple_lora_trainer())
    try:
        async with asyncio.timeout(1):
            while not manager._cancel_events[task_id].is_set():
                await asyncio.sleep(0)
        stopping.cancel()
        with pytest.raises(asyncio.CancelledError):
            await stopping
        assert not runner.done()
        selected = task_manager.get_simple_lora_trainer(db=database)
        assert selected is manager
        with pytest.raises(task_manager.HTTPException) as caught:
            await selected.start_training("held", dataset, config)
        assert caught.value.status_code == 409
    finally:
        release.set()
        await asyncio.gather(runner, return_exceptions=True)
        await manager.shutdown(timeout=1)
    await task_manager.shutdown_simple_lora_trainer()
    assert task_manager._simple_lora_trainer is None
