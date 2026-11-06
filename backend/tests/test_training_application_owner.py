"""Training API and workers retain their database and actual thread ownership."""

import asyncio
import builtins
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import psutil
import pytest
from fastapi import FastAPI
from test_owned_startup_resources import lifecycle as lifecycle
from test_training_runtime_store import runtime as runtime

from api import training
from app.config import create_access_token
from app.runtime import RuntimeContainer
from db.database import SQLiteDB
from training import task_manager
from training import trainer as trainer_module
from training.preprocessor import DatasetPreprocessor


@pytest.fixture
async def rig(tmp_path, monkeypatch):
    monkeypatch.setenv("LORA_PATH", str(tmp_path / "adapters"))
    processor = DatasetPreprocessor(base_dir=tmp_path)
    processor.prepare_training_data(
        [
            {
                "instruction": f"完整题设{index}：两张票，每张3元，一共多少元？",
                "input": "没有其他费用。",
                "output": "一共6元。",
            }
            for index in range(10)
        ],
        output_name="complete-input",
    )
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"qwen3"}')
    monkeypatch.setattr("training.preprocessor.get_dataset_preprocessor", lambda: processor)
    monkeypatch.setattr(
        task_manager,
        "ALL_GPU_CONFIGS",
        {"synthetic": task_manager.RTX4060Config(model_name_or_path=str(model), fp16=False, load_in_4bit=False)},
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, mem_get_info=lambda _: (16 * 1024**3, 24 * 1024**3))
        ),
    )
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=64 * 1024**3))
    forbidden = Mock(side_effect=AssertionError("training API used global database"))
    monkeypatch.setattr(
        training, "db", SimpleNamespace(get_all_training_tasks=forbidden, save_training_task=forbidden), raising=False
    )
    state = SimpleNamespace(stores=[], clients=[], started={}, forbidden=forbidden, managers=[])

    async def controlled(self, task_id, *args):
        state.managers.append(self)
        state.started[task_id] = self
        while not self._cancel_events[task_id].is_set():
            await asyncio.sleep(0)

    monkeypatch.setattr(task_manager.SimpleLoRATrainer, "_run_training", controlled)
    for index in range(2):
        store = SQLiteDB(tmp_path / f"owner-{index}.sqlite")
        store.add_user("synthetic-admin", "unused-signed-token-hash")
        store.execute_sql("UPDATE users SET role='admin' WHERE username='synthetic-admin'")
        store.save_training_task(
            "shared-history",
            {"lora_name": f"history-owner-{index}", "status": "completed", "progress": 100, "config": {}},
        )
        state.stores.append(store)
        app = FastAPI()
        app.state.runtime_container = RuntimeContainer(db=store, is_pg_mode=lambda: False, startup_env={})
        app.include_router(training.router)
        state.clients.append(
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
                base_url="https://isolated-training.test",
                headers={"Authorization": "Bearer " + create_access_token("synthetic-admin", 1, "admin")},
            )
        )

    async def start(index):
        return await state.clients[index].post(
            "/api/training/start",
            json={"lora_name": f"owned-{index}", "dataset_name": "complete-input", "model_type": "synthetic"},
        )

    state.start = start
    try:
        yield state
    finally:
        managers = set(state.managers + [getattr(store, "_lora_trainer", None) for store in state.stores])
        for manager in managers - {None}:
            await manager.shutdown(timeout=1)
        for client in state.clients:
            await client.aclose()
        for store in state.stores:
            store.close_connection()


async def test_factories_reuse_only_the_database_owned_manager(rig):
    first = task_manager.get_simple_lora_trainer(db=rig.stores[0])
    second = task_manager.get_simple_lora_trainer(db=rig.stores[1])
    assert first is not second and first.db is rig.stores[0] and second.db is rig.stores[1]
    assert task_manager.get_simple_lora_trainer(db=rig.stores[0]) is first
    assert task_manager.get_simple_lora_trainer(db=rig.stores[1]) is second
    assert (await first.get_task_status("shared-history"))["lora_name"] == "history-owner-0"
    assert (await second.get_task_status("shared-history"))["lora_name"] == "history-owner-1"


async def test_parallel_first_resolution_constructs_one_owner(rig, monkeypatch):
    original = task_manager.SimpleLoRATrainer

    def construct(*args, **kwargs):
        time.sleep(0.02)
        return original(*args, **kwargs)

    factory = Mock(side_effect=construct)
    monkeypatch.setattr(task_manager, "SimpleLoRATrainer", factory)
    managers = await asyncio.gather(
        *(asyncio.to_thread(task_manager.get_simple_lora_trainer, db=rig.stores[0]) for _ in range(8))
    )
    assert all(manager is managers[0] for manager in managers)
    assert factory.call_count == 1


async def test_http_start_list_status_cancel_share_the_application_owner(rig):
    task_ids = []
    for index, api in enumerate(rig.clients):
        history = await api.get("/api/training/tasks/shared-history")
        assert history.status_code == 200 and history.json()["task"]["lora_name"] == f"history-owner-{index}"
        created = await rig.start(index)
        assert created.status_code == 200
        task_ids.append(created.json()["task_id"])
    await asyncio.sleep(0)
    assert rig.started[task_ids[0]].db is rig.stores[0] and rig.started[task_ids[1]].db is rig.stores[1]
    assert (await rig.clients[1].get("/api/training/tasks/" + task_ids[0])).status_code == 404
    stopped = await rig.clients[0].post("/api/training/tasks/" + task_ids[0] + "/cancel")
    assert stopped.status_code == 200 and stopped.json()["success"] is True
    first = await rig.clients[0].get("/api/training/tasks")
    second = await rig.clients[1].get("/api/training/tasks")
    assert {task["task_id"] for task in first.json()["tasks"]} == {"shared-history", task_ids[0]}
    assert {task["task_id"] for task in second.json()["tasks"]} == {"shared-history", task_ids[1]}
    assert rig.stores[0].get_training_task(task_ids[0])["status"] == "cancelled"
    assert not rig.started[task_ids[1]]._cancel_events[task_ids[1]].is_set()
    rig.forbidden.assert_not_called()


async def test_store_failure_is_private_and_repaired_store_accepts_start(rig, monkeypatch, caplog):
    manager = task_manager.get_simple_lora_trainer(db=rig.stores[0])
    save = rig.stores[0].save_training_task
    monkeypatch.setattr(
        rig.stores[0], "save_training_task", Mock(side_effect=OSError("synthetic-private-admission-content"))
    )
    response = await rig.start(0)
    assert response.status_code == 500 and not manager._runner_tasks
    assert "synthetic-private" not in response.text and "synthetic-private" not in caplog.text
    monkeypatch.setattr(rig.stores[0], "save_training_task", save)
    response = await rig.start(0)
    assert response.status_code == 200 and rig.stores[0].get_training_task(response.json()["task_id"])


async def test_shutdown_releases_only_its_database_owner(rig):
    owners = [task_manager.get_simple_lora_trainer(db=store) for store in rig.stores]
    ids = [(await rig.start(index)).json()["task_id"] for index in range(2)]
    await asyncio.sleep(0)
    await task_manager.shutdown_simple_lora_trainer(rig.stores[0])
    assert rig.stores[0]._lora_trainer is None
    assert rig.stores[1]._lora_trainer is owners[1]
    assert rig.stores[0].get_training_task(ids[0])["status"] == "interrupted"
    assert not owners[1]._cancel_events[ids[1]].is_set()


async def test_application_lifespan_passes_the_actual_database_to_training_shutdown(lifecycle, monkeypatch):
    owners = []

    async def close(*args):
        owners.append(args)

    monkeypatch.setattr(task_manager, "shutdown_simple_lora_trainer", close)
    first, first_db = lifecycle.application(postgres=True, db_path=False)
    second, second_db = lifecycle.application(postgres=True, db_path=False)
    async with lifecycle.main.lifespan(first):
        pass
    async with lifecycle.main.lifespan(second):
        pass
    assert owners == [(first_db,), (second_db,)]


async def test_shutdown_timeout_keeps_the_actual_training_thread_and_owner(runtime, monkeypatch):
    _, database, dataset, config = runtime
    manager = task_manager.get_simple_lora_trainer(db=database)
    entered, release = threading.Event(), threading.Event()

    class ControlledTrainer:
        def __init__(self, configuration):
            self.configuration = configuration

        def train(self):
            self.progress_fn(60, 6, 10)
            entered.set()
            assert release.wait(timeout=5)
            return Path(self.configuration.output_dir)

    monkeypatch.setattr(trainer_module, "LoRATrainer", ControlledTrainer)
    task_id = await manager.start_training("held-thread", dataset, config)
    runner = manager._runner_tasks[task_id]
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        with pytest.raises(TimeoutError):
            await manager.shutdown(timeout=0.01)
        assert not runner.done() and manager._runner_tasks[task_id] is runner
        assert task_manager.get_simple_lora_trainer(db=database) is manager
        assert manager._cancel_events[task_id].is_set()
    finally:
        release.set()
        await asyncio.gather(runner, return_exceptions=True)
    assert database.get_training_task(task_id)["status"] == "interrupted"
    await task_manager.shutdown_simple_lora_trainer(database)
    assert database._lora_trainer is None


@pytest.mark.parametrize("fault", ["dependency", "probe"])
async def test_gpu_probe_fault_cannot_become_cpu_preflight(rig, monkeypatch, fault):
    monkeypatch.setattr(training, "db", rig.stores[0], raising=False)
    factory = Mock(wraps=task_manager.get_simple_lora_trainer)
    monkeypatch.setattr(task_manager, "get_simple_lora_trainer", factory)
    if fault == "dependency":
        original = builtins.__import__

        def missing(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("synthetic-required-torch-unavailable")
            return original(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", missing)
    else:
        monkeypatch.setitem(
            sys.modules,
            "torch",
            SimpleNamespace(
                cuda=SimpleNamespace(is_available=Mock(side_effect=RuntimeError("synthetic-gpu-probe-fault")))
            ),
        )
    response = await rig.start(0)
    assert response.status_code == 500
    factory.assert_not_called()
    assert not rig.started
