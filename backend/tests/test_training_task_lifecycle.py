"""Lifecycle tests for in-process LoRA training jobs."""

from __future__ import annotations

import asyncio

from training import task_manager


def test_training_shutdown_tracks_tasks_and_preserves_interrupted_state(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("LORA_PATH", str(tmp_path / "loras"))
    trainer = task_manager.SimpleLoRATrainer(base_dir=tmp_path)

    async def scenario():
        started = asyncio.Event()

        async def wait_forever(task_id, lora_name, dataset_path, config):
            started.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(trainer, "_run_training", wait_forever)
        task_id = await trainer.start_training(
            "test-adapter",
            tmp_path / "dataset",
            {},
        )
        await asyncio.wait_for(started.wait(), timeout=1)

        assert task_id in trainer._runner_tasks
        await trainer.shutdown(timeout=0)

        status = await trainer.get_task_status(task_id)
        assert status is not None
        assert status["status"] == "interrupted"
        assert task_id not in trainer._runner_tasks

    asyncio.run(scenario())


def test_global_training_shutdown_releases_singleton(monkeypatch, tmp_path):
    monkeypatch.setenv("LORA_PATH", str(tmp_path / "loras"))
    trainer = task_manager.SimpleLoRATrainer(base_dir=tmp_path)
    monkeypatch.setattr(task_manager, "_simple_lora_trainer", trainer)

    asyncio.run(task_manager.shutdown_simple_lora_trainer())

    assert task_manager._simple_lora_trainer is None