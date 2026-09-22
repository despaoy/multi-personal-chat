"""Lifecycle tests for in-process LoRA training jobs."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import pytest

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


def test_dataset_directory_prefers_training_jsonl_over_metadata(monkeypatch, tmp_path):
    monkeypatch.setenv("LORA_PATH", str(tmp_path / "loras"))
    trainer = task_manager.SimpleLoRATrainer(base_dir=tmp_path)
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    data = tmp_path / "train.jsonl"
    data.write_text('{}\n', encoding="utf-8")
    assert trainer._find_dataset_file(tmp_path) == data


def test_dataset_directory_rejects_ambiguous_files(monkeypatch, tmp_path):
    monkeypatch.setenv("LORA_PATH", str(tmp_path / "loras"))
    trainer = task_manager.SimpleLoRATrainer(base_dir=tmp_path)
    for name in ("metadata.json", "dialogues.jsonl"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="明确指定"):
        trainer._find_dataset_file(tmp_path)


def test_training_config_preserves_canonical_names_and_fractional_epochs(monkeypatch, tmp_path):
    monkeypatch.setenv("LORA_PATH", str(tmp_path / "loras"))
    monkeypatch.setitem(sys.modules, "training.trainer", SimpleNamespace(LoRATrainingConfig=SimpleNamespace))
    trainer = task_manager.SimpleLoRATrainer(base_dir=tmp_path)
    data = tmp_path / "train.jsonl"
    data.write_text('{}\n', encoding="utf-8")
    config = trainer._build_config("adapter", data, {
        "base_model_path": "chosen-model",
        "lora_r": 64,
        "num_train_epochs": 0.5,
        "gradient_checkpointing": False,
        "lora_target_modules": "q_proj,v_proj",
    })
    assert config.base_model_path == "chosen-model"
    assert config.lora_r == 64
    assert config.num_train_epochs == 0.5
    assert config.gradient_checkpointing is False
    assert config.target_modules == ["q_proj", "v_proj"]
    assert not hasattr(config, "lora_target_modules")


def test_frontend_training_aliases_take_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("LORA_PATH", str(tmp_path / "loras"))
    monkeypatch.setitem(sys.modules, "training.trainer", SimpleNamespace(LoRATrainingConfig=SimpleNamespace))
    trainer = task_manager.SimpleLoRATrainer(base_dir=tmp_path)
    data = tmp_path / "train.jsonl"
    data.write_text('{}\n', encoding="utf-8")
    config = trainer._build_config("adapter", data, {
        "lora_rank": 32, "lora_r": 64,
        "target_modules": " all-linear ", "lora_target_modules": "q_proj",
    })
    assert config.lora_r == 32
    assert config.target_modules is None


def test_cleanup_keeps_newest_completed_tasks(monkeypatch, tmp_path):
    monkeypatch.setenv("LORA_PATH", str(tmp_path / "loras"))
    trainer = task_manager.SimpleLoRATrainer(base_dir=tmp_path)
    trainer.tasks = {
        "old": {"status": "completed", "created_at": "2026-01-01"},
        "new": {"status": "completed", "created_at": "2026-02-01"},
    }
    trainer._cleanup_old_tasks(max_completed=1)
    assert set(trainer.tasks) == {"new"}


def test_cancelled_worker_blocks_same_adapter_until_exit(monkeypatch, tmp_path):
    monkeypatch.setenv("LORA_PATH", str(tmp_path / "loras"))
    trainer = task_manager.SimpleLoRATrainer(base_dir=tmp_path)

    async def scenario():
        started = asyncio.Event()
        finish = asyncio.Event()

        async def worker(*args):
            started.set()
            await finish.wait()

        monkeypatch.setattr(trainer, "_run_training", worker)
        task_id = await trainer.start_training("adapter", tmp_path, {})
        await started.wait()
        assert await trainer.cancel_task(task_id)
        with pytest.raises(task_manager.HTTPException) as error:
            await trainer.start_training("adapter", tmp_path, {})
        assert error.value.status_code == 409
        finish.set()
        await trainer.shutdown(timeout=1)

    asyncio.run(scenario())


def test_training_callback_observes_cancellation():
    import threading
    from training.trainer import LoRATrainer, ProgressCallback

    event = threading.Event()
    event.set()
    control = SimpleNamespace(should_training_stop=False, should_save=True, should_evaluate=True)
    callback = ProgressCallback(cancel_event=event)
    callback.on_step_end(None, SimpleNamespace(), control)
    assert control.should_training_stop
    assert not control.should_save
    assert not control.should_evaluate
    trainer = LoRATrainer()
    trainer.cancel_event = event
    with pytest.raises(RuntimeError, match="cancelled"):
        trainer._check_cancelled()
