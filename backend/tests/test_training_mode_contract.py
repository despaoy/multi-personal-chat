"""Requested training modes must match the worker that actually runs."""

import asyncio
import json
from unittest.mock import Mock

import pytest

from training import trainer as module
from training.task_manager import SimpleLoRATrainer


@pytest.mark.parametrize("mode", ["default", False, True, None, "false"])
def test_training_mode_is_validated_before_worker_creation(monkeypatch, tmp_path, mode):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "qwen3"}), encoding="utf-8")
    data = tmp_path / "train.jsonl"
    data.write_text(json.dumps({"messages": [
        {"role": "user", "content": "What is 2 plus 2?"},
        {"role": "assistant", "content": "4."},
    ]}) + "\n", encoding="utf-8")
    monkeypatch.setenv("LORA_PATH", str(tmp_path / "loras"))
    output = tmp_path / "loras" / "synthetic-adapter"
    worker = Mock()
    worker.train.return_value = output
    factory = Mock(return_value=worker)
    monkeypatch.setattr(module, "LoRATrainer", factory)
    manager = SimpleLoRATrainer(base_dir=tmp_path)
    config = {"base_model_path": str(model), "fp16": False}
    if mode != "default":
        config["use_deepspeed"] = mode

    async def run():
        task_id = await manager.start_training("synthetic-adapter", data, config)
        await asyncio.gather(*list(manager._runner_tasks.values()))
        return await manager.get_task_status(task_id)

    status = asyncio.run(run())
    if mode == "default" or mode is False:
        assert status["status"] == "completed" and status["progress"] == 100
        factory.assert_called_once()
        worker.train.assert_called_once_with()
        saved = module.LoRATrainingConfig.load(output / "training_config_auto.json")
        assert saved.use_deepspeed is False and saved.validate() == []
    else:
        assert status["status"] == "failed" and "DeepSpeed" in status["error_message"]
        factory.assert_not_called()
        worker.train.assert_not_called()
        assert not (output / "training_config_auto.json").exists()
