from pathlib import Path

import pytest

from training.evaluator import build_training_evaluation, write_training_evaluation_report


def test_training_evaluation_summarizes_losses_and_provenance(tmp_path: Path):
    report = build_training_evaluation(
        eval_results={"eval_loss": 1.25, "eval_runtime": 2.0},
        log_history=[
            {"loss": 2.0},
            {"loss": 1.5, "eval_loss": 1.4},
            {"loss": 1.2, "eval_loss": 1.1},
        ],
        config={"lora_r": 32, "packing": True, "seed": 42, "ignored": "value"},
    )

    assert report["schema_version"] == 1
    assert report["metrics"]["final_train_loss"] == 1.2
    assert report["metrics"]["final_eval_loss"] == 1.25
    assert report["metrics"]["best_eval_loss"] == 1.1
    assert report["metrics"]["eval_perplexity"] is not None
    assert report["provenance"] == {"lora_r": 32, "packing": True, "seed": 42}

    report_path = write_training_evaluation_report(
        tmp_path,
        {"eval_loss": 1.25},
        [],
        {"lora_r": 32},
    )
    assert report_path == tmp_path / "training_evaluation.json"
    assert report_path.exists()


def test_training_evaluation_handles_invalid_or_extreme_losses():
    report = build_training_evaluation(
        eval_results={"eval_loss": 100.0},
        log_history=[{"loss": "not-a-number"}],
        config={},
    )

    assert report["metrics"]["final_train_loss"] is None
    assert report["metrics"]["eval_perplexity"] is None

def test_training_config_allows_dora_and_rslora_together():
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from training.trainer import LoRATrainingConfig

    errors = LoRATrainingConfig(
        use_dora=True,
        use_rslora=True,
    ).validate()

    assert "DoRA and RSLoRA cannot be enabled together" not in errors

def test_non_quantized_model_uses_transformers_4_compatible_dtype(monkeypatch, tmp_path: Path):
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    import torch
    import training.trainer as trainer_module

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type":"qwen2"}', encoding="utf-8")
    captured = {}
    sentinel = object()

    def fake_from_pretrained(path, **kwargs):
        captured["path"] = path
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        trainer_module.AutoModelForCausalLM,
        "from_pretrained",
        fake_from_pretrained,
    )
    config = trainer_module.LoRATrainingConfig(base_model_path=str(model_dir))
    loaded = trainer_module.LoRATrainer(config)._load_model()

    assert loaded is sentinel
    assert captured["torch_dtype"] is torch.float16
    assert "dtype" not in captured

def test_gradient_checkpointing_enables_input_grads_and_disables_cache(monkeypatch):
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    import training.trainer as trainer_module

    class DummyConfig:
        use_cache = True

    class DummyModel:
        config = DummyConfig()
        input_grads_enabled = False

        def enable_input_require_grads(self):
            self.input_grads_enabled = True

    model = DummyModel()
    monkeypatch.setattr(trainer_module, "get_peft_model", lambda value, config: value)
    trainer = trainer_module.LoRATrainer(
        trainer_module.LoRATrainingConfig(gradient_checkpointing=True)
    )

    configured = trainer._configure_lora_model(model)

    assert configured is model
    assert configured.input_grads_enabled is True
    assert configured.config.use_cache is False
