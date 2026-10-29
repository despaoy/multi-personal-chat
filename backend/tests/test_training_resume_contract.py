"""Explicit resume intent survives validation, CLI loading and experiment dispatch."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from training import trainer as module
from training.preprocessor import DatasetPreprocessor
from training.task_manager import SimpleLoRATrainer


@pytest.fixture
def training_runtime(monkeypatch, tmp_path):
    from test_training_chat_dataset import TinyChatTokenizer

    monkeypatch.setenv("LORA_PATH", str(tmp_path / "loras"))
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    directory, _ = DatasetPreprocessor(base_dir=tmp_path).prepare_training_data(
        [
            {
                "instruction": f"第{index}条完整预约：周四14点，尚未付款，下雨取消。",
                "input": "改期须书面确认。",
                "output": "周四14点，尚未付款，下雨取消；改期须书面确认。",
            }
            for index in range(10)
        ],
        output_name="complete-resume-data",
    )
    config = SimpleLoRATrainer(base_dir=tmp_path)._build_config(
        "synthetic-resume",
        directory,
        {
            "base_model_path": str(model_dir),
            "fp16": False,
            "packing": False,
            "max_seq_length": 512,
            "report_to": "none",
        },
    )
    calls = []

    def artifact(directory, name):
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / name).write_text("explicit synthetic artifact; no model weights")

    model = SimpleNamespace(
        parameters=lambda: [],
        print_trainable_parameters=lambda: None,
        save_pretrained=lambda path: artifact(path, "model.txt"),
    )
    tokenizer = TinyChatTokenizer()
    monkeypatch.setattr(tokenizer, "save_pretrained", lambda path: artifact(path, "tokenizer.txt"), raising=False)
    load_model = Mock(return_value=model)
    monkeypatch.setattr(module.LoRATrainer, "cleanup_gpu", lambda self: None)
    monkeypatch.setattr(module.LoRATrainer, "print_gpu_memory", lambda *args: None)
    monkeypatch.setattr(module.LoRATrainer, "_load_model", load_model)
    monkeypatch.setattr(module.LoRATrainer, "_load_tokenizer", Mock(return_value=tokenizer))
    monkeypatch.setattr(module.LoRATrainer, "_configure_lora_model", lambda self, value: value)

    class ControlledArguments:
        def __init__(self, max_length=None, **kwargs):
            self.max_length = max_length

    class ControlledExecution:
        def __init__(self, **kwargs):
            self.model = kwargs["model"]
            assert len(kwargs["train_dataset"]) == 9 and len(kwargs["eval_dataset"]) == 1
            self.state = SimpleNamespace(
                best_model_checkpoint=None, best_metric=0.5, log_history=[{"loss": 0.6}, {"eval_loss": 0.5}]
            )

        def train(self, resume_from_checkpoint=None):
            calls.append(resume_from_checkpoint)
            if resume_from_checkpoint is not None and not Path(resume_from_checkpoint).is_dir():
                raise FileNotFoundError(resume_from_checkpoint)

        def evaluate(self):
            return {"eval_loss": 0.5}

    monkeypatch.setattr(module, "SFTConfig", ControlledArguments)
    monkeypatch.setattr(module, "SFTTrainer", ControlledExecution)
    return config, calls, load_model


@pytest.mark.parametrize("requested", ["fresh", "existing", "missing", "empty"])
def test_actual_training_preserves_resume_intent_before_loading_resources(training_runtime, tmp_path, requested):
    config, calls, load_model = training_runtime
    checkpoint = tmp_path / "checkpoint-10"
    if requested == "existing":
        checkpoint.mkdir()
        (checkpoint / "trainer_state.json").write_text('{"global_step": 10}')
    config.resume_from_checkpoint = None if requested == "fresh" else "" if requested == "empty" else str(checkpoint)
    if requested in {"missing", "empty"}:
        with pytest.raises(ValueError, match="resume_from_checkpoint"):
            module.LoRATrainer(config).train()
        assert calls == []
        load_model.assert_not_called()
        assert not Path(config.output_dir).exists()
        return
    result = module.LoRATrainer(config).train()
    assert result == Path(config.output_dir) / "final"
    assert calls == [config.resume_from_checkpoint]
    assert (result / "model.txt").is_file() and (result / "tokenizer.txt").is_file()
    saved = json.loads((result.parent / "training_config.json").read_text())
    assert saved["resume_from_checkpoint"] == config.resume_from_checkpoint
    assert json.loads((result.parent / "training_evaluation.json").read_text())["metrics"]["final_eval_loss"] == 0.5
    if requested == "existing":
        assert (checkpoint / "trainer_state.json").read_text() == '{"global_step": 10}'


def test_json_and_cli_resume_paths_use_project_base_and_explicit_override(monkeypatch, tmp_path):
    for name in ["model", "checkpoint-5", "checkpoint-9", "unrelated-cwd"]:
        (tmp_path / name).mkdir()
    (tmp_path / "train.jsonl").write_text('{"instruction":"完整问题", "input":"尚未付款", "output":"尚未付款"}\n')
    configuration = {
        "base_model_path": "model",
        "train_data_path": "train.jsonl",
        "output_dir": "output",
        "resume_from_checkpoint": "checkpoint-5",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(configuration))
    monkeypatch.setattr(module, "_BACKEND_DIR", tmp_path)
    monkeypatch.chdir(tmp_path / "unrelated-cwd")
    loaded = module.LoRATrainingConfig.load(path)
    assert loaded.resume_from_checkpoint == str(tmp_path / "checkpoint-5")
    assert loaded.validate() == []
    observed = []

    class ExplicitCliExecution:
        def __init__(self, config):
            self.config = config

        def train(self):
            errors = self.config.validate()
            if errors:
                raise ValueError("; ".join(errors))
            observed.append(self.config)
            return tmp_path / "output"

    monkeypatch.setattr(module, "LoRATrainer", ExplicitCliExecution)
    monkeypatch.setattr(sys, "argv", ["trainer", "--config", str(path), "--resume", "checkpoint-9"])
    module.main()
    assert observed[0].resume_from_checkpoint == str(tmp_path / "checkpoint-9")
    assert observed[0].base_model_path == loaded.base_model_path
    monkeypatch.setattr(sys, "argv", ["trainer", "--config", str(path), "--resume", ""])
    with pytest.raises(ValueError, match="resume_from_checkpoint"):
        module.main()
    assert len(observed) == 1


def test_experiment_resume_requires_checkpoint_and_keeps_frozen_inputs(monkeypatch, tmp_path, capsys):
    project = Path(__file__).resolve().parents[2]
    specification = importlib.util.spec_from_file_location(
        "explicit_resume_experiment", project / "scripts/run_kisaki_experiment.py"
    )
    runner = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(runner)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    prompt = "完整虚构角色；保留否定、条件和时间。"
    (tmp_path / "prompt.txt").write_text(prompt)
    config = dict(
        _experiment_id="synthetic",
        _dataset_version="complete-fixture",
        base_model_path=str(model),
        system_prompt_policy="replace",
        system_prompt=prompt,
        _prompt_content_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        _prompt_policy_version="synthetic-v1",
    )
    dataset = dict(
        status="frozen",
        freeze_blockers=[],
        dataset_id="complete-fixture",
        prompt_policy=dict(required_training_policy="replace", path="prompt.txt", version="synthetic-v1"),
    )
    for split, field, key, count in [
        ("train", "train_data_path", "_train_data_sha256", 10),
        ("validation", "eval_data_path", "_validation_data_sha256", 2),
    ]:
        path = tmp_path / (split + ".jsonl")
        path.write_text(
            "".join(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": f"第{index}条：尚未付款，下雨取消。"},
                            {"role": "assistant", "content": "尚未付款，下雨取消。"},
                        ]
                    },
                    ensure_ascii=False,
                )
                + "\n"
                for index in range(count)
            )
        )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        config.update({field: str(path), key: digest})
        dataset[split] = dict(path=str(path), sha256=digest, count=count)
    config_path = tmp_path / "experiment.json"
    config_path.write_text(json.dumps(config))
    manifest = tmp_path / "dataset.json"
    manifest.write_text(json.dumps(dataset))
    original = {
        path: path.read_bytes()
        for path in [config_path, manifest, tmp_path / "train.jsonl", tmp_path / "validation.jsonl"]
    }
    for name, value in dict(
        PROJECT_ROOT=tmp_path,
        BACKEND=tmp_path,
        SERVER_ROOT=tmp_path / "lab",
        CONFIGS={"e1": config_path},
        DATASET_MANIFEST=manifest,
    ).items():
        monkeypatch.setattr(runner, name, value)
    monkeypatch.setattr(runner, "environment_snapshot", lambda path: {"explicit_test_environment": True})
    monkeypatch.setattr(runner, "package_versions", lambda: {})
    monkeypatch.setattr(runner, "gpu_snapshot", lambda: [])
    calls = []

    def controlled_process(command, **kwargs):
        if "training.trainer" in command:
            resolved = json.loads(Path(command[-1]).read_text())
            calls.append(resolved)
            output = Path(resolved["output_dir"]) / "final"
            output.mkdir(parents=True, exist_ok=True)
            (output / "explicit-synthetic-artifact.txt").write_text("no model execution")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", controlled_process)
    monkeypatch.setattr(sys, "argv", ["experiment", "--experiment", "e1", "--resume"])
    assert runner.main() == 2
    assert "resume_checkpoint_missing=" in capsys.readouterr().err and calls == []
    run = tmp_path / "lab/runtime/experiments/kisaki/r1v4/e1/seed42"
    assert not run.exists()
    output = tmp_path / "lab/runtime/loras/kisaki/r1v4/e1/seed42"
    for step in [2, 10]:
        checkpoint = output / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "trainer_state.json").write_text(json.dumps({"best_model_checkpoint": str(checkpoint)}))
    before = {path: path.read_bytes() for path in output.glob("checkpoint-*/trainer_state.json")}
    assert runner.main() == 0
    assert calls[0]["resume_from_checkpoint"] == str(output / "checkpoint-10")
    assert json.loads((run / "run_manifest.json").read_text())["status"] == "training_complete"
    assert all(path.read_bytes() == value for path, value in {**original, **before}.items())
    monkeypatch.setattr(sys, "argv", ["experiment", "--experiment", "e1"])
    assert runner.main() == 2 and len(calls) == 1
    assert "refusing_to_overwrite_nonempty_output=" in capsys.readouterr().err
