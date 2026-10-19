"""Checkpoint metadata owns loading; malformed configurations never reach weights."""

import json
from unittest.mock import Mock

import pytest
from transformers import PretrainedConfig

from training import trainer as module


@pytest.mark.parametrize("method", [None, "awq", "gptq", "bitsandbytes"])
def test_complete_checkpoint_configuration_is_given_to_native_loader(monkeypatch, tmp_path, method):
    quantization = {
        "awq": {"quant_method": "awq", "bits": 4, "group_size": 64, "version": "gemv", "zero_point": False, "backend": "autoawq"},
        "gptq": {"quant_method": "gptq", "bits": 4, "group_size": 64, "desc_act": True, "sym": False},
        "bitsandbytes": {"quant_method": "bitsandbytes", "load_in_4bit": True, "bnb_4bit_quant_type": "nf4"},
    }.get(method)
    data = {"model_type": "qwen3"}
    if quantization is not None:
        data["quantization_config"] = quantization
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    original = path.read_bytes()
    model = object()
    prepared = object()
    load = Mock(return_value=model)
    prepare = Mock(return_value=prepared)
    monkeypatch.setattr(module.AutoModelForCausalLM, "from_pretrained", load)
    monkeypatch.setattr(module, "prepare_model_for_kbit_training", prepare)
    trainer = module.LoRATrainer(module.LoRATrainingConfig(base_model_path=str(tmp_path)))
    assert trainer._load_model() is (prepared if method else model)
    load.assert_called_once()
    options = load.call_args.kwargs
    assert isinstance(options["config"], PretrainedConfig) and options["config"].model_type == "qwen3"
    assert getattr(options["config"], "quantization_config", None) == quantization
    assert "quantization_config" not in options and options["local_files_only"] is True
    assert prepare.call_count == (1 if method else 0) and path.read_bytes() == original


@pytest.mark.parametrize("fault,error", [("json", json.JSONDecodeError), ("missing", FileNotFoundError), ("model_type", TypeError), ("unknown", ValueError)])
def test_invalid_checkpoint_configuration_stops_before_weights(monkeypatch, tmp_path, fault, error):
    if fault != "missing":
        data = {} if fault == "model_type" else {"model_type": "unknown-synthetic-architecture"}
        (tmp_path / "config.json").write_text("{broken" if fault == "json" else json.dumps(data), encoding="utf-8")
    load = Mock()
    prepare = Mock()
    monkeypatch.setattr(module.AutoModelForCausalLM, "from_pretrained", load)
    monkeypatch.setattr(module, "prepare_model_for_kbit_training", prepare)
    trainer = module.LoRATrainer(module.LoRATrainingConfig(base_model_path=str(tmp_path)))
    with pytest.raises(error):
        trainer._load_model()
    load.assert_not_called()
    prepare.assert_not_called()
    assert trainer._model is None


def test_native_quantized_loader_error_is_not_retried_or_published(monkeypatch, tmp_path):
    data = {"model_type": "qwen3", "quantization_config": {"quant_method": "awq", "bits": 4, "version": "gemv"}}
    (tmp_path / "config.json").write_text(json.dumps(data), encoding="utf-8")
    failure = TypeError("synthetic-native-loader-failure")
    load = Mock(side_effect=failure)
    prepare = Mock()
    monkeypatch.setattr(module.AutoModelForCausalLM, "from_pretrained", load)
    monkeypatch.setattr(module, "prepare_model_for_kbit_training", prepare)
    trainer = module.LoRATrainer(module.LoRATrainingConfig(base_model_path=str(tmp_path)))
    with pytest.raises(TypeError) as caught:
        trainer._load_model()
    assert caught.value is failure and trainer._model is None
    load.assert_called_once()
    assert load.call_args.kwargs["config"].quantization_config == data["quantization_config"]
    prepare.assert_not_called()
