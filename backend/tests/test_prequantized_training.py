import json
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("prequantized_config,loaded_flag", [(True, False), (False, True)])
def test_sft_prepares_prequantized_checkpoint_without_requantizing(
    monkeypatch, tmp_path, prequantized_config, loaded_flag
):
    pytest.importorskip("peft")
    import training.trainer as module

    config = (
        {"quantization_config": {"quant_method": "bitsandbytes", "load_in_4bit": True}} if prequantized_config else {}
    )
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    model = SimpleNamespace(is_loaded_in_4bit=loaded_flag, is_loaded_in_8bit=False)
    prepared = object()
    calls = []

    def load(path, **kwargs):
        assert "quantization_config" not in kwargs
        return model

    def prepare(value):
        assert value is model
        calls.append(value)
        return prepared

    monkeypatch.setattr(module.AutoModelForCausalLM, "from_pretrained", load)
    monkeypatch.setattr(module, "prepare_model_for_kbit_training", prepare)
    trainer = module.LoRATrainer(module.LoRATrainingConfig(base_model_path=str(tmp_path), load_in_4bit=False))
    assert trainer._load_model() is prepared
    assert len(calls) == 1


def test_preference_prepares_auto_quantized_weights_and_attaches_adapter(monkeypatch, tmp_path):
    import sys

    pytest.importorskip("peft")
    pytest.importorskip("training.trainer")  # Import callbacks before stubbing model libraries.
    from training.preference_trainer import PreferenceTrainer, PreferenceTrainingConfig

    calls = []
    model = SimpleNamespace(is_loaded_in_4bit=True, is_loaded_in_8bit=False)
    tokenizer = SimpleNamespace(pad_token="pad", eos_token="eos")

    def prepare(value):
        assert value is model
        calls.append("prepare")
        return value

    def attach(value, config):
        assert calls == ["prepare"]
        calls.append("adapter")
        return value

    def stop_before_training(rows):
        raise RuntimeError("mock stop before training")

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **kw: model),
            BitsAndBytesConfig=lambda **kw: kw,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(
            PeftModel=object,
            LoraConfig=lambda **kw: kw,
            get_peft_model=attach,
            prepare_model_for_kbit_training=prepare,
            TaskType=SimpleNamespace(CAUSAL_LM="causal"),
        ),
    )
    monkeypatch.setitem(
        sys.modules, "datasets", SimpleNamespace(Dataset=SimpleNamespace(from_list=stop_before_training))
    )
    monkeypatch.setattr(
        "training.preference_compat.resolve_preference_backend", lambda method: (object, lambda max_length: None)
    )
    result = PreferenceTrainer(
        PreferenceTrainingConfig(
            base_model_path=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            load_in_4bit=False,
        )
    ).train([{"prompt": "q", "chosen": "a", "rejected": "b"}])
    assert result.error == "mock stop before training"
    assert calls == ["prepare", "adapter"]
    assert not (tmp_path / "out").exists()
