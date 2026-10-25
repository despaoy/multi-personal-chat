"""Exercise the real local provider lifecycle without weights or network.

External tokenizer/model/PEFT constructors are controlled fixtures. Generation
reports the adapter actually attached to the synthetic base, so changing only
provider metadata cannot make an adapter-to-base transition pass. NF4 is the
existing first loading strategy; no new precision setting is invented here.

"""

import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from inference import model_manager

REPLIES = {
    None: "Synthetic base model received the complete request.",
    "adapter-a": "Synthetic adapter A received the complete request.",
    "adapter-b": "Synthetic adapter B received the complete request.",
}
TOKENS = {None: 10, "adapter-a": 20, "adapter-b": 30}


def arrange_transformer_lifecycle(tmp_path, monkeypatch):
    """Use real local paths and the actual provider, replacing external ML only."""
    base_path = tmp_path / "base-model"
    base_path.mkdir()
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "generation_config.json"):
        (base_path / name).write_text("{}", encoding="utf-8")
    (base_path / "model.safetensors").write_bytes(b"synthetic-base-weights-never-read")
    adapters = {}
    for label in ("adapter-a", "adapter-b"):
        path = tmp_path / label
        path.mkdir()
        (path / "adapter_config.json").write_text("{}", encoding="utf-8")
        (path / "adapter_model.safetensors").write_bytes(b"synthetic-adapter-weights-never-read")
        adapters[label] = path

    rig = SimpleNamespace(
        base_path=base_path,
        adapters=adapters,
        faults={},
        tokenizer_calls=[],
        base_calls=[],
        adapter_calls=[],
        evaluations=[],
        generations=[],
        generation_options=[],
        template_options=[],
        bases=[],
        cuda_available=False,
        cache_clear_calls=0,
    )

    def step(stage):
        failures = rig.faults.get(stage, [])
        if failures:
            raise failures.pop(0)

    class Tensor:
        shape = (1, 2)

        def to(self, device):
            return self

    class Tokenizer:
        eos_token_id = 0

        def apply_chat_template(self, messages, **kwargs):
            rig.template_options.append(dict(kwargs))
            assert messages[-1]["role"] == "user" and messages[-1]["content"] == "Complete isolated request."
            return SimpleNamespace(input_ids=Tensor())

        def decode(self, tokens, **kwargs):
            label = next(label for label, token in TOKENS.items() if tokens == [token])
            return REPLIES[label]

    class BaseModel:
        device = "cpu"

        def __init__(self):
            self.adapter = None

        def eval(self):
            rig.evaluations.append(("base", self))
            step("base_eval")
            return self

        def generate(self, input_ids, **kwargs):
            rig.generations.append(self.adapter)
            rig.generation_options.append(dict(kwargs))
            return [[1, 2, TOKENS[self.adapter]]]

    class AdapterModel:
        def __init__(self, base):
            self.base_model = SimpleNamespace(model=base)
            self.device = base.device

        def eval(self):
            rig.evaluations.append((self.base_model.model.adapter, self))
            step("adapter_eval")
            return self

        def generate(self, *args, **kwargs):
            return self.base_model.model.generate(*args, **kwargs)

    def tokenizer_loader(path, **kwargs):
        rig.tokenizer_calls.append((path, kwargs))
        step("tokenizer")
        return Tokenizer()

    def base_loader(path, **kwargs):
        rig.base_calls.append((path, kwargs))
        step("base_load")
        model = BaseModel()
        rig.bases.append(model)
        return model

    def adapter_loader(base, path, **kwargs):
        path = Path(path)
        if not path.is_dir():
            raise FileNotFoundError(path)
        rig.adapter_calls.append((base, path, kwargs))
        # PEFT may mutate the supplied base before a later loading failure.
        base.adapter = path.name
        step("adapter_load")
        return AdapterModel(base)

    def clear_cuda_cache():
        rig.cache_clear_calls += 1
        step("cache_clear")

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        float16="synthetic-float16",
        long="synthetic-long",
        no_grad=nullcontext,
        tensor=lambda value, **kwargs: Tensor(),
        cuda=SimpleNamespace(
            memory_allocated=lambda: 0,
            empty_cache=clear_cuda_cache,
            is_available=lambda: rig.cuda_available,
        ),
    ))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_loader),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=base_loader),
        BitsAndBytesConfig=lambda **kwargs: SimpleNamespace(**kwargs),
    ))
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(
        PeftModel=SimpleNamespace(from_pretrained=adapter_loader),
    ))
    monkeypatch.setenv("BASE_MODEL_PATH", str(base_path))
    monkeypatch.setattr(model_manager, "_get_db_config", lambda: {"temperature": 0.2, "maxTokens": 32})
    rig.provider = model_manager.TransformersPeftProvider()
    return rig


def generate(rig):
    return rig.provider.generate("Complete isolated request.")[0]


@pytest.mark.parametrize("stage", ["tokenizer", "base_eval", "adapter_load", "adapter_eval"])
def test_failed_local_initialization_never_publishes_a_partial_model(tmp_path, monkeypatch, stage):
    rig = arrange_transformer_lifecycle(tmp_path, monkeypatch)
    expected = None
    if stage.startswith("adapter"):
        expected = "adapter-a"
        rig.provider.set_lora_adapter(str(rig.adapters[expected]))
    failure = RuntimeError("isolated initialization failure")
    rig.faults[stage] = [failure]

    with pytest.raises(RuntimeError) as caught:
        generate(rig)
    assert caught.value is failure and not rig.generations
    assert rig.provider.get_status()["loaded"] is False
    assert rig.provider._model is None

    assert generate(rig) == REPLIES[expected]
    assert rig.generations == [expected] and rig.provider.get_status()["loaded"] is True


def test_failed_nf4_load_preserves_error_and_never_tries_another_precision(tmp_path, monkeypatch):
    rig = arrange_transformer_lifecycle(tmp_path, monkeypatch)
    failure = RuntimeError("isolated NF4 load failure")
    rig.faults["base_load"] = [failure]
    with pytest.raises(RuntimeError) as caught:
        generate(rig)
    assert caught.value is failure
    assert len(rig.base_calls) == 1 and rig.generations == []
    assert rig.provider._model is None and not rig.provider.get_status()["loaded"]
    options = rig.base_calls[0][1]
    assert options["quantization_config"].load_in_4bit is True
    assert options["quantization_config"].bnb_4bit_quant_type == "nf4"

    assert generate(rig) == REPLIES[None]
    assert len(rig.base_calls) == 2 and rig.generations == [None]
    assert all(call[1]["quantization_config"].load_in_4bit is True for call in rig.base_calls)


def test_adapter_to_base_and_next_adapter_change_actual_generation(tmp_path, monkeypatch):
    rig = arrange_transformer_lifecycle(tmp_path, monkeypatch)
    for label in ("adapter-a", None, "adapter-b", None):
        path = str(rig.adapters[label]) if label else None
        rig.provider.set_lora_adapter(path)
        assert generate(rig) == REPLIES[label]
        assert rig.provider.get_status()["loraAdapter"] == path
    assert rig.generations == ["adapter-a", None, "adapter-b", None]


def test_selecting_same_adapter_reuses_loaded_model_and_actual_output(tmp_path, monkeypatch):
    rig = arrange_transformer_lifecycle(tmp_path, monkeypatch)
    path = str(rig.adapters["adapter-a"])
    rig.provider.set_lora_adapter(path)
    assert generate(rig) == REPLIES["adapter-a"]
    model, tokenizer = rig.provider._model, rig.provider._tokenizer

    rig.provider.set_lora_adapter(path)
    assert generate(rig) == REPLIES["adapter-a"]
    assert rig.provider._model is model and rig.provider._tokenizer is tokenizer
    assert len(rig.tokenizer_calls) == len(rig.base_calls) == len(rig.adapter_calls) == 1
    assert rig.generations == ["adapter-a", "adapter-a"]
    assert rig.provider.get_status()["loaded"] is True
    assert rig.provider.get_status()["loraAdapter"] == path


def test_close_clears_state_and_preserves_cache_release_error_then_can_reload(tmp_path, monkeypatch):
    rig = arrange_transformer_lifecycle(tmp_path, monkeypatch)
    path = str(rig.adapters["adapter-a"])
    rig.provider.set_lora_adapter(path)
    assert generate(rig) == REPLIES["adapter-a"]
    rig.cuda_available = True
    failure = RuntimeError("isolated CUDA cache release failure")
    rig.faults["cache_clear"] = [failure]

    with pytest.raises(RuntimeError) as caught:
        rig.provider.close()
    assert caught.value is failure and rig.cache_clear_calls == 1
    assert rig.provider._model is None and rig.provider._tokenizer is None
    assert rig.provider.get_status()["loaded"] is False

    rig.provider.set_lora_adapter(path)
    assert generate(rig) == REPLIES["adapter-a"]
    assert len(rig.tokenizer_calls) == len(rig.base_calls) == len(rig.adapter_calls) == 2
    assert rig.generations == ["adapter-a", "adapter-a"]
    assert rig.provider.get_status()["loaded"] is True


@pytest.mark.parametrize("loaded", [False, True])
def test_missing_requested_adapter_is_an_error_without_clearing_previous_selection(tmp_path, monkeypatch, loaded):
    rig = arrange_transformer_lifecycle(tmp_path, monkeypatch)
    previous = str(rig.adapters["adapter-a"])
    rig.provider.set_lora_adapter(previous)
    if loaded:
        assert generate(rig) == REPLIES["adapter-a"]
    before = dict(rig.provider.get_status())
    model = rig.provider._model
    with pytest.raises(FileNotFoundError):
        rig.provider.set_lora_adapter(str(tmp_path / "missing-requested-adapter"))
    assert rig.provider.get_status() == before and rig.provider._model is model
    assert generate(rig) == REPLIES["adapter-a"]


def test_adapter_disappearing_after_selection_cannot_turn_into_base_success(tmp_path, monkeypatch):
    rig = arrange_transformer_lifecycle(tmp_path, monkeypatch)
    path = rig.adapters["adapter-a"]
    rig.provider.set_lora_adapter(str(path))
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        (path / name).unlink()
    path.rmdir()
    with pytest.raises(FileNotFoundError):
        generate(rig)
    assert rig.generations == [] and not rig.provider.get_status()["loaded"]


@pytest.mark.parametrize("stage", ["adapter_load", "adapter_eval"])
def test_failed_adapter_switch_does_not_publish_requested_adapter_as_loaded(tmp_path, monkeypatch, stage):
    rig = arrange_transformer_lifecycle(tmp_path, monkeypatch)
    old_path, new_path = (str(rig.adapters[label]) for label in ("adapter-a", "adapter-b"))
    rig.provider.set_lora_adapter(old_path)
    assert generate(rig) == REPLIES["adapter-a"]
    failure = RuntimeError("isolated adapter switch failure")
    rig.faults[stage] = [failure]
    with pytest.raises(RuntimeError) as caught:
        rig.provider.set_lora_adapter(new_path)
        generate(rig)
    assert caught.value is failure and rig.generations == ["adapter-a"]
    assert rig.provider.get_status()["loaded"] is False
    assert rig.provider._model is None and rig.provider._tokenizer is None
    rig.provider.set_lora_adapter(new_path)
    assert generate(rig) == REPLIES["adapter-b"]
    assert rig.provider.get_status()["loraAdapter"] == new_path



@pytest.mark.asyncio
async def test_unsupported_frequency_penalty_fails_before_local_model_loading(tmp_path, monkeypatch):
    rig = arrange_transformer_lifecycle(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="frequency_penalty"):
        await rig.provider.async_complete([{"role": "user", "content": "Complete isolated request."}],
            temperature=.17, max_tokens=733, top_p=.23, frequency_penalty=.31)
    assert not rig.base_calls and not rig.tokenizer_calls and not rig.generations
