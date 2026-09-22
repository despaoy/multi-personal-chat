"""Offline evaluation must never silently truncate complete evidence."""

import pytest

from evaluation.offline_reviewer import final_answer_text, validate_generation_budget


@pytest.mark.parametrize("count", [0, -1, 4097])
def test_reject_incomplete_prompt(count):
    with pytest.raises(ValueError, match="no truncation"):
        validate_generation_budget(count, 4096)


@pytest.mark.parametrize("count", [1, 4096])
def test_accept_complete_prompt(count):
    validate_generation_budget(count, 4096)


@pytest.fixture
def fake_offline_backend(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    import torch

    from evaluation.offline_reviewer import OfflineTransformersReviewer

    observations = {}

    class Inputs(dict):
        def to(self, device):
            assert device == "cpu"
            return self

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 2

        def apply_chat_template(self, messages, **kwargs):
            observations["template"] = kwargs
            return "complete prompt"

        def __call__(self, text, **kwargs):
            observations["tokenize"] = kwargs
            return Inputs(input_ids=torch.tensor([[1, 3, 4]]), attention_mask=torch.tensor([[1, 1, 1]]))

        def decode(self, tokens, **kwargs):
            return "{}"

    class Model:
        device = "cpu"
        config = SimpleNamespace(quantization_config={"quant_method": "bitsandbytes"})
        generation_config = SimpleNamespace(eos_token_id=[2, 9])

        def eval(self):
            return self

        def generate(self, **kwargs):
            observations["generate"] = kwargs
            observations["grad_enabled"] = torch.is_grad_enabled()
            return torch.tensor([[1, 3, 4, 8, observations.get("end_token", 9)]])

    def load_model(path, **kwargs):
        observations["load"] = kwargs
        return Model()

    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model),
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: Tokenizer()),
        ),
    )
    return OfflineTransformersReviewer(tmp_path), observations


async def test_local_generation_preserves_full_input_and_gradient_state(fake_offline_backend):
    import torch

    reviewer, observed = fake_offline_backend
    with torch.enable_grad():
        assert await reviewer([{"role": "user", "content": "input"}]) == "{}"
        assert torch.is_grad_enabled()
    assert not observed["grad_enabled"]
    assert observed["load"]["local_files_only"] is True
    assert observed["load"]["trust_remote_code"] is False
    assert observed["load"]["use_safetensors"] is True
    assert observed["tokenize"]["truncation"] is False
    assert observed["template"]["enable_thinking"] is False
    assert observed["generate"]["max_time"] == 90
    assert observed["generate"]["pad_token_id"] == 0
    assert reviewer.calls[0]["complete"] is True


async def test_unfinished_generation_is_not_reported_as_complete_json(fake_offline_backend):
    reviewer, observed = fake_offline_backend
    observed["end_token"] = 8
    with pytest.raises(TimeoutError, match="limit"):
        await reviewer([{"role": "user", "content": "input"}])
    assert reviewer.calls[0]["complete"] is False


def test_thinking_boundary_strips_scratch_text_only_in_explicit_mode():
    assert final_answer_text('private scratch</think> {"decisions":[]}', thinking=True) == '{"decisions":[]}'
    assert final_answer_text('  {"decisions":[]} ', thinking=False) == '{"decisions":[]}'


@pytest.mark.parametrize("text", ["{}", "scratch</think>", "a</think>b</think>{}", "a</think><think>x"])
def test_unfinished_or_ambiguous_thinking_cannot_be_used(text):
    with pytest.raises(ValueError):
        final_answer_text(text, thinking=True)


async def test_sampled_ablation_uses_declared_parameters_and_restores_rng(fake_offline_backend):
    import torch

    reviewer, observed = fake_offline_backend
    reviewer.decoding = "sampled"
    reviewer.metadata["sampling_parameters"] = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0}
    before = torch.get_rng_state().clone()
    assert await reviewer([{"role": "user", "content": "input"}]) == "{}"
    assert torch.equal(before, torch.get_rng_state())
    assert observed["generate"]["do_sample"] is True
    assert observed["generate"]["temperature"] == 0.7
