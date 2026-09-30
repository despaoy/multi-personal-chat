"""PPO math and real, randomly initialized tiny-Qwen training (no external calls)."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from test_persona_alignment_workflow import scene

from training.persona_ppo import PersonaPPO, PPOConfig, verify_calibration
from training.ppo_math import action_statistics, clipped_loss, generalized_advantages, sampled_kl


def test_action_alignment_and_detached_critic():
    logits = torch.randn(1, 5, 8, requires_grad=True)
    hidden = torch.randn(1, 5, 3, requires_grad=True)
    critic = torch.nn.Linear(3, 1)

    def model(**kw):
        return SimpleNamespace(logits=logits, hidden_states=[hidden])

    tokens = torch.tensor([[0, 1, 2, 3, 4]])
    probs, values, _ = action_statistics(model, critic, tokens, 3)
    expected = logits[0, 2:4].log_softmax(-1)[torch.arange(2), torch.tensor([3, 4])]
    torch.testing.assert_close(probs, expected)
    values.sum().backward()
    assert hidden.grad is None and logits.grad is None
    assert critic.weight.grad is not None


def test_gae_and_clipping():
    advantages, returns = generalized_advantages(torch.tensor([0.0, 1.0, 0.0, 2.0]), torch.ones(4), lam=1)
    torch.testing.assert_close(returns, torch.tensor([3.0, 3.0, 2.0, 2.0]))
    torch.testing.assert_close(advantages, returns - 1)
    zero = torch.zeros(2)
    loss, metrics = clipped_loss(
        torch.tensor([1.5, 0.5]).log(), zero, zero, zero, zero, torch.tensor([1.0, -1.0]), zero, value_coef=0
    )
    assert loss.item() == pytest.approx(-0.2)
    assert metrics["clip_fraction"] == 1
    assert sampled_kl(zero, zero).sum() == 0
    with pytest.raises(ValueError, match="nonfinite"):
        sampled_kl(zero, torch.tensor([float("inf"), 0.0]))


@pytest.fixture
def tiny(tmp_path):
    from peft import LoraConfig, get_peft_model
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    torch.set_num_threads(1)
    torch.manual_seed(42)
    base, sft = tmp_path / "base", tmp_path / "sft"
    core = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "[EOS]": 2, "hello": 3, "yes": 4}, unk_token="[UNK]"))
    core.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=core, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]")
    tok.chat_template = "{% for m in messages %}{{ m['role'] + ' ' + m['content'] + ' [EOS] ' }}{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
    tok.save_pretrained(base)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=5,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=1024,
            pad_token_id=0,
            eos_token_id=2,
            attention_dropout=0,
        )
    )
    model.save_pretrained(base)
    model.name_or_path = str(base)
    model = get_peft_model(
        model, LoraConfig(r=2, lora_alpha=2, lora_dropout=0, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    )
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.fill_(0.03)
    model.save_pretrained(sft)
    return PPOConfig(
        str(base),
        str(sft),
        str(tmp_path / "full"),
        updates=2,
        rollout_batch_size=8,
        ppo_epochs=2,
        learning_rate=0.001,
        critic_learning_rate=0.01,
        max_prompt_length=900,
        max_new_tokens=24,
        max_length=1024,
        max_turns=2,
        allow_uncalibrated_pilot=True,
    )


class SyntheticReward:
    contract = {"kind": "synthetic_test", "judge_id": "test-v1"}

    def __init__(self, abstain=False):
        self.records = []
        self.abstain = abstain

    def score(self, record):
        self.records.append(record)
        return {
            **{k: record[k] for k in ("candidate_id", "context_sha256", "response_sha256")},
            "judge_id": "test-v1",
            "status": "abstained" if self.abstain else "scored",
            "aggregation": {"reward": 0.8 if "yes" in record["response"] else -0.2},
        }


def test_real_ppo_update_resume_and_export(tiny, tmp_path):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    data = [{**scene(), "continuations": ["hello"]}]
    reward = SyntheticReward()
    full = PersonaPPO(tiny, reward)
    result = full.train(data, validation=[scene(2, "validation")])
    assert result["updates_completed"] == 2
    assert result["reference_verified_unchanged"]
    assert reward.records  # Real sampling reached complete nonempty responses.
    assert any(len(r["scene"]["history"]) > 2 for r in reward.records)
    assert full.critic.weight.abs().sum() > 0
    parameters = dict(full.model.named_parameters())
    assert any(
        not torch.equal(p, parameters[n.replace(".default.", ".reference.")])
        for n, p in parameters.items()
        if "lora_" in n and ".default." in n
    )
    partial = PersonaPPO(replace(tiny, output_dir=str(tmp_path / "partial")), SyntheticReward())
    first = partial.train(data, validation=[scene(2, "validation")], stop_after=1)
    resumed = PersonaPPO(replace(tiny, output_dir=str(tmp_path / "resumed")), SyntheticReward())
    final = resumed.train(data, validation=[scene(2, "validation")], resume=first["last_checkpoint"])
    assert final["history"] == result["history"]
    for name, p in resumed.model.named_parameters():
        torch.testing.assert_close(p, parameters[name], rtol=0, atol=0)
    torch.testing.assert_close(full.critic.weight, resumed.critic.weight, rtol=0, atol=0)
    loaded = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(tiny.base_model_path), result["adapter_path"]
    )
    assert set(loaded.peft_config) == {"default"}
    assert json.loads((tmp_path / "full" / "validation-000002" / "summary.json").read_text())["total"] == 2
    bad = PersonaPPO(replace(tiny, output_dir=str(tmp_path / "bad"), kl_coef=0.2), SyntheticReward())
    with pytest.raises(ValueError, match="mismatch"):
        bad.train(data, validation=[scene(2, "validation")], resume=first["last_checkpoint"])
    checkpoint = __import__("pathlib").Path(first["last_checkpoint"])
    with (checkpoint / "state.pt").open("ab") as f:
        f.write(b"tamper")
    bad = PersonaPPO(replace(tiny, output_dir=str(tmp_path / "tampered")), SyntheticReward())
    with pytest.raises(ValueError, match="integrity"):
        bad.train(data, validation=[scene(2, "validation")], resume=checkpoint)


def test_calibration_gate(tiny):
    with pytest.raises(ValueError, match="calibration"):
        verify_calibration(replace(tiny, allow_uncalibrated_pilot=False), SyntheticReward.contract, None)
    report = dict(
        status="measured_not_approved_for_ppo",
        judge_id="test-v1",
        pair_count=30,
        source_group_count=10,
        agreement=0.8,
        coverage=0.9,
    )
    assert "thresholds_met" in verify_calibration(tiny, SyntheticReward.contract, report)
    with pytest.raises(ValueError, match="exact judge"):
        verify_calibration(tiny, SyntheticReward.contract, {**report, "judge_id": "other"})


def test_final_test_and_context_budget_rejected(tiny):
    trainer = PersonaPPO(tiny, SyntheticReward())
    with pytest.raises(ValueError, match="train scenes"):
        trainer.train([scene(1, "test")])
    trainer._load()
    trainer.config = replace(tiny, max_prompt_length=1)
    with pytest.raises(ValueError, match="budget"):
        trainer._prompt([{"role": "user", "content": "hello hello hello"}])


def test_controlled_abstention_and_reward_binding(tiny, tmp_path):
    """Controlled actions exercise error paths, not sampling effectiveness."""
    reward = SyntheticReward(abstain=True)
    trainer = PersonaPPO(tiny, reward)
    trainer._load()
    trainer.run_signature = "synthetic-unit-test"
    trainer._generate = lambda *a, **kw: (torch.tensor([[1, 3, 2]]), 1, "hello", True)
    segments, trace = trainer._collect_episode(scene(), 0)
    assert not segments and trace["status"] == "abstained"
    reward.abstain = False
    score = reward.score
    reward.score = lambda record: {**score(record), "response_sha256": "wrong"}
    with pytest.raises(ValueError, match="another response"):
        trainer._collect_episode(scene(), 0)
    reward.score = lambda record: {**score(record), "judge_id": "wrong"}
    with pytest.raises(ValueError, match="judge identity"):
        trainer._collect_episode(scene(), 0)
    reward.score = score
    reward.abstain = True
    result = trainer.train([scene()])
    assert result["status"] == "stopped_no_scorable_rollouts"
    assert result["updates_completed"] == 0


def test_preflight_binds_base_and_checks_budget(tiny):
    from pathlib import Path

    trainer = PersonaPPO(tiny, SyntheticReward())
    first = trainer.preflight([scene()])
    assert trainer.model is None and not Path(tiny.output_dir).exists()
    config_path = Path(tiny.base_model_path) / "config.json"
    config_path.write_text(config_path.read_text() + "\n")
    assert trainer.preflight([scene()])["run_signature"] != first["run_signature"]
    trainer.config = replace(tiny, max_prompt_length=1)
    with pytest.raises(ValueError, match="budget"):
        trainer.preflight([scene()])


def test_local_validation_exports_to_blind_review(tiny, tmp_path):
    from training.persona_alignment import load_run
    from training.persona_review import make_packet

    trainer = PersonaPPO(tiny, SyntheticReward())
    trainer._load()
    trainer.run_signature = "synthetic-evaluation-test"
    trainer._generate = lambda *a, **kw: (
        None,
        1,
        "hello" if trainer.model.active_adapter == "reference" else "yes",
        True,
    )
    path = tmp_path / "evaluation"
    trainer._evaluate([scene(2, "validation")], path)
    records = load_run(path)
    packet, key, template = make_packet(records, purpose="evaluation", baseline="sft_reference", candidate="ppo_policy")
    assert len(packet["items"]) == 1
    assert not template["human_confirmed"]


def test_cli_check_only_without_model_load_or_judge_call(tiny, tmp_path, monkeypatch, capsys):
    import sys
    from dataclasses import asdict

    from training.persona_ppo import base_model_identity, main

    config_path, scenes_path, judge_path = [tmp_path / name for name in ("ppo.json", "scenes.jsonl", "judge.json")]
    config_path.write_text(json.dumps(asdict(tiny)), encoding="utf-8")
    scenes_path.write_text(json.dumps(scene()) + "\n", encoding="utf-8")
    judge_path.write_text(
        json.dumps(
            {"name": "unreachable-test", "model": "test", "revision": "test-v1", "base_url": "http://127.0.0.1:1/v1"}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "persona_ppo",
            "--config",
            str(config_path),
            "--train-scenes",
            str(scenes_path),
            "--judge-config",
            str(judge_path),
            "--check-only",
        ],
    )
    main()
    report = json.loads(capsys.readouterr().out)
    assert report["train_scenes"] == 1 and report["reward_status"] == "uncalibrated_pilot"
    assert not __import__("pathlib").Path(tiny.output_dir).exists()
    with pytest.raises(ValueError, match="immutable"):
        base_model_identity(replace(tiny, base_model_path="org/model", base_revision="main"))
