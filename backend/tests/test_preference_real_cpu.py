"""Real tiny-model integration: no downloads, GPU, role data or production assets."""

import json

import pytest


@pytest.mark.parametrize("method,conversational", [("dpo", False), ("dpo", True), ("orpo", False)])
def test_dpo_sft_reference_eval_and_policy_export_on_cpu(tmp_path, method, conversational):
    torch = pytest.importorskip("torch")
    pytest.importorskip("trl")
    pytest.importorskip("peft")
    from peft import LoraConfig, PeftModel, get_peft_model
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    from training.preference_trainer import PreferenceTrainer, PreferenceTrainingConfig

    base, sft, output = (tmp_path / name for name in ("base", "sft", "dpo"))
    vocab = {
        word: i
        for i, word in enumerate(
            [
                "[PAD]",
                "[UNK]",
                "[EOS]",
                "question",
                "first",
                "second",
                "heldout",
                "correct",
                "wrong",
            ]
        )
    }
    tokenizer_core = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer_core.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_core,
        pad_token="[PAD]",
        unk_token="[UNK]",
        eos_token="[EOS]",
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' [EOS] ' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ 'assistant ' }}{% endif %}"
    )
    tokenizer.save_pretrained(base)
    torch.manual_seed(42)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=len(vocab),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=128,
            pad_token_id=0,
            eos_token_id=2,
            attention_dropout=0,
        )
    )
    model.save_pretrained(base)
    model.name_or_path = str(base)
    adapted = get_peft_model(
        model,
        LoraConfig(
            r=2,
            lora_alpha=2,
            lora_dropout=0,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    # Nonzero SFT weights distinguish an SFT reference from the base model.
    with torch.no_grad():
        for name, parameter in adapted.named_parameters():
            if "lora_B" in name:
                parameter.fill_(0.03)
    adapted.save_pretrained(sft)

    def row(question, group):
        result = {
            "id": group,
            "prompt": f"question {question} ",
            "chosen": "correct",
            "rejected": "wrong",
            "metadata": {"source_group": group},
            "review_status": "approved",
        }
        if conversational:
            result.update(
                prompt=[{"role": "user", "content": result["prompt"]}],
                chosen=[{"role": "assistant", "content": result["chosen"]}],
                rejected=[{"role": "assistant", "content": result["rejected"]}],
            )
        return result

    config = PreferenceTrainingConfig(
        method=method,
        base_model_path=str(base),
        adapter_path=str(sft),
        output_dir=str(output),
        load_in_4bit=False,
        gradient_checkpointing=False,
        learning_rate=1e-3,
        gradient_accumulation_steps=1,
        max_length=32,
        max_prompt_length=16,
    )
    result = PreferenceTrainer(config).train([row("first", "s1"), row("second", "s2")], [row("heldout", "s3")])
    assert result.error == "", result.error
    assert result.train_steps == 2
    assert result.data_counts == {"train": 2, "validation": 1}
    assert result.baseline_eval_metrics and result.eval_metrics
    assert result.eval_accuracy is not None
    if method == "dpo":
        assert result.reference_snapshot["verified_unchanged"] is True
        assert result.reference_snapshot["sha256_before"] == result.reference_snapshot["sha256_after"]
        assert result.baseline_eval_metrics["eval_rewards/chosen"] == pytest.approx(0, abs=1e-6)
        assert result.baseline_eval_metrics["eval_rewards/rejected"] == pytest.approx(0, abs=1e-6)
    else:
        assert result.reference_snapshot == {}
    assert "not generated-response" in result.metric_note
    assert (output / "adapter_config.json").is_file()
    assert not (output / "reference").exists()
    reloaded = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base), output)
    assert set(reloaded.peft_config) == {"default"}
    original = dict(adapted.named_parameters())
    assert any(
        not torch.equal(parameter, original[name]) for name, parameter in reloaded.named_parameters() if "lora_" in name
    )
    report = PreferenceTrainer(config).save_report(result, output)
    assert json.loads(report.read_text())["result"]["environment"]["trl"]
    assert len(result.data_sha256["train"]) == 64
    assert result.data_sha256["train"] != result.data_sha256["validation"]
