import copy
import json

import pytest

from training.preference_compat import reference_adapter_kwargs
from training.preference_trainer import PreferenceTrainer, PreferenceTrainingConfig
from training.preference_validation import (
    group_split,
    validate_adapter_path,
    validate_partitions,
    validate_token_budgets,
)


def pair(index=1, group=None):
    return {
        "id": str(index),
        "prompt": f"question {index}",
        "chosen": "grounded answer",
        "rejected": "invented history",
        "review_status": "approved",
        "metadata": {"source_group": group or f"scene-{index}"},
    }


@pytest.mark.parametrize("field,value", [("chosen", ""), ("rejected", "grounded answer"), ("review_status", "pending")])
def test_invalid_preferences_fail_before_model_loading(field, value):
    row = pair()
    row[field] = value
    result = PreferenceTrainer(PreferenceTrainingConfig(base_model_path="unused")).train([row])
    assert result.error
    assert not result.environment  # Expensive model libraries have not been imported.


def test_missing_adapter_cannot_fall_back_to_base(tmp_path):
    result = PreferenceTrainer(PreferenceTrainingConfig(adapter_path=str(tmp_path / "missing"))).train([pair()])
    assert "refusing base-model fallback" in result.error
    assert not result.environment


def test_adapter_requires_config_weights_and_lora_contract(tmp_path):
    (tmp_path / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA"}))
    with pytest.raises(ValueError, match="weights"):
        validate_adapter_path(str(tmp_path))
    (tmp_path / "adapter_model.safetensors").write_bytes(b"placeholder")
    validate_adapter_path(str(tmp_path))
    (tmp_path / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "bias": "all"}))
    with pytest.raises(ValueError, match="bias=none"):
        validate_adapter_path(str(tmp_path))


@pytest.mark.parametrize("key", ["source_group", "source_ids", "evidence_text_sha256", "scene_id"])
def test_validation_rejects_provenance_leakage(key):
    train, validation = pair(1), pair(2)
    value = ["shared"] if key in {"source_ids", "evidence_text_sha256"} else "shared"
    train["metadata"][key] = value
    validation["metadata"][key] = value
    with pytest.raises(ValueError, match="cross-split"):
        validate_partitions([train], [validation])


@pytest.mark.parametrize("key", ["id", "prompt"])
def test_validation_rejects_duplicate_ids_or_normalized_prompts(key):
    train, validation = pair(1), pair(2)
    validation[key] = train[key] if key == "id" else " QUES TION 1 "
    with pytest.raises(ValueError, match="cross-split"):
        validate_partitions([train], [validation])


def test_final_test_never_used_for_validation():
    validation = pair(2)
    validation["metadata"]["split"] = "test"
    with pytest.raises(ValueError, match="partition"):
        validate_partitions([pair()], [validation])


def test_group_split_is_deterministic_transitive_and_does_not_mutate():
    rows = [pair(i) for i in range(8)]
    rows[0]["metadata"]["source_ids"] = ["source-a"]
    rows[1]["metadata"].update(source_ids=["source-a", "source-b"])
    rows[2]["metadata"]["source_ids"] = ["source-b"]
    original = copy.deepcopy(rows)
    train, validation = group_split(rows)
    assert (train, validation) == group_split(list(reversed(rows)))
    assert rows == original
    groups = [{row["id"] for row in items} for items in (train, validation)]
    assert any({"0", "1", "2"} <= group for group in groups)
    validate_partitions(train, validation)
    assert {row["metadata"]["split"] for row in validation} == {"validation"}


def test_group_split_refuses_missing_or_single_source():
    with pytest.raises(ValueError, match="two independent"):
        group_split([pair(1, "same"), pair(2, "same")])
    rows = [pair(1), pair(2)]
    rows[0]["metadata"] = {}
    with pytest.raises(ValueError, match="requires source"):
        group_split(rows)


def test_reference_capability_fails_closed_for_implicit_new_api():
    def supported(model_adapter_name=None, ref_adapter_name=None):
        pass

    assert reference_adapter_kwargs(supported, method="dpo", adapter_path="sft") == {
        "model_adapter_name": "default",
        "ref_adapter_name": "reference",
    }
    with pytest.raises(RuntimeError, match="explicit SFT reference"):
        reference_adapter_kwargs(lambda: None, method="dpo", adapter_path="sft")
    assert reference_adapter_kwargs(lambda: None, method="orpo", adapter_path="sft") == {}


def test_plain_text_budgets_cannot_silently_truncate():
    def tokenizer(text, **kwargs):
        return {"input_ids": list(range(len(text)))}

    validate_token_budgets([pair()], tokenizer, max_length=100, max_prompt_length=50)
    with pytest.raises(ValueError, match="max_prompt_length"):
        validate_token_budgets([pair()], tokenizer, max_length=100, max_prompt_length=2)
    with pytest.raises(ValueError, match="max_length"):
        validate_token_budgets([pair()], tokenizer, max_length=20, max_prompt_length=15)


def test_reference_weight_integrity_detects_mutation_and_trainable_reference():
    torch = pytest.importorskip("torch")
    from training.preference_reference import assert_initial_reference, assert_reference_unchanged

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora = torch.nn.ParameterDict(
                {
                    "default": torch.nn.Parameter(torch.ones(2)),
                    "reference": torch.nn.Parameter(torch.ones(2), requires_grad=False),
                }
            )

        def named_parameters(self):
            return iter([(f"layer.lora.{key}.weight", value) for key, value in self.lora.items()])

    model = Model()
    digest = assert_initial_reference(model)
    with torch.no_grad():
        model.lora["default"].add_(1)
    assert_reference_unchanged(model, digest)
    with torch.no_grad():
        model.lora["reference"].add_(1)
    with pytest.raises(RuntimeError, match="changed"):
        assert_reference_unchanged(model, digest)
    model.lora["reference"].requires_grad_(True)
    with pytest.raises(RuntimeError, match="frozen"):
        assert_initial_reference(model)


def test_validation_loader_preserves_partition_and_rejects_test(tmp_path):
    from training.evidence_dataset import load_preference_training_rows

    row = pair()
    row["metadata"]["split"] = "validation"
    path = tmp_path / "validation.jsonl"
    path.write_text(json.dumps(row))
    assert load_preference_training_rows(path, split="validation")[0]["id"] == "1"
    with pytest.raises(ValueError, match="partition"):
        load_preference_training_rows(path)
    with pytest.raises(ValueError, match="train/validation"):
        load_preference_training_rows(path, split="test")


def test_grouped_freezer_outputs_loadable_independent_partitions(monkeypatch, tmp_path):
    from scripts.prepare_kisaki_dpo_v3 import main

    from training.evidence_dataset import load_preference_training_rows

    rows = [pair(index) for index in range(10)]
    for row in rows:
        row["annotator"] = "manual"
        row["metadata"]["persona"] = "kisaki"
    source, output = tmp_path / "reviewed.jsonl", tmp_path / "frozen"
    source.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["prepare", "--input", str(source), "--output-dir", str(output), "--minimum", "10"])
    assert main() == 0
    train = load_preference_training_rows(output / "kisaki_dpo_train.jsonl")
    validation = load_preference_training_rows(output / "kisaki_dpo_heldout.jsonl", split="validation")
    validate_partitions(train, validation)
    assert len(train) + len(validation) == 10
    assert json.loads((output / "manifest.json").read_text())["schema_version"] == 2
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        main()


def test_budget_accounts_for_separate_completion_tokenization():
    def tokenizer(text, **kwargs):
        # A synthetic merge at the prompt/completion boundary: concatenation
        # hides the real DPO length, since DPO tokenizes the fields separately.
        return {"input_ids": [1] * (1 if text.startswith("question 1grounded") else len(text))}

    with pytest.raises(ValueError, match="max_length"):
        validate_token_budgets([pair()], tokenizer, max_length=22, max_prompt_length=14)


def test_existing_adapter_output_is_never_overwritten(tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}")
    result = PreferenceTrainer(PreferenceTrainingConfig(base_model_path="unused", output_dir=str(tmp_path))).train(
        [pair()]
    )
    assert "refusing to overwrite" in result.error
    assert (tmp_path / "adapter_config.json").read_text() == "{}"


def test_direct_python_entry_cannot_bypass_review_by_omitting_status():
    row = pair()
    del row["review_status"]
    result = PreferenceTrainer(PreferenceTrainingConfig(base_model_path="unused")).train([row])
    assert "explicit approved" in result.error
    assert not result.environment
