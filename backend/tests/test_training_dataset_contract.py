"""Created data reaches the actual trainer with complete context and fixed splits."""

import copy
import json
from pathlib import Path

import pytest

from training.chat_dataset import normalize_chat_record
from training.preprocessor import DatasetPreprocessor
from training.task_manager import SimpleLoRATrainer


PROMPT = "系统专属标记：你是虚构角色甲，保留未付款、天气和书面改期条件。"
QUESTION = "用户专属标记：小林预约周四14点，3张票每张2元，尚未付款，下雨取消。"
EXTRA = "附加输入专属标记：改期需要书面确认，预约不代表已付款。"
ANSWER = "助手专属标记：小林周四14点预约，共6元，尚未付款，下雨取消，改期需要书面确认。"
HISTORY = [["历史用户专属标记：可以自动改期吗？", "历史助手专属标记：需要书面确认，不能自动改期。"]]


@pytest.mark.parametrize("kind", ["alpaca", "prompt", "conversations", "messages", "mixed"])
def test_created_data_reaches_actual_training_preparation(monkeypatch, tmp_path, kind):
    from test_training_chat_dataset import TinyChatTokenizer
    from training.trainer import LoRATrainer

    rows = []
    for index in range(12):
        shape = ["alpaca", "prompt", "conversations", "messages"][index % 4] if kind == "mixed" else kind
        question = f"样本{index}。" + QUESTION
        if shape == "alpaca":
            row = {"instruction": question, "input": EXTRA, "output": ANSWER, "history": copy.deepcopy(HISTORY)}
        elif shape == "prompt":
            row = {"prompt": question + "\n" + EXTRA, "response": ANSWER, "history": copy.deepcopy(HISTORY)}
        else:
            row = {shape: [{"role": "user", "content": HISTORY[0][0]}, {"role": "assistant", "content": HISTORY[0][1]},
                          {"role": "user", "content": question + "\n" + EXTRA}, {"role": "assistant", "content": ANSWER}]}
        rows.append(row)
    original = copy.deepcopy(rows)
    creator = DatasetPreprocessor(base_dir=tmp_path)
    directory, stats = creator.prepare_training_data(rows, output_name="verified", custom_prompt=PROMPT)
    assert rows == original and stats["invalid_samples"] == 0
    stored = [*json.loads((directory / "train.json").read_text()), *json.loads((directory / "eval.json").read_text())]
    assert sorted(json.dumps(row, sort_keys=True) for row in stored) == sorted(json.dumps(row, sort_keys=True) for row in original)
    monkeypatch.setenv("LORA_PATH", str(tmp_path / "loras"))
    config = SimpleLoRATrainer(base_dir=tmp_path)._build_config("fiction", directory, {"packing": False, "max_seq_length": 512})
    assert config.system_prompt == PROMPT and Path(config.eval_data_path) == directory / "eval.json"
    tokenizer = TinyChatTokenizer()
    prepared = LoRATrainer(config)._load_and_preprocess_data(tokenizer)
    assert len(prepared["train"]) == 10 and len(prepared["eval"]) == 2
    for partition in prepared.values():
        for row in partition:
            full = tokenizer.decode(row["input_ids"])
            supervised = tokenizer.decode([token for token, label in zip(row["input_ids"], row["labels"], strict=True) if label != -100])
            assert PROMPT in full and QUESTION in full and EXTRA in full and HISTORY[0][0] in full
            assert ANSWER in supervised and HISTORY[0][1] in supervised
            assert "用户专属标记" not in supervised and "系统专属标记" not in supervised


def test_manifest_selection_and_explicit_options_have_clear_precedence(monkeypatch, tmp_path):
    (tmp_path / "chosen.json").write_text("[]")
    (tmp_path / "train.json").write_text("[]")
    (tmp_path / "dataset_info.json").write_text(json.dumps({"file_name": "chosen.json", "file_name_eval": "unused-missing.json", "custom_prompt": PROMPT}))
    monkeypatch.setenv("LORA_PATH", str(tmp_path / "loras"))
    manager = SimpleLoRATrainer(base_dir=tmp_path)
    config = manager._build_config("fiction", tmp_path, {"system_prompt": "用户明确覆盖的人设", "eval_data_path": None})
    assert Path(config.train_data_path) == tmp_path / "chosen.json"
    assert config.system_prompt == "用户明确覆盖的人设" and config.eval_data_path is None


@pytest.mark.parametrize("fault", ["missing", "escape", "directory", "metadata_only"])
def test_manifest_errors_do_not_select_another_training_file(monkeypatch, tmp_path, fault):
    directory = tmp_path / "dataset"
    directory.mkdir()
    info = {"custom_prompt": PROMPT}
    if fault != "metadata_only":
        (directory / "train.json").write_text("[]")
        info["file_name"] = "missing.json"
    if fault == "escape":
        (tmp_path / "outside.json").write_text("[]")
        info["file_name"] = "../outside.json"
    elif fault == "directory":
        info["file_name"] = "."
    (directory / "dataset_info.json").write_text(json.dumps(info))
    monkeypatch.setenv("LORA_PATH", str(tmp_path / "loras"))
    with pytest.raises(FileNotFoundError if fault in {"missing", "metadata_only"} else ValueError):
        SimpleLoRATrainer(base_dir=tmp_path)._build_config("fiction", directory, {})


@pytest.mark.parametrize("change", ["input_type", "history_type", "partial_pair", "empty_turn"])
def test_flat_records_require_complete_context(change):
    record = {"instruction": QUESTION, "input": EXTRA, "output": ANSWER, "history": copy.deepcopy(HISTORY)}
    if change == "input_type":
        record["input"] = [EXTRA]
    elif change == "history_type":
        record["history"] = "not a pair list"
    elif change == "partial_pair":
        record["history"] = [[HISTORY[0][0]]]
    else:
        record["history"] = [[HISTORY[0][0], " "]]
    with pytest.raises(ValueError):
        normalize_chat_record(record, default_system_prompt=PROMPT)


def test_unspecified_training_persona_does_not_invent_a_character():
    from training.trainer import LoRATrainingConfig

    config = LoRATrainingConfig()
    messages = normalize_chat_record({"prompt": QUESTION + "\n" + EXTRA, "response": ANSWER}, default_system_prompt=config.system_prompt)
    assert config.system_prompt == "" and messages == [{"role": "user", "content": QUESTION + "\n" + EXTRA}, {"role": "assistant", "content": ANSWER}]


def test_null_required_answer_is_never_repaired():
    with pytest.raises(ValueError):
        normalize_chat_record({"instruction": QUESTION, "input": EXTRA, "output": None}, default_system_prompt=PROMPT)
