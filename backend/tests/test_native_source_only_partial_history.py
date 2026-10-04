"""Independent quoted-source controls, not actual native histories."""

import copy
import json

import pytest
from evaluation.native_seed_provenance import expected_seed_record_count
from tests.test_native_partial_history import contract, persist


def source_only_contract(tmp_path):
    phase, root, seed, data, proof = contract(tmp_path)
    turns = data["history-advancement-progress.json"]
    tasks = data["case.json"]["history_advancement_tasks"]
    snapshot = data["partial-snapshot.json"]
    for turn in turns[-3:]:
        task = tasks[turn["index"]]
        old = task["message"]
        task["message"] += '\n第三方给定数据：[{"label":"synthetic stock"}]'
        for row in snapshot["successful_native_message_rows"]:
            if row["message"] == old:
                row["message"] = task["message"]
        for row in snapshot["sources"]:
            if row.get("body") == old:
                row["body"] = task["message"]
        for call in data["cloud-calls.json"]:
            for message in call["request"]["messages"]:
                if old in message["content"]:
                    message["content"] = message["content"].replace(old, task["message"])
    ref = proof["native_partial_history_seed_origin"]
    ref["source_only_history"] = True
    persist(root, seed, data, proof)
    (root / "writer-admission.json").unlink()
    assert expected_seed_record_count(proof, phase) == 6
    return phase, root, seed, data, proof


def test_source_only_proves_missing_parser_artifact_without_fabricating_one(tmp_path):
    phase, root, _, _, proof = source_only_contract(tmp_path)
    assert expected_seed_record_count(proof, phase) == 6
    assert not (root / "writer-admission.json").exists()


@pytest.mark.parametrize("change", ["writer_call", "personal_fact", "existing_parser_artifact", "no_explicit_branch"])
def test_source_only_missing_artifact_cannot_waive_write_evidence(tmp_path, change):
    phase, root, seed, data, proof = source_only_contract(tmp_path)
    assert expected_seed_record_count(proof, phase) == 6
    if change == "writer_call":
        call = copy.deepcopy(data["cloud-calls.json"][-1])
        call["request"]["messages"] = [dict(role="user", content=json.dumps(dict(current_user_message="我喜欢红茶。")))]
        data["cloud-calls.json"].append(call)
        data["evaluation-budget-stop.json"]["actual_calls"] += 1
    elif change == "personal_fact":
        data["case.json"]["history_advancement_tasks"][-1]["message"] = '我喜欢红茶。附言："样例"。'
    elif change == "no_explicit_branch":
        proof["native_partial_history_seed_origin"].pop("source_only_history")
    persist(root, seed, data, proof)
    if change != "existing_parser_artifact":
        (root / "writer-admission.json").unlink()
    with pytest.raises(ValueError):
        expected_seed_record_count(proof, phase)
