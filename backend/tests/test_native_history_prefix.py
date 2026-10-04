"Independent successful-prefix contracts, never native seeds or paid receipts."

import copy
import hashlib
import json

import pytest

from evaluation.native_seed_provenance import expected_seed_record_count
from tests.test_native_history_snapshot import history_contract, save


def contract(tmp_path):
    phase, root, origin, calls, proof = history_contract(tmp_path)
    origin.update(http_status=500, primary_calls=2, writer_admission=[],
                  prepared=[dict(task=0), dict(task=1), dict(blocked_question=True)])
    before = copy.deepcopy(origin)
    before["prepared"] = before["prepared"][:-1]
    before["scheduler_before_question"] = before["scheduler_final"]
    before["writer_admission_before_question"] = before["writer_admission"]
    blocked = dict(model="deepseek-flash", max_tokens=2048,
                   messages=[dict(role="user", content="<user_query>请依据全部来源核对当前紫茶偏好与旧版本。</user_query>")])
    proof["history_advancement_turns_completed"] = 2
    proof["native_history_seed_origin"]["successful_prefix_only"] = True
    persist(root, origin, calls, proof, before, blocked)
    return phase, root, origin, calls, proof, before, blocked


def persist(root, origin, calls, proof, before, blocked):
    save(root, origin, calls, proof)
    ref = proof["native_history_seed_origin"]
    for name, data, key in [("before-question.json", before, "before_question_sha256"),
                            ("last-sent-request.json", blocked, "blocked_request_sha256")]:
        raw = json.dumps(data).encode()
        (root / name).write_bytes(raw)
        ref[key] = hashlib.sha256(raw).hexdigest()


def test_completed_native_prefix_can_restore_without_faking_terminal_success(tmp_path):
    phase, _, origin, _, proof, _, _ = contract(tmp_path)
    assert expected_seed_record_count(proof, phase) == 6
    assert origin["http_status"] == 500


@pytest.mark.parametrize("change", ["flag", "provider_after_prefix", "generated_answer",
                                    "snapshot_rows", "writer", "task_failure", "blocked_sent",
                                    "prepared_extra", "hash", "prefix_count", "backup_answer"])
def test_successful_prefix_requires_exact_completed_before_question_artifacts(tmp_path, change):
    phase, root, origin, calls, proof, before, blocked = contract(tmp_path)
    if change == "flag":
        proof["native_history_seed_origin"].pop("successful_prefix_only")
    elif change == "provider_after_prefix":
        calls.append(copy.deepcopy(calls[-1]))
        origin["cloud_calls"] = 3
    elif change == "generated_answer":
        origin["generation"].append(dict(observed_primary_calls=1))
    elif change == "snapshot_rows":
        before["seed_records"][0]["content"] = "substituted before-question record"
    elif change == "writer":
        origin["writer_admission"] = [dict(accepted=[dict(operation="ADD")])]
    elif change == "task_failure":
        origin["history_advancement_turns"][-1]["status"] = 500
    elif change == "blocked_sent":
        blocked = copy.deepcopy(calls[-1]["request"])
    elif change == "prepared_extra":
        origin["prepared"].append(dict(other=True))
    elif change == "hash":
        proof["native_history_seed_origin"]["before_question_sha256"] = "0" * 64
    elif change == "prefix_count":
        before["calls_before_question"] = 1
    elif change == "backup_answer":
        proof["before_question_backup"]["prior_same_task_answers"] = 1
    persist(root, origin, calls, proof, before, blocked)
    if change == "hash":
        proof["native_history_seed_origin"]["before_question_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        expected_seed_record_count(proof, phase)
