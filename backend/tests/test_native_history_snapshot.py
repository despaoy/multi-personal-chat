"""Independent complete contract fixtures, not native seeds or paid evidence."""

import copy
import hashlib
import json

import pytest

from evaluation.native_seed_provenance import expected_seed_record_count
from tests.test_native_successor_snapshot import contract


def history_contract(tmp_path):
    phase, _, seed, _, _, snapshot, prior = contract(tmp_path)
    raw = json.dumps(prior).encode()
    (seed / "before-question.json").write_bytes(raw)
    root = phase / "native-pg-history"
    root.mkdir()
    rows = copy.deepcopy(prior["seed_records"])
    sources = copy.deepcopy(prior["durable_seed_sources"])
    calls, turns = [], []
    for i in range(2):
        message = f"第三方完整紫茶目录{i}：甲费7元时限3小时容量4、乙费9元时限2小时容量5；全部前提登记完成，失败暂停。本表不描述本人。请列出两项并按费用排序。"
        sources.append(
            dict(
                source_message_id=f"task-{i}",
                body=message,
                state="recorded",
                observed_at=f"2026-10-04T04:00:0{i}+00:00",
            )
        )
        calls.append(
            dict(
                http_status=200,
                request=dict(
                    model="deepseek-flash",
                    max_tokens=2048,
                    messages=[dict(role="user", content=f"<user_query>{message}</user_query>")],
                ),
                response=dict(choices=[dict(message=dict(content="甲7元3小时4；乙9元2小时5；登记完成，失败暂停。"))]),
            )
        )
        turns.append(
            dict(index=i, task_id=f"contract-{i}", status=200, response=dict(abstained=False), input_chars=len(message))
        )
    dump = b"Complete opaque independent advanced-history snapshot, not native PostgreSQL"
    (root / "before-question.dump").write_bytes(dump)
    backup = dict(
        database_sha256=hashlib.sha256(dump).hexdigest(),
        bytes=len(dump),
        prior_same_task_answers=0,
        successful_seed_claim_ids=[r["id"] for r in rows],
    )
    origin = dict(
        synthetic_only=True,
        transport="authenticated_ASGI",
        database_mode="PostgreSQL",
        http_status=200,
        chat_auth_statuses=[200, 200],
        seed_template_verified=True,
        seed_scope=prior["seed_scope"],
        seed_records=rows,
        user_fact_records_after=rows,
        durable_seed_verified_before_question=True,
        before_question_backup=backup,
        durable_seed_sources=sources,
        scheduler_final=dict(saved=0, erased=0, failed=0),
        cloud_calls=2,
        history_advancement_turns_inherited=0,
        history_advancement_turns=turns,
        history_advancement_turns_completed=2,
        calls_before_question=2,
        primary_output_tokens=2048,
        generation=[dict(observed_primary_calls=1), dict(observed_primary_calls=1)],
    )
    proof = dict(
        seed_scope=prior["seed_scope"],
        seed_records=rows,
        durable_seed_sources=sources,
        durable_seed_verified_before_question=True,
        before_question_backup=backup,
        history_advancement_turns=turns,
        native_history_seed_origin=dict(
            phase="stage104",
            variant="native-pg-history",
            restored_seed=dict(
                phase="stage104", variant="native-pg-successor-seed", proof_sha256=hashlib.sha256(raw).hexdigest()
            ),
        ),
    )
    save(root, origin, calls, proof)
    return phase, root, origin, calls, proof


def save(root, origin, calls, proof):
    for name, data, key in [
        ("result.json", origin, "result_sha256"),
        ("cloud-calls.json", calls, "cloud_calls_sha256"),
    ]:
        raw = json.dumps(data).encode()
        (root / name).write_bytes(raw)
        proof["native_history_seed_origin"][key] = hashlib.sha256(raw).hexdigest()


def test_verified_native_history_snapshot_retains_actual_seed_count(tmp_path):
    phase, _, _, _, proof = history_contract(tmp_path)
    assert expected_seed_record_count(proof, phase) == 6


@pytest.mark.parametrize(
    "change",
    [
        "owner",
        "record",
        "saved",
        "erased",
        "source",
        "status",
        "dump",
        "prefix",
        "no_primary",
        "prior_hash",
        "self_origin",
        "answer",
        "cloud_status",
    ],
)
def test_history_receipt_is_not_a_count_or_success_override(tmp_path, change):
    phase, root, origin, calls, proof = history_contract(tmp_path)
    if change == "owner":
        proof["seed_scope"] = dict(owner="1")
    elif change == "record":
        origin["user_fact_records_after"][0]["content"] = "substituted"
    elif change == "saved":
        origin["scheduler_final"]["saved"] = 1
    elif change == "erased":
        origin["scheduler_final"]["erased"] = 1
    elif change == "source":
        origin["durable_seed_sources"][-1]["body"] = "Not on any completed task request"
    elif change == "status":
        origin["history_advancement_turns"][-1]["status"] = 500
    elif change == "dump":
        (root / "before-question.dump").write_bytes(b"other")
    elif change == "prefix":
        origin["calls_before_question"] = 1
    elif change == "no_primary":
        origin["generation"][0]["observed_primary_calls"] = 0
    elif change == "prior_hash":
        proof["native_history_seed_origin"]["restored_seed"]["proof_sha256"] = "0" * 64
    elif change == "self_origin":
        proof["native_history_seed_origin"]["restored_seed"]["variant"] = "native-pg-history"
    elif change == "answer":
        proof["before_question_backup"]["prior_same_task_answers"] = 1
    elif change == "cloud_status":
        calls[0]["http_status"] = 402
    save(root, origin, calls, proof)
    with pytest.raises(ValueError):
        expected_seed_record_count(proof, phase)
