"Independent partial-history artifacts; never injected as native records or paid evidence."

import copy
import hashlib
import json

import pytest

from evaluation.native_seed_provenance import expected_seed_record_count
from tests.test_native_history_snapshot import history_contract


def contract(tmp_path):
    prior_phase, prior_root, _, _, prior = history_contract(tmp_path)
    prior_raw = json.dumps(prior).encode()
    (prior_root / "before-question.json").write_bytes(prior_raw)
    phase = tmp_path / "stage105"
    root = phase / "native-pg-partial"
    seed = phase / "native-pg-snapshot"
    root.mkdir(parents=True)
    seed.mkdir()
    scope_row = prior["seed_records"][0]
    owner = prior["seed_scope"]["owner"]
    owner_key = json.dumps((scope_row["platform"], scope_row["adapter"], owner))
    scope_key = json.dumps((scope_row["character_id"], scope_row["platform"],
                            scope_row["adapter"], owner, "private", owner))
    tasks = [dict(id=t["task_id"], message=prior["durable_seed_sources"][-2+i]["body"])
             for i, t in enumerate(prior["history_advancement_turns"])]
    turns = copy.deepcopy(prior["history_advancement_turns"])
    sources = copy.deepcopy(prior["durable_seed_sources"])
    messages, calls = [], []
    for i in range(2):
        text = f"第三方完整紫馆{i}：费用9元、预约提前8小时、上限40人，全部必要材料是登记页与有效票证，不符合时暂停。请只依据全文列出。"
        source_id = f"independent-partial-{i}"
        index = len(tasks)
        tasks.append(dict(id=f"new-{i}", message=text))
        reply = "费用9元，预约提前8小时，上限40人，登记页与有效票证缺一不可，不符合时暂停。"
        turns.append(dict(index=index, task_id=f"new-{i}", status=200, response=dict(abstained=False, reply=reply)))
        sources.append(dict(source_message_id=source_id, body=text, state="recorded", observed_at="2026-10-04T08:00:00+00:00", owner_key=owner_key, scope_key=scope_key))
        messages.append(dict(message=text, reply=reply, senderId=owner, userId=owner,
                             characterId=scope_row["character_id"], platform=scope_row["platform"],
                             adapter=scope_row["adapter"], conversationType="private", branchId=None,
                             sessionId="stage105-advance-"+str(index+1), sourceMessageId=source_id))
        calls.append(dict(http_status=200, request=dict(model="deepseek-flash", max_tokens=2048,
                         messages=[dict(role="user", content="<user_query>"+text+"</user_query>")]),
                         response=dict(choices=[dict(message=dict(content=reply))])))
    failed_index = len(tasks)
    tasks.append(dict(id="pending", message="完整第三方资料：丙费用8元，时限2小时，请只按原文列出。"))
    progress = [*turns, dict(index=failed_index, task_id="pending", status=500, response=dict(detail="budget stopped before main"))]
    clock = "2026-10-04T08:02:00+00:00"
    sources.append(dict(source_message_id="pending-source", body=None, state="pending",
                        observed_at=clock, owner_key=owner_key, scope_key=scope_key))
    calls.append(dict(http_status=200, request=dict(model="deepseek-flash", max_tokens=768,
                 messages=[dict(role="user", content=tasks[-1]["message"])]),
                 response=dict(choices=[dict(message=dict(content="{}"))])))
    preparation = dict(synthetic_only=True, transport="authenticated_ASGI", database_mode="PostgreSQL",
                       database="independent-partial-contract", chat_auth_statuses=[200,200],
                       seed_template_verified=True, seed_scope=prior["seed_scope"],
                       generation=[dict(model_invoked=True, observed_primary_calls=1) for _ in range(2)],
                       prepared=[{}, {}, dict(received_at=clock)])
    dump = b"Opaque independent partial contract bytes, never a real PostgreSQL backup"
    (seed / "before-question.dump").write_bytes(dump)
    rows = copy.deepcopy(prior["seed_records"])
    backup = dict(bytes=len(dump), database_sha256=hashlib.sha256(dump).hexdigest(),
                  prior_same_task_answers=0, successful_seed_claim_ids=[r["id"] for r in rows])
    snapshot = dict(transport="actual_readonly_PostgreSQL_then_pg_dump", database=preparation["database"],
                    seed_scope=prior["seed_scope"], records=rows, sources=sources, source_fence="",
                    backup=backup, successful_history_receipts=turns, failed_native_receipt=progress[-1],
                    failed_native_message_rows=[], failed_history_kept_without_success_override=True,
                    unexecuted_final_query_absent=True, failed_task_no_paid_main=True,
                    successful_native_message_rows=messages)
    ref = dict(phase="stage105", variant=root.name,
               prior_seed=dict(phase=prior_phase.name, variant=prior_root.name,
                               proof_sha256=hashlib.sha256(prior_raw).hexdigest()),
               snapshot=dict(phase="stage105", variant=seed.name))
    proof = dict(seed_scope=prior["seed_scope"], seed_records=rows, durable_seed_sources=sources,
                 before_question_backup=backup, history_advancement_turns=turns,
                 history_advancement_turns_completed=len(turns), native_partial_history_seed_origin=ref)
    artifacts = {"preparation-observed.json":preparation, "history-advancement-progress.json":progress,
                 "cloud-calls.json":calls, "writer-admission.json":[],
                 "evaluation-budget-stop.json":dict(next_send_blocked=True,actual_calls=len(calls)),
                 "partial-snapshot.json":snapshot, "case.json":dict(history_advancement_tasks=tasks)}
    persist(root, seed, artifacts, proof)
    assert expected_seed_record_count(proof, phase) == 6
    return phase, root, seed, artifacts, proof


def persist(root, seed, data, proof):
    ref = proof["native_partial_history_seed_origin"]
    for name, key in [("preparation-observed.json","preparation_sha256"),
                      ("history-advancement-progress.json","progress_sha256"),
                      ("cloud-calls.json","cloud_calls_sha256"),
                      ("writer-admission.json","admission_sha256"),
                      ("evaluation-budget-stop.json","budget_stop_sha256"),
                      ("partial-snapshot.json","snapshot_sha256"), ("case.json","case_sha256")]:
        raw = json.dumps(data[name]).encode()
        snapshot_file = name in {"partial-snapshot.json","case.json"}
        ((seed if snapshot_file else root)/name).write_bytes(raw)
        (ref["snapshot"] if snapshot_file else ref)[key] = hashlib.sha256(raw).hexdigest()


def test_real_success_prefix_and_pending_failure_are_distinct(tmp_path):
    phase, _, _, data, proof = contract(tmp_path)
    assert expected_seed_record_count(proof, phase) == 6
    assert data["history-advancement-progress.json"][-1]["status"] == 500
    assert data["partial-snapshot.json"]["sources"][-1]["body"] is None


@pytest.mark.parametrize("change", ["record", "old_source", "reply", "pending_body", "pending_clock",
                                    "message_scope", "source_id", "failed_success", "paid_final",
                                    "accepted_write", "answer_backup", "dump", "inherited"])
def test_partial_prefix_rejects_fabricated_success_or_changed_durable_evidence(tmp_path, change):
    phase, root, seed, data, proof = contract(tmp_path)
    snap = data["partial-snapshot.json"]
    if change == "record":
        snap["records"][0]["content"] = "substituted"
    elif change == "old_source":
        snap["sources"][0]["body"] = "substituted"
    elif change == "reply":
        snap["successful_native_message_rows"][0]["reply"] = "substituted"
    elif change == "pending_body":
        snap["sources"][-1]["body"] = "manufactured successful source"
    elif change == "pending_clock":
        snap["sources"][-1]["observed_at"] = "2028-01-01"
    elif change == "message_scope":
        snap["successful_native_message_rows"][0]["senderId"] = "1"
    elif change == "source_id":
        snap["successful_native_message_rows"][0]["sourceMessageId"] = "invented"
    elif change == "failed_success":
        data["history-advancement-progress.json"][-1]["status"] = 200
    elif change == "paid_final":
        data["cloud-calls.json"].append(copy.deepcopy(data["cloud-calls.json"][0]))
        data["evaluation-budget-stop.json"]["actual_calls"] += 1
    elif change == "accepted_write":
        data["writer-admission.json"].append(dict(accepted=[dict(operation="ADD")]))
    elif change == "answer_backup":
        proof["before_question_backup"]["prior_same_task_answers"] = 1
    elif change == "dump":
        (seed / "before-question.dump").write_bytes(b"substituted")
    elif change == "inherited":
        data["history-advancement-progress.json"][0]["task_id"] = "substituted"
    persist(root, seed, data, proof)
    with pytest.raises(ValueError):
        expected_seed_record_count(proof, phase)
