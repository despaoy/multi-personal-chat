"Independent contract fixtures; these are never model requests or native seed evidence."

import copy
import hashlib
import json

import pytest

from evaluation.native_seed_provenance import expected_seed_record_count
from tests.test_native_successor_snapshot import contract as deferred_contract


def contract(tmp_path):
    previous_phase, _, prior_root, _, _, _, prior = deferred_contract(tmp_path)
    prior_raw = json.dumps(prior).encode()
    (prior_root / "before-question.json").write_bytes(prior_raw)
    phase = tmp_path / "stage105"
    root = phase / "native-pg-current"
    root.mkdir(parents=True)
    old = copy.deepcopy(prior["seed_records"])
    previous = next(r for r in old if r["id"] == 5)
    source = "我现在不喜欢紫茶。请将我以前喜欢紫茶的偏好替换为这条，不改变其他偏好。"
    successor = dict(previous, id=7, content="我不喜欢紫茶。", status="active",
                     relation_type="SUPERSEDE", supersedes_memory_id=5,
                     source_message_ids=["independent-current"], evidence=["我现在不喜欢紫茶。"],
                     observed_at="2026-10-04T04:00:00+00:00", valid_from=None, valid_to=None,
                     attributed_to="user", qualifiers={}, metadata={})
    rows = [successor, *copy.deepcopy(old)]
    next(r for r in rows if r["id"] == 5).update(status="superseded", valid_to=successor["observed_at"])
    proposal = dict(operation="SUPERSEDE", target_memory_id="5", target_memory_key="slot1",
                    memory=dict(content=successor["content"], memory_key="slot1"),
                    evidence=successor["evidence"][0], attributed_to="user", qualifiers={})
    origin = dict(synthetic_only=True, transport="authenticated_ASGI", database_mode="PostgreSQL",
                  database="independent-current-contract", http_status=200,
                  chat_auth_statuses=[200, 200], seed_scope=prior["seed_scope"],
                  primary_calls=1, cloud_calls=1, seed_records=old, user_fact_records_after=rows,
                  before_question_backup=prior["before_question_backup"],
                  durable_seed_sources=prior["durable_seed_sources"], author_only=True,
                  seed_template_verified=True, additional_source_status=200,
                  additional_source_response=dict(abstained=False),
                  generation=[dict(model_invoked=True, guard_retried=False)],
                  scheduler_final=dict(saved=1, failed=0, erased=0,
                                       recent_results=[dict(source_message_id="independent-current")]),
                  writer_admission=[dict(raw_response="independent-current-response",
                                         input=dict(source_message=source), accepted=[proposal])])
    calls = [dict(http_status=200, request=dict(model="deepseek-flash"),
                  response=dict(choices=[dict(message=dict(content="independent-current-response"))]))]
    dump = b"Independent opaque bytes, never an actual PostgreSQL backup"
    (root / "before-question.dump").write_bytes(dump)
    backup = dict(database_sha256=hashlib.sha256(dump).hexdigest(), bytes=len(dump),
                  prior_same_task_answers=0, successful_seed_claim_ids=[r["id"] for r in rows])
    sources = [*copy.deepcopy(prior["durable_seed_sources"]),
               dict(source_message_id="independent-current", body=source, state="recorded",
                    observed_at=successor["observed_at"])]
    snap = dict(transport="actual_readonly_PostgreSQL_then_pg_dump", database=origin["database"],
                seed_scope=prior["seed_scope"], records=rows, sources=sources, backup=backup,
                source_fence="", prior_source_fence="")
    provenance = dict(phase="stage105", variant="native-pg-current", transition="current_supersede",
                      prior_seed=dict(phase=previous_phase.name, variant=prior_root.name,
                                      proof_sha256=hashlib.sha256(prior_raw).hexdigest()),
                      post_update_snapshot=dict(phase="stage105", variant="native-pg-current"))
    proof = dict(seed_scope=prior["seed_scope"], seed_records=rows, durable_seed_sources=sources,
                 before_question_backup=backup, native_update_seed_origin=provenance)
    save(root, origin, calls, snap, proof)
    return phase, root, origin, calls, snap, proof


def save(root, origin, calls, snap, proof):
    provenance = proof["native_update_seed_origin"]
    for name, data, ref, key in [
        ("result.json", origin, provenance, "result_sha256"),
        ("cloud-calls.json", calls, provenance, "cloud_calls_sha256"),
        ("post-update-snapshot.json", snap, provenance["post_update_snapshot"], "snapshot_sha256"),
    ]:
        raw = json.dumps(data).encode()
        (root / name).write_bytes(raw)
        ref[key] = hashlib.sha256(raw).hexdigest()


def test_recursive_six_record_current_supersession_contract(tmp_path):
    phase, _, _, _, _, proof = contract(tmp_path)
    assert expected_seed_record_count(proof, phase) == 7


@pytest.mark.parametrize("change", ["unrelated", "target_content", "scope", "old_active",
                                    "future_deferred", "provider", "source", "clock", "dump",
                                    "fence", "prior", "duplicate", "extra", "not_author_only"])
def test_current_update_requires_exact_real_delta_and_bound_artifacts(tmp_path, change):
    phase, root, origin, calls, snap, proof = contract(tmp_path)
    new = origin["user_fact_records_after"][0]
    if change == "unrelated":
        origin["user_fact_records_after"][1]["content"] = "unrelated rewritten"
    elif change == "target_content":
        next(r for r in origin["user_fact_records_after"] if r["id"] == 5)["content"] = "rewritten old content"
    elif change == "scope":
        new["conversation_id"] = "other"
    elif change == "old_active":
        next(r for r in origin["user_fact_records_after"] if r["id"] == 5)["status"] = "active"
    elif change == "future_deferred":
        new["metadata"] = dict(deferred_mutation={})
    elif change == "provider":
        origin["writer_admission"][0]["raw_response"] = "forced proposal"
    elif change == "source":
        snap["sources"][-1]["body"] = "substituted original"
    elif change == "clock":
        snap["sources"][-1]["observed_at"] = "2028-01-01"
    elif change == "dump":
        (root / "before-question.dump").write_bytes(b"substituted snapshot")
    elif change == "fence":
        snap["source_fence"] = "erased"
    elif change == "prior":
        proof["native_update_seed_origin"]["prior_seed"]["proof_sha256"] = "0" * 64
    elif change == "duplicate":
        new["id"] = origin["seed_records"][0]["id"]
    elif change == "extra":
        origin["writer_admission"].append(copy.deepcopy(origin["writer_admission"][0]))
    elif change == "not_author_only":
        origin["author_only"] = False
    save(root, origin, calls, snap, proof)
    with pytest.raises(ValueError):
        expected_seed_record_count(proof, phase)


def test_twenty_completed_history_receipts_survive_authoring_without_replay():
    from evaluation.native_current_snapshot import inherit_history_receipts

    turns = [dict(index=i, status=200, response=dict(abstained=False)) for i in range(20)]
    prior = dict(history_advancement_turns=turns, history_advancement_turns_completed=20)
    proof = dict(seed_records=[dict(id=7)])
    inherit_history_receipts(proof, prior)
    assert proof["history_advancement_turns"] == turns
    assert proof["history_advancement_turns"] is not turns
    assert proof["seed_records"] == [dict(id=7)]
    assert list(range(28))[len(proof["history_advancement_turns"]):] == list(range(20, 28))


@pytest.mark.parametrize("change", ["count", "failed"])
def test_inherited_history_requires_complete_successful_receipts(change):
    from evaluation.native_current_snapshot import inherit_history_receipts

    prior = dict(history_advancement_turns=[dict(status=200, response=dict(abstained=False))],
                 history_advancement_turns_completed=1)
    if change == "count":
        prior["history_advancement_turns_completed"] = 2
    else:
        prior["history_advancement_turns"][0]["status"] = 500
    with pytest.raises(AssertionError):
        inherit_history_receipts({}, prior)


@pytest.mark.parametrize("change", ["missing", "substituted"])
def test_current_snapshot_rejects_lost_history_provenance(tmp_path, change):
    phase, root, origin, calls, snap, proof = contract(tmp_path)
    ref = proof["native_update_seed_origin"]["prior_seed"]
    path = tmp_path / ref["phase"] / ref["variant"] / "before-question.json"
    prior = json.loads(path.read_bytes())
    turns = [dict(index=i, status=200, response=dict(abstained=False)) for i in range(20)]
    prior.update(history_advancement_turns=turns, history_advancement_turns_completed=20)
    raw = json.dumps(prior).encode()
    path.write_bytes(raw)
    ref["proof_sha256"] = hashlib.sha256(raw).hexdigest()
    if change == "substituted":
        proof.update(history_advancement_turns=turns[:-1], history_advancement_turns_completed=19)
    save(root, origin, calls, snap, proof)
    with pytest.raises(ValueError):
        expected_seed_record_count(proof, phase)
    proof.update(history_advancement_turns=turns, history_advancement_turns_completed=20)
    assert expected_seed_record_count(proof, phase) == 7
