"""Contract fixtures below are not native evaluation seeds or provider evidence."""

import copy
import hashlib
import json

import pytest
from evaluation.native_seed_provenance import expected_seed_record_count


def fixture(tmp_path):
    phase = tmp_path / "stage97"
    root = phase / "native-pg-update"
    root.mkdir(parents=True)
    scope = dict(owner="6")
    old = [
        dict(
            id=i,
            memory_key=f"slot{i}",
            status="active",
            content="contract value",
            evidence=["contract source"],
            source_message_ids=["old"],
            metadata={},
            sender_id="6",
        )
        for i in range(1, 5)
    ]
    new = copy.deepcopy(old)
    new[0]["status"] = "superseded"
    new.append(
        dict(
            id=5,
            memory_key="slot1",
            status="active",
            relation_type="SUPERSEDE",
            supersedes_memory_id=1,
            sender_id="6",
            source_message_ids=["new"],
            evidence=["contract source"],
        )
    )
    origin = dict(
        synthetic_only=True,
        transport="authenticated_ASGI",
        database_mode="PostgreSQL",
        http_status=200,
        chat_auth_statuses=[200, 200],
        seed_scope=scope,
        primary_calls=1,
        cloud_calls=1,
        seed_records=old,
        user_fact_records_after=new,
        scheduler_final=dict(saved=1, failed=0, erased=0, recent_results=[dict(source_message_id="new")]),
        writer_admission=[
            dict(
                raw_response="writer contract",
                input=dict(source_message="contract source"),
                accepted=[
                    dict(
                        operation="SUPERSEDE",
                        target_memory_id="1",
                        target_memory_key="slot1",
                        evidence="contract source",
                    )
                ],
            )
        ],
    )
    cloud = [
        dict(
            http_status=200,
            request=dict(model="deepseek-flash"),
            response=dict(choices=[dict(message=dict(content="writer contract"))]),
        )
    ]
    proof = dict(
        seed_scope=scope,
        seed_records=new,
        before_question_backup=dict(successful_seed_claim_ids=[r["id"] for r in new]),
        native_update_seed_origin=dict(phase="stage97", variant="native-pg-update"),
    )
    write(root, origin, cloud, proof)
    return phase, root, origin, cloud, proof


def write(root, origin, cloud, proof):
    for name, data, key in [
        ("result.json", origin, "result_sha256"),
        ("cloud-calls.json", cloud, "cloud_calls_sha256"),
    ]:
        raw = json.dumps(data).encode()
        (root / name).write_bytes(raw)
        proof["native_update_seed_origin"][key] = hashlib.sha256(raw).hexdigest()


def test_exact_completed_successor_artifact_can_authorize_five_record_seed(tmp_path):
    phase, _, _, _, proof = fixture(tmp_path)
    assert expected_seed_record_count(proof, phase) == 5


@pytest.mark.parametrize("authored,count", [(False, 3), (True, 4)])
def test_original_legacy_seed_limits_are_preserved_without_new_provenance(tmp_path, authored, count):
    assert expected_seed_record_count(dict(additional_source_written_by_real_native_turn=authored), tmp_path) == count


@pytest.mark.parametrize("key", ["result_sha256", "cloud_calls_sha256"])
def test_changed_native_receipt_is_not_a_count_override(tmp_path, key):
    phase, _, _, _, proof = fixture(tmp_path)
    proof["native_update_seed_origin"][key] = "0" * 64
    with pytest.raises(ValueError):
        expected_seed_record_count(proof, phase)


@pytest.mark.parametrize("change", ["records", "owner", "variant"])
def test_seed_does_not_depart_from_authored_records_owner_or_bounded_origin(tmp_path, change):
    phase, _, _, _, proof = fixture(tmp_path)
    if change == "records":
        proof["seed_records"] = copy.deepcopy(proof["seed_records"])
        proof["seed_records"][-1]["content"] = "unauthored"
    elif change == "owner":
        proof["seed_scope"] = dict(owner="other")
    else:
        proof["native_update_seed_origin"]["variant"] = "../outside"
    with pytest.raises(ValueError):
        expected_seed_record_count(proof, phase)


@pytest.mark.parametrize("change", ["operation", "old_active", "old_content", "new_scope", "source_id"])
def test_hashes_alone_do_not_prove_valid_semantic_transition(tmp_path, change):
    phase, root, origin, cloud, proof = fixture(tmp_path)
    if change == "operation":
        origin["user_fact_records_after"][-1]["relation_type"] = "ADD"
    elif change == "old_active":
        origin["user_fact_records_after"][0]["status"] = "active"
    elif change == "old_content":
        origin["user_fact_records_after"][0]["content"] = "rewritten history"
    elif change == "new_scope":
        origin["user_fact_records_after"][-1]["sender_id"] = "other"
    else:
        origin["user_fact_records_after"][-1]["source_message_ids"] = ["invented"]
    write(root, origin, cloud, proof)
    with pytest.raises(ValueError):
        expected_seed_record_count(proof, phase)
