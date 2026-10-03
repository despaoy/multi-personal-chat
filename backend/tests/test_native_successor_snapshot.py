"""Independent contract artifacts only; never used as native seeds/provider evidence."""

import copy
import hashlib
import json

import pytest

from evaluation.native_seed_provenance import expected_seed_record_count
from tests.test_native_seed_provenance import fixture as original_contract
from tests.test_native_seed_provenance import write


def contract(tmp_path):
    old_phase, _, _, _, prior = original_contract(tmp_path)
    phase = tmp_path / "stage104"
    phase.mkdir()
    parent_root = tmp_path / "stage102" / "native-pg-source-seed"
    parent_root.mkdir(parents=True)
    old = copy.deepcopy(prior["seed_records"])
    # The original SUPERSEDE contract helper omitted optional typed fields.
    # Enrich its completed receipt consistently, never a real/native seed.
    fields = dict(
        character_id="purple",
        platform="web",
        adapter="contract",
        conversation_type="private",
        conversation_id="6",
        scope_level="conversation",
    )
    origin97 = json.loads((old_phase / "native-pg-update" / "result.json").read_text())
    for row in [*origin97["seed_records"], *origin97["user_fact_records_after"]]:
        row.update(fields)
    old = origin97["user_fact_records_after"]
    old[-1].update(memory_key="slot1")
    prior["seed_records"] = old
    write(
        old_phase / "native-pg-update",
        origin97,
        json.loads((old_phase / "native-pg-update" / "cloud-calls.json").read_text()),
        prior,
    )
    prior.update(
        durable_seed_sources=[],
        before_question_backup=dict(
            database_sha256="original",
            bytes=9000,
            prior_same_task_answers=0,
            successful_seed_claim_ids=[x["id"] for x in old],
        ),
    )
    raw_prior = json.dumps(prior).encode()
    (parent_root / "before-question.json").write_bytes(raw_prior)
    source = "我从2027年1月8日起只在完整登记页已提交且紫色签章校验通过时选择紫茶。"
    evidence = source
    clock = "2026-10-04T04:00:00+00:00"
    qualifiers = dict(condition="完整登记页已提交且紫色签章校验通过", time="2027年1月8日起")
    new = dict(
        id=6,
        memory_key="slot1",
        status="active",
        relation_type="COEXIST",
        parent_memory_id=5,
        supersedes_memory_id=None,
        sender_id="6",
        content="完整的独立紫茶条件",
        source_message_ids=["new-native-contract"],
        evidence=[evidence],
        observed_at=clock,
        valid_from=None,
        valid_to=None,
        attributed_to="user",
        qualifiers=qualifiers,
        **fields,
    )
    new["metadata"] = dict(
        qualifiers=qualifiers,
        temporal_provenance=dict(validity_authority="unverified"),
        deferred_mutation=dict(
            original_operation="SUPERSEDE",
            expression="2027年1月8日",
            source_observed_at=clock,
            earliest_possible_start="2027-01-08T00:00:00+08:00",
            not_a_scheduled_replacement=True,
        ),
    )
    rows = [new, *old]
    accepted = dict(
        operation="SUPERSEDE",
        target_memory_id="5",
        target_memory_key="slot1",
        memory=dict(content=new["content"], memory_key="slot1"),
        evidence=evidence,
        qualifiers=qualifiers,
        attributed_to="user",
    )
    origin = dict(
        synthetic_only=True,
        transport="authenticated_ASGI",
        database_mode="PostgreSQL",
        database="independent-contract",
        http_status=200,
        chat_auth_statuses=[200, 200],
        seed_scope=prior["seed_scope"],
        primary_calls=1,
        cloud_calls=1,
        seed_records=old,
        user_fact_records_after=rows,
        before_question_backup=prior["before_question_backup"],
        durable_seed_sources=[],
        scheduler_final=dict(
            saved=1, failed=0, erased=0, recent_results=[dict(source_message_id="new-native-contract")]
        ),
        writer_admission=[
            dict(
                raw_response="unchanged contract writer response",
                input=dict(source_message=source),
                accepted=[accepted],
            )
        ],
    )
    cloud = [
        dict(
            http_status=200,
            request=dict(model="deepseek-flash"),
            response=dict(choices=[dict(message=dict(content="unchanged contract writer response"))]),
        )
    ]
    origin_root = tmp_path / "stage102" / "native-pg-index"
    origin_root.mkdir()
    seed_root = phase / "native-pg-successor-seed"
    seed_root.mkdir()
    dump = b"Independent opaque snapshot bytes, not PostgreSQL native evidence"
    (seed_root / "before-question.dump").write_bytes(dump)
    backup = dict(
        database_sha256=hashlib.sha256(dump).hexdigest(),
        bytes=len(dump),
        prior_same_task_answers=0,
        successful_seed_claim_ids=[x["id"] for x in rows],
    )
    sources = [dict(source_message_id="new-native-contract", state="recorded", body=source, observed_at=clock)]
    snap = dict(
        transport="actual_readonly_PostgreSQL_then_pg_dump",
        database=origin["database"],
        seed_scope=prior["seed_scope"],
        records=rows,
        sources=sources,
        source_fence="",
        prior_source_fence="",
        backup=backup,
    )
    provenance = dict(
        phase="stage102",
        variant="native-pg-index",
        transition="deferred_coexist",
        prior_seed=dict(
            phase="stage102", variant="native-pg-source-seed", proof_sha256=hashlib.sha256(raw_prior).hexdigest()
        ),
        post_update_snapshot=dict(phase="stage104", variant="native-pg-successor-seed"),
    )
    proof = dict(
        seed_scope=prior["seed_scope"],
        seed_records=rows,
        durable_seed_sources=sources,
        before_question_backup=backup,
        native_update_seed_origin=provenance,
    )
    save(origin_root, seed_root, origin, cloud, snap, proof)
    return phase, origin_root, seed_root, origin, cloud, snap, proof


def save(origin_root, seed_root, origin, cloud, snap, proof):
    for path, data, ref, key in [
        (origin_root / "result.json", origin, proof["native_update_seed_origin"], "result_sha256"),
        (origin_root / "cloud-calls.json", cloud, proof["native_update_seed_origin"], "cloud_calls_sha256"),
        (
            seed_root / "post-update-snapshot.json",
            snap,
            proof["native_update_seed_origin"]["post_update_snapshot"],
            "snapshot_sha256",
        ),
    ]:
        raw = json.dumps(data).encode()
        path.write_bytes(raw)
        ref[key] = hashlib.sha256(raw).hexdigest()


def test_complete_authenticated_deferred_successor_contract(tmp_path):
    phase, _, _, _, _, _, proof = contract(tmp_path)
    assert expected_seed_record_count(proof, phase) == 6


@pytest.mark.parametrize(
    "change",
    [
        "prior_hash",
        "prior_snapshot",
        "dump",
        "future_prior",
        "snapshot_path",
        "source_body",
        "source_clock",
        "source_state",
        "source_id",
        "fence",
        "old_rewrite",
        "relation",
        "scope",
        "expiry",
        "schedule",
        "model_authority",
        "future_clock",
        "expression",
        "official_response",
        "content",
        "qualifier",
        "operation",
        "target",
        "claim_id",
        "admission_count",
    ],
)
def test_hashes_and_count_alone_never_authorize_successor(tmp_path, change):
    phase, root, seed, origin, cloud, snap, proof = contract(tmp_path)
    provenance = proof["native_update_seed_origin"]
    new = origin["user_fact_records_after"][0]
    ad = origin["writer_admission"][0]
    if change == "prior_hash":
        provenance["prior_seed"]["proof_sha256"] = "0" * 64
    elif change == "prior_snapshot":
        origin["before_question_backup"]["database_sha256"] = "substituted"
    elif change == "dump":
        (seed / "before-question.dump").write_bytes(b"substituted")
    elif change == "future_prior":
        provenance["prior_seed"]["phase"] = "stage105"
    elif change == "snapshot_path":
        provenance["post_update_snapshot"]["variant"] = "../outside"
    elif change == "source_body":
        snap["sources"][0]["body"] = "我喜欢紫茶。"
    elif change == "source_clock":
        snap["sources"][0]["observed_at"] = "2026-10-05T04:00:00+00:00"
    elif change == "source_state":
        snap["sources"][0]["state"] = "revoked"
    elif change == "source_id":
        new["source_message_ids"] = ["invented"]
    elif change == "fence":
        snap["source_fence"] = "advanced"
    elif change == "old_rewrite":
        origin["user_fact_records_after"][1]["content"] = "rewritten"
    elif change == "relation":
        new["relation_type"] = "SUPERSEDE"
    elif change == "scope":
        new["conversation_id"] = "other"
    elif change == "expiry":
        new["valid_from"] = "2027-01-08"
    elif change == "schedule":
        new["metadata"]["deferred_mutation"]["not_a_scheduled_replacement"] = False
    elif change == "model_authority":
        new["metadata"]["temporal_provenance"]["validity_authority"] = "model"
    elif change == "future_clock":
        new["metadata"]["deferred_mutation"]["earliest_possible_start"] = "2026-10-03T00:00:00+00:00"
    elif change == "expression":
        new["metadata"]["deferred_mutation"]["expression"] = "2027年2月8日"
    elif change == "official_response":
        ad["raw_response"] = "forced"
    elif change == "content":
        new["content"] = "substituted condition"
    elif change == "qualifier":
        new["qualifiers"] = {}
    elif change == "operation":
        ad["accepted"][0]["operation"] = "ADD"
    elif change == "target":
        ad["accepted"][0]["target_memory_id"] = "4"
    elif change == "claim_id":
        new["id"] = True
    elif change == "admission_count":
        origin["writer_admission"].append(copy.deepcopy(ad))
    save(root, seed, origin, cloud, snap, proof)
    with pytest.raises(ValueError):
        expected_seed_record_count(proof, phase)


@pytest.mark.parametrize("change", ["wrapped", "duplicate", "quoted", "not_literal"])
def test_normalized_full_clause_remains_bound_to_single_unquoted_source(tmp_path, change):
    phase, root, seed, origin, cloud, snap, proof = contract(tmp_path)
    source = "从2027年1月8日起 我\n只在完整登记页已提交且紫色签章校验通过时选择紫茶。"
    evidence = "".join(source.split())
    if change == "duplicate":
        source = source + "\n" + source
    elif change == "quoted":
        source = "甲方说：“" + source + "”"
    elif change == "not_literal":
        source = source.replace("紫色签章", "黄色签章")
    origin["writer_admission"][0]["input"]["source_message"] = source
    origin["writer_admission"][0]["accepted"][0]["evidence"] = evidence
    origin["user_fact_records_after"][0]["evidence"] = [evidence]
    snap["sources"][0]["body"] = source
    save(root, seed, origin, cloud, snap, proof)
    if change == "wrapped":
        assert expected_seed_record_count(proof, phase) == 6
    else:
        with pytest.raises(ValueError):
            expected_seed_record_count(proof, phase)


def test_persisted_model_bound_and_pair_qualifiers_remain_proposals(tmp_path):
    phase, root, seed, origin, cloud, snap, proof = contract(tmp_path)
    proposal = origin["writer_admission"][0]["accepted"][0]
    new = origin["user_fact_records_after"][0]
    proposal["valid_from"] = new["valid_from"] = "2027-01-07T16:00:00+00:00"
    proposal["qualifiers"] = list(proposal["qualifiers"].items())
    save(root, seed, origin, cloud, snap, proof)
    assert expected_seed_record_count(proof, phase) == 6
    assert new["metadata"]["temporal_provenance"]["validity_authority"] == "unverified"
    assert new["metadata"]["deferred_mutation"]["not_a_scheduled_replacement"] is True
