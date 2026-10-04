"""Independent fictional protocol contracts, never native seeds or provider evidence."""

import copy
import hashlib
import json
from dataclasses import asdict

import pytest
from evaluation.mixed_subject_history_probe import validate_native_messages
from evaluation.native_seed_provenance import expected_seed_record_count
from pydantic import ValidationError

from character.memory_llm import parse_llm_proposals


def fixture(tmp_path):
    phase = tmp_path / "stage2"
    phase.mkdir()
    root = tmp_path / "stage1" / "native-pg-addition"
    prior_root = tmp_path / "stage1" / "native-pg-seed"
    root.mkdir(parents=True)
    prior_root.mkdir()
    fields = dict(
        character_id="contract-role",
        platform="web",
        adapter="web-character",
        sender_id="42",
        conversation_type="private",
        conversation_id="42",
        scope_level="conversation",
    )
    old = [dict(id=i, **fields, content="independent old value", status="active") for i in range(1, 4)]
    scope = dict(owner="42", conversation="42", character="contract-role")
    source = "我喜欢花栗茶。条件：复印完成；场景：雨后；频率：每季三次；程度：稍微喜欢；确定性：已经确定；例外：刮风不适用；地点：丙台；时间：每年夏季。"
    quals = dict(
        condition="复印完成",
        context="雨后",
        frequency="每季三次",
        preference_strength="稍微喜欢",
        certainty="已经确定",
        exception="刮风不适用",
        location="丙台",
        time="每年夏季",
    )
    output = json.dumps(
        dict(
            memories=[
                dict(kind="like", value="花栗茶", evidence=source, confidence=0.98, operation="ADD", qualifiers=quals)
            ]
        ),
        ensure_ascii=False,
    )
    kwargs = dict(
        source_message=source,
        history=[],
        existing_memories=[],
        confidence_threshold=0.85,
        feedback_target_ids=[],
        source_type="user",
    )
    (proposal,) = [json.loads(json.dumps(asdict(p))) for p in parse_llm_proposals(output, **kwargs)]
    added = dict(
        id=4,
        **fields,
        content=proposal["memory"]["content"],
        memory_key=proposal["memory"]["memory_key"],
        memory_type=proposal["memory"]["memory_type"],
        relation_type="ADD",
        status="active",
        parent_memory_id=None,
        supersedes_memory_id=None,
        evidence=[proposal["evidence"]],
        qualifiers=quals,
        metadata=dict(qualifiers=quals),
        attributed_to="user",
        valid_from=None,
        valid_to=None,
        observed_at="2030-01-01T01:00:00+00:00",
        source_message_ids=["new-contract-source"],
    )
    new = [*old, added]
    sources = [dict(source_message_id="old-contract-source", body="complete independent old source", state="recorded")]
    new_source = dict(
        source_message_id="new-contract-source",
        body=source,
        state="recorded",
        observed_at=added["observed_at"],
        owner_key=json.dumps((fields["platform"], fields["adapter"], fields["sender_id"])),
        scope_key=json.dumps(
            tuple(
                fields[k]
                for k in ("character_id", "platform", "adapter", "sender_id", "conversation_type", "conversation_id")
            )
        ),
    )
    dump = b"independent fictional snapshot; no actual PostgreSQL rows" * 30
    (root / "before-question.dump").write_bytes(dump)
    backup = dict(
        database_sha256=hashlib.sha256(dump).hexdigest(),
        bytes=len(dump),
        prior_same_task_answers=0,
        successful_seed_claim_ids=[r["id"] for r in new],
    )
    prior = dict(
        seed_scope=scope,
        seed_records=old,
        durable_seed_sources=sources,
        history_advancement_turns=[],
        history_advancement_turns_completed=0,
    )
    status = dict(
        saved=1,
        failed=0,
        erased=0,
        recent_results=[dict(source_message_id=new_source["source_message_id"], persisted=1)],
    )
    entry = dict(raw_response=output, input=kwargs, accepted=[proposal])
    snap = dict(
        seed_scope=scope,
        seed_records=new,
        durable_seed_sources=[*sources, new_source],
        durable_seed_verified_before_question=True,
        calls_before_question=2,
        generation=[dict(model_invoked=True, guard_fallback=False, observed_primary_calls=1)],
        scheduler_before_question=status,
        writer_admission_before_question=[entry],
        history_advancement_turns=[],
        history_advancement_turns_completed=0,
        before_question_backup=backup,
    )
    origin = dict(
        synthetic_only=True,
        transport="authenticated_ASGI",
        database_mode="PostgreSQL",
        http_status=200,
        chat_auth_statuses=[200, 200],
        seed_template_verified=True,
        additional_source_status=200,
        additional_source_response=dict(abstained=False),
        additional_source_written_by_real_native_turn=True,
        seed_scope=scope,
        seed_records=new,
        user_fact_records_after=new,
        cloud_calls=2,
        scheduler_final=status,
        durable_seed_sources=snap["durable_seed_sources"],
        history_advancement_turns=[],
        history_advancement_turns_completed=0,
        before_question_backup=backup,
    )
    calls = [
        dict(
            http_status=200,
            request=dict(
                model="deepseek-flash",
                max_tokens=2048,
                messages=[dict(role="user", content="<user_query>" + source + "</user_query>")],
            ),
            response=dict(choices=[dict(message=dict(content="contract acknowledgment"))]),
        ),
        dict(
            http_status=200,
            request=dict(
                model="deepseek-flash",
                max_tokens=768,
                messages=[dict(role="user", content=json.dumps(dict(current_user_message=source)))],
            ),
            response=dict(choices=[dict(message=dict(content=output))]),
        ),
    ]
    proof = {**copy.deepcopy(snap)}

    def persist():
        def save(path, obj):
            raw = json.dumps(obj, ensure_ascii=False).encode()
            path.write_bytes(raw)
            return hashlib.sha256(raw).hexdigest()

        proof["native_additive_seed_origin"] = dict(
            phase="stage1",
            variant="native-pg-addition",
            result_sha256=save(root / "result.json", origin),
            cloud_calls_sha256=save(root / "cloud-calls.json", calls),
            snapshot_sha256=save(root / "before-question.json", snap),
            prior_seed=dict(
                phase="stage1", variant="native-pg-seed", proof_sha256=save(prior_root / "before-question.json", prior)
            ),
        )

    persist()
    return phase, proof, snap, origin, calls, persist


def test_complete_addition_derives_count_from_actual_delta(tmp_path):
    phase, proof, *_ = fixture(tmp_path)
    assert expected_seed_record_count(proof, phase) == 4


@pytest.mark.parametrize(
    "change",
    [
        "scope",
        "old_row",
        "qualifier",
        "source_scope",
        "source_body",
        "source_state",
        "source_clock",
        "writer_output",
        "writer_input",
        "wrong_prefix",
        "failed_auth",
        "failed_native",
        "no_native_writer",
        "failed_storage",
        "snapshot_records",
        "dump_hash",
        "count_only",
        "history_override",
        "duplicate_row",
        "cycle",
    ],
)
def test_hashes_and_counts_cannot_replace_actual_addition_bindings(tmp_path, change):
    phase, proof, snap, origin, calls, persist = fixture(tmp_path)
    assert expected_seed_record_count(proof, phase) == 4
    if change == "scope":
        proof["seed_scope"] = dict(owner="99")
    elif change == "old_row":
        snap["seed_records"][0]["content"] = "changed old value"
    elif change == "qualifier":
        snap["writer_admission_before_question"][0]["accepted"][0]["qualifiers"].pop()
    elif change == "source_scope":
        snap["durable_seed_sources"][-1]["scope_key"] = "foreign scope"
    elif change == "source_body":
        snap["durable_seed_sources"][-1]["body"] = "different statement"
    elif change == "source_state":
        snap["durable_seed_sources"][-1]["state"] = "revoked"
    elif change == "source_clock":
        snap["durable_seed_sources"][-1]["observed_at"] = "other time"
    elif change == "writer_output":
        calls[-1]["response"]["choices"][0]["message"]["content"] = '{"memories":[]}'
    elif change == "writer_input":
        calls[-1]["request"]["messages"][-1]["content"] = json.dumps(dict(current_user_message="different source"))
    elif change == "wrong_prefix":
        snap["calls_before_question"] = 1
    elif change == "failed_auth":
        origin["chat_auth_statuses"] = [401, 200]
    elif change == "failed_native":
        origin["additional_source_status"] = 500
    elif change == "no_native_writer":
        origin["additional_source_written_by_real_native_turn"] = False
    elif change == "failed_storage":
        snap["scheduler_before_question"]["saved"] = 0
    elif change == "snapshot_records":
        proof["seed_records"] = proof["seed_records"][:-1]
    elif change == "dump_hash":
        snap["before_question_backup"]["database_sha256"] = "invalid"
    elif change == "count_only":
        proof["expected_seed_count"] = 8
        proof["seed_records"] += [copy.deepcopy(proof["seed_records"][-1])]
    elif change == "history_override":
        proof["history_advancement_turns_completed"] = 41
    elif change == "duplicate_row":
        snap["seed_records"][-1]["id"] = snap["seed_records"][0]["id"]
    elif change == "cycle":
        proof["native_history_seed_origin"] = dict(phase="stage1", variant="native-pg-addition")
    persist()
    with pytest.raises(ValueError, match="additive"):
        expected_seed_record_count(proof, phase)


def test_history_only_validates_remaining_real_requests_not_unsent_question():
    fixture = dict(
        question="x" * 8001,
        history_advancement_tasks=[dict(message="旧任务"), dict(message="完整的新合成数据：甲=2，乙=3，请计算和。")],
    )
    validate_native_messages(fixture, advance_history=True, completed_history=1, history_only=True)
    with pytest.raises(ValidationError):
        validate_native_messages(fixture, advance_history=True, completed_history=1)


def test_history_only_does_not_waive_an_invalid_remaining_task():
    fixture = dict(
        question="有效的未来问题。",
        history_advancement_tasks=[dict(message="完整输入"), dict(message="完整新任务数据：甲=2，乙=3，请计算和。")],
    )
    validate_native_messages(fixture, advance_history=True, completed_history=1, history_only=True)
    fixture["history_advancement_tasks"][1]["message"] = "x" * 8001
    with pytest.raises(ValidationError):
        validate_native_messages(fixture, advance_history=True, completed_history=1, history_only=True)
