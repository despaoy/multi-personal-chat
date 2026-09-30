import copy
import json

import pytest

from training.persona_judge import calibration_from_review, judge_candidate, validate_judgment
from training.persona_review import export_preferences, lock_decisions, make_packet, summarize_evaluation
from training.persona_reward import DIMENSIONS, VIOLATIONS
from training.persona_sampling import (
    digest,
    sample_candidates,
    sampling_contract,
    validate_candidate,
    validate_run,
    validate_scenes,
)


def scene(index=1, split="train"):
    return {
        "id": f"scene-{index}",
        "persona": "synthetic-test-character",
        "source_group": f"group-{index}",
        "source_ids": [f"source-{index}"],
        "split": split,
        "review_status": "approved",
        "persona_profile": "测试人物，说话简短，重视约定。",
        "relationship": "普通熟人",
        "situation": "借阅后归还物品。",
        "evidence": [{"id": "e1", "source_id": f"source-{index}", "content": f"第{index}次约定归还小说"}],
        "history": [{"role": "user", "content": "十分钟后还你。"}, {"role": "assistant", "content": "好。"}],
        "user_message": "我先走，可以吗？",
    }


def model(name="baseline"):
    return {"name": name, "model": f"local-{name}", "revision": "test-revision", "base_url": "http://127.0.0.1:9000"}


GENERATION = {"temperature": 0.7, "top_p": 0.9, "max_tokens": 128, "seed": 42}


def candidates(split="train", count=1, n=1):
    return sample_candidates(
        [scene(i, split) for i in range(count)],
        [model(), model("candidate")],
        GENERATION,
        samples_per_model=n,
        call=lambda m, messages, g: (
            ("记得归还小说。" if m["name"] == "candidate" else "记得带上笔记本。") + str(g["seed"])
        ),
    )


def completed(packet, key, winner="candidate"):
    decisions = {
        "packet_sha256": packet["packet_sha256"],
        "reviewer": "synthetic-human-fixture",
        "human_confirmed": True,
        "review_method": "human_only",
        "decisions": [],
    }
    for item in packet["items"]:
        side = next(side for side in ("A", "B") if key["rows"][item["id"]][side]["model"]["name"] == winner)
        decisions["decisions"].append(
            {
                "id": item["id"],
                "binding_sha256": item["binding_sha256"],
                "winner": side,
                "dimensions": dict.fromkeys(DIMENSIONS, side),
                "hard_errors": {"A": [], "B": []},
                "reason": "Synthetic test decision.",
            }
        )
    return decisions


def judge_response(**changes):
    result = {
        "abstain": False,
        "scores": dict.fromkeys(DIMENSIONS, 0.8),
        "violations": dict.fromkeys(VIOLATIONS, False),
        "reasons": dict.fromkeys(DIMENSIONS, "根据当前关系和约定"),
        "evidence_ids": ["e1"],
        "reason": "依据当前情境",
    }
    result.update(changes)
    return json.dumps(result, ensure_ascii=False)


def test_generation_records_exact_context_and_never_ranks_candidates():
    records = candidates(n=2)
    assert len(records) == 4
    for row in records:
        validate_candidate(row)
        assert "chosen" not in row and "human_final_approved" not in row
        assert "retrieved_evidence" in row["messages"][-1]["content"]
    assert {row["generation"]["seed"] for row in records} == {42, 43}


def test_sampling_failure_has_no_template_fallback_or_leaked_exception():
    def fail(*args):
        raise RuntimeError("secret-test-key")

    rows = sample_candidates([scene()], [model()], GENERATION, samples_per_model=1, call=fail)
    assert rows[0]["status"] == "error" and "response" not in rows[0]
    assert "secret-test-key" not in json.dumps(rows)
    with pytest.raises(ValueError, match="successful"):
        make_packet(rows, purpose="preferences")


@pytest.mark.parametrize("change", ["response", "messages", "generation"])
def test_candidate_tampering_is_rejected(change):
    row = candidates()[0]
    if change == "response":
        row[change] += "changed"
    elif change == "messages":
        row[change][-1]["content"] = "changed"
    else:
        row[change]["seed"] += 1
    with pytest.raises(ValueError, match="binding"):
        validate_candidate(row)


def test_run_manifest_detects_partial_journal():
    rows = candidates()
    contract = sampling_contract([scene(0)], [model(), model("candidate")], GENERATION, 1)
    validate_run(contract, rows)
    with pytest.raises(ValueError, match="incomplete"):
        validate_run(contract, rows[:-1])


def test_scene_partition_isolation_checks_evidence_and_sources():
    first, second = scene(1), scene(2, "validation")
    second["evidence"][0]["content"] = first["evidence"][0]["content"]
    with pytest.raises(ValueError, match="cross-split"):
        validate_scenes([first, second])


def test_blind_packet_hides_model_identity_and_requires_explicit_human_decision():
    packet, key, template = make_packet(candidates(), purpose="preferences")
    serialized = json.dumps(packet)
    assert "local-baseline" not in serialized and "local-candidate" not in serialized
    with pytest.raises(ValueError, match="reviewer"):
        lock_decisions(packet, template)
    template["reviewer"] = "fixture"
    with pytest.raises(ValueError, match="human-confirmed"):
        lock_decisions(packet, template)
    decision = completed(packet, key)
    del decision["decisions"][0]["dimensions"]["relationship"]
    with pytest.raises(ValueError, match="seven"):
        lock_decisions(packet, decision)


def test_reviewed_preference_export_roundtrips_real_loader_and_freezer(tmp_path):
    from training.evidence_dataset import load_preference_training_rows
    from training.preference_validation import group_split, validate_partitions

    packet, key, _ = make_packet(candidates(count=3), purpose="preferences")
    locked = lock_decisions(packet, completed(packet, key))
    result = export_preferences(packet, locked, key)
    assert len(result["pairs"]) == 3
    train, validation = group_split(result["pairs"])
    validate_partitions(train, validation)
    path = tmp_path / "validation.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in validation), encoding="utf-8")
    loaded = load_preference_training_rows(path, split="validation")
    assert loaded == validation
    loaded[0]["chosen"][0]["content"] = "changed after review"
    path.write_text(json.dumps(loaded[0]))
    with pytest.raises(ValueError, match="binding"):
        load_preference_training_rows(path, split="validation")


@pytest.mark.parametrize("split", ["validation", "test"])
def test_final_evaluation_scenes_cannot_become_training_pairs(split):
    with pytest.raises(ValueError, match="never validation/test"):
        make_packet(candidates(split), purpose="preferences")
    packet, key, _ = make_packet(candidates(split), purpose="evaluation", baseline="baseline", candidate="candidate")
    locked = lock_decisions(packet, completed(packet, key))
    with pytest.raises(ValueError, match="cannot become"):
        export_preferences(packet, locked, key)


def test_ties_invalid_and_winners_with_hard_errors_not_exported():
    packet, key, _ = make_packet(candidates(count=3), purpose="preferences")
    decision = completed(packet, key)
    decision["decisions"][0]["winner"] = "tie"
    decision["decisions"][1]["winner"] = "invalid"
    last = decision["decisions"][2]
    last["hard_errors"][last["winner"]] = ["fabricated_fact"]
    result = export_preferences(packet, lock_decisions(packet, decision), key)
    assert result["pairs"] == []
    assert sum(result["excluded"].values()) == 3


def test_blind_report_counts_actual_generations_and_dimension_votes():
    packet, key, _ = make_packet(
        candidates("validation", count=2), purpose="evaluation", baseline="baseline", candidate="candidate"
    )
    locked = lock_decisions(packet, completed(packet, key))
    result = summarize_evaluation(packet, locked, key)
    assert result["counts"] == {"win": 2}
    assert result["by_dimension"]["relationship"] == {"win": 2}
    assert result["win_share_ties_half"] == 1
    assert result["group_bootstrap_95ci"] == [1, 1]
    assert "not_release" in result["status"]
    tampered = copy.deepcopy(locked)
    tampered["review"]["decisions"][0]["winner"] = "tie"
    with pytest.raises(ValueError, match="locked decisions hash"):
        summarize_evaluation(packet, tampered, key)


def test_eval_requires_same_decoding_and_full_model_pair():
    rows = candidates("validation")
    with pytest.raises(ValueError, match="exactly one"):
        make_packet(rows[:1], purpose="evaluation", baseline="baseline", candidate="candidate")
    rows[1]["generation"]["temperature"] = 0.1
    rows[1]["request_sha256"] = digest(
        {"model": rows[1]["model"], "messages": rows[1]["messages"], "generation": rows[1]["generation"]}
    )
    with pytest.raises(ValueError, match="decoding"):
        make_packet(rows, purpose="evaluation", baseline="baseline", candidate="candidate")


def test_judge_is_context_bound_and_cannot_approve_training_data():
    row = candidates()[0]
    captured = []

    def call(model, messages, generation, **kwargs):
        captured.append(messages)
        return judge_response()

    result = judge_candidate(row, model("judge"), call=call)
    validate_judgment(result, row)
    assert result["review_status"] == "pending" and result["feedback_source"] == "ai"
    assert "local-baseline" not in captured[0][1]["content"]
    assert row["response"] in captured[0][1]["content"]
    with pytest.raises(ValueError, match="another candidate"):
        validate_judgment(result, candidates()[1])


@pytest.mark.parametrize(
    "reply",
    ["not json", judge_response(evidence_ids=["invented"]), judge_response(scores={}), judge_response(abstain="false")],
)
def test_judge_invalid_output_fails_closed(reply):
    with pytest.raises(ValueError):
        judge_candidate(candidates()[0], model("judge"), call=lambda *a, **kw: reply)


def test_calibration_joins_by_candidate_hash_and_reports_abstentions():
    rows = candidates("validation", count=2)
    packet, key, _ = make_packet(rows, purpose="evaluation", baseline="baseline", candidate="candidate")
    locked = lock_decisions(packet, completed(packet, key))
    judgments = [judge_candidate(row, model("judge"), call=lambda *a, **kw: judge_response()) for row in rows]
    judgments[0] = judge_candidate(
        rows[0], model("judge"), call=lambda *a, **kw: json.dumps({"abstain": True, "reason": "证据不足"})
    )
    result = calibration_from_review(packet, locked, key, judgments)
    assert result["report"]["coverage"] == 0.5
    assert result["report"]["skipped"]["judge_abstention"] == 1
    assert result["report"]["agreement"] == 0  # Equal judge scores cannot agree with a human strict preference.


def test_test_split_cannot_calibrate_rewards():
    rows = candidates("test")
    packet, key, _ = make_packet(rows, purpose="evaluation", baseline="baseline", candidate="candidate")
    locked = lock_decisions(packet, completed(packet, key))
    with pytest.raises(ValueError, match="final test"):
        calibration_from_review(packet, locked, key, [])


def test_legacy_template_generator_disabled_for_real_training(tmp_path):
    from data.gen_preference_pairs import generate_pairs

    with pytest.raises(ValueError, match="Legacy Gold/template"):
        generate_pairs(tmp_path / "gold.json", tmp_path / "sft.json", tmp_path / "out.jsonl")
    assert not (tmp_path / "out.jsonl").exists()


def test_mock_preferences_rejected_even_with_approved_label():
    from training.preference_validation import validate_pairs

    with pytest.raises(ValueError, match="mock"):
        validate_pairs(
            [{"prompt": "q", "chosen": "a", "rejected": "b", "review_status": "approved", "metadata": {"mock": True}}],
            split="train",
        )
