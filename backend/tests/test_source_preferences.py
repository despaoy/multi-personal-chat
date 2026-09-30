import copy

import pytest
from test_persona_alignment_workflow import GENERATION, completed, model

from training.persona_review import export_preferences, lock_decisions, make_packet
from training.persona_sampling import validate_run
from training.source_preferences import prepare_sources, sample_source_pairs


def inputs():
    raw = [
        {
            "id": "event1",
            "character": "Test",
            "speaker_kind": "direct",
            "source_role": "canonical",
            "source_file": "chapter.txt",
            "scene_block_id": "chapter:scene1",
            "source_line_start": 3,
            "text": "I will keep my promise.",
        }
    ]
    train = [
        {
            "id": "row1",
            "messages": [
                {"role": "user", "content": "Will you come?"},
                {"role": "assistant", "content": raw[0]["text"]},
            ],
            "metadata": {
                "character": "Test",
                "data_source": "game_extraction",
                "target_event_ids": ["event1"],
                "source_file": "chapter.txt",
                "scene_block_id": "chapter:scene1",
            },
        }
    ]
    return train, raw


def approved():
    train, raw = inputs()
    rows, report = prepare_sources(train, [], raw, "A fictional character.")
    rows[0]["scene"].update(review_status="approved", relationship="friends", situation="A prior appointment.")
    rows[0]["context_review"] = dict(
        reviewer="synthetic-reviewer",
        approved=True,
        sufficient_pre_reply_context=True,
        no_target_or_future_leakage=True,
    )
    return rows, raw


def test_prepare_pending_and_heldout_exclusion():
    train, raw = inputs()
    rows, report = prepare_sources(train, [], raw, "profile")
    assert rows[0]["scene"]["review_status"] == "pending"
    assert not rows[0]["context_review"]["approved"]
    assert not prepare_sources(train, train, raw, "profile")[0]
    train[0]["messages"][-1]["content"] = "modified source"
    assert prepare_sources(train, [], raw, "profile")[1]["exclusion_counts"] == {"source_verification_failed": 1}


def test_source_candidate_review_export_and_no_answer_leak():
    rows, raw = approved()
    calls = []

    def generate(identity, messages, params):
        calls.append(messages)
        assert all(raw[0]["text"] not in m["content"] for m in messages)
        return "As an assistant I cannot attend."

    contract, records = sample_source_pairs(rows, raw, model("ordinary"), GENERATION, call=generate)
    assert len(calls) == 1
    validate_run(contract, records)
    assert {r["status"] for r in records} == {"generated", "source_excerpt"}
    packet, key, template = make_packet(records, purpose="preferences")
    locked = lock_decisions(packet, completed(packet, key, winner="canonical_original"))
    pairs = export_preferences(packet, locked, key)["pairs"]
    assert pairs[0]["chosen"][0]["content"] == raw[0]["text"]
    assert pairs[0]["metadata"]["canonical_source"]["event_ids"] == ["event1"]
    locked = lock_decisions(packet, completed(packet, key, winner="ordinary"))
    assert export_preferences(packet, locked, key)["excluded"] == {"original_not_preferred": 1}
    with pytest.raises(ValueError, match="not model evaluation"):
        make_packet(records, purpose="evaluation")


def test_unreviewed_leaked_or_modified_sources_cannot_call_model():
    rows, raw = approved()

    def forbidden(*args):
        pytest.fail("model should not be called")

    for mutate, error in [
        (lambda r: r[0]["context_review"].update(approved=False), "explicit review"),
        (lambda r: r[0]["scene"].update(situation="Before: " + raw[0]["text"]), "leakage"),
        (lambda r: r[0]["source_reply"].update(text="changed"), "binding"),
    ]:
        changed = copy.deepcopy(rows)
        mutate(changed)
        with pytest.raises(ValueError, match=error):
            sample_source_pairs(changed, raw, model(), GENERATION, call=forbidden)
