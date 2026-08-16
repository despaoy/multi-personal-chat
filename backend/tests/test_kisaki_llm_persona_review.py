from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
V4 = ROOT / "backend/data/character_dialogues/experiments/v4"
ARTIFACTS = V4 / "augmentation_candidates/llm_persona_review_20260816"
PROMOTER = ROOT / "scripts/promote_kisaki_v41_round06.py"
REVIEW_ID = "KISAKI-V41-LLM-PERSONA-REVIEW-20260816"


def _module():
    spec = importlib.util.spec_from_file_location("kisaki_promoter", PROMOTER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_llm_persona_review_covers_every_record_and_assistant_turn():
    summary = _json(ARTIFACTS / "summary.json")
    reviews = _jsonl(ARTIFACTS / "record_reviews.jsonl")
    assert summary["status"] == "approved_and_promoted"
    assert summary["review_id"] == REVIEW_ID
    assert len(reviews) == summary["record_count"] == 426
    assert sum(len(review["turn_reviews"]) for review in reviews) == 1549
    assert sum(bool(review["revised_assistant_turns"]) for review in reviews) == 39
    assert all(
        all(all(turn["checks"].values()) for turn in review["turn_reviews"])
        for review in reviews
    )


def test_llm_persona_review_preserves_users_and_tracks_exact_revisions():
    original = _jsonl(ARTIFACTS / "original_llm_records.jsonl")
    reviewed = _jsonl(ARTIFACTS / "reviewed_llm_records.jsonl")
    assert [record["id"] for record in original] == [record["id"] for record in reviewed]
    changed_turns = 0
    for before, after in zip(original, reviewed):
        assert _module()._user_texts(before) == _module()._user_texts(after)
        before_assistants = [m["content"] for m in before["messages"] if m["role"] == "assistant"]
        after_assistants = [m["content"] for m in after["messages"] if m["role"] == "assistant"]
        changed_turns += sum(left != right for left, right in zip(before_assistants, after_assistants))
    assert changed_turns == 39


def test_downstream_review_match_requires_exact_reviewed_snapshot():
    module = _module()
    original = _jsonl(ARTIFACTS / "original_llm_records.jsonl")
    reviewed = _jsonl(ARTIFACTS / "reviewed_llm_records.jsonl")
    changed_index = next(
        index for index, (before, after) in enumerate(zip(original, reviewed))
        if before["messages"] != after["messages"]
    )
    assert module._matches_authorized_downstream_review(
        reviewed[changed_index], original[changed_index]
    )
    tampered = json.loads(json.dumps(reviewed[changed_index], ensure_ascii=False))
    next(message for message in tampered["messages"] if message["role"] == "assistant")["content"] += "篡改"
    assert not module._matches_authorized_downstream_review(
        tampered, original[changed_index]
    )
