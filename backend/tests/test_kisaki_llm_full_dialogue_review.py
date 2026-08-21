from __future__ import annotations

import importlib.util
import json
import re
import zipfile
from datetime import datetime, time, timezone
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
V4 = ROOT / "backend/data/character_dialogues/experiments/v4"
TRAIN = V4 / "train.jsonl"
AUXILIARY = V4 / "cleanup/candidate/technical_auxiliary.jsonl"
ARTIFACTS = V4 / "augmentation_candidates/llm_full_dialogue_review_20260816"
PROMOTER = ROOT / "scripts/promote_kisaki_v41_round06.py"
REVIEW_ID = "KISAKI-V41-LLM-FULL-DIALOGUE-REVIEW-20260816"


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _module():
    spec = importlib.util.spec_from_file_location("kisaki_promoter_full_review_test", PROMOTER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _record(record_id: str) -> dict:
    records = _jsonl(TRAIN) + _jsonl(AUXILIARY)
    return next(record for record in records if record["id"] == record_id)


def _last_python_namespace(record_id: str) -> dict:
    record = _record(record_id)
    content = [message["content"] for message in record["messages"] if message["role"] == "assistant"][-1]
    blocks = re.findall(r"```python\n(.*?)```", content, re.S)
    assert blocks
    namespace = {"__name__": "reviewed_sample"}
    exec(compile(blocks[-1], f"<{record_id}>", "exec"), namespace)
    return namespace


def test_full_dialogue_review_covers_every_record_and_turn():
    summary = _json(ARTIFACTS / "summary.json")
    reviews = _jsonl(ARTIFACTS / "record_reviews.jsonl")
    assert summary["status"] == "approved_and_promoted"
    assert summary["review_id"] == REVIEW_ID
    assert len(reviews) == summary["record_count"] == 426
    assert sum(len(review["turn_reviews"]) for review in reviews) == 1549
    assert sum(bool(review["revised_assistant_turns"]) for review in reviews) == 14
    assert summary["revised_assistant_turn_count"] == 16
    assert summary["rejected_record_count"] == 0
    assert all(
        all(all(turn["checks"].values()) for turn in review["turn_reviews"])
        for review in reviews
    )
    assert all(
        all(set(turn["checks"]) == set(summary["rubric"]) for turn in review["turn_reviews"])
        for review in reviews
    )


def test_full_dialogue_review_preserves_users_ids_and_order():
    original = _jsonl(ARTIFACTS / "original_llm_records.jsonl")
    reviewed = _jsonl(ARTIFACTS / "reviewed_llm_records.jsonl")
    assert [record["id"] for record in original] == [record["id"] for record in reviewed]
    promoter = _module()
    changed_turns = 0
    for before, after in zip(original, reviewed):
        assert promoter._user_texts(before) == promoter._user_texts(after)
        before_turns = [m["content"] for m in before["messages"] if m["role"] == "assistant"]
        after_turns = [m["content"] for m in after["messages"] if m["role"] == "assistant"]
        changed_turns += sum(left != right for left, right in zip(before_turns, after_turns))
    assert changed_turns == 16


def test_motif_stacking_and_duplicate_openings_are_removed():
    daily = _record("kisaki_v41_round06_daily_chat")
    answers = [m["content"] for m in daily["messages"] if m["role"] == "assistant"]
    assert not any(re.search("星星|猫|小说|红茶", answer) for answer in answers)
    cluster = _record("kisaki_v41_auto_b058_handling_cluster_robust_inference_with_few_clusters")
    first = next(m["content"] for m in cluster["messages"] if m["role"] == "assistant")
    assert "学校数。不一定" not in first
    assert first.count("不等于独立信息很多") == 1


def test_incremental_jsonl_parser_sample_handles_chunk_boundaries():
    parser_class = _last_python_namespace(
        "kisaki_v41_auto_b012_incremental_jsonl_byte_parser"
    )["JsonlStreamParser"]
    parser = parser_class(max_line_bytes=32)
    assert parser.feed(b'{"a":') == []
    assert parser.feed(b"1}\r\n\n") == [{"a": 1}]
    assert parser.feed(b'{"b":2}') == []
    assert parser.finalize() == [{"b": 2}]


def test_timezone_scheduler_sample_handles_dst_boundaries():
    next_daily_run = _last_python_namespace(
        "kisaki_v41_auto_b013_timezone_daily_scheduler"
    )["next_daily_run"]
    spring = next_daily_run(
        datetime(2024, 3, 10, 6, 0, tzinfo=timezone.utc),
        time(2, 30),
        "America/New_York",
    )
    assert spring == datetime(2024, 3, 10, 7, 0, tzinfo=timezone.utc)
    autumn = next_daily_run(
        datetime(2024, 11, 3, 4, 0, tzinfo=timezone.utc),
        time(1, 30),
        "America/New_York",
    )
    assert autumn == datetime(2024, 11, 3, 5, 30, tzinfo=timezone.utc)


def test_safe_zip_sample_extracts_regular_file_and_rejects_traversal(tmp_path):
    safe_extract_zip = _last_python_namespace(
        "kisaki_v41_auto_b016_safe_zip_extraction"
    )["safe_extract_zip"]
    archive = tmp_path / "safe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("folder/readme.txt", "ok")
    destination = tmp_path / "out"
    safe_extract_zip(archive, destination)
    assert (destination / "folder/readme.txt").read_text(encoding="utf-8") == "ok"

    bad_archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad_archive, "w") as handle:
        handle.writestr("../escape.txt", "bad")
    with pytest.raises(ValueError, match="unsafe ZIP path"):
        safe_extract_zip(bad_archive, tmp_path / "bad-out")


def test_complete_code_samples_execute_their_claimed_minimum_behavior(tmp_path):
    counter_namespace = _last_python_namespace("kisaki_v41_round06_coding_debug")
    sample = tmp_path / "sample.jsonl"
    sample.write_text('{"data_source":"a"}\n[]\n{broken}\n', encoding="utf-8")
    counts, bad_count, lines = counter_namespace["count_by_data_source"](sample)
    assert dict(counts) == {"a": 1}
    assert (bad_count, lines) == (2, [2, 3])

    grouping = _last_python_namespace(
        "kisaki_v41_auto_b040_fixing_shared_lists_created_with_dict_fromkeys"
    )["group_tasks"]
    groups = grouping(["todo", "done"], [("write", "todo")])
    assert groups == {"todo": ["write"], "done": []}
    assert groups["todo"] is not groups["done"]
