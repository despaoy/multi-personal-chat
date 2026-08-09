import csv
import json

import pytest

from scripts.review_kisaki_gold_v2 import export_review, import_review, review_summary


def _data():
    return {
        "prompts": [
            {
                "id": "gold-1",
                "category": "persona",
                "prompt": "测试问题",
                "expected_behavior": "保持角色",
                "review_status": "pending",
                "review_notes": "",
            }
        ]
    }


def _edit_decision(path, decision):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = rows[0].keys()
    rows[0]["human_decision"] = decision
    rows[0]["review_notes"] = "人工确认"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_review_round_trip_only_updates_human_metadata(tmp_path):
    data = _data()
    review_file = tmp_path / "review.csv"
    export_review(data, review_file)
    _edit_decision(review_file, "approved")
    updated = import_review(data, review_file, reviewer="student")
    assert updated["prompts"][0]["prompt"] == "测试问题"
    assert updated["prompts"][0]["review_status"] == "approved"
    assert updated["prompts"][0]["reviewed_by"] == "student"
    assert review_summary(updated)["decisions"] == {"approved": 1}


def test_review_import_rejects_content_hash_mismatch(tmp_path):
    data = _data()
    review_file = tmp_path / "review.csv"
    export_review(data, review_file)
    with review_file.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["content_sha256"] = "0" * 64
    with review_file.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="content changed"):
        import_review(data, review_file, reviewer="student")


def test_review_import_rejects_non_human_decision_values(tmp_path):
    data = _data()
    review_file = tmp_path / "review.csv"
    export_review(data, review_file)
    _edit_decision(review_file, "ai_approved")
    with pytest.raises(ValueError, match="Invalid human_decision"):
        import_review(data, review_file, reviewer="student")
