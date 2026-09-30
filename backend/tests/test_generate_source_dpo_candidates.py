import copy

import pytest
from scripts.generate_source_dpo_candidates import build_contexts, process, validate_judgment
from test_source_preferences import inputs

from training.source_preferences import prepare_sources


def fixture(tmp_path):
    train, raw = inputs()
    raw[0]["source_line_end"] = 3
    raw[0]["scene_block_id"] = "chapter.txt:scene-block:1"
    m = train[0]["metadata"]
    m.update(
        scene_block_id=raw[0]["scene_block_id"],
        context_line_start=2,
        response_line_start=3,
        response_line_end=3,
        context_speaker_label="friend",
    )
    (tmp_path / "chapter.txt").write_text(
        "A prior promise.\n[friend] Will you come?\n[Test] I will keep my promise.\nFuture secret.", encoding="utf-8"
    )
    drafts, _ = prepare_sources(train, [], raw, "profile")
    return drafts, raw


def test_window_stops_before_target_and_checks_gold(tmp_path):
    drafts, raw = fixture(tmp_path)
    rows, excluded = build_contexts(drafts, raw, [], {}, tmp_path)
    assert len(rows) == 1 and not excluded
    assert "Future secret" not in str(rows[0]["prompt"])
    assert raw[0]["text"] not in str(rows[0]["prompt"])
    raw[0]["id"] = "tsukiyashiro_kisaki_raw_ab12"
    drafts[0]["source_row"]["metadata"]["target_event_ids"] = [raw[0]["id"]]
    rows, excluded = build_contexts(drafts, raw, [], {"evidence_refs": [raw[0]["id"]]}, tmp_path)
    assert not rows and excluded[0]["reason"] == "heldout_scene"


def test_actual_script_mismatch_and_validation_window(tmp_path):
    drafts, raw = fixture(tmp_path)
    validation = [{"metadata": {"source_file": "chapter.txt", "source_line_start": 1, "source_line_end": 1}}]
    assert build_contexts(drafts, raw, validation, {}, tmp_path)[1][0]["reason"] == "heldout_window_overlap"
    (tmp_path / "chapter.txt").write_text("before\nquestion\nDIFFERENT\nafter", encoding="utf-8")
    assert build_contexts(drafts, raw, [], {}, tmp_path)[1][0]["reason"] == "literal_source_mismatch"


def test_invalid_judgments_rejected():
    j = {
        "context_sufficient": True,
        "winner": "A",
        "confidence": 0.9,
        "reason": "reason",
        "evidence_lines": [1],
        "hard_errors": {"A": [], "B": []},
        "dimensions": dict.fromkeys(
            ("persona_decision", "relationship", "grounding", "continuity", "naturalness"), "A"
        ),
    }
    row = {"context_start": 1, "context_end": 2}
    validate_judgment(j, row)
    bad = copy.deepcopy(j)
    bad["evidence_lines"] = [3]
    with pytest.raises(ValueError, match="evidence"):
        validate_judgment(bad, row)


def test_api_errors_do_not_become_preferences(tmp_path, monkeypatch):
    drafts, raw = fixture(tmp_path)
    row = build_contexts(drafts, raw, [], {}, tmp_path)[0][0]

    def fail(*a, **kw):
        raise RuntimeError("private transport details")

    monkeypatch.setattr("scripts.generate_source_dpo_candidates.chat_completion", fail)
    result = process(row, {})
    assert result["status"] == "error" and result["review_status"] == "pending"
    assert "private transport details" not in str(result)
