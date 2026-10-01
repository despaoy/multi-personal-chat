"""Do not confuse an adjacent public citation with a private claim source."""

import json
from pathlib import Path

import pytest

from evaluation.native_mixed_context_probe import audit_private_public_citation_wire, private_public_marker_checks


def test_semicolon_separates_private_receipt_from_cited_public_code():
    raw = "PB-681-Q 是你本人确认书的私人回执，不是公开编号；YY-573-R 是工坊公开课程编号[S1]。"
    assert all(private_public_marker_checks(raw, "PB-681-Q", "YY-573-R", {"S1"}).values())


def test_actual_private_source_marker_is_rejected():
    raw = "你的私人回执是PB-681-Q[S1]。公开课程号是YY-573-R[S1]。"
    checks = private_public_marker_checks(raw, "PB-681-Q", "YY-573-R", {"S1"})
    assert checks["private_receipt_present"]
    assert checks["public_code_has_own_authorized_citation"]
    assert not checks["private_claims_not_cited_as_public"]


@pytest.mark.parametrize("public_suffix", ["", "[S99]"])
def test_missing_or_unknown_public_marker_cannot_prove_public_provenance(public_suffix):
    raw = "你的私人回执是PB-681-Q。公开课程号是YY-573-R" + public_suffix
    assert not private_public_marker_checks(raw, "PB-681-Q", "YY-573-R", {"S1"})["public_code_has_own_authorized_citation"]


def test_public_fact_alone_cannot_prove_private_receipt_was_answered():
    checks = private_public_marker_checks("课程号YY-573-R[S1]。", "PB-681-Q", "YY-573-R", {"S1"})
    assert not checks["private_receipt_present"]
    assert not checks["private_claims_not_cited_as_public"]


@pytest.mark.parametrize("swapped", [False, True])
def test_complete_history_cannot_compensate_for_missing_or_wrong_private_memory_lane(swapped):
    path = Path(__file__).parent / "fixtures/deepseek_private_public_citation_cases.json"
    fixture = json.loads(path.read_text(encoding="utf-8"))
    source, doc = fixture["cases"][0]["message"], fixture["documents"][0]
    private = doc["content"] if swapped else ""
    public = source if swapped else doc["content"]
    wire = "<character_memory>\n- " + json.dumps({"evidence": [private]}, ensure_ascii=False) + "\n</character_memory>"
    wire += "\n<retrieved_evidence>\n" + public + "\n</retrieved_evidence>"
    raw = "你已收到书面确认，尚未参加课程，私人回执是PB-681-Q。公开课程编号YY-573-R[S1]。"
    proof = {"generation": [{"cloud_call_range": [0, 1], "response": {"reply": raw.replace("[S1]", ""),
        "citations": [{"source_id": "doc_1_chunk_0", "source_title": doc["title"], "key": "S1"}]}}],
        "claims": [{"evidence_json": json.dumps([source])}],
        "prepared_diagnostics": [{"selection_status": "selected", "used_memory_ids": ["1"],
            "semantic_status": "applied", "policy_status": "applied"}]}
    calls = [{"request": {"max_tokens": 1024, "messages": [{"role": "system", "content": "应用规则"},
        {"role": "user", "content": source}, {"role": "user", "content": wire}]},
        "response": {"choices": [{"message": {"content": raw}}]}}]
    checks = audit_private_public_citation_wire(proof, calls, fixture)
    assert checks["complete_private_source_in_actual_history"]
    assert not checks["complete_private_source_in_selected_memory"]
    if swapped:
        assert not checks["complete_public_course_in_actual_evidence"]
        assert not checks["private_receipt_outside_public_evidence"]
        assert not checks["public_course_outside_private_memory"]
