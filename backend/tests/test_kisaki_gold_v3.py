from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/build_kisaki_gold_v3.py"


def load_builder():
    spec = importlib.util.spec_from_file_location("build_kisaki_gold_v3", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_gold_v3_candidate_is_balanced_and_not_formally_usable():
    data = json.loads(
        (ROOT / "backend/evaluation/kisaki_gold_set_v3_candidates.json").read_text(
            encoding="utf-8"
        )
    )
    assert data["status"] == "pending_human_review"
    assert data["evaluation_role"] == "final_held_out_candidate"
    assert data["formal_use_allowed"] is False
    assert data["total_prompts"] == 150
    assert Counter(row["category"] for row in data["prompts"]) == {
        "persona": 30,
        "factual": 20,
        "persona_knowledge": 10,
        "multiturn": 30,
        "safety": 30,
        "rag_grounded": 30,
    }


def test_gold_v3_has_unique_ids_and_pending_human_decisions():
    data = json.loads(
        (ROOT / "backend/evaluation/kisaki_gold_set_v3_candidates.json").read_text(
            encoding="utf-8"
        )
    )
    ids = [row["id"] for row in data["prompts"]]
    assert len(ids) == len(set(ids)) == 150
    assert all(row["review_status"] == "pending_human_review" for row in data["prompts"])
    assert all(row["contamination_status"] == "clean" for row in data["prompts"])


def test_gold_v3_reaudit_is_clean_and_bound_to_current_frozen_data():
    audit = json.loads(
        (ROOT / "backend/evaluation/kisaki_gold_set_v3_contamination_audit.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads(
        (ROOT / "backend/data/character_dialogues/experiments/v4/canonical_dataset_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    gold = json.loads(
        (ROOT / "backend/evaluation/kisaki_gold_set_v3.json").read_text(
            encoding="utf-8"
        )
    )
    approval = json.loads(
        (
            ROOT
            / "docs/research/review_packets/kisaki_v4/07_GOLD_V3/gold_v3_final_approval.json"
        ).read_text(encoding="utf-8")
    )
    assert audit["schema_version"] == 2
    assert audit["status"] == "clean"
    assert audit["candidate_count"] == 150
    assert audit["frozen_reference_count"] == 996
    assert audit["frozen_train_count"] == manifest["train"]["count"] == 926
    assert audit["frozen_validation_count"] == manifest["validation"]["count"] == 70
    assert audit["text_overlap_matches"] == []
    assert audit["duplicate_normalized_prompts"] == []
    assert audit["rag_evidence_event_overlaps"] == []
    assert audit["candidate_content_sha256"] == gold["content_sha256"]
    assert gold["content_sha256"] == approval["content_sha256"]
    assert manifest["status"] == "frozen"
    assert manifest["freeze_blockers"] == []
    assert manifest["gold_v3"]["status"] == "frozen"
    assert manifest["gold_v3"]["formal_use_allowed"] is True
    assert manifest["gold_v3"]["contamination_reaudit"]["status"] == "clean"
    assert audit["frozen_train_sha256"] == manifest["train"]["sha256"]
    assert audit["frozen_validation_sha256"] == manifest["validation"]["sha256"]


def test_gold_v3_builder_is_deterministic():
    module = load_builder()
    first = (
        module.build_persona()
        + module.build_factual()
        + module.build_multiturn()
        + module.build_safety()
        + module.build_rag()
    )
    second = (
        module.build_persona()
        + module.build_factual()
        + module.build_multiturn()
        + module.build_safety()
        + module.build_rag()
    )
    assert module.canonical_hash(first) == module.canonical_hash(second)


def test_gold_v3_rag_uses_held_out_events_only():
    data = json.loads(
        (ROOT / "backend/evaluation/kisaki_gold_set_v3_candidates.json").read_text(
            encoding="utf-8"
        )
    )
    rag = [row for row in data["prompts"] if row["category"] == "rag_grounded"]
    assert len(rag) == 30
    answerable = [row for row in rag if row["expected_action"] == "answer"]
    unanswerable = [row for row in rag if row["expected_action"] == "abstain"]
    assert all(row["expected_refs"] and row["evidence_refs"] for row in answerable)
    assert all(
        not row["expected_refs"] and row["evidence_refs"] and row["distractor_refs"]
        for row in unanswerable
    )
    assert Counter(row["rag_case_type"] for row in rag) == {
        "single_evidence": 15,
        "multi_evidence": 5,
        "hard_negative": 5,
        "unanswerable": 5,
    }
    assert Counter(row["expected_action"] for row in rag) == {"answer": 25, "abstain": 5}
    hard_negatives = [row for row in rag if row["rag_case_type"] == "hard_negative"]
    assert all(row["distractor_refs"] for row in hard_negatives)
    assert all(
        set(row["distractor_refs"])
        <= {evidence["document_id"] for evidence in row["evidence_refs"]}
        for row in hard_negatives + unanswerable
    )
    assert all(
        not any(f"证据 {number}" in fact for number in range(1, 34))
        for row in hard_negatives
        for fact in row["required_answer_facts"]
    )
    by_id = {row["id"]: row for row in rag}
    expected_visible_order = {
        "kisaki_v3_rag_021": ["tsukiyashiro_kisaki_doc_021", "tsukiyashiro_kisaki_doc_016"],
        "kisaki_v3_rag_022": ["tsukiyashiro_kisaki_doc_017", "tsukiyashiro_kisaki_doc_018"],
        "kisaki_v3_rag_023": ["tsukiyashiro_kisaki_doc_016", "tsukiyashiro_kisaki_doc_021"],
        "kisaki_v3_rag_024": ["tsukiyashiro_kisaki_doc_029", "tsukiyashiro_kisaki_doc_026"],
        "kisaki_v3_rag_025": ["tsukiyashiro_kisaki_doc_016", "tsukiyashiro_kisaki_doc_030"],
    }
    for sample_id, expected_order in expected_visible_order.items():
        assert [ref["document_id"] for ref in by_id[sample_id]["evidence_refs"]] == expected_order

    rag_029 = by_id["kisaki_v3_rag_029"]
    assert rag_029["prompt"] == "证据是否列出了把妃当成陌生人的同班同学姓名？"
    assert rag_029["expected_refs"] == []
    assert rag_029["distractor_refs"] == ["tsukiyashiro_kisaki_doc_030"]
    assert [ref["document_id"] for ref in rag_029["evidence_refs"]] == [
        "tsukiyashiro_kisaki_doc_030"
    ]
    assert rag_029["expected_action"] == "abstain"
    assert rag_029["gold_answer"] == "证据只说明同班同学把妃当成陌生人，没有列出具体姓名。"
    assert rag_029["required_answer_facts"] == ["证据未提供具体姓名"]
