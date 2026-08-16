import importlib.util
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts/build_kisaki_gold_v21.py"
DATASET = PROJECT_ROOT / "backend/evaluation/kisaki_gold_set_v21_candidates.json"
AUDIT = PROJECT_ROOT / "backend/evaluation/kisaki_gold_set_v21_contamination_audit.json"


def _module():
    spec = importlib.util.spec_from_file_location("build_kisaki_gold_v21", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_gold_v21_has_balanced_structured_development_contract():
    dataset = _load(DATASET)
    rows = dataset["prompts"]

    assert dataset["status"] == "pending_human_review"
    assert dataset["evaluation_role"] == "development_only"
    assert dataset["formal_use_allowed"] is False
    assert len(rows) == dataset["total_prompts"] == 150
    assert Counter(row["category"] for row in rows) == {
        "persona": 30,
        "factual": 20,
        "persona_knowledge": 10,
        "multiturn": 30,
        "safety": 30,
        "rag_grounded": 30,
    }
    required_fields = {
        "cluster_id", "evaluation_role", "required_facts", "required_behaviors", "optional_style_traits",
        "forbidden_claims", "evidence_refs", "expected_action", "rubric",
        "contamination_status", "review_status",
    }
    assert all(required_fields <= row.keys() for row in rows)
    assert all(abs(sum(item["weight"] for item in row["rubric"]) - 1.0) < 1e-9 for row in rows)
    assert all(row["contamination_status"] == "clean" for row in rows)


def test_gold_v21_cluster_and_subtype_distribution_matches_preregistration():
    rows = _load(DATASET)["prompts"]
    persona = [row for row in rows if row["category"] == "persona"]
    multiturn = [row for row in rows if row["category"] == "multiturn"]
    safety = [row for row in rows if row["category"] == "safety"]
    rag = [row for row in rows if row["category"] == "rag_grounded"]

    assert len({row["cluster_id"] for row in persona}) == 15
    assert set(Counter(row["cluster_id"] for row in persona).values()) == {2}
    assert len({row["cluster_id"] for row in multiturn}) == 30
    assert Counter(row["expected_action"] for row in safety) == {
        "refuse": 10,
        "allow": 8,
        "clarify": 5,
        "clarify_supportive": 1,
        "allow_with_confirmation": 1,
        "safe_alternative": 3,
        "crisis_support": 1,
        "allow_with_redaction": 1,
    }
    assert Counter(row["rag_case_type"] for row in rag) == {
        "single_evidence": 10, "multi_evidence": 5, "hard_negative": 5,
        "unanswerable": 5, "paraphrase": 5,
    }


def test_multiturn_prompts_are_fixed_natural_progressions_without_old_coercive_templates():
    rows = [row for row in _load(DATASET)["prompts"] if row["category"] == "multiturn"]
    assert len({row["cluster_id"] for row in rows}) == 30
    assert all(len(row["conversation"]) == len(row["turn_rubrics"]) == 3 for row in rows)
    text = "\n".join(message["content"] for row in rows for message in row["conversation"])
    assert "你不愿意直接回答吗" not in text
    assert "可你刚才的语气不像这么想" not in text


def test_factual_and_rag_evidence_are_reproducible_and_answers_are_judgeable():
    rows = _load(DATASET)["prompts"]
    factual = [row for row in rows if row["category"] == "factual"]
    rag = [row for row in rows if row["category"] == "rag_grounded"]
    assert len(factual) == 20
    for row in factual:
        assert row["evidence_refs"]
        for evidence in row["evidence_refs"]:
            assert evidence["source_event_id"].startswith("tsukiyashiro_kisaki_raw_")
            assert evidence["source_path"].startswith("gametext/纸上魔法使/")
            assert evidence["source_line_start"] > 0
            assert evidence["source_line_end"] >= evidence["source_line_start"]
    assert all("gold_answer" in row and row["required_answer_facts"] for row in rag)
    assert all(row.get("distractor_refs") for row in rag if row["rag_case_type"] == "hard_negative")
    for row in rag:
        for evidence in row["evidence_refs"]:
            assert evidence["source_event_ids"]
            assert evidence["source_line_end"] >= evidence["source_line_start"]
            assert evidence["source_lineage"]
        assert len(row["kb_revision"]) == 64

    by_id = {row["id"]: row for row in rag}
    assert by_id["kisaki_v21_rag_005"]["required_answer_facts"] == ["需要时间"]
    assert by_id["kisaki_v21_rag_015"]["expected_refs"] == [
        "tsukiyashiro_kisaki_doc_031",
        "tsukiyashiro_kisaki_doc_032",
        "tsukiyashiro_kisaki_doc_033",
    ]

    documents = _load(
        PROJECT_ROOT / "backend/data/character_dialogues/experiments/research/character_rag_seed_documents.json"
    )["documents"]
    documents_by_id = {row["id"]: row for row in documents}
    assert [item["source_line_start"] for item in documents_by_id["tsukiyashiro_kisaki_doc_011"]["source_lineage"]] == [2049, 2052, 2054]


def test_gold_v21_has_no_current_train_or_validation_text_overlap():
    audit = _load(AUDIT)
    assert audit["status"] == "clean"
    assert audit["threshold"] == 0.90
    assert audit["id_overlaps"] == []
    assert audit["text_overlap_matches"] == []
    assert audit["rag_evidence_event_count"] == 23
    assert audit["rag_evidence_event_overlaps"] == []
