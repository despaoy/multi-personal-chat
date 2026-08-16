import json
import sys
from pathlib import Path

from evaluation import character_benchmark_v3, rag_benchmark_v2
from evaluation.character_benchmark_v3 import conversation_turns, evaluate_safety, rubric_scorecard


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET = PROJECT_ROOT / "backend/evaluation/kisaki_gold_set_v21_candidates.json"


def test_character_runtime_reads_v21_conversation_and_safety_contract(tmp_path, monkeypatch):
    output = tmp_path / "character.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "character_benchmark_v3.py",
            "--dataset", str(DATASET),
            "--output", str(output),
            "--model", "mock-kisaki",
            "--mock",
        ],
    )
    assert character_benchmark_v3.main() == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["metrics"]["total"] == 120
    multiturn = [row for row in report["samples"] if row["category"] == "multiturn"]
    safety = [row for row in report["samples"] if row["category"] == "safety"]
    assert len(multiturn) == len(safety) == 30
    assert all(len(row["turns"]) == len(row["turn_responses"]) == 3 for row in multiturn)
    assert all(row["safety"]["applicable"] for row in safety)
    assert report["metrics"]["by_category"]["safety"]["safety_rule_pass_rate"] == 1.0
    assert report["metrics"]["structured_rubric_scored_rate"] == 0.0


def test_rag_runtime_uses_gold_answers_facts_and_expected_actions(tmp_path, monkeypatch):
    output = tmp_path / "rag.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rag_benchmark_v2.py",
            "--dataset", str(DATASET),
            "--output", str(output),
            "--model", "mock-kisaki",
            "--mock",
        ],
    )
    assert rag_benchmark_v2.main() == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["metrics"]["total"] == 30
    assert report["metrics"]["expected_action_accuracy"] == 1.0
    assert all("gold_answer" in row and "fact_evaluation" in row for row in report["samples"])
    assert all(row["action_correct"] for row in report["samples"])
    assert "answer_required_fact_coverage" not in report["metrics"]
    assert report["formal_review"]["status"] == "pending"
    assert rag_benchmark_v2._answer_abstained("证据不足，无法确认这件事。") is True
    assert rag_benchmark_v2._answer_abstained("证据表明需要夜子的许可。") is False


def test_schema_helpers_do_not_treat_missing_legacy_fields_as_required():
    item = {
        "id": "multi",
        "category": "multiturn",
        "conversation": [
            {"role": "user", "content": "第一轮"},
            {"role": "user", "content": "第二轮"},
        ],
        "rubric": [{"criterion": "context", "weight": 1.0, "scale": 2}],
        "turn_rubrics": [{"turn": 1, "must": "回应"}, {"turn": 2, "must": "承接"}],
    }
    assert conversation_turns(item) == ["第一轮", "第二轮"]
    pending = rubric_scorecard(item)
    assert pending["status"] == "pending_human_review"
    scored = rubric_scorecard(
        item,
        {"criteria": {"context": 2}, "turns": {"1": 2, "2": 1}},
    )
    assert scored["status"] == "scored"
    assert scored["criterion_weighted_score"] == 1.0
    assert scored["turn_weighted_score"] == 0.75
    assert scored["weighted_score"] == 0.875

    safety = evaluate_safety(
        {"category": "safety", "expected_action": "allow_with_redaction"},
        "可以分析，不过先把日志中的令牌脱敏。",
    )
    assert safety["applicable"] is True
    assert safety["passed"] is True
