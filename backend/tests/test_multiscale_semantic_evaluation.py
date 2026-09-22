from types import SimpleNamespace

from scripts.evaluate_multiscale_semantic_rerank import matches, resolve_gold, summarize


def _doc(key, subject="妃", relation="哥哥"):
    return SimpleNamespace(
        id=key,
        document_type="relation",
        title="亲属关系",
        entities=["妃", "琉璃"],
        metadata={"subject": subject, "relation": relation, "target": "琉璃"},
    )


def test_gold_resolution_preserves_relation_direction_and_or_clauses():
    first, second = _doc("a"), _doc("b", subject="琉璃", relation="妹妹")
    criteria = {"document_type": "relation", "subject": "妃", "relation_type": "哥哥"}
    assert matches(first, criteria)
    assert not matches(second, criteria)
    case = {"expected": {"criteria": {"either_of": [criteria, {"subject": "琉璃"}]}}}
    assert resolve_gold(case, {"a": first, "b": second}) == ({"a", "b"}, "resolved")


def test_missing_gold_not_silently_treated_as_negative():
    case = {"expected": {"expected_document_ids": ["missing"]}}
    assert resolve_gold(case, {}) == ({"missing"}, "unresolved")
    assert resolve_gold({"category": "no_answer", "expected": {"criteria": {}}}, {}) == (set(), "negative")


def test_summarize_reports_denominators_and_unresolved_count():
    metrics = summarize(
        [
            {"gold_status": "unresolved", "latency_ms": 1, "actual_rerank_method": "deterministic"},
            {"gold_status": "negative", "latency_ms": 1, "actual_rerank_method": "none", "retrieved_ids": []},
        ]
    )
    assert metrics["queries"] == 2
    assert metrics["unresolved_annotations"] == 1
    assert metrics["hit_at_5"] is None
    assert metrics["negative_domain_hit_rate"] == 0
    assert metrics["negative_nonabstention_rate"] is None


def test_domain_hit_and_abstention_are_separate_metrics():
    metrics = summarize(
        [
            {
                "gold_status": "negative",
                "latency_ms": 1,
                "actual_rerank_method": "cross_encoder",
                "retrieved_ids": ["a"],
                "abstained": True,
            },
            {
                "gold_status": "negative",
                "latency_ms": 1,
                "actual_rerank_method": "cross_encoder",
                "retrieved_ids": ["b"],
                "abstained": False,
            },
        ]
    )
    assert metrics["negative_domain_hit_rate"] == 1
    assert metrics["negative_nonabstention_rate"] == 0.5
