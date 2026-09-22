import pytest

from scripts.audit_rag_ranking_disagreements import build_queue


def test_queue_keeps_original_labels_and_does_not_approve_equivalence():
    old = {"id": "q", "gold_ids": ["a"], "retrieved_ids": ["a", "b"], "hit_at_1": 1}
    new = {**old, "retrieved_ids": ["b", "a"], "hit_at_1": 0}
    report = {"variants": {"deterministic": {"per_query": [old]}, "cross_encoder": {"per_query": [new]}}}
    cases = [{"id": "q", "query": "query", "expected": {"expected_document_ids": ["a"]}}]
    queue = build_queue(report, cases, {"a": {"id": "a"}, "b": {"id": "b"}})
    assert queue[0]["review_status"] == "pending"
    assert queue[0]["review_judgment"] is None
    assert queue[0]["frozen_gold_ids"] == ["a"]
    assert queue[0]["hit_at_1_delta"] == -1
    assert old["retrieved_ids"] == ["a", "b"]
    with pytest.raises(ValueError, match="missing"):
        build_queue(report, cases, {"a": {"id": "a"}})
