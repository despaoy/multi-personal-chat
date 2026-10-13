"""Same document identity cannot conceal changed evidence between rounds."""
from copy import deepcopy
from unittest.mock import Mock

import pytest

from knowledge.grounded_answer.corrective import CorrectiveRetrievalAdapter


def row():
    return dict(id="permit", title="permit", summary="written permit", content="Written approval is required.",
                source={"source_path": "rules.txt", "line_start": 1}, domain_id="lab", index_version="v1",
                metadata={"owner": "lab"}, reality_status="objective", temporal_scope="current",
                content_scope="main_story", fused_score=0.2)


def bundle(record, *, abstained):
    return dict(results=[record], citations=[dict(source_id=record["id"])], confidence=0.2 if abstained else 0.8, abstained=abstained)


@pytest.mark.parametrize("field,value", [
    ("document_type", "event"), ("title", "another title"), ("summary", "No approval"), ("content", "Approval is not required."),
    ("source", {"source_path": "other.txt", "line_start": 9}), ("domain_id", "other"),
    ("index_version", "v2"), ("metadata", {"owner": "other"}), ("reality_status", "fictional"),
    ("temporal_scope", "past"), ("content_scope", "other"),
])
def test_changed_authority_cannot_merge_with_new_confidence(field, value):
    first, second = row(), row()
    second[field] = value
    retrieve = Mock(side_effect=[bundle(first, abstained=True), bundle(second, abstained=False)])
    with pytest.raises(ValueError, match="Conflicting document evidence"):
        CorrectiveRetrievalAdapter(retrieve).retrieve_with_correction("entry", top_k=3, filters=None)


def test_same_evidence_different_scores_deduplicates_without_mutating_inputs():
    first, second = row(), row()
    second["fused_score"] = 0.8
    original = deepcopy([first, second])
    merged = CorrectiveRetrievalAdapter._merge_bundles(bundle(first, abstained=True), bundle(second, abstained=False))
    assert len(merged["results"]) == len(merged["citations"]) == 1
    assert merged["results"][0]["content"] == first["content"]
    assert merged["results"][0]["retrieval_rank"] == 1
    assert merged["confidence"] == 0.8 and merged["abstained"] is False
    assert [first, second] == original
