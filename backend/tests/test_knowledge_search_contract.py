"""Compatibility tests for the public knowledge-search response."""

from api.knowledge import _knowledge_search_response


def test_evidence_mode_preserves_legacy_search_type() -> None:
    response = _knowledge_search_response("query", [{"documentId": 1}], "evidence")

    assert response["retrievalMode"] == "evidence"
    assert response["searchType"] == "rag_pipeline"
    assert response["results"] == [{"documentId": 1}]


def test_non_evidence_modes_keep_existing_values() -> None:
    for mode in ("hybrid", "keyword", "empty"):
        response = _knowledge_search_response("query", [], mode)

        assert response["retrievalMode"] == mode
        assert response["searchType"] == mode
