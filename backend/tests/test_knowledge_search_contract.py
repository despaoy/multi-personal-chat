"""Completed knowledge-search results, identity and explicit failures."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from api import knowledge
from api.knowledge import _knowledge_search_response
from db.schemas import KnowledgeSearchRequest
from knowledge import rag_helper


def test_evidence_mode_preserves_legacy_search_type() -> None:
    response = _knowledge_search_response("query", [{"documentId": 1}])

    assert response["retrievalMode"] == "evidence"
    assert response["searchType"] == "rag_pipeline"
    assert response["results"] == [{"documentId": 1}]


def test_empty_results_have_explicit_empty_labels() -> None:
    response = _knowledge_search_response("query", [])
    assert response["retrievalMode"] == "empty"
    assert response["searchType"] == "empty"
    assert response["results"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize('stage', ['index', 'factory', 'retrieval', 'database'])
async def test_required_search_failure_is_503_without_private_detail(monkeypatch, stage, caplog):
    error = OSError('private-search-detail')
    monkeypatch.setattr(knowledge, '_ensure_vector_index', Mock(side_effect=error) if stage == 'index' else lambda: True)
    retrieve = Mock(side_effect=error)
    factory = Mock(side_effect=error) if stage == 'factory' else Mock(return_value=SimpleNamespace(retrieve_context=retrieve))
    monkeypatch.setattr(rag_helper, 'get_rag_helper', factory)
    options = {}
    if stage == 'database':
        monkeypatch.setattr(knowledge, 'db', SimpleNamespace(get_knowledge_bases=Mock(side_effect=error)))
        options['knowledgeBaseName'] = 'requested-library'
    with pytest.raises(HTTPException) as caught:
        await knowledge.search_knowledge(KnowledgeSearchRequest(query='档案馆开放时间', **options))
    assert caught.value.status_code == 503 and caught.value.__cause__ is error
    assert 'private-search-detail' not in str(caught.value.detail) + caplog.text
    assert retrieve.call_count == (1 if stage == 'retrieval' else 0)


@pytest.mark.asyncio
async def test_empty_completed_search_does_not_trigger_another_retriever(monkeypatch):
    monkeypatch.setattr(knowledge, '_ensure_vector_index', lambda: True)
    retrieve = Mock(return_value=[])
    monkeypatch.setattr(rag_helper, 'get_rag_helper', lambda: SimpleNamespace(retrieve_context=retrieve))
    from knowledge import vector_db
    vector = Mock(side_effect=AssertionError('unexpected second search'))
    monkeypatch.setattr(vector_db, 'get_vector_db', vector)
    result = await knowledge.search_knowledge(KnowledgeSearchRequest(query='档案馆开放时间'))
    assert result['success'] and result['results'] == [] and result['retrievalMode'] == 'empty'
    retrieve.assert_called_once()
    vector.assert_not_called()


@pytest.fixture
def search_hit(monkeypatch):
    row = knowledge._vector_chunk_document({
        "documentId": 37, "chunkIndex": 2, "doc_title": "完整办理规则",
        "content": "周五18点前提交原件，收到书面确认才算成功；编号不能证明已确认。",
        "doc_category": "规定", "doc_kb_id": 21,
    }, {21: "已授权资料库"})
    row["normalized_score"] = .75
    retrieve = Mock(return_value=[row])
    monkeypatch.setattr(knowledge, "_ensure_vector_index", lambda: True)
    monkeypatch.setattr(rag_helper, "get_rag_helper", lambda: SimpleNamespace(retrieve_context=retrieve))
    return row, retrieve


@pytest.mark.asyncio
async def test_result_uses_document_identity_and_complete_required_metadata(search_hit):
    row, retrieve = search_hit
    response = await knowledge.search_knowledge(KnowledgeSearchRequest(query="办理完整规则"))
    assert response["results"] == [{
        "documentId": 37, "documentTitle": row["title"], "chunkIndex": 2,
        "content": row["content"], "score": .75, "searchType": "rag_pipeline",
    }]
    assert response["retrievalMode"] == "evidence"
    retrieve.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["document_id", "normalized_score"])
async def test_incomplete_result_is_failure_without_substitute_identity_or_score(search_hit, missing):
    row, retrieve = search_hit
    del row[missing]
    with pytest.raises(HTTPException) as caught:
        await knowledge.search_knowledge(KnowledgeSearchRequest(query="办理完整规则"))
    assert caught.value.status_code == 503
    assert isinstance(caught.value.__cause__, KeyError) and caught.value.__cause__.args == (missing,)
    retrieve.assert_called_once()
