"""Search results must describe the knowledge revision actually retrieved."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException
from api import knowledge
from db.schemas import KnowledgeSearchRequest
from knowledge.index_lifecycle import default_knowledge_index as index_lifecycle
from knowledge import rag_helper
from test_knowledge_cross_process_freshness import readers as readers


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['stable', 'update', 'delete', 'read_failure'])
async def test_search_checks_revision_and_recovers_on_next_request(readers, monkeypatch, change):
    reader, writer, document, _new_base, vector = readers
    old = document['content']
    new = '完整新规则：旧安排作废，周五提交电子材料，下周一凭书面确认领取。'
    read_revision = index_lifecycle._get_rebuild_revision
    failure = OSError('synthetic revision read failure')
    calls = []

    def retrieve(*args, **kwargs):
        calls.append(1)
        rows = [dict(deepcopy(row), normalized_score=0.9) for row in vector.metadata]
        if len(calls) == 1:
            if change == 'update':
                writer.save_knowledge_document(dict(content=new), doc_id=document['id'], chunks=[new])
            elif change == 'delete':
                writer.delete_knowledge_document(document['id'])
            elif change == 'read_failure':
                monkeypatch.setattr(index_lifecycle, '_get_rebuild_revision', Mock(side_effect=failure))
        return rows
    monkeypatch.setattr(rag_helper, 'get_rag_helper', lambda: SimpleNamespace(retrieve_context=retrieve))
    request = KnowledgeSearchRequest(query='完整办理条件是什么')
    try:
        if change == 'stable':
            response = await knowledge.search_knowledge(request, dict(role='admin'), database=index_lifecycle.db)
            assert response['success'] and old in response['results'][0]['content']
        else:
            with pytest.raises(HTTPException) as caught:
                await knowledge.search_knowledge(request, dict(role='admin'), database=index_lifecycle.db)
            assert caught.value.status_code == 503
            if change == 'read_failure':
                assert caught.value.__cause__ is failure
                monkeypatch.setattr(index_lifecycle, '_get_rebuild_revision', read_revision)
            else:
                assert str(caught.value.__cause__) == 'Knowledge authority changed during search'
            assert len(calls) == 1
            response = await knowledge.search_knowledge(request, dict(role='admin'), database=index_lifecycle.db)
            assert response['success'] and len(calls) == 2
            if change == 'delete':
                assert response['results'] == [] and response['retrievalMode'] == 'empty'
            else:
                body = response['results'][0]['content']
                assert (new if change == 'update' else old) in body
                if change == 'update':
                    assert old not in body
    finally:
        reader.close_connection()
        writer.close_connection()
