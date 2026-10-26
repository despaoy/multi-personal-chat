from types import SimpleNamespace
"""An unavailable index dependency cannot authorize a successful search."""

from knowledge.index_lifecycle import default_knowledge_index as index_lifecycle
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from api import knowledge
from db.schemas import KnowledgeSearchRequest


@pytest.mark.parametrize('stage', ['dependency', 'revision'])
async def test_index_preparation_failure_reaches_search_without_private_detail(monkeypatch, caplog, stage):
    private = 'synthetic-private-index-detail'
    monkeypatch.setattr(index_lifecycle, '_vector_index_revision', 7 if stage == 'revision' else None)
    monkeypatch.setattr(index_lifecycle, '_get_rebuild_revision', Mock(return_value=7))
    if stage == 'dependency':
        monkeypatch.setattr(index_lifecycle, '_vector_provider', Mock(side_effect=ImportError(private)))
    else:
        monkeypatch.setattr(index_lifecycle, '_vector_provider', lambda: SimpleNamespace(get_stats=lambda: dict(total_documents=0, index_size=0, bm25_corpus_size=0)))
        monkeypatch.setattr(index_lifecycle, '_get_rebuild_revision', Mock(side_effect=OSError(private)))
    with pytest.raises(HTTPException) as caught:
        await knowledge.search_knowledge(KnowledgeSearchRequest(query='完整合成规则的办理条件'), dict(role='admin'), database=index_lifecycle.db)
    assert caught.value.status_code == 503
    assert private not in str(caught.value.detail) and private not in caplog.text
    assert ('ImportError' if stage == 'dependency' else 'OSError') in caplog.text
    if stage == 'dependency':
        assert index_lifecycle._vector_index_revision is None
