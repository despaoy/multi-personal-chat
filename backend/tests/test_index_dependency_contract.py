"""An unavailable index dependency cannot authorize a successful search."""
import builtins
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from api import knowledge
from app import config
from db.schemas import KnowledgeSearchRequest


@pytest.mark.parametrize('stage',['dependency','revision'])
async def test_index_preparation_failure_reaches_search_without_private_detail(monkeypatch,caplog,stage):
    private='synthetic-private-index-detail'
    monkeypatch.setattr(knowledge,'_vector_index_built',stage=='revision')
    monkeypatch.setattr(knowledge,'_vector_index_revision',7 if stage=='revision' else None)
    monkeypatch.setattr(knowledge,'_get_rebuild_revision',Mock(return_value=7))
    if stage=='dependency':
        monkeypatch.setattr(config,'VECTOR_DB_AVAILABLE',False)
        original=builtins.__import__
        def unavailable(name,*args,**kwargs):
            if name=='knowledge.vector_db':
                raise ImportError(private)
            return original(name,*args,**kwargs)
        monkeypatch.setattr(builtins,'__import__',unavailable)
    else:
        monkeypatch.setattr(knowledge,'_get_rebuild_revision',Mock(side_effect=OSError(private)))
    with pytest.raises(HTTPException) as caught:
        await knowledge.search_knowledge(KnowledgeSearchRequest(query='完整合成规则的办理条件'),dict(role='admin'))
    assert caught.value.status_code==503
    assert private not in str(caught.value.detail) and private not in caplog.text
    assert ('ImportError' if stage=='dependency' else 'OSError') in caplog.text
    if stage=='dependency':
        assert not knowledge._vector_index_built and knowledge._vector_index_revision is None
