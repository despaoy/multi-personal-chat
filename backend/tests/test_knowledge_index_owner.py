"""Document transactions own revisions; one rebuild path publishes the index."""
from unittest.mock import Mock

import numpy as np
import pytest

from api import knowledge
from app import config
from db.database import SQLiteDB
from db.schemas import KnowledgeDocumentUpdate
from knowledge import vector_db as vectors


@pytest.mark.parametrize('operation',['update','delete'])
@pytest.mark.parametrize('fail_rebuild',[False,True])
async def test_committed_document_change_uses_one_index_publication(tmp_path,monkeypatch,operation,fail_rebuild):
    db=SQLiteDB(tmp_path/'source.sqlite')
    monkeypatch.setattr(knowledge,'db',db)
    monkeypatch.setattr(knowledge,'VECTOR_DB_AVAILABLE',True)
    monkeypatch.setattr(config,'VECTOR_DB_AVAILABLE',True)
    monkeypatch.setattr(knowledge,'_vector_index_built',False)
    monkeypatch.setattr(knowledge,'_vector_index_revision',None)
    monkeypatch.setattr(vectors.VectorDatabase,'_check_gpu_availability',lambda self:False)
    vector=vectors.VectorDatabase(str(tmp_path/'vectors'))
    encode=Mock(side_effect=lambda texts:np.tile(np.eye(1,384,dtype=np.float32),(len(texts),1)))
    monkeypatch.setattr(vector,'_model',object())
    monkeypatch.setattr(vector,'_get_embeddings_batch',encode)
    monkeypatch.setattr(vectors,'get_vector_db',lambda:vector)
    forbidden=Mock(side_effect=AssertionError('CRUD must not run separate incremental indexing'))
    monkeypatch.setattr(config,'get_vector_db',forbidden)
    source='完整旧规则：10月12日14点在一楼办理，必须携带书面确认。'
    updated='完整新规则：10月19日15点在二楼办理，必须携带书面确认。'
    row=db.save_knowledge_document(dict(title='合成规则',content=source),chunks=[source])
    assert knowledge._ensure_vector_index()
    calls=encode.call_count
    revision=knowledge._get_rebuild_revision()
    if operation=='update':
        result=await knowledge.update_knowledge_document(row['id'],KnowledgeDocumentUpdate(content=updated),dict(role='admin'))
        assert ''.join(c['content'] for c in db.get_knowledge_chunks(row['id']))==updated
    else:
        result=await knowledge.delete_knowledge_document(row['id'],dict(role='admin'))
        assert db.get_knowledge_document(row['id']) is None
    assert result['success'] and knowledge._get_rebuild_revision()==revision+1
    assert knowledge._read_rebuild_status()[0]=='dirty' and not knowledge._vector_index_built
    forbidden.assert_not_called()
    assert encode.call_count==calls
    if fail_rebuild:
        monkeypatch.setattr(vector,'clear_all',Mock(side_effect=OSError('synthetic publication failure')))
        assert knowledge._ensure_vector_index() is False
        assert knowledge._read_rebuild_status()[0]!='complete' and not knowledge._vector_index_built
    else:
        assert knowledge._ensure_vector_index()
        assert knowledge._read_rebuild_status()[0]=='complete'
        reopened=vectors.VectorDatabase(str(tmp_path/'vectors'))
        assert reopened.metadata==vector.metadata and reopened.snapshot_validated
        assert reopened.index.ntotal==len(reopened.metadata)==(1 if operation=='update' else 0)
        if operation=='update':
            assert updated in reopened.metadata[0]['content'] and source not in reopened.metadata[0]['content']
            assert encode.call_count==calls+1
