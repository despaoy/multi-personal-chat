"""Persistence errors belong to each receipt, not a later global status read."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
from repositories.character_memory import DatabaseCharacterMemoryRepository

from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig
from character.models import UserScope
from db.database import SQLiteDB


@pytest.mark.parametrize('partial',[False,True])
async def test_write_error_is_scoped_to_receipt_and_preserves_commits(tmp_path,monkeypatch,caplog,partial):
    source='请记住，我叫林澈，我喜欢红茶。'
    def row(kind,value,evidence):
        return dict(operation='ADD',kind=kind,value=value,evidence=evidence,confidence=.98,attributed_to='user')
    first=json.dumps(dict(memories=[row('name','林澈','我叫林澈'),row('like','红茶','我喜欢红茶')]),ensure_ascii=False)
    second=json.dumps(dict(memories=[row('dislike','薄荷糖','我不喜欢薄荷糖')]),ensure_ascii=False)
    repo=DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path/'receipts.sqlite'))
    scope=UserScope('web','receipt','fiction','fiction','private')
    original=repo.append_claim
    private='synthetic-private-storage-detail'
    failing=True
    async def append(*args,**kwargs):
        if failing and (not partial or kwargs['memory_key']=='user_name'):
            raise OSError(private)
        return await original(*args,**kwargs)
    monkeypatch.setattr(repo,'append_claim',append)
    completion=SimpleNamespace(complete=AsyncMock(side_effect=[first,second]),close=AsyncMock())
    worker=MemoryEnrichmentScheduler(config=MemoryLlmConfig(True,'unused','fixture'),completion=completion,
        embedding_provider=SimpleNamespace(embed_texts=lambda texts:np.ones((len(texts),2),dtype=np.float32)))
    try:
        receipt=await worker.schedule_and_wait(repository=repo,character_id='role',user_scope=scope,
            message=source,rule_hints=(),source_message_id='first')
        assert receipt['status']==('partial' if partial else 'failed')
        assert receipt['error']=='OSError' and receipt['stage']=='persistence'
        assert receipt['accepted']==2 and receipt['persisted']==int(partial)
        assert receipt['operation_outcomes'].count('failed')==2-int(partial)
        records=await repo.list_memory_records('role',scope)
        assert len(records)==int(partial)
        if partial:
            assert records[0]['memory_key']=='preference_红茶'
        assert private not in json.dumps(receipt) and private not in caplog.text
        failing=False
        success=await worker.schedule_and_wait(repository=repo,character_id='role',user_scope=scope,
            message='请记住，我不喜欢薄荷糖。',rule_hints=(),source_message_id='second')
        assert success['status']=='saved' and success['persisted']==1 and 'error' not in success
        assert receipt['error']=='OSError'
    finally:
        await worker.shutdown(timeout=1)
