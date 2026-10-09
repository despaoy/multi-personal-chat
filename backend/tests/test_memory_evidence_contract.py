"""Malformed evidence is a protocol failure, distinct from unsupported evidence."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from repositories.character_memory import DatabaseCharacterMemoryRepository

from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig, parse_llm_memories
from character.models import UserScope
from db.database import SQLiteDB

SOURCE = '请记住，我喜欢红茶。'

def response(evidence='我喜欢红茶', omit=False):
    row = dict(operation='ADD', kind='like', value='红茶', content='用户喜欢红茶',
               attributed_to='user', confidence=.95, qualifiers={})
    if not omit:
        row['evidence'] = evidence
    return json.dumps({'memories': [row]}, ensure_ascii=False)

@pytest.mark.parametrize('evidence', [None, False, 0, 1, [], {}, '', '  '])
def test_malformed_evidence_is_explicit_failure(evidence):
    with pytest.raises(ValueError, match='evidence'):
        parse_llm_memories(response(evidence), source_message=SOURCE)

def test_missing_evidence_is_explicit_failure():
    with pytest.raises(ValueError, match='evidence'):
        parse_llm_memories(response(omit=True), source_message=SOURCE)

def test_valid_evidence_preserves_success_and_unsupported_evidence_is_rejected():
    assert len(parse_llm_memories(response(), source_message=SOURCE)) == 1
    assert parse_llm_memories(response('我喜欢咖啡'), source_message=SOURCE) == []

async def test_protocol_failure_reaches_receipt_without_claim_write(tmp_path):
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path/'evidence.sqlite'))
    scope = UserScope('web','evidence','fiction','fiction','private')
    completion = SimpleNamespace(complete=AsyncMock(return_value=response(None)),close=AsyncMock())
    worker = MemoryEnrichmentScheduler(config=MemoryLlmConfig(True,'unused','fixture'),completion=completion)
    try:
        result = await worker.schedule_and_wait(repository=repo,character_id='role',user_scope=scope,
            message=SOURCE,rule_hints=(),source_message_id='evidence-request')
        assert result['status']=='failed' and result['stage']=='proposal_validation'
        assert result['error']=='ValueError' and result['accepted']==result['persisted']==0
        completion.complete.assert_awaited_once()
        assert await repo.list_memory_records('role',scope)==[]
    finally:
        await worker.shutdown(timeout=1)
