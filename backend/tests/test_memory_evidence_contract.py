"""Declared evidence must be grounded before semantic admission."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from repositories.character_memory import DatabaseCharacterMemoryRepository

from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig, MemoryResponseError, parse_llm_proposals
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
        parse_llm_proposals(response(evidence), source_message=SOURCE)

def test_missing_evidence_is_explicit_failure():
    with pytest.raises(ValueError, match='evidence'):
        parse_llm_proposals(response(omit=True), source_message=SOURCE)

def test_valid_evidence_preserves_success_and_unsupported_evidence_is_rejected():
    assert len(parse_llm_proposals(response(), source_message=SOURCE)) == 1
    with pytest.raises(MemoryResponseError) as caught:
        parse_llm_proposals(response('我喜欢咖啡'), source_message=SOURCE)
    assert caught.value.field == 'evidence'


@pytest.mark.parametrize('operation,confidence', [('ADD', .1), ('NOOP', .95)])
def test_invalid_quote_cannot_hide_behind_admission_or_noop(operation, confidence):
    payload = json.loads(response('我喜欢咖啡'))
    payload['memories'][0].update(operation=operation, confidence=confidence)
    with pytest.raises(MemoryResponseError) as caught:
        parse_llm_proposals(json.dumps(payload), source_message=SOURCE)
    assert caught.value.field == 'evidence'


def test_literal_evidence_keeps_whitespace_contract_and_normal_nonadmission():
    payload = json.loads(response('我喜欢\n红茶'))
    (proposal,) = parse_llm_proposals(json.dumps(payload), source_message=SOURCE)
    assert proposal.evidence == '我喜欢 红茶'
    payload['memories'][0]['confidence'] = .1
    assert parse_llm_proposals(json.dumps(payload), source_message=SOURCE) == []
    assert parse_llm_proposals('{"memories": []}', source_message=SOURCE) == []

@pytest.mark.parametrize('evidence', [None, '我喜欢咖啡'])
async def test_protocol_failure_reaches_receipt_without_claim_write(tmp_path, evidence):
    database = SQLiteDB(tmp_path/'evidence.sqlite')
    repo = DatabaseCharacterMemoryRepository(database)
    scope = UserScope('web','evidence','fiction','fiction','private')
    completion = SimpleNamespace(complete=AsyncMock(return_value=response(evidence)),close=AsyncMock())
    worker = MemoryEnrichmentScheduler(config=MemoryLlmConfig(True,'unused','fixture'),completion=completion)
    try:
        result = await worker.schedule_and_wait(repository=repo,character_id='role',user_scope=scope,
            message=SOURCE,rule_hints=(),source_message_id='evidence-request')
        assert result['status']=='failed' and result['stage']=='proposal_validation'
        assert result['error']=='MemoryResponseError' and result['accepted']==result['persisted']==0
        assert result['reason'] == 'invalid_model_response' and result['field'] == 'evidence'
        assert result['source_capture'] == 'recorded' and result['retryable'] is False
        assert '我喜欢咖啡' not in json.dumps(result, ensure_ascii=False)
        completion.complete.assert_awaited_once()
        assert await repo.list_memory_records('role',scope)==[]
    finally:
        await worker.shutdown(timeout=1)
        database.close_connection()
