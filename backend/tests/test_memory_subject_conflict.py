import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from repositories.character_memory import DatabaseCharacterMemoryRepository

from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig, parse_llm_proposals
from character.memory_subject import claims_other_subject_as_user, explicitly_other_subject
from character.models import UserScope
from db.database import SQLiteDB

SOURCE = '我的大学专业是物理学，不是考古学；学考古学的是我的朋友。'


def candidate(content):
    return dict(kind='other_user_fact', value='考古学', evidence='学考古学的是我的朋友',
                content=content, confidence=.95, operation='ADD')


@pytest.mark.parametrize('predicate,value', [('学', '考古学'), ('喜欢', '咖啡'), ('讨厌', '红茶')])
@pytest.mark.parametrize('owner', ['我的朋友', '我妹妹', '她'])
def test_inverse_subject_keeps_original_actor(predicate, value, owner):
    source = f'{predicate}{value}的是{owner}。'
    assert explicitly_other_subject(source=source, evidence=value, value=value)


@pytest.mark.parametrize('source', ['学考古学的是我。', '学考古学的不是我的朋友。',
                                  '学考古学的是我，我的朋友学习考古学。'])
def test_self_negated_or_mixed_subject_is_not_proof_of_exclusive_other_owner(source):
    assert not explicitly_other_subject(source=source, evidence=source.rstrip('。'), value='考古学')


@pytest.mark.parametrize('content,conflict', [('用户学习考古学', True),
    ('用户的专业是考古学', True), ('用户的朋友学习考古学', False)])
def test_generic_summary_cannot_transfer_known_friend_fact_to_user(content, conflict):
    assert claims_other_subject_as_user(source=SOURCE, evidence='学考古学的是我的朋友',
        value='考古学', content=content) is conflict
    raw = json.dumps({'memories': [candidate(content)]})
    if conflict:
        with pytest.raises(ValueError, match='主体'):
            parse_llm_proposals(raw, source_message=SOURCE)
    else:
        result = parse_llm_proposals(raw, source_message=SOURCE)
        assert len(result) == 1 and result[0].memory.content == content


@pytest.mark.asyncio
async def test_subject_conflict_reports_failure_before_any_fact_is_written(tmp_path, caplog):
    database = SQLiteDB(tmp_path / 'subject.sqlite')
    repo = DatabaseCharacterMemoryRepository(database)
    scope = UserScope('web', 'subject-contract', 'reader', 'reader', 'private')
    valid = dict(kind='major', value='物理学', evidence='我的大学专业是物理学',
                 content='用户的大学专业是物理学', confidence=.95, operation='ADD')
    raw = json.dumps({'memories': [valid, candidate('用户学习考古学')]})
    completion = SimpleNamespace(complete=AsyncMock(return_value=raw), close=AsyncMock())
    worker = MemoryEnrichmentScheduler(config=MemoryLlmConfig(True, 'unused', 'fixture'), completion=completion)
    try:
        receipt = await worker.schedule_and_wait(repository=repo, character_id='role', user_scope=scope,
            message=SOURCE, rule_hints=(), source_message_id='source')
        assert receipt['status'] == 'failed' and receipt['stage'] == 'proposal_validation'
        assert receipt['source_capture'] == 'recorded' and receipt['error'] == 'ValueError'
        assert receipt['accepted'] == receipt['persisted'] == 0
        assert await repo.list_memory_records('role', scope) == []
        assert (await repo.list_sources('role', scope, source_message_ids=('source',)))[0]['body'] == SOURCE
        assert '考古学' not in json.dumps(receipt) and '考古学' not in caplog.text
    finally:
        await worker.shutdown(timeout=1)
        database.close_connection()
