"""Complete destructive-operation evidence cannot be silently discarded at Top-K."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from repositories.character_memory import DatabaseCharacterMemoryRepository

from character.erasure_authority import partial_erasure_plan
from character.evidence_selector import InputBudgetError
from character.memory_llm import (
    MAX_EXISTING_MEMORIES,
    MemoryEnrichmentScheduler,
    MemoryLlmConfig,
    _search_existing_memories,
)
from character.models import UserScope
from db.database import SQLiteDB


def selection(count):
    rows, clauses = [], []
    for i in range(count):
        text = f'这是合成来源编号{i:02d}的完整第一句，仅描述一次虚构工坊记录。'
        sid = f'source-{i}'
        rows.append(dict(id=str(i), memory_key=f'event_{i}', memory_type='shared_event', status='active',
            content=text, observed_at='2026-10-09T08:00:00+00:00', source_message_id=sid,
            source_message_ids=[sid], evidence=[text], metadata=dict(content_semantics='quoted_source',
                speaker_role='user', described_subject='not_resolved', erasure_evidence_projection=dict(
                    version=1, kind='original_source_fragments', complete_original_source=False,
                    sources=[dict(source_message_id=sid,spans=[[0,len(text)]])]))))
        action = '从长期记忆中删除' if i == 0 else '保留'
        clauses.append(f'{action}原话以“{text}”开头的记录。')
    return tuple(rows), ''.join(clauses)

@pytest.mark.parametrize('count', [MAX_EXISTING_MEMORIES, MAX_EXISTING_MEMORIES+1])
def test_complete_required_selection_has_explicit_capacity_boundary(count):
    rows, message = selection(count)
    plan = partial_erasure_plan(message, rows)
    assert plan.valid and not plan.unresolved_protection
    assert len(plan.allowed_ids) == 1 and len(plan.protected_ids) == count-1
    provider = SimpleNamespace(embed_texts=Mock(side_effect=AssertionError('ranking cannot drop required evidence')))
    if count > MAX_EXISTING_MEMORIES:
        with pytest.raises(InputBudgetError):
            _search_existing_memories(rows, message, (), (), provider)
    else:
        assert _search_existing_memories(rows, message, (), (), provider) == rows
    provider.embed_texts.assert_not_called()

async def test_capacity_failure_reaches_receipt_before_model_or_write(tmp_path, monkeypatch):
    rows, message = selection(MAX_EXISTING_MEMORIES+1)
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path/'capacity.sqlite'))
    scope = UserScope('web','capacity','fiction','fiction','private')
    original_read = repo.list_memory_records
    monkeypatch.setattr(repo,'list_memory_records',AsyncMock(return_value=list(rows)))
    completion = SimpleNamespace(complete=AsyncMock(side_effect=AssertionError('over-budget erasure must not call model')),close=AsyncMock())
    worker = MemoryEnrichmentScheduler(config=MemoryLlmConfig(True,'unused','fixture'),completion=completion)
    try:
        result = await worker.schedule_and_wait(repository=repo,character_id='role',user_scope=scope,
            message=message,rule_hints=(),source_message_id='capacity-request')
        assert result['status']=='failed' and result['stage']=='memory_search'
        assert result['reason']=='input_budget' and result['error']=='InputBudgetError'
        assert result['accepted']==result['persisted']==0
        completion.complete.assert_not_awaited()
        assert await original_read('role',scope)==[]
    finally:
        await worker.shutdown(timeout=1)
