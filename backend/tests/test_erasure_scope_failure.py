from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from repositories.character_memory import DatabaseCharacterMemoryRepository

from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig
from character.memory_operation import operation_receipt_context
from character.models import MemoryItem, UserScope
from db.database import SQLiteDB


@pytest.mark.asyncio
async def test_unresolved_mixed_erasure_fails_before_model_and_preserves_all_claims(tmp_path):
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / 'scope.sqlite'))
    scope = UserScope('test', 'erasure', 'reader', 'reader', 'private')
    for key, content in [('fact_pet', '用户的猫叫豆沙、糯米'), ('user_major', '用户的大学专业是海洋工程')]:
        await repo.append_claim('role', scope, MemoryItem('', 'user_fact', content, .9), memory_key=key)
    before = await repo.list_memory_records('role', scope, include_inactive=True)
    completion = SimpleNamespace(complete=AsyncMock(), close=AsyncMock())
    worker = MemoryEnrichmentScheduler(config=MemoryLlmConfig(True, 'unused', 'fixture'), completion=completion,
        embedding_provider=SimpleNamespace(embed_texts=lambda texts: [[1., 0.] for _ in texts]))
    try:
        result = await worker.schedule_and_wait(repository=repo, character_id='role', user_scope=scope,
            message='请忘掉我的猫名字和新增猫的记录，保留大学专业。', rule_hints=[], source_message_id='erase', timeout_seconds=2)
        assert result['status'] == 'failed' and result['reason'] == 'unresolved_erasure_scope'
        assert result['stage'] == 'erasure_authorization' and result['persisted'] == 0
        completion.complete.assert_not_awaited()
        assert await repo.list_memory_records('role', scope, include_inactive=True) == before
        assert await repo.list_sources('role', scope) == []
        context = operation_receipt_context(result)
        assert '未能确认删除与保留范围' in context and '未执行删除' in context and '提供相关原话' in context
    finally:
        await worker.shutdown(timeout=1)


def test_raw_reason_content_cannot_become_operation_context():
    context = operation_receipt_context(dict(status='failed', reason='private-untrusted-detail', persisted=0))
    assert 'private-untrusted-detail' not in context and '操作未成功完成' in context
