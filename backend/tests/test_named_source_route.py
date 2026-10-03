"""Named document tasks own their source route despite character topic words."""
from threading import RLock
from types import SimpleNamespace

import pytest

from knowledge.source_expansion import requested_document_titles


@pytest.mark.parametrize('query', [
    '请核对演练场景中《合成偏好》的当前修订：月社妃当前和上一版偏好分别是什么？',
    '核查《合成偏好》的末尾限制。',
    '请查看场景中《合成偏好》的两个版本。',
    '查阅《合成偏好》。',
    '读取《合成偏好》全文。',
])
def test_complete_named_read_with_scope_is_a_source_request(query):
    assert requested_document_titles(query) == ('合成偏好',)


@pytest.mark.parametrize('query', [
    '朋友让我核对《合成偏好》，我只在转述这件事。',
    '不要核对《合成偏好》。',
    '请核对场景中《合成偏好》，别读取这份。',
    '请核对场景中《合成偏好》，再核对《另一个标题》。',
    '请核对' + '很长的未知范围' * 20 + '中《合成偏好》。',
])
def test_background_negative_or_incomplete_named_read_does_not_grant_a_route(query):
    assert requested_document_titles(query) == ()


async def test_named_role_document_uses_fresh_generic_authority_and_complete_body(monkeypatch):
    from api import generate, knowledge
    from knowledge import rag_helper, vector_db
    from knowledge.multiscale_rag import runtime

    body = '合成角色当前偏好甲；上一版本乙仅为历史；不证明已经操作。'
    record = dict(id='doc_1_chunk_0', document_id=1, chunk_index=0, title='合成偏好',
                  category='演练', knowledge_base_id=7, content=body)
    indexed = SimpleNamespace(_lock=RLock(), cache_generation=11, snapshot_validated=True,
                              metadata=[record])
    calls = []

    def character_forbidden():
        pytest.fail('Named document task must not first load an unrelated curated index')

    monkeypatch.setattr(runtime, 'get_multiscale_rag_service', character_forbidden)
    monkeypatch.setattr(knowledge, '_ensure_vector_index', lambda: calls.append('fresh') or True)
    monkeypatch.setattr(knowledge, '_vector_index_revision', 77)
    monkeypatch.setattr(knowledge, '_get_rebuild_revision', lambda: 77)
    monkeypatch.setattr(vector_db, 'get_vector_db', lambda: indexed)
    monkeypatch.setattr(generate, 'db', SimpleNamespace(get_knowledge_document=lambda identity:
        dict(id=identity, title='合成偏好', category='演练', knowledge_base_id=7, content=body)))
    monkeypatch.setenv('CORRECTIVE_RAG_ENABLED', 'false')

    def retrieve(query, **kwargs):
        calls.append('generic')
        assert '月社妃' in query and kwargs['filters'] is None
        return dict(results=[record], confidence=0.8, abstained=False)

    monkeypatch.setattr(rag_helper, 'get_rag_helper', lambda: SimpleNamespace(
        retrieve_with_citations=retrieve, build_citations=lambda records: []))
    result = await generate._retrieve_rag_bundle(
        '请核对演练场景中《合成偏好》的当前修订：月社妃当前最喜欢哪个方案？', 3, None)
    assert calls == ['fresh', 'generic']
    assert result['requested_source_titles'] == ['合成偏好']
    assert result['original_source_packets'][0]['original_body'] == body
    assert result['source_coverage'][0]['original_source_receipt']['authority_revision'] == 77


async def test_named_read_with_unready_generic_index_fails_without_curated_substitution(monkeypatch):
    from api import generate, knowledge
    from knowledge.multiscale_rag import runtime

    monkeypatch.setattr(knowledge, '_ensure_vector_index', lambda: False)
    monkeypatch.setattr(runtime, 'get_multiscale_rag_service', lambda:
        pytest.fail('A missing named document index must not substitute curated character material'))
    with pytest.raises(RuntimeError, match='Generic knowledge index is not ready'):
        await generate._retrieve_rag_bundle('核对《合成偏好》中月社妃的记录。', 3, None)
