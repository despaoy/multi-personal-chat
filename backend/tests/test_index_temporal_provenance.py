from types import SimpleNamespace

import pytest

from knowledge.multiscale_rag.index_builder import CharacterKnowledgeIndexBuilder
from knowledge.retrieval_core.documents import KnowledgeIndexDocument


@pytest.mark.parametrize('scope', ['current', 'unknown', 'flashback', 'reconstruction', 'hypothetical'])
@pytest.mark.parametrize('text', ['他们现在仍是儿时好友', '他否认小时候见过她', '这是一本名为童年的书', '当年'])
def test_index_builder_never_relabels_source_time_from_past_words(scope, text):
    source = KnowledgeIndexDocument(
        id='relation', domain_id='test', document_type='relation', title=text,
        summary=text, content='保留原文证据', embedding_text='', temporal_scope=scope,
        metadata={'scale': 'card', 'subject': '人物甲', 'target': '人物乙', 'relation': text})
    builder = object.__new__(CharacterKnowledgeIndexBuilder)
    builder.hierarchy_builder = SimpleNamespace(build=lambda *_: SimpleNamespace(
        documents=(source,), counts={'relation': 1}, exact_evidence_matches=1))
    result = builder.build(None, None)
    built, = result.documents
    assert built.temporal_scope == scope
    assert built.metadata['source_temporal_scope'] == scope
    assert built.metadata['semantic_temporal_scope'] == scope
    assert f'时间：{scope}' in built.embedding_text
    assert source.temporal_scope == scope
    assert 'source_temporal_scope' not in source.metadata
    assert built.content == source.content
