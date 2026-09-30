from types import SimpleNamespace

import pytest

from knowledge.entity_scope import explicit_identity_subject, in_identity_scope


@pytest.mark.parametrize('name', ['人物甲', '林远', 'Alex'])
@pytest.mark.parametrize('form', ['{x}是谁？', '请问{x}是什么人？', '{x}的身份是什么。'])
def test_identity_scope_is_not_character_specific(name, form):
    analysis = SimpleNamespace(entities=[name], normalized_query=form.format(x=name))
    assert explicit_identity_subject(analysis) == name


@pytest.mark.parametrize('query', ['林远的哥哥是谁？', '林远是谁派来的？', '林远是谁？以及谁帮助他？',
                                  '为什么林远隐瞒身份？', '林远的身份怎么影响安宁？'])
def test_bridge_causal_and_mixed_queries_keep_open_scope(query):
    assert not explicit_identity_subject(SimpleNamespace(entities=['林远'], normalized_query=query))


def test_mentions_do_not_establish_ownership():
    unrelated = SimpleNamespace(document_type='fact', entities=['林远', '安宁'], metadata={'subject': '安宁'})
    assert not in_identity_scope(unrelated, '林远')
    unrelated.metadata = {}
    assert in_identity_scope(unrelated, '林远')
    relation = SimpleNamespace(document_type='relation', metadata={'subject': '安宁', 'target': '林远'})
    assert in_identity_scope(relation, '林远')


@pytest.mark.parametrize('query', [
    '我只想聊聊自己的感受，林远是谁？',
    '不要给我建议，林远是什么人？',
    '林远是谁，不要分析。',
])
def test_identity_task_survives_conversation_controls(query):
    assert explicit_identity_subject(SimpleNamespace(entities=['林远'], normalized_query=query)) == '林远'


@pytest.mark.parametrize('query', [
    '我刚才说了什么，林远是谁？',
    '不要分析这个故事，林远是谁？',
    '我只想聊聊自己的感受，林远是谁，告诉我他的经历。',
    '请翻译“不要分析，林远是谁？”',
    '我只想聊聊自己的感受，林远的哥哥是谁？',
    '不要分析？林远是谁？',
    '我只想聊聊自己的感受，如果林远是谁会怎样？',
])
def test_control_projection_never_drops_other_content_tasks(query):
    assert not explicit_identity_subject(SimpleNamespace(entities=['林远'], normalized_query=query))


def test_alias_normalization_and_multiple_targets(tmp_path):
    from knowledge.retrieval_core.query import QueryAnalyzer
    from knowledge.retrieval_core.registry import KnowledgeDomainConfig

    config = KnowledgeDomainConfig('transfer', tmp_path, lambda _: [],
                                   aliases={'小林': '林远', '林远': '林远'})
    analysis = QueryAnalyzer([config]).analyze('小林是谁？')
    assert explicit_identity_subject(analysis) == '林远'
    assert not explicit_identity_subject(SimpleNamespace(
        entities=['林远', '安宁'], normalized_query='林远和安宁是什么人？'))


@pytest.mark.parametrize('query', ['林远是谁？', '我只想聊聊自己的感受，林远是谁？',
                                 '先说我的专业，再告诉我林远是谁。'])
def test_filter_runs_before_ranking_and_preserves_retrieval_diagnostics(monkeypatch, query):
    from knowledge.multiscale_rag import service
    from knowledge.retrieval_core.documents import KnowledgeIndexDocument
    from knowledge.retrieval_core.retrieval import RetrievalCandidate

    direct = KnowledgeIndexDocument('direct', 'test', 'fact', '林远身份', '学生', '学生', '学生',
                                     metadata={'subject': '林远'})
    unrelated = KnowledgeIndexDocument('other', 'test', 'fact', '安宁身份', '医生', '林远认识医生', '医生',
                                        entities=['林远', '安宁'], metadata={'subject': '安宁'})
    candidates = [RetrievalCandidate(0, unrelated), RetrievalCandidate(1, direct)]
    instance = object.__new__(service.RoutedMultiScaleService)
    instance.identity_coverage = False
    instance.config = SimpleNamespace(domain_id='test')
    instance.analyzer = instance.reranker = instance.extractor = None
    instance.indexes = {}
    instance.by_id = instance.evidence_by_parent = {}
    instance.retrievers = {service.CARD_TYPES: SimpleNamespace(search=lambda *a, **kw: candidates)}
    monkeypatch.setattr(service, 'analyze_explicit_domain', lambda *a: SimpleNamespace(
        entities=['林远'], normalized_query=query))
    monkeypatch.setattr(service, 'choose_card_types', lambda *a: service.CARD_TYPES)

    def rank(_analysis, recalled, **kwargs):
        assert [c.document.id for c in recalled] == ['direct']
        return recalled

    monkeypatch.setattr(service, 'rerank_with_title_frames', rank)
    result = instance.retrieve(query)
    assert result['identity_scope'] == {'subject': '林远', 'excluded_candidates': 1}
    assert result['identity_task'] == ({} if '我的专业' in query else {'query': query, 'subject': '林远'})
    assert result['relation_scope']['excluded_candidates'] == 0
    assert [c['id'] for c in result['citations']] == ['direct']
    assert '医生' not in result['context_text']
