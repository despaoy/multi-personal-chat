from types import SimpleNamespace

import pytest

from knowledge.relation_scope import explicit_relation_pair, in_relation_scope


@pytest.mark.parametrize('names', [('人物甲', '人物乙'), ('莉娜', '亚历克斯')])
@pytest.mark.parametrize('format', ['说说{a}和{b}的关系。', '{b}与{a}是什么关系？',
                                   '{a}跟{b}之间到底有什么关系？',
                                   '{a}和{b}在原作里是什么关系？',
                                   '请解释{a}与{b}在小说中的关系。',
                                   '{a}跟{b}在作品里是什么关系？我画过他们的同人。'])
def test_direct_pair_with_different_names(names, format):
    a, b = names
    assert explicit_relation_pair(SimpleNamespace(entities=names, normalized_query=format.format(a=a, b=b))) == set(names)


@pytest.mark.parametrize('query', [
    '人物甲的哥哥和人物乙是什么关系？',
    '人物甲和人物乙的关系为什么变差？',
    '人物甲和人物乙的关系如何影响其他人？',
    '人物甲和人物乙各自有哪些朋友？',
    '人物甲与人物乙共同认识谁？',
    '人物甲和人物乙在原作里各自有哪些朋友？',
    '人物甲和人物乙在故事中通过谁建立关系？',
    '人物甲和人物乙在原著里的关系为什么改变？',
])
def test_bridge_causal_and_open_questions_are_not_direct_pairs(query):
    assert not explicit_relation_pair(SimpleNamespace(entities=['人物甲', '人物乙'], normalized_query=query))


def test_structured_endpoints_not_incidental_mentions_control_scope():
    pair = frozenset({'a', 'b'})
    doc = SimpleNamespace(document_type='relation', entities=['a', 'b', 'c'],
                          metadata={'subject': 'a', 'target': 'c'})
    assert not in_relation_scope(doc, pair)
    doc.metadata['target'] = 'b'
    assert in_relation_scope(doc, pair)
    doc.metadata = {}
    assert in_relation_scope(doc, pair)  # Unknown legacy data is not fabricated.


def test_multi_party_query_is_not_forced_into_one_pair():
    assert not explicit_relation_pair(SimpleNamespace(entities=['a', 'b', 'c'], normalized_query='a和b的关系与c有关吗？'))


def test_aliases_are_normalized_before_matching_endpoints(tmp_path):
    from knowledge.retrieval_core.query import QueryAnalyzer
    from knowledge.retrieval_core.registry import KnowledgeDomainConfig

    config = KnowledgeDomainConfig('transfer', tmp_path, lambda _: [],
        aliases={'小林': '林远', '林远': '林远', '小安': '安宁', '安宁': '安宁'})
    analysis = QueryAnalyzer([config]).analyze('小林与小安是什么关系？')
    pair = explicit_relation_pair(analysis)
    assert pair == {'林远', '安宁'}
    reverse = SimpleNamespace(document_type='relation', metadata={'subject': '安宁', 'target': '林远'})
    assert in_relation_scope(reverse, pair)


@pytest.mark.parametrize('query,scoped', [('a和b是什么关系？', True), ('a和b的关系为什么改变？', False),
                                       ('a和b在原作里是什么关系？画他们的是谁？', True)])
@pytest.mark.parametrize('include_direct', [True, False])
def test_scope_precedes_reranking_and_applies_to_timeline(monkeypatch, query, scoped, include_direct):
    from knowledge.multiscale_rag import service
    from knowledge.retrieval_core.documents import KnowledgeIndexDocument
    from knowledge.retrieval_core.retrieval import RetrievalCandidate

    def doc(key, target):
        return KnowledgeIndexDocument(key, 'test', 'relation', key, key, key, key,
                                      entities=['a', 'b', 'c'], metadata={'subject': 'a', 'target': target})

    direct, unrelated = doc('direct', 'b'), doc('unrelated', 'c')
    candidates = [RetrievalCandidate(0, unrelated)]
    if include_direct:
        candidates.append(RetrievalCandidate(1, direct))
    instance = object.__new__(service.RoutedMultiScaleService)
    instance.identity_coverage = False
    instance.config = SimpleNamespace(domain_id='test')
    instance.analyzer = None
    instance.indexes = {frozenset({'relation'}): SimpleNamespace(documents=[c.document for c in candidates])}
    instance.retrievers = {frozenset({'relation'}): SimpleNamespace(search=lambda *a, **kw: candidates)}
    instance.reranker = None
    instance.extractor = None
    instance.by_id = {}
    instance.evidence_by_parent = {}
    monkeypatch.setattr(service, 'analyze_explicit_domain', lambda *a: SimpleNamespace(
        entities=['a', 'b'], normalized_query=query))
    monkeypatch.setattr(service, 'choose_card_types', lambda *a: frozenset({'relation'}))

    def rank(_analysis, recalled, **kwargs):
        assert len(recalled) == int(include_direct) + int(not scoped)
        return recalled

    monkeypatch.setattr(service, 'rerank_with_title_frames', rank)
    result = instance.retrieve(query)
    expected = ({'direct'} if include_direct else set()) | (set() if scoped else {'unrelated'})
    assert {d['id'] for d in result['results']} == expected
    assert {d['id'] for d in result['relation_timeline']} == expected
    assert result['relation_scope']['excluded_candidates'] == int(scoped)
    if scoped:
        assert 'unrelated' not in result['context_text']
