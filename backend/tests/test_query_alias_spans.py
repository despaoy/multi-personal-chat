from knowledge.retrieval_core.loaders import AliasEntityNormalizer
from knowledge.retrieval_core.query import QueryAnalyzer
from knowledge.retrieval_core.registry import KnowledgeDomainConfig


def analyzer(tmp_path, aliases):
    return QueryAnalyzer([KnowledgeDomainConfig('test', tmp_path, lambda _: [], aliases=aliases)])


def test_embedded_short_name_is_not_a_second_person(tmp_path):
    aliases = {'安宁': '安宁', '宁': '宁老师'}
    query = '安宁是谁？'
    result = analyzer(tmp_path, aliases).analyze(query)
    assert result.entities == AliasEntityNormalizer(aliases).scan_text(query) == ['安宁']
    assert result.normalized_query == query


def test_replacement_text_is_not_reinterpreted_as_another_alias(tmp_path):
    result = analyzer(tmp_path, {'小林': '林远', '远': '远老师'}).analyze('小林是谁？')
    assert result.normalized_query == '林远是谁？'
    assert result.entities == ['林远']


def test_short_alias_outside_long_name_still_counts(tmp_path):
    result = analyzer(tmp_path, {'安宁': '安宁', '宁': '宁老师'}).analyze('安宁和宁是什么关系？')
    assert result.entities == ['安宁', '宁老师']
    assert result.normalized_query == '安宁和宁老师是什么关系？'


def test_repeated_and_adjacent_mentions_are_preserved(tmp_path):
    result = analyzer(tmp_path, {'小林': '林远', '阿周': '周明'}).analyze('小林阿周，小林认识阿周吗？')
    assert set(result.entities) == {'林远', '周明'}
    assert result.normalized_query == '林远周明，林远认识周明吗？'


def test_alias_expansion_cannot_duplicate_canonical_prefix(tmp_path):
    result = analyzer(tmp_path, {'Alex': 'Alex Smith'}).analyze('Alex Smith是谁？')
    assert result.normalized_query == 'Alex Smith是谁？'


def test_unknown_query_remains_unchanged(tmp_path):
    result = analyzer(tmp_path, {'小林': '林远'}).analyze('今天下雨了吗？')
    assert not result.entities
    assert result.normalized_query == result.original_query


def test_configured_domain_aliases_preserve_identity_subjects():
    from knowledge.entity_scope import explicit_identity_subject
    from knowledge.retrieval_core.registry import get_default_registry

    config = get_default_registry().require('tsukiyashiro_kisaki')
    query_analyzer = QueryAnalyzer([config])
    for alias, canonical in config.aliases.items():
        for suffix in ('是谁？', '是什么人？', '的身份是什么？'):
            result = query_analyzer.analyze(alias + suffix, domain_id=config.domain_id)
            assert result.normalized_query == canonical + suffix, (alias, result)
            assert explicit_identity_subject(result) == canonical, (alias, result)


def test_equivalent_identity_forms_search_the_same_partition(tmp_path):
    from knowledge.multiscale_rag.service import choose_card_types

    query_analyzer = analyzer(tmp_path, {'小林': '林远', '林远': '林远'})
    for name in ('小林', '林远'):
        for suffix in ('是谁？', '是什么人？', '是什么身份？', '的身份是什么？'):
            query = name + suffix
            result = query_analyzer.analyze(query)
            assert choose_card_types(result, query) == frozenset({'fact', 'relation'})
