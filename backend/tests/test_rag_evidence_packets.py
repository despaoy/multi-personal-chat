"""Claim evidence reaches ordinary generation whole, ahead of broad background."""
from types import SimpleNamespace

from knowledge.multiscale_rag import service
from knowledge.retrieval_core.documents import SourceReference


def make_service(monkeypatch, evidence_texts):
    source = SourceReference('test.txt', 1, 24)
    docs = [SimpleNamespace(id=str(i), document_type='fact', title=f'事实{i}',
                            summary='概括', metadata={'scene_id': 'scene'}, source=source)
            for i in range(len(evidence_texts))]
    candidates = [SimpleNamespace(document=d, to_dict=lambda: {}) for d in docs]
    instance = object.__new__(service.RoutedMultiScaleService)
    instance.context_max_chars = 6000
    instance.identity_coverage = False
    instance.config = SimpleNamespace(domain_id='test')
    instance.analyzer = instance.reranker = instance.extractor = None
    instance.indexes = {}
    instance.retrievers = {service.CARD_TYPES: SimpleNamespace(search=lambda *a, **kw: candidates)}
    instance.by_id = {'scene': SimpleNamespace(id='scene', title='场景', summary='背景' * 1500)}
    instance.evidence_by_parent = {str(i): SimpleNamespace(content=text)
                                   for i, text in enumerate(evidence_texts) if text is not None}
    monkeypatch.setattr(service, 'analyze_explicit_domain', lambda *a: SimpleNamespace(entities=[]))
    monkeypatch.setattr(service, 'choose_card_types', lambda *a: service.CARD_TYPES)
    monkeypatch.setattr(service, 'rerank_with_title_frames', lambda *a, **kw: candidates)
    return instance


def test_ordinary_answer_gets_complete_claim_evidence_before_background(monkeypatch):
    instance = make_service(monkeypatch, ['承诺将来照顾。\n但还没有发生。', '另一事件。'])
    result = instance.retrieve('解释人物经历')
    text = result['context_text']
    assert '承诺将来照顾。\n但还没有发生。' in text
    assert text.index('另一事件。') < text.index('【父场景】')
    assert '【原文摘录】' not in text  # no source-file verification was performed
    assert result['raw_source_status'] == 'not_requested'


def test_budget_omits_whole_packet_instead_of_its_qualification(monkeypatch):
    instance = make_service(monkeypatch, ['甲' * 3500 + '限制一', '乙' * 3500 + '限制二'])
    result = instance.retrieve('解释人物经历')
    assert '限制一' in result['context_text']
    assert '乙' not in result['context_text']
    assert '【父场景】' not in result['context_text']
    assert result['context_budget']['skipped_blocks'] == 2
    assert [c['id'] for c in result['citations']] == ['0']
    assert len(result['context_text']) == result['context_budget']['used_chars'] <= 6000


def test_missing_evidence_is_labeled_summary_not_original(monkeypatch):
    result = make_service(monkeypatch, [None]).retrieve('解释人物经历')
    assert '【fact摘要】' in result['context_text']
    assert '【卡片关联证据】' not in result['context_text']


def test_cloud_budget_preserves_long_evidence_without_changing_local_instance(monkeypatch):
    texts = ['甲' * 3500 + '限制一', '乙' * 3500 + '限制二']
    local = make_service(monkeypatch, texts)
    cloud = make_service(monkeypatch, texts)
    cloud.context_max_chars = 16384
    expanded = cloud.retrieve('解释人物经历')
    assert all(text in expanded['context_text'] for text in texts)
    assert expanded['context_budget']['max_chars'] == 16384
    assert expanded['context_budget']['used_chars'] > 6000
    assert [c['id'] for c in expanded['citations']] == ['0', '1']
    assert '限制二' not in local.retrieve('解释人物经历')['context_text']


def test_background_cannot_masquerade_as_evidence_when_all_packets_excluded(monkeypatch):
    result = make_service(monkeypatch, ['长' * 6100]).retrieve('解释人物经历')
    assert result['context_text'] == '' and result['citations'] == []
    assert result['context_budget']['admitted_ids'] == []


def test_background_requires_its_own_admitted_evidence(monkeypatch):
    instance = make_service(monkeypatch, ['有效证据', '长' * 6100])
    candidates = instance.retrievers[service.CARD_TYPES].search()
    candidates[1].document.metadata['scene_id'] = 'excluded-scene'
    instance.by_id['scene'].summary = '有效场景'
    instance.by_id['excluded-scene'] = SimpleNamespace(id='excluded-scene', title='孤立场景', summary='不应入模')
    result = instance.retrieve('解释人物经历')
    assert '有效场景' in result['context_text']
    assert '不应入模' not in result['context_text']
    background = [p for p in result['evidence_packets'] if p['kind'] == 'background']
    assert background[0]['supporting_document_ids'] == ['0']


def test_shared_background_survives_when_later_sibling_evidence_fits(monkeypatch):
    instance = make_service(monkeypatch, ['长' * 6100, '有效证据'])
    instance.by_id['scene'].summary = '共享背景'
    result = instance.retrieve('解释人物经历')
    assert '共享背景' in result['context_text']
    assert [c['id'] for c in result['citations']] == ['1']
    background = [p for p in result['evidence_packets'] if p['kind'] == 'background']
    assert len(background) == 1
    assert background[0]['supporting_document_ids'] == ['0', '1']
