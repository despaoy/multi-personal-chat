from inference.generation_request import GenerationRequest, RetrievalResult, build_generation_request


def test_final_window_admits_whole_packets_and_updates_citations():
    packets = ({'text': '甲' * 900 + '但是不能做', 'document_ids': ['a'], 'kind': 'evidence'},
               {'text': '乙' * 900 + '但是尚未发生', 'document_ids': ['b'], 'kind': 'evidence'})
    request = GenerationRequest(message='问题', persona_prompt='设定' * 100,
                                apply_prompt_policy=False, max_tokens=300, context_window_tokens=2400,
                                retrieval=RetrievalResult(status='ok', evidence='\n\n'.join(p['text'] for p in packets),
                                    evidence_packets=packets, citations=({'id': 'a'}, {'id': 'b'})))
    plan = build_generation_request(request)
    assert '但是不能做' in plan.messages[-1]['content']
    assert '乙' not in plan.messages[-1]['content']
    assert plan.retrieval.citations == ({'id': 'a'},)
    assert plan.retrieval.evidence_packets == packets[:1]


def test_no_packet_fits_does_not_use_background_as_fact():
    packets = ({'text': '甲' * 8000, 'document_ids': ['a'], 'kind': 'evidence'},
               {'text': '背景', 'document_ids': [], 'kind': 'background'})
    request = GenerationRequest(message='问题', max_tokens=100, context_window_tokens=2000,
                                retrieval=RetrievalResult(status='ok', evidence='甲' * 8000,
                                    evidence_packets=packets, citations=({'id': 'a'},)))
    plan = build_generation_request(request)
    assert plan.retrieval.status == 'character_abstention'
    assert plan.retrieval.reason == 'evidence_budget_exhausted'
    assert not plan.retrieval.citations
    assert '背景' not in plan.messages[-1]['content']


def test_token_budget_rechecks_background_ownership():
    packets = ({'text': '有效证据', 'document_ids': ['a'], 'kind': 'evidence'},
               {'text': '乙' * 8000, 'document_ids': ['b'], 'kind': 'evidence'},
               {'text': '孤立场景', 'document_ids': [], 'kind': 'background', 'supporting_document_ids': ['b']},
               {'text': '共享场景', 'document_ids': [], 'kind': 'background', 'supporting_document_ids': ['a', 'b']})
    plan = build_generation_request(GenerationRequest(
        message='问题', apply_prompt_policy=False, max_tokens=100, context_window_tokens=2000,
        retrieval=RetrievalResult(status='ok', evidence='有效证据', evidence_packets=packets,
                                  citations=({'id': 'a'}, {'id': 'b'}))))
    assert '孤立场景' not in plan.messages[-1]['content']
    assert '共享场景' in plan.messages[-1]['content']
    assert plan.retrieval.citations == ({'id': 'a'},)
