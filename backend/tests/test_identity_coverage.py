from dataclasses import replace
from types import SimpleNamespace

import pytest

from knowledge.multiscale_rag.service import rerank_with_title_frames, select_identity_coverage
from knowledge.retrieval_core.retrieval import RetrievalCandidate


def candidate(index, kind='fact', subject='林远', target='安宁'):
    doc = SimpleNamespace(id=str(index), document_type=kind, title='', summary='',
                          metadata={'subject': subject, 'target': target})
    return RetrievalCandidate(index, doc, rerank_score=1 - index / 100)


def test_coverage_preserves_best_fact_and_bound_without_mutating_candidates():
    rows = [candidate(i) for i in range(25)] + [candidate(25, 'relation')]
    selected = select_identity_coverage(rows, 3, '林远')
    assert [c.row for c in selected] == [0, 1, 25]
    assert len(rows) == 26 and rows[2].row == 2


@pytest.mark.parametrize('subject,target', [('他人', '另一人'), ('', ''), (None, None)])
def test_incidental_mentions_or_missing_endpoints_cannot_reserve_slot(subject, target):
    rows = [candidate(i) for i in range(3)] + [candidate(3, 'relation', subject, target)]
    assert select_identity_coverage(rows, 3, '林远') == rows[:3]


def test_existing_relation_not_duplicated_and_inverse_owner_is_eligible():
    rows = [candidate(0), candidate(1, 'relation', '安宁', '林远'), candidate(2)]
    assert select_identity_coverage(rows, 3, '林远') == rows
    assert select_identity_coverage(rows, 1, '林远') == rows[:1]
    assert select_identity_coverage(rows, 0, '林远') == []


@pytest.mark.parametrize('method', ['deterministic', 'cross_encoder'])
@pytest.mark.parametrize('enabled', [False, True])
def test_relation_participates_before_rank_truncation_only_when_explicitly_enabled(method, enabled):
    rows = [candidate(i) for i in range(25)] + [candidate(25, 'relation')]
    limits = []

    def rank(_analysis, candidates, top_k):
        limits.append(top_k)
        return [replace(c, rerank_method=method) for c in candidates[:top_k]]

    analysis = SimpleNamespace(entities=['林远'], normalized_query='林远是谁？', original_query='林远是谁？')
    result = rerank_with_title_frames(analysis, rows, top_k=3,
                                     reranker=SimpleNamespace(rerank=rank), identity_coverage=enabled)
    assert limits == [26 if enabled else 20]
    assert [c.row for c in result] == ([0, 1, 25] if enabled else [0, 1, 2])
    assert rows[0].rerank_method == 'none'


def test_open_or_multi_entity_query_never_reserves_relation():
    rows = [candidate(i) for i in range(3)] + [candidate(3, 'relation')]
    analysis = SimpleNamespace(entities=['林远', '安宁'], normalized_query='林远为什么帮助安宁？',
                               original_query='林远为什么帮助安宁？')
    rank = SimpleNamespace(rerank=lambda a, c, top_k: [replace(x) for x in c[:top_k]])
    assert len(rerank_with_title_frames(analysis, rows, top_k=3, reranker=rank, identity_coverage=True)) == 3
    assert select_identity_coverage(rows, 3, '') == rows[:3]
