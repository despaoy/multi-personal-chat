import numpy as np
import pytest

from evaluation.episode_retriever_audit import cosine_scores, ranked


def test_normalization_is_scale_invariant_and_keeps_negative_scores():
    assert cosine_scores([2, 0], [[10, 0], [-3, 0], [0, 4]]) == [1, -1, 0]
    assert ranked([{'id': 'a'}, {'id': 'b'}, {'id': 'c'}], [1, -1, 0])[0]['id'] == 'a'


@pytest.mark.parametrize('q,docs', [([0, 0], [[1, 1]]), ([1, 0], [[0, 0]]),
    ([1, np.nan], [[1, 1]]), ([1, 0], [[1]]), ([[1, 0]], [[1, 0]])])
def test_invalid_vectors_fail_closed(q, docs):
    with pytest.raises(ValueError):
        cosine_scores(q, docs)


def test_semantic_encoding_filters_scope_and_prohibited_sessions_first():
    from types import SimpleNamespace

    from evaluation.episode_retriever_audit import scoped_semantic_hits
    from evaluation.episodic_recall_audit import Episode

    class Provider:
        def embed_texts(self, texts):
            assert texts == ['问题', '可编码原话']
            return [[1, 0], [1, 0]]

    rows = [Episode('1', ('user',), 'a', '', '可编码原话'),
            Episode('2', ('other',), 'b', '', '其他用户'),
            Episode('3', ('user',), 'c', '', '不要保存这个修改。')]
    result = scoped_semantic_hits(SimpleNamespace(episodes=rows, scope=('user',)), '问题', Provider(), 2)
    assert result == [{'id': '1', 'score': 1.0}]
