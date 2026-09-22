from types import SimpleNamespace

import pytest

from evaluation.hierarchical_memory_retrieval import hierarchical_rankings, pack_turn_excerpts


def documents():
    return [
        SimpleNamespace(metadata={"session_id": session, "turn_id": turn})
        for session, turn in [("a", "a1"), ("a", "a2"), ("a", "a1"), ("b", "b1"), ("c", "c1")]
    ]


def test_session_selection_never_uses_gold_and_dense_order_is_retained():
    result = hierarchical_rankings(documents(), [0, 1, 2, 3, 4], [4, 1, 0, 2, 3], session_k=2)
    assert result["session_rrf_then_dense"] == [1, 0, 2, 3]
    assert result["session_rrf_round_robin_dense"] == [1, 3, 0, 2]
    assert all(4 not in indices for indices in result.values())


def test_round_robin_retains_all_chunks_without_duplicate_turn_priority():
    result = hierarchical_rankings(documents(), [0, 3, 4, 1, 2], [0, 2, 1, 3, 4], session_k=3)
    assert result["session_rrf_round_robin_dense"] == [0, 3, 4, 1, 2]
    assert sorted(result["session_rrf_round_robin_dense"]) == list(range(5))


def test_packing_is_atomic_under_same_budget_and_counts_turn_once():
    selected, tokens = pack_turn_excerpts(documents(), [0, 2, 1, 3, 4], [300, 300, 10, 80, 60], budget=450)
    assert selected == [0, 3, 4]
    assert tokens == 440
    # A cheaper duplicate chunk cannot masquerade as the highest ranked excerpt.
    assert 2 not in selected


@pytest.mark.parametrize("lengths,budget", [([1], 100), ([1] * 5, 0), ([1, 1, 0, 1, 1], 10)])
def test_invalid_token_budget_rejected(lengths, budget):
    with pytest.raises(ValueError):
        pack_turn_excerpts(documents(), list(range(5)), lengths, budget=budget)
