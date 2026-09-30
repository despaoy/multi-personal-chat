import pytest

from evaluation.chat_tokenization_probe import token_difference


@pytest.mark.parametrize('expected,actual,first', [([1, 2], [1, 2], None),
    ([1, 2], [1, 3], 1), ([1], [1, 2], 1), ([1, 2], [1], 1), ([], [1], 0)])
def test_exact_ids_not_just_token_counts(expected, actual, first):
    result = token_difference(expected, actual)
    assert result['first_difference'] == first
    assert result['equal'] == (first is None)


def test_boolean_not_accepted_as_token_id():
    with pytest.raises(ValueError):
        token_difference([True], [1])
