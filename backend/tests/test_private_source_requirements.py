"""A case declares required originals; default complete-source gates stay strict."""

import pytest

from evaluation.mixed_subject_history_probe import required_private_sources


def fixture(indices=None):
    case = dict(source_message="original", additional_private_source_messages=["past strongest", "current replacement"])
    if indices is not None:
        case["required_private_source_indices"] = indices
    return case


def test_legacy_cases_require_every_complete_private_original():
    assert required_private_sources(fixture()) == ("original", "past strongest", "current replacement")


def test_current_only_case_declares_current_source_without_rewriting_retained_originals():
    case = fixture([2])
    assert required_private_sources(case) == ("current replacement",)
    assert case["source_message"] == "original" and case["additional_private_source_messages"] == [
        "past strongest",
        "current replacement",
    ]


@pytest.mark.parametrize("indices", [[], [True], [-1], [3], [2, 2], ["2"]])
def test_missing_invalid_or_duplicated_required_identity_is_rejected(indices):
    with pytest.raises(ValueError, match="Invalid required private source indices"):
        required_private_sources(fixture(indices))
