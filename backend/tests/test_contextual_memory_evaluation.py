import copy

import pytest
from scripts.evaluate_contextual_memory import evaluate, validate_cases

CASE = {"id": "a", "query": "hi", "memories": [{"id": "m", "content": "用户喜欢苹果"}], "gold_ids": []}


@pytest.mark.parametrize(
    "change", ["duplicate", "missing_gold", "double_gold", "wrong_query", "single_pair", "duplicate_memory"]
)
def test_bad_fixture_rejected_before_model_loading(change):
    case = copy.deepcopy(CASE)
    cases = [case]
    if change == "duplicate":
        cases.append(copy.deepcopy(case))
    elif change == "missing_gold":
        case["gold_ids"] = ["absent"]
    elif change == "double_gold":
        case["gold_ids"] = ["m", "m"]
    elif change == "wrong_query":
        case["query"] = None
    elif change == "single_pair":
        case["pair_id"] = "p"
    else:
        case["memories"] *= 2
    with pytest.raises(ValueError):
        validate_cases(cases)


async def test_explicit_diagnostic_subset_reports_unscored_pair():
    case = {**CASE, "pair_id": "p"}
    result = await evaluate([case], "lexical_baseline")
    assert result["incomplete_pairs_not_scored"] == ["p"]
    assert result["paired_metrics"]["pairs"] == 0


def test_complete_pair_is_valid():
    validate_cases([{**CASE, "pair_id": "p"}, {**CASE, "id": "b", "pair_id": "p"}])
