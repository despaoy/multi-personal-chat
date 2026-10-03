"""A paid-evaluation ceiling must stop before a request is sent."""

from decimal import Decimal

import pytest
from evaluation.provider_budget import EvaluationBudgetExceeded, admitted_request_budget


def receipt(model="deepseek-flash", prompt=60000, cached=50000, output=1000, status=200):
    return {
        "http_status": status,
        "request": {"model": model},
        "response": {
            "usage": {"prompt_tokens": prompt, "prompt_cache_hit_tokens": cached, "completion_tokens": output}
        },
    }


def test_cost_cap_includes_actual_cached_usage_and_full_next_output_reservation():
    kwargs = dict(model="deepseek-flash", input_tokens=60000, output_tokens=2048, max_calls=12)
    record = admitted_request_budget([receipt()], max_cost_cny="0.166384", **kwargs)
    assert Decimal(record["prior_spend_peak_upper_bound_cny"]) == Decimal("0.030")
    assert record["projected_peak_upper_bound_cny"] == "0.166384"
    with pytest.raises(EvaluationBudgetExceeded):
        admitted_request_budget([receipt()], max_cost_cny="0.166383", **kwargs)


def test_call_cap_counts_failed_sends_and_preserves_response_ledger():
    calls = [receipt(status=500), receipt()]
    with pytest.raises(EvaluationBudgetExceeded):
        admitted_request_budget(calls, model="deepseek-flash", input_tokens=100, output_tokens=10, max_calls=2)
    assert len(calls) == 2 and calls[0]["http_status"] == 500


def test_unlimited_legacy_default_and_mixed_model_usage_remain_honest():
    record = admitted_request_budget(
        [receipt(model="deepseek-v4-pro", prompt=1000, cached=0, output=0)],
        model="deepseek-flash",
        input_tokens=1000,
        output_tokens=0,
    )
    assert record["prior_spend_peak_upper_bound_cny"] == "0.009"
    assert record["projected_peak_upper_bound_cny"] == "0.011"
