"""Conservative paid-request admission for isolated DeepSeek evaluations."""

from decimal import Decimal

_PEAK_RMB_PER_MILLION = {
    "deepseek-flash": (Decimal("0.04"), Decimal("2"), Decimal("8")),
    "deepseek-v4-pro": (Decimal("0.30"), Decimal("9"), Decimal("27")),
}


class EvaluationBudgetExceeded(RuntimeError):
    """Preserve the existing run and stop before another paid request."""


def admitted_request_budget(calls, *, model, input_tokens, output_tokens, max_calls=None, max_cost_cny=None):
    """Reserve uncached input and the full output cap at peak prices.

    Usage from completed responses updates the conservative spend. This is an
    evaluation ceiling, not an invoice or a production response policy.
    """
    if max_calls is not None and len(calls) >= max_calls:
        raise EvaluationBudgetExceeded("Cloud request limit reached; preserve this run without replay")
    if type(input_tokens) is not int or input_tokens < 0 or type(output_tokens) is not int or output_tokens < 0:
        raise ValueError("Request reservation requires nonnegative integer token bounds")
    rates = _PEAK_RMB_PER_MILLION[model]
    spend = Decimal(0)
    for call in calls:
        response = call.get("response")
        usage = response.get("usage") if isinstance(response, dict) else None
        if call.get("request_started") and (call["http_status"] is None
                or (call["http_status"] == 200 and not isinstance(usage, dict))):
            budget = call.get("budget", {})
            pending_input, pending_output = budget.get("input_bound"), budget.get("output_reserved")
            if type(pending_input) is not int or pending_input < 0 or type(pending_output) is not int or pending_output < 0:
                raise ValueError("Unknown provider usage lacks a complete request reservation")
            _, miss, out = _PEAK_RMB_PER_MILLION[call["request"]["model"]]
            spend += (pending_input * miss + pending_output * out) / Decimal(1_000_000)
            continue
        if call["http_status"] != 200:
            continue
        if not isinstance(usage, dict):
            raise ValueError("Successful call lacks usage and a complete request reservation")
        prompt, output = usage["prompt_tokens"], usage["completion_tokens"]
        cached = usage.get("prompt_cache_hit_tokens", usage.get("prompt_tokens_details", {}).get("cached_tokens", 0))
        if not (0 <= cached <= prompt and output >= 0):
            raise ValueError("Invalid provider usage; inspect this run before continuing")
        hit, miss, out = _PEAK_RMB_PER_MILLION[call["request"]["model"]]
        spend += (cached * hit + (prompt - cached) * miss + output * out) / Decimal(1_000_000)
    reserved = (input_tokens * rates[1] + output_tokens * rates[2]) / Decimal(1_000_000)
    if max_cost_cny is not None and spend + reserved > Decimal(str(max_cost_cny)):
        raise EvaluationBudgetExceeded("Estimated RMB ceiling reached; preserve this run without replay")
    return {
        "price_policy": "evaluation peak reservation rates, not invoice/current quote; uncached input and full output reserved",
        "input_bound": input_tokens,
        "output_reserved": output_tokens,
        "prior_calls": len(calls),
        "prior_spend_peak_upper_bound_cny": str(spend),
        "next_request_peak_reservation_cny": str(reserved),
        "projected_peak_upper_bound_cny": str(spend + reserved),
        "max_cloud_calls": max_calls,
        "max_estimated_cost_cny": str(max_cost_cny) if max_cost_cny is not None else None,
    }
