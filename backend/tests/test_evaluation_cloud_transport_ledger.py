"""A lost response must not erase the actual send or its spend reservation."""

import asyncio
import json
from decimal import Decimal

import httpx
import pytest
from evaluation.cloud_transport_ledger import record_cloud_send
from evaluation.mixed_subject_history_probe import EvaluationProviderBlocked, send_with_balance_stop
from evaluation.provider_budget import EvaluationBudgetExceeded, admitted_request_budget

PAYLOAD = dict(
    model="deepseek-flash",
    messages=[dict(role="user", content="完整合成来源：费用27元，周三，原件必带；预约不免原件。")],
    max_tokens=8,
)
BUDGET = dict(input_bound=17, output_reserved=8)


async def send(root, calls, state, sender):
    async with httpx.AsyncClient() as client:
        request = client.build_request(
            "POST",
            "https://api.deepseek.com/chat/completions",
            json=PAYLOAD,
            headers={"Authorization": "Bearer synthetic-secret-never-log"},
        )
        return await record_cloud_send(
            root, calls, PAYLOAD, BUDGET, client, request, sender, state, send_with_balance_stop
        )


async def positive(root):
    calls = []
    state = {}

    async def sender(_client, request, **_kwargs):
        saved = json.loads((root / "cloud-calls.json").read_text())
        assert len(saved) == 1 and saved[0]["request_started"] and saved[0]["http_status"] is None
        assert (
            saved[0]["request"] == PAYLOAD
            and "synthetic-secret-never-log" not in (root / "cloud-calls.json").read_text()
        )
        return httpx.Response(
            200,
            json=dict(
                usage=dict(prompt_tokens=17, prompt_cache_hit_tokens=0, completion_tokens=8),
                choices=[dict(message=dict(content="完整回答"))],
            ),
            request=request,
        )

    response = await send(root, calls, state, sender)
    assert response.status_code == 200 and len(calls) == 1 and calls[0]["response"]["usage"]["completion_tokens"] == 8
    return calls


@pytest.mark.asyncio
async def test_actual_send_is_durable_before_response_and_recorded_once(tmp_path):
    await positive(tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [httpx.ReadError, asyncio.CancelledError])
async def test_lost_response_or_cancellation_preserves_started_send_and_full_reservation(tmp_path, error_type):
    await positive(tmp_path)
    calls = []

    async def failed(_client, _request, **_kwargs):
        raise error_type("simulated lost response")

    with pytest.raises(error_type):
        await send(tmp_path, calls, {}, failed)
    saved = json.loads((tmp_path / "cloud-calls.json").read_text())
    assert (
        len(saved) == 1
        and saved[0]["request_started"]
        and saved[0]["http_status"] is None
        and saved[0]["transport_error"] == error_type.__name__
    )
    bound = admitted_request_budget(saved, model="deepseek-flash", input_tokens=17, output_tokens=8)
    assert Decimal(bound["prior_spend_peak_upper_bound_cny"]) == Decimal(".000098")
    with pytest.raises(EvaluationBudgetExceeded):
        admitted_request_budget(saved, model="deepseek-flash", input_tokens=17, output_tokens=8, max_cost_cny=".00018")


@pytest.mark.asyncio
async def test_real_402_is_recorded_but_blocked_followup_is_not_a_send(tmp_path):
    await positive(tmp_path)
    calls = []
    state = {}
    seen = []

    async def reject(_client, request, **_kwargs):
        seen.append(request)
        return httpx.Response(402, json=dict(error=dict(message="Insufficient Balance")), request=request)

    first = await send(tmp_path, calls, state, reject)
    assert first.status_code == 402
    with pytest.raises(EvaluationProviderBlocked):
        await send(tmp_path, calls, state, reject)
    assert len(calls) == len(seen) == 1 and calls[0]["http_status"] == 402 and state["blocked_send_attempts"] == 1


@pytest.mark.asyncio
async def test_non_json_error_response_retains_status_and_exact_body(tmp_path):
    await positive(tmp_path)
    calls = []

    async def reject(_client, request, **_kwargs):
        return httpx.Response(500, text="original unavailable response", request=request)

    with pytest.raises(ValueError):
        await send(tmp_path, calls, {}, reject)
    saved = json.loads((tmp_path / "cloud-calls.json").read_text())
    assert (
        len(saved) == 1
        and saved[0]["http_status"] == 500
        and saved[0]["response_text"] == "original unavailable response"
    )


def test_unknown_usage_without_full_reservation_cannot_be_treated_as_free():
    with pytest.raises(ValueError):
        admitted_request_budget(
            [dict(request_started=True, http_status=None, request=dict(model="deepseek-flash"))],
            model="deepseek-flash",
            input_tokens=17,
            output_tokens=8,
        )
