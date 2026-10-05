"""Admissions and the real ledger must compose across in-flight responses.

Transport is an explicit HTTP mock, not a cloud/model success claim.
"""

import asyncio
import json
from decimal import Decimal

import httpx
import pytest
from evaluation.cloud_transport_ledger import record_cloud_send
from evaluation.mixed_subject_history_probe import send_with_balance_stop
from evaluation.provider_budget import EvaluationBudgetExceeded, admitted_request_budget

PAYLOAD = dict(model="deepseek-flash", messages=[dict(role="user", content="完整虚构费用来源：37元。")], max_tokens=8)
USAGE = dict(
    usage=dict(prompt_tokens=17, prompt_cache_hit_tokens=0, completion_tokens=8),
    choices=[dict(message=dict(content="合成测试响应"))],
)


async def send(root, calls, client, sender):
    budget = admitted_request_budget(
        calls, model="deepseek-flash", input_tokens=17, output_tokens=8, max_calls=2, max_cost_cny=".000196"
    )
    request = client.build_request("POST", "https://api.deepseek.com/v1/chat/completions", json=PAYLOAD)
    return await record_cloud_send(root, calls, PAYLOAD, budget, client, request, sender, {}, send_with_balance_stop)


@pytest.mark.asyncio
async def test_pending_admission_and_ledger_keep_the_second_request_inside_the_ceiling(tmp_path):
    calls = []
    entered = asyncio.Event()
    release = asyncio.Event()

    async def first_sender(_client, request, **_kwargs):
        entered.set()
        await release.wait()
        return httpx.Response(200, json=USAGE, request=request)

    async def second_sender(_client, request, **_kwargs):
        return httpx.Response(200, json=USAGE, request=request)

    async with httpx.AsyncClient() as client:
        first = asyncio.create_task(send(tmp_path, calls, client, first_sender))
        await asyncio.wait_for(entered.wait(), 3)
        try:
            pending = json.loads((tmp_path / "cloud-calls.json").read_text())
            assert len(pending) == 1 and pending[0]["http_status"] is None
            bounded = admitted_request_budget(
                pending, model="deepseek-flash", input_tokens=17, output_tokens=8, max_cost_cny=".000196"
            )
            assert Decimal(bounded["prior_spend_peak_upper_bound_cny"]) == Decimal(".000098")
            with pytest.raises(EvaluationBudgetExceeded):
                admitted_request_budget(
                    pending, model="deepseek-flash", input_tokens=17, output_tokens=8, max_cost_cny=".000195"
                )
            assert (await send(tmp_path, calls, client, second_sender)).status_code == 200
            with pytest.raises(EvaluationBudgetExceeded):
                admitted_request_budget(calls, model="deepseek-flash", input_tokens=17, output_tokens=8, max_calls=2)
        finally:
            release.set()
            await first
    saved = json.loads((tmp_path / "cloud-calls.json").read_text())
    assert len(saved) == 2 and all(row["http_status"] == 200 and row["request_started"] for row in saved)


@pytest.mark.asyncio
async def test_received_headers_without_usage_keep_the_inflight_reservation(tmp_path):
    calls = []
    entered = asyncio.Event()
    release = asyncio.Event()

    class PendingBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            entered.set()
            await release.wait()
            yield json.dumps(USAGE).encode()

    async def sender(_client, request, **_kwargs):
        return httpx.Response(200, stream=PendingBody(), request=request)

    async with httpx.AsyncClient() as client:
        first = asyncio.create_task(send(tmp_path, calls, client, sender))
        await asyncio.wait_for(entered.wait(), 3)
        try:
            pending = json.loads((tmp_path / "cloud-calls.json").read_text())
            assert pending[0]["http_status"] == 200 and pending[0]["response"] is None
            bounded = admitted_request_budget(
                pending, model="deepseek-flash", input_tokens=17, output_tokens=8, max_cost_cny=".000196"
            )
            assert Decimal(bounded["prior_spend_peak_upper_bound_cny"]) == Decimal(".000098")
            with pytest.raises(EvaluationBudgetExceeded):
                admitted_request_budget(
                    pending, model="deepseek-flash", input_tokens=17, output_tokens=8, max_cost_cny=".000195"
                )
        finally:
            release.set()
            await first


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [httpx.ReadError, asyncio.CancelledError])
async def test_admission_generated_reservation_survives_lost_response(tmp_path, error_type):
    calls = []

    async def sender(_client, _request, **_kwargs):
        raise error_type("explicit mock response loss")

    async with httpx.AsyncClient() as client:
        with pytest.raises(error_type):
            await send(tmp_path, calls, client, sender)
    saved = json.loads((tmp_path / "cloud-calls.json").read_text())
    assert len(saved) == 1 and saved[0]["transport_error"] == error_type.__name__
    bounded = admitted_request_budget(saved, model="deepseek-flash", input_tokens=17, output_tokens=8)
    assert Decimal(bounded["prior_spend_peak_upper_bound_cny"]) == Decimal(".000098")
    with pytest.raises(EvaluationBudgetExceeded):
        admitted_request_budget(saved, model="deepseek-flash", input_tokens=17, output_tokens=8, max_cost_cny=".000195")


@pytest.mark.parametrize("input_tokens,output_tokens", [(-1, 8), (17, -1), (True, 8), (17, False)])
def test_invalid_new_bounds_cannot_mint_an_inflight_reservation(input_tokens, output_tokens):
    with pytest.raises(ValueError):
        admitted_request_budget([], model="deepseek-flash", input_tokens=input_tokens, output_tokens=output_tokens)
