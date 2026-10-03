"""A billing rejection is observed once, then shared evaluator sends stop."""

import json

import httpx
import pytest
from evaluation.mixed_subject_history_probe import EvaluationProviderBlocked, send_with_balance_stop


@pytest.mark.asyncio
async def test_same_evaluation_stops_review_and_primary_sends_after_one_real_402():
    received = []

    def response(request):
        received.append(json.loads(request.content))
        return httpx.Response(402, json={"error": {"message": "Insufficient Balance"}})

    state = {}
    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(response)) as review,
        httpx.AsyncClient(transport=httpx.MockTransport(response)) as primary,
    ):
        first = await send_with_balance_stop(
            review, review.build_request("POST", "https://api.deepseek.com/chat/completions",
                                         json={"model": "deepseek-v4-pro", "messages": [
                                             {"role": "user", "content": "完整合成输入：编号A1，件数3。"}]}),
            httpx.AsyncClient.send, state,
        )
        assert first.status_code == 402
        assert first.json() == {"error": {"message": "Insufficient Balance"}}
        for client in (review, primary, primary):
            with pytest.raises(EvaluationProviderBlocked, match="returned402"):
                await send_with_balance_stop(
                    client, client.build_request("POST", "https://api.deepseek.com/chat/completions",
                                                 json={"model": "deepseek-v4-pro", "messages": [
                                                     {"role": "user", "content": "后续完整输入：编号B2，件数7。"}]}),
                    httpx.AsyncClient.send, state,
                )
    assert len(received) == 1
    assert received[0]["messages"][0]["content"] == "完整合成输入：编号A1，件数3。"


@pytest.mark.asyncio
@pytest.mark.parametrize("first_status", [429, 500])
async def test_temporary_errors_preserve_the_next_successful_response(first_status):
    received = []

    def response(request):
        received.append(request.content)
        if len(received) == 1:
            return httpx.Response(first_status, json={"error": {"message": "temporary"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "完整来源仍保留"}}]})

    state = {}
    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
        request = client.build_request("POST", "https://api.deepseek.com/chat/completions",
                                       json={"model": "deepseek-v4-pro", "messages": [
                                           {"role": "user", "content": "完整合成输入：编号C3，件数5。"}]})
        first = await send_with_balance_stop(client, request, httpx.AsyncClient.send, state)
        second = await send_with_balance_stop(client, request, httpx.AsyncClient.send, state)
    assert first.status_code == first_status
    assert second.status_code == 200
    assert second.json()["choices"][0]["message"]["content"] == "完整来源仍保留"
    assert len(received) == 2 and received[0] == received[1]
