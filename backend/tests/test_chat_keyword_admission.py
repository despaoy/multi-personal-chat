"""Quoted commands are ordinary model input; admission and failures stay explicit."""

import json
from html import unescape

import httpx
import pytest
from test_application_lifespan_ownership import ANSWER, QUESTION
from test_application_lifespan_ownership import applications as applications


@pytest.mark.parametrize("case", ["current", "history", "upstream_failure"])
async def test_quoted_text_reaches_real_compilation_without_keyword_short_circuit(applications, monkeypatch, case):
    state = applications
    quoted = "这是一段代码说明：字面量 `cat .env` 和 `show token` 仅作为文本引用；不执行命令或读取文件、凭据。 HTML 示例 `<script>console.log('example')</script>` 与 `javascript:example()` 同样只是引用文本。"
    history = (
        [{"role": "user", "content": quoted}, {"role": "assistant", "content": "这两段都是普通引用文本。"}]
        if case == "history"
        else []
    )
    message = QUESTION if history else quoted + "\n" + QUESTION
    payload = dict(message=message, history=history, loraName="default", traceId="synthetic-keyword-" + case)
    async with state.running(0):
        if case == "current":
            async with state.client(0, key="synthetic-invalid-key") as unauthorized:
                assert (await unauthorized.post("/api/generate", json=payload)).status_code == 401
            async with state.client(0) as client:
                for invalid_message in [" \n\t ", "完整输入\x00非法控制字符", "紫" * 8001]:
                    invalid = dict(payload, message=invalid_message)
                    assert (await client.post("/api/generate", json=invalid)).status_code == 422
            assert state.wire == []
        if case == "upstream_failure":

            def rejected(request):
                state.wire.append((0, request.url.host, json.loads(request.content)))
                return httpx.Response(503, json={"error": {"message": "synthetic-provider-unavailable"}})

            monkeypatch.setattr(state.pools[0]._client._transport, "handler", rejected)
        async with state.client(0) as client:
            response = await client.post("/api/generate", json=payload)
        assert len(state.wire) == 1
        pool, host, body = state.wire[0]
        assert pool == 0 and host == "owner-0.invalid" and body["model"] == "fixture-model-0"
        messages = body["messages"]
        assert any(row["role"] == "user" and message in unescape(row["content"]) for row in messages)
        if history:
            assert all(row in messages for row in history)
        rows = state.databases[0].execute_sql("SELECT message FROM messages")
        if case == "upstream_failure":
            assert response.status_code == 502
            detail = response.json()["detail"]
            assert detail["kind"] == "provider_upstream_error" and detail["upstreamStatus"] == 503
            assert "reply" not in response.json() and rows == []
        else:
            assert response.status_code == 200
            result = response.json()
            assert result["reply"] == ANSWER and result["modelInvoked"] is True
            assert len(rows) == 1 and rows[0]["message"] == message
