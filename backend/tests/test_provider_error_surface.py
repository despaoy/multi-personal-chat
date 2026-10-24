""" real provider status survives ordinary/SSE API boundaries safely.

Reuse the real SQLite/provider/grounded-service fixture; only the provider HTTP
transport's complete response is controlled. The actual application handlers,
ASGI routes, request context and answer cache remain active. No live HTTP or
paid model is used.
"""

import json
import logging
from contextlib import asynccontextmanager, contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from test_ask_request_context import ANSWER, SOURCE, USER, request
from test_ask_request_context import ask_rig as ask_rig

from api import ask, generate
from app.main import create_app
from app.runtime import RuntimeContainer
from inference import model_manager
from inference.openai_protocol import ProviderResponseError
from inference.request_context import get_request_context

pytestmark = pytest.mark.usefixtures("no_live_context_review", "literal_grounded_query")
PRIVATE_DETAIL = "fiction-private-provider-diagnostic-do-not-publish"
PRIVATE_KEY = "fiction-private-provider-key-do-not-publish"
PRIVATE_URL = "https://fiction-private-diagnostic.invalid/private-path"
FAILURES = [
    pytest.param(401, "provider_auth_error", 502, id="upstream-401"),
    pytest.param(403, "provider_auth_error", 502, id="upstream-403"),
    pytest.param(429, "provider_rate_limited", 503, id="upstream-429"),
    pytest.param(503, "provider_upstream_error", 502, id="upstream-503"),
    pytest.param(402, "provider_balance_error", 503, id="upstream-402"),
]


@pytest.fixture
async def provider_api(ask_rig, monkeypatch):
    rig = ask_rig
    rig.set_config(openaiCompatApiKey=PRIVATE_KEY)
    state = SimpleNamespace(
        status=200, failure=None, contexts=[], scope_exits=[], clients_closed=[], rig=rig,
    )
    forbidden_fallback = AsyncMock(side_effect=AssertionError("Provider rejection cannot switch models"))
    monkeypatch.setattr(generate, "get_vllm_client", forbidden_fallback)
    monkeypatch.setattr(rig.manager, "async_generate", forbidden_fallback)
    state.forbidden_fallback = forbidden_fallback

    async def respond(http_request):
        wire = json.loads(http_request.content)
        rig.calls.append(dict(body=wire, endpoint=str(http_request.url), auth=http_request.headers["Authorization"]))
        state.contexts.append(get_request_context())
        assert SOURCE in str(wire["messages"])
        assert wire["temperature"] == .2 and wire["max_tokens"] == 128 and wire["top_p"] == 1.0
        assert http_request.headers["Authorization"] == "Bearer " + PRIVATE_KEY
        if state.failure is not None:
            raise state.failure
        if state.status != 200:
            return httpx.Response(state.status, json={"error": {
                "message": PRIVATE_DETAIL + " " + PRIVATE_KEY + " " + PRIVATE_URL,
            }})
        return httpx.Response(200, json={"usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}, "choices": [{
            "finish_reason": "stop", "message": {"role": "assistant", "content": ANSWER + "[S1]"},
        }]})

    transport = httpx.MockTransport(respond)

    @asynccontextmanager
    async def acquire(_provider, timeout):
        client = httpx.AsyncClient(transport=transport, timeout=timeout)
        try:
            async with client:
                yield client
        finally:
            state.clients_closed.append(client.is_closed)

    monkeypatch.setattr(model_manager.OpenAICompatProvider, "_acquire_http_client", acquire)
    actual_scope = ask.use_request_context

    @contextmanager
    def observe_scope(context):
        try:
            with actual_scope(context):
                yield context
        finally:
            # Runs in the actual endpoint/stream consumer task after reset,
            # rather than inspecting only the unrelated ASGI caller's context.
            state.scope_exits.append(get_request_context())

    monkeypatch.setattr(ask, "use_request_context", observe_scope)
    app = create_app(RuntimeContainer(
        db=rig.database, is_pg_mode=lambda: False, inference_runtime=None,
        startup_env={"SECURITY_MIDDLEWARE_ENABLED": "false", "MODEL_PROVIDER": "openai_compat", "VLLM_ENABLED": "false"},
    ))
    app.dependency_overrides[ask.get_current_user] = lambda: USER
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://fiction-provider-api",
        ) as client:
            yield SimpleNamespace(client=client, state=state)
    finally:
        await app.state.readiness_probe.shutdown()
        await transport.aclose()


def events(response):
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


def assert_one_actual_call_and_cleanup(state):
    rig = state.rig
    assert len(rig.calls) == rig.retrievals == rig.snapshots.call_count == 1
    assert state.contexts[0] is rig.retrieval_contexts[0] and state.contexts[0] is not None
    assert state.scope_exits == [None] and get_request_context() is None
    assert state.clients_closed == [True]
    state.forbidden_fallback.assert_not_awaited()
    assert rig.database.get_knowledge_document(int(rig.document_id))["content"] == SOURCE


def assert_private_diagnostics_absent(response, caplog):
    visible = response.text + caplog.text
    for private in (PRIVATE_DETAIL, PRIVATE_KEY, PRIVATE_URL):
        assert private not in visible


def assert_reason_is_explicit(detail, kind):
    message = detail["message"]
    assert isinstance(message, str) and message.strip()
    assert message not in {"服务器内部错误", "服务内部错误", "Internal Server Error"}
    # These assert the public reason, not the vendor's private diagnostic text.
    markers = {
        "provider_auth_error": ("鉴权", "认证", "权限", "API Key"),
        "provider_balance_error": ("余额", "账户", "充值"),
        "provider_rate_limited": ("限流", "频率", "频繁", "配额", "请求过多"),
        "provider_upstream_error": ("上游", "模型服务", "提供商", "503"),
    }
    assert any(marker in message for marker in markers[kind])


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True], ids=["ordinary", "sse"])
async def test_real_provider_200_keeps_normal_answer_and_success_events(provider_api, caplog, streaming):
    caplog.set_level(logging.WARNING)
    api = provider_api
    response = await api.client.post("/api/ask/stream" if streaming else "/api/ask", json=request().model_dump())
    assert response.status_code == 200
    if streaming:
        sequence = events(response)
        assert [event["type"] for event in sequence] == ["meta", "delta", "citations", "done"]
        assert sequence[1]["text"] == ANSWER and sequence[-1]["model_invoked"] is True
        assert sequence[-1]["abstained"] is False
        assert api.state.rig.service.cache.stats()["entries"] == 0
    else:
        payload = response.json()
        assert payload["answer"] == ANSWER and payload["modelInvoked"] is True and payload["cacheHit"] is False
        assert api.state.rig.service.cache.stats()["entries"] == 1
    assert_one_actual_call_and_cleanup(api.state)
    assert_private_diagnostics_absent(response, caplog)
    row, = api.state.rig.database.execute_sql("SELECT * FROM model_invocations")
    assert row["purpose"] == "knowledge_answer" and row["totalTokens"] == 120
    assert row["usageSource"] == "provider" and row["errorType"] == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True], ids=["ordinary", "sse"])
@pytest.mark.parametrize("status,kind,http_status", FAILURES)
async def test_real_upstream_status_is_safe_explicit_and_never_retried(provider_api, caplog, streaming, status, kind, http_status):
    caplog.set_level(logging.WARNING)
    api = provider_api
    api.state.status = status
    response = await api.client.post("/api/ask/stream" if streaming else "/api/ask", json=request().model_dump())
    assert response.status_code != 401  # The user's app login is still valid.
    if streaming:
        assert response.status_code == 200
        sequence = events(response)
        assert [event["type"] for event in sequence] == ["meta", "error"]
        detail = sequence[-1]
        assert detail["kind"] == kind
    else:
        assert response.status_code == http_status
        detail = response.json()
        assert detail["success"] is False and detail["error"] == kind
        assert "answer" not in detail
    assert detail["upstreamStatus"] == status
    assert_reason_is_explicit(detail, kind)
    assert_one_actual_call_and_cleanup(api.state)
    assert api.state.rig.service.cache.stats()["entries"] == 0
    assert_private_diagnostics_absent(response, caplog)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True], ids=["ordinary", "sse"])
@pytest.mark.parametrize("error_type", [PermissionError, RuntimeError], ids=["filesystem-permission", "untyped-runtime"])
async def test_untyped_failure_is_not_misclassified_as_provider_auth(provider_api, caplog, streaming, error_type):
    caplog.set_level(logging.WARNING)
    api = provider_api
    api.state.failure = error_type(PRIVATE_DETAIL + " status=401 " + PRIVATE_KEY + " " + PRIVATE_URL)
    response = await api.client.post("/api/ask/stream" if streaming else "/api/ask", json=request().model_dump())
    if streaming:
        assert response.status_code == 200
        sequence = events(response)
        assert [event["type"] for event in sequence] == ["meta", "error"]
        detail = sequence[-1]
        assert detail["kind"] == "internal_error" and detail["message"] == "服务内部错误"
    else:
        assert response.status_code == 500
        detail = response.json()
        assert detail["success"] is False and detail["error"] == "Internal Server Error"
        assert detail["message"] == "服务器内部错误" and "answer" not in detail
    assert "upstreamStatus" not in detail
    assert "provider_auth_error" not in response.text and "provider_rate_limited" not in response.text
    assert_one_actual_call_and_cleanup(api.state)
    assert api.state.rig.service.cache.stats()["entries"] == 0
    assert_private_diagnostics_absent(response, caplog)


def test_provider_response_error_is_a_safe_status_only_runtime_error():
    error = ProviderResponseError(401)
    assert isinstance(error, RuntimeError)
    assert type(error.status_code) is int and error.status_code == 401 and error.http_status == 502
    detail = error.public_detail()
    assert set(detail) == {"kind", "message", "upstreamStatus"}
    assert detail["kind"] == "provider_auth_error" and detail["upstreamStatus"] == 401
    assert_reason_is_explicit(detail, detail["kind"])
    assert not any(hasattr(error, name) for name in ("body", "url", "response", "request"))
