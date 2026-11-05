"""A complete retrieved source that cannot fit is capacity failure, not absence."""

from copy import deepcopy

import httpx
import pytest
from fastapi import FastAPI

from app.main import global_exception_handler
from inference.context_budget import InputBudgetError
from inference.generation_request import GenerationRequest, RetrievalResult, generate_character_response
from inference.request_context import InferenceRequestContext, use_request_context

SOURCE = (
    "合成档案馆规程：周二09:00至17:00只对预约观众开放，预约不免除原件核验。"
    + "本段是完整虚构资料的背景，不增加额外事实。" * 120
    + "末尾限制：周一不开放；雨天只关闭室外展区，暴雨全馆暂停且预约也不例外。"
)
QUERY = "按这份完整虚构规程说明周二开放时间、预约条件、周一及雨天暴雨的限制。"
ANSWER = "周二09:00至17:00只对预约观众开放，仍需核验原件；周一不开放，雨天只关闭室外展区，暴雨全馆暂停且预约也不例外。"


@pytest.mark.parametrize("budget", ["context", "evidence", "sufficient"])
async def test_shared_generation_capacity_is_explicit_at_http_boundary(budget):
    request = GenerationRequest(
        message=QUERY,
        apply_prompt_policy=False,
        max_tokens=128,
        context_window_tokens=1600 if budget == "context" else 20000,
        evidence_max_chars=len(SOURCE) - 1 if budget == "evidence" else 0,
        retrieval=RetrievalResult(
            status="ok",
            evidence=SOURCE,
            evidence_packets=({"kind": "evidence", "text": SOURCE, "document_ids": ["source-a"]},),
        ),
    )
    before = deepcopy(request)
    calls = []

    async def model(**kwargs):
        calls.append(kwargs)
        assert SOURCE in kwargs["messages"][-1]["content"]
        assert QUERY in kwargs["messages"][-1]["content"]
        return ANSWER

    app = FastAPI()
    app.add_exception_handler(InputBudgetError, global_exception_handler)

    @app.post("/answer")
    async def answer():
        result = await generate_character_response(request, model)
        return {"reply": result.reply, "modelInvoked": result.model_invoked}

    # Explicit arithmetic-only counter. The actual shared pipeline and global
    # application error handler execute; no external provider is used here.
    with use_request_context(InferenceRequestContext("vllm", None, len)):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://capacity-case") as client:
            response = await client.post("/answer")
    assert request == before
    if budget == "sufficient":
        assert response.status_code == 200
        assert response.json() == {"reply": ANSWER, "modelInvoked": True}
        assert len(calls) == 1
    else:
        assert response.status_code == InputBudgetError.http_status == 503
        payload = response.json()
        assert payload["success"] is False and payload["error"] == "context_budget_exceeded"
        assert payload["stage"] == "context_budget" and payload["reason"] == "capacity_exceeded"
        assert "reply" not in payload and calls == []
