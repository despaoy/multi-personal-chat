"""Final coverage uses complete synthetic receipts and send-time source authority."""

from services import chat_answer

from copy import deepcopy
from html import unescape
from unittest.mock import AsyncMock

import pytest
from scope_fixtures import install_public_source_reader
from tests.test_public_fact_coverage import FEE, FULL, QUERY, bundle, deps, obligations, run

from api import generate
from db.schemas import MessageRequest
from knowledge import public_task_evidence, retrieval_query_plan
from knowledge.retrieval_query_plan import RetrievalQueryPlan


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,expected", [("fee", "partial_answer"), ("full", "grounded_answer")])
async def test_unprepared_public_state_uses_final_fact_coverage(monkeypatch, mode, expected, provider_budget):
    reviewed, _ = await run(mode)
    install_public_source_reader(monkeypatch, generate, reviewed)

    async def planner(query, *, history, context_budget):
        assert query == QUERY and history == ()
        assert context_budget is provider_budget.review
        return RetrievalQueryPlan(dependencies=deps(), public_obligations=obligations())

    async def no_binding(*_args, **_kwargs):
        return None

    async def retrieved_or_reviewed(*_args, **_kwargs):
        return deepcopy(reviewed)

    inputs = []

    async def model(**kwargs):
        wire = unescape("\n".join(m["content"] for m in kwargs["messages"]))
        assert QUERY in wire and (FEE if mode == "fee" else FULL) in wire
        inputs.append(wire)
        return "明确单元占位回复，不声称真实模型语义成功。"

    monkeypatch.setattr(retrieval_query_plan, "plan_retrieval", planner)
    monkeypatch.setattr("knowledge.public_question_binding.resolve_question_binding", no_binding)
    monkeypatch.setattr(chat_answer, 'retrieve_chat_knowledge', retrieved_or_reviewed)
    monkeypatch.setattr(public_task_evidence, "review_public_candidates", retrieved_or_reviewed)
    _, used, meta = await chat_answer.generate_chat_answer(
        MessageRequest(message=QUERY), None,
        runtime_config=dict(maxTokens=2048, useKnowledgeBase=True), model_generate=model,
    )
    assert len(inputs) == 1 and used and meta["answerMode"] == expected
    assert meta["abstained"] is (expected != "grounded_answer")
    assert ("partial_public_evidence" in (meta.get("warnings") or [])) is (expected != "grounded_answer")


@pytest.mark.asyncio
async def test_fact_scope_failure_stops_before_generation(monkeypatch, semantic_plan, controlled_public_bundle):
    semantic_plan(QUERY, dict(public_knowledge=[0], private_memory=[1]))

    async def retrieve(query, *_args, **_kwargs):
        assert query == QUERY
        return bundle()

    async def failing_review(*_args, **_kwargs):
        return (await run("full", scope_failure=True))[0]

    model = AsyncMock(side_effect=AssertionError("Failed evidence review must stop before generation"))
    monkeypatch.setattr(chat_answer, 'retrieve_chat_knowledge', retrieve)
    monkeypatch.setattr(public_task_evidence, "review_public_candidates", failing_review)
    with pytest.raises(chat_answer.ChatRetrievalError) as error:
        await chat_answer.generate_chat_answer(
            MessageRequest(message=QUERY), None,
            runtime_config=dict(maxTokens=2048, useKnowledgeBase=True), model_generate=model,
        )
    assert error.value.http_status == 503 and isinstance(error.value.__cause__, RuntimeError)
    assert str(error.value.__cause__) == "reviewer unavailable"
    model.assert_not_awaited()


@pytest.fixture(params=["vllm", "openai_compat"], ids=["local-budget", "cloud-budget"])
def provider_budget(monkeypatch, request):
    from types import SimpleNamespace
    from inference.provider_context import get_provider_context_budget

    manager = SimpleNamespace(_current_provider=SimpleNamespace(value=request.param))
    budget = get_provider_context_budget(manager, env={
        "VLLM_MAX_MODEL_LEN": "8192", "OPENAI_COMPAT_CONTEXT_WINDOW_TOKENS": "8192",
    })
    assert budget.window_tokens > 0
    if request.param == "openai_compat":
        assert budget.review.window_tokens > 0
    else:
        assert budget.review is None
    monkeypatch.setattr(chat_answer, "get_provider_context_budget", lambda: budget)
    return budget
