"""Declared unit fixtures; real final receipt parsing, no cloud/auth/history claim."""

from copy import deepcopy
from html import unescape

import pytest
from tests.test_public_fact_coverage import FEE, FULL, QUERY, deps, obligations, run

from api import generate
from db.schemas import MessageRequest
from knowledge import intent_detector, public_task_evidence, retrieval_query_plan
from knowledge.retrieval_query_plan import RetrievalQueryPlan


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,scope_failure,expected",
    [("fee", False, "partial_answer"), ("full", False, "grounded_answer"), ("full", True, "abstention")],
)
async def test_unprepared_public_state_uses_final_fact_coverage(monkeypatch, mode, scope_failure, expected):
    # Existing fictional unit reviewer stubs construct receipts via strict native
    # parsing; they are not actual model responses or real stored business data.
    reviewed, _ = await run(mode, scope_failure=scope_failure)

    async def planner(_query):
        return RetrievalQueryPlan(status="applied", dependencies=deps(), public_obligations=obligations())

    async def no_binding(*_args, **_kwargs):
        return None

    async def retrieve(*_args, **_kwargs):
        return deepcopy(reviewed)

    async def reviewed_fixture(*_args, **_kwargs):
        return deepcopy(reviewed)

    inputs = []

    async def model(**kwargs):
        wire = unescape("\n".join(m["content"] for m in kwargs["messages"]))
        assert QUERY in wire and (FEE if mode == "fee" else FULL) in wire
        inputs.append(wire)
        return "明确单元占位回复，不声称真实模型语义成功。"

    from knowledge import public_question_binding

    monkeypatch.setattr(retrieval_query_plan, "plan_retrieval_views", planner)
    monkeypatch.setattr(public_question_binding, "resolve_question_binding", no_binding)
    monkeypatch.setattr(intent_detector, "needs_rag", lambda _: (True, "declared-unit", None))
    monkeypatch.setattr(generate, "_retrieve_rag_bundle", retrieve)
    monkeypatch.setattr(public_task_evidence, "review_public_candidates", reviewed_fixture)
    _, used, meta = await generate._generate_with_retrieval(
        MessageRequest(message=QUERY),
        None,
        runtime_config=dict(maxTokens=2048, useKnowledgeBase=True),
        model_generate=model,
    )
    assert len(inputs) == 1 and used and meta["answerMode"] == expected
    assert meta["abstained"] is (expected != "grounded_answer")
    assert ("partial_public_evidence" in (meta.get("warnings") or [])) is (expected != "grounded_answer")
