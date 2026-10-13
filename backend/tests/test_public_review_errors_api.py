"""Public review failures stop the actual chat entry before answer generation."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from tests.test_public_fact_coverage import FULL, QUERY, TASK, bundle, evidence_reply, scope_reply, source_reply

from api import generate
from db.schemas import MessageRequest
from inference.context_budget import ReviewContextBudget
from knowledge import public_question_binding, public_task_evidence

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures("no_live_context_review")]
STAGES = ("object", "fact_scope", "source", "fact_evidence")


@pytest.mark.parametrize(
    ("stage", "defect"),
    [(stage, defect) for stage in STAGES for defect in ("provider", "timeout", "invalid")]
    + [("capacity", "capacity")],
)
async def test_public_review_errors_api_stop_before_answer(monkeypatch, semantic_plan, stage, defect):
    planning = semantic_plan(QUERY, dict(public_knowledge=[0], private_memory=[1]))
    actual_review = public_task_evidence.review_public_candidates
    events, answers = [], []
    error = asyncio.TimeoutError("Declared review timeout") if defect == "timeout" else RuntimeError("Declared provider failure")

    def reviewer(name, response):
        async def review(messages):
            payload = json.loads(messages[-1]["content"])
            assert payload["query"] == QUERY
            if name in ("source", "fact_evidence"):
                assert payload["sources"][0]["original_body"] == FULL
            events.append(name)
            if name == stage:
                if defect in ("provider", "timeout"):
                    raise error
                return "{"
            return json.dumps(response)
        return review

    async def binding(*args, **kwargs):
        return None

    async def retrieve(*args, **kwargs):
        return bundle()

    async def review(*args, **kwargs):
        return await actual_review(
            *args, **kwargs,
            scope_reviewer=reviewer("object", dict(scopes=[dict(task_id=TASK, objects=["柳港续租"])])),
            fact_scope_reviewer=reviewer("fact_scope", scope_reply()),
            reviewer=reviewer("source", source_reply()),
            fact_reviewer=reviewer("fact_evidence", evidence_reply()),
            span_references=False,
        )

    async def answer(**kwargs):
        answers.append(kwargs)
        pytest.fail("A failed public review must stop answer generation")

    budget = SimpleNamespace(
        review=ReviewContextBudget(65536), window_tokens=1024 if stage == "capacity" else 65536,
        evidence_max_chars=0,
    )
    monkeypatch.setattr(generate, "get_provider_context_budget", lambda: budget)
    monkeypatch.setattr(generate, "_retrieve_rag_bundle", retrieve)
    monkeypatch.setattr(public_question_binding, "resolve_question_binding", binding)
    monkeypatch.setattr(public_task_evidence, "review_public_candidates", review)
    with pytest.raises(HTTPException) as caught:
        await generate._generate_with_retrieval(
            MessageRequest(message=QUERY), None,
            runtime_config=dict(maxTokens=2048, useKnowledgeBase=True), model_generate=answer,
        )
    assert caught.value.status_code == 503 and len(planning) == 1 and not answers
    if stage == "capacity":
        assert not events and isinstance(caught.value.__cause__, ValueError)
        assert "scope input exceeds context budget" in str(caught.value.__cause__)
    else:
        assert events == list(STAGES[:STAGES.index(stage) + 1])
        if defect in ("provider", "timeout"):
            assert caught.value.__cause__ is error
        else:
            assert isinstance(caught.value.__cause__, ValueError)
