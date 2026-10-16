"""Semantic plans own complete dependencies and tasks, never search hints or grants."""
import asyncio
import json

import pytest

from knowledge import retrieval_query_plan
from knowledge.retrieval_query_plan import QueryPlanReviewError, plan_retrieval
from knowledge.turn_dependencies import query_segments

QUERY = "请按青川通道公共办理规则回答费用、时长和核验条件。" + "假设方案资料完整，实际情况未知。" * 40


def complete_plan():
    return dict(
        dependencies=dict(private_memory=[], public_knowledge=[0],
                          current_input=list(range(1, len(query_segments(QUERY)))),
                          control=[], unresolved_source=[]),
        public_partitions=[dict(segment_id=0, cuts=[])],
    )


@pytest.mark.parametrize("field,value", [("source_grants", ["私密库"]), ("filters", {"knowledge_base_id": 1}),
    ("search_views", []), ("search_views", [["历史对象"]])])
async def test_complete_plan_cannot_add_search_hints_filters_or_source_permissions(field, value):
    proposal = complete_plan()
    proposal[field] = value

    async def review(_messages):
        return json.dumps(proposal)

    with pytest.raises(QueryPlanReviewError, match="query_plan_invalid") as caught:
        await plan_retrieval(QUERY, reviewer=review)
    assert str(caught.value.__cause__) == "Unexpected task-plan fields"


async def test_complete_plan_rejects_duplicate_keys_before_contract_validation():
    raw = json.dumps(complete_plan()).replace('"dependencies":', '"dependencies": {}, "dependencies":', 1)

    async def review(_messages):
        return raw

    with pytest.raises(QueryPlanReviewError, match="query_plan_invalid") as caught:
        await plan_retrieval(QUERY, reviewer=review)
    assert str(caught.value.__cause__) == "Duplicate query-plan key"


async def test_complete_model_plan_is_decoded_only_once(monkeypatch):
    raw = json.dumps(complete_plan())
    original = json.loads
    calls = []

    def load(value, **kwargs):
        calls.append(value)
        return original(value, **kwargs)

    async def review(_messages):
        return raw

    monkeypatch.setattr(retrieval_query_plan.json, "loads", load)
    result = await plan_retrieval(QUERY, reviewer=review)
    assert "".join(result.dependencies.segments) == QUERY and result.public_obligations
    assert calls == [raw]


@pytest.mark.parametrize("container", [bytes, bytearray])
async def test_non_text_complete_model_plan_is_not_silently_decoded(container):
    raw = container(json.dumps(complete_plan()).encode("utf-8"))

    async def review(_messages):
        return raw

    with pytest.raises(QueryPlanReviewError, match="query_plan_invalid") as caught:
        await plan_retrieval(QUERY, reviewer=review)
    assert str(caught.value.__cause__) == "Query planner must return text"


@pytest.mark.asyncio
async def test_full_question_reaches_reviewer_without_source_or_answer_injection():
    async def review(messages):
        payload = json.loads(messages[-1]["content"])
        assert payload["query"] == QUERY
        assert "".join(segment["text"] for segment in payload["segments"]) == QUERY
        assert [segment["id"] for segment in payload["segments"]] == list(range(len(payload["segments"])))
        return json.dumps(complete_plan())
    result = await plan_retrieval(QUERY, reviewer=review)
    assert result.status == "applied"
    assert result.public_obligations[0]["query"] == query_segments(QUERY)[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["invalid", "exception", "timeout"])
async def test_failed_review_raises_with_original_cause(mode):
    async def review(messages):
        if mode == "exception":
            raise RuntimeError("Declared unit provider failure")
        if mode == "timeout":
            raise asyncio.TimeoutError("Declared unit timeout")
        value = complete_plan()
        value["search_views"] = [["invented"]]
        return json.dumps(value)
    with pytest.raises(QueryPlanReviewError) as caught:
        await plan_retrieval(QUERY, reviewer=review)
    expected = {"invalid": ("invalid", ValueError), "exception": ("provider_error", RuntimeError),
                "timeout": ("timeout", asyncio.TimeoutError)}
    status, cause = expected[mode]
    assert caught.value.status == status and isinstance(caught.value.__cause__, cause)


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["dependencies", "public_partitions", "legacy_unknown"])
async def test_incomplete_legacy_plan_is_rejected(field):
    async def review(_messages):
        value = complete_plan()
        if field == "legacy_unknown":
            value["dependencies"]["unknown"] = value["dependencies"].pop("unresolved_source")
        else:
            del value[field]
        return json.dumps(value)
    with pytest.raises(QueryPlanReviewError, match="query_plan_invalid") as caught:
        await plan_retrieval(QUERY, reviewer=review)
    assert caught.value.status == "invalid" and type(caught.value.__cause__) is ValueError


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", ["0", "0.09", "30.1", "nan", "inf", "bad"])
async def test_invalid_timeout_is_not_clamped_or_sent(monkeypatch, timeout):
    monkeypatch.setenv("RAG_TASK_PLANNER_TIMEOUT_SECONDS", timeout)
    async def forbidden(_messages):
        pytest.fail("Invalid timeout must fail before a model call")
    with pytest.raises(ValueError, match="RAG_TASK_PLANNER_TIMEOUT_SECONDS"):
        await plan_retrieval(QUERY, reviewer=forbidden)


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", ["0.1", "30"])
async def test_valid_timeout_boundaries_are_used_as_configured(monkeypatch, timeout):
    monkeypatch.setenv("RAG_TASK_PLANNER_TIMEOUT_SECONDS", timeout)
    original = asyncio.wait_for
    seen = []
    async def wait_for(coroutine, *, timeout):
        seen.append(timeout)
        return await original(coroutine, timeout=timeout)
    async def review(_messages):
        return json.dumps(complete_plan())
    monkeypatch.setattr(asyncio, "wait_for", wait_for)
    assert (await plan_retrieval(QUERY, reviewer=review)).status == "applied"
    assert seen == [float(timeout)]


@pytest.mark.asyncio
async def test_cancellation_propagates():
    async def review(_messages):
        raise asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await plan_retrieval(QUERY, reviewer=review)
