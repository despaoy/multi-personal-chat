"""Planner proposals are extractive ranking views, never source grants."""
import asyncio
import json

import pytest

from knowledge.retrieval_query_plan import QueryPlanReviewError, parse_search_views, plan_retrieval_views
from knowledge.turn_dependencies import query_segments

QUERY = "请按青川通道公共办理规则回答费用、时长和核验条件。" + "假设方案资料完整，实际情况未知。" * 40


def complete_plan():
    return dict(
        search_views=[["青川通道", "公共办理规则", "核验条件"]],
        dependencies=dict(private_memory=[], public_knowledge=[0],
                          current_input=list(range(1, len(query_segments(QUERY)))),
                          control=[], unresolved_source=[]),
        public_partitions=[dict(segment_id=0, cuts=[])],
    )


def test_literal_public_views_keep_original_query_and_deduplicate():
    raw = json.dumps({"search_views": [["青川通道", "公共办理规则", "费用", "时长"], ["青川通道", "公共办理规则", "费用", "时长"]]})
    assert parse_search_views(raw, QUERY) == ("青川通道 公共办理规则 费用 时长",)
    assert QUERY.startswith("请按青川通道")


@pytest.mark.parametrize("raw", [
    '{"search_views":[["未提供的答案"]]}',
    '{"search_views":[[true]]}',
    '{"search_views":{},"source_grants":["私密库"]}',
    '{"search_views":[],"search_views":[["青川通道"]]}',
    '{"search_views":[[]]}',
    '{"search_views":[["青川通道"]],"filters":{"knowledge_base_id":1}}',
])
def test_untrusted_output_cannot_invent_values_filters_or_permissions(raw):
    with pytest.raises(ValueError):
        parse_search_views(raw, QUERY)


@pytest.mark.asyncio
async def test_full_question_reaches_reviewer_without_source_or_answer_injection(monkeypatch):
    monkeypatch.setenv("RAG_TASK_PLANNER_ENABLED", "true")
    async def review(messages):
        payload = json.loads(messages[-1]["content"])
        assert payload["query"] == QUERY
        assert "".join(segment["text"] for segment in payload["segments"]) == QUERY
        assert [segment["id"] for segment in payload["segments"]] == list(range(len(payload["segments"])))
        return json.dumps(complete_plan())
    result = await plan_retrieval_views(QUERY, reviewer=review)
    assert result.status == "applied" and result.views == ("青川通道 公共办理规则 核验条件",)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["invalid", "exception", "timeout"])
async def test_failed_review_raises_with_original_cause(monkeypatch, mode):
    monkeypatch.setenv("RAG_TASK_PLANNER_ENABLED", "true")
    async def review(messages):
        if mode == "exception":
            raise RuntimeError("Declared unit provider failure")
        if mode == "timeout":
            raise asyncio.TimeoutError("Declared unit timeout")
        value = complete_plan()
        value["search_views"] = [["invented"]]
        return json.dumps(value)
    with pytest.raises(QueryPlanReviewError) as caught:
        await plan_retrieval_views(QUERY, reviewer=review)
    expected = {"invalid": ("invalid", ValueError), "exception": ("provider_error", RuntimeError),
                "timeout": ("timeout", asyncio.TimeoutError)}
    status, cause = expected[mode]
    assert caught.value.status == status and isinstance(caught.value.__cause__, cause)


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["dependencies", "public_partitions"])
async def test_incomplete_legacy_plan_is_rejected(monkeypatch, field):
    monkeypatch.setenv("RAG_TASK_PLANNER_ENABLED", "true")
    async def review(_messages):
        value = complete_plan()
        del value[field]
        return json.dumps(value)
    with pytest.raises(QueryPlanReviewError, match="query_plan_invalid"):
        await plan_retrieval_views(QUERY, reviewer=review)


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", ["0", "0.09", "30.1", "nan", "inf", "bad"])
async def test_invalid_timeout_is_not_clamped_or_sent(monkeypatch, timeout):
    monkeypatch.setenv("RAG_TASK_PLANNER_ENABLED", "true")
    monkeypatch.setenv("RAG_TASK_PLANNER_TIMEOUT_SECONDS", timeout)
    async def forbidden(_messages):
        pytest.fail("Invalid timeout must fail before a model call")
    with pytest.raises(ValueError, match="RAG_TASK_PLANNER_TIMEOUT_SECONDS"):
        await plan_retrieval_views(QUERY, reviewer=forbidden)


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", ["0.1", "30"])
async def test_valid_timeout_boundaries_are_used_as_configured(monkeypatch, timeout):
    monkeypatch.setenv("RAG_TASK_PLANNER_ENABLED", "true")
    monkeypatch.setenv("RAG_TASK_PLANNER_TIMEOUT_SECONDS", timeout)
    original = asyncio.wait_for
    seen = []
    async def wait_for(coroutine, *, timeout):
        seen.append(timeout)
        return await original(coroutine, timeout=timeout)
    async def review(_messages):
        return json.dumps(complete_plan())
    monkeypatch.setattr(asyncio, "wait_for", wait_for)
    assert (await plan_retrieval_views(QUERY, reviewer=review)).status == "applied"
    assert seen == [float(timeout)]


@pytest.mark.asyncio
async def test_cancellation_propagates(monkeypatch):
    monkeypatch.setenv("RAG_TASK_PLANNER_ENABLED", "true")
    async def review(_messages):
        raise asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await plan_retrieval_views(QUERY, reviewer=review)


@pytest.mark.asyncio
async def test_flat_search_view_strings_are_rejected(monkeypatch):
    monkeypatch.setenv("RAG_TASK_PLANNER_ENABLED", "true")
    async def review(messages):
        assert "必须是二维字符串数组" in messages[0]["content"]
        value = complete_plan()
        value["search_views"] = ["青川通道 公共办理规则 核验条件"]
        return json.dumps(value)
    with pytest.raises(QueryPlanReviewError, match="query_plan_invalid") as caught:
        await plan_retrieval_views(QUERY, reviewer=review)
    assert isinstance(caught.value.__cause__, ValueError)
    assert str(caught.value.__cause__) == "Invalid search-view spans"
