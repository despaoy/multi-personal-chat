"""Planner proposals are extractive ranking views, never source grants."""
import json

import pytest

from knowledge.retrieval_query_plan import parse_search_views, plan_retrieval_views

QUERY = "请按青川通道公共办理规则回答费用、时长和核验条件。" + "假设方案资料完整，实际情况未知。" * 40


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
        assert json.loads(messages[-1]["content"]) == {"query": QUERY}
        return json.dumps({"search_views": [["青川通道", "公共办理规则", "核验条件"]]})
    result = await plan_retrieval_views(QUERY, reviewer=review)
    assert result.status == "applied" and result.views == ("青川通道 公共办理规则 核验条件",)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["disabled", "short", "corrective"])
async def test_unsupported_or_short_tasks_do_not_add_a_paid_call(monkeypatch, mode):
    monkeypatch.setenv("RAG_TASK_PLANNER_ENABLED", "false" if mode == "disabled" else "true")
    monkeypatch.setenv("CORRECTIVE_RAG_ENABLED", "true" if mode == "corrective" else "false")
    async def review(messages):
        pytest.fail("Reviewer must not be called")
    result = await plan_retrieval_views("短问题" if mode == "short" else QUERY, reviewer=review)
    assert not result.views


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["invalid", "exception"])
async def test_failed_review_retains_original_retrieval_path(monkeypatch, mode):
    monkeypatch.setenv("RAG_TASK_PLANNER_ENABLED", "true")
    async def review(messages):
        if mode == "exception":
            raise RuntimeError("Unavailable")
        return '{"search_views":[["invented"]]}'
    result = await plan_retrieval_views(QUERY, reviewer=review)
    assert not result.views and result.status in {"invalid", "unavailable"}
