"""Independent history contracts; model stubs here are explicit unit controls."""

from services import chat_answer

import json
from types import SimpleNamespace

import pytest

from inference.context_budget import InputBudgetError
from inference.context_budget import ReviewContextBudget
from knowledge.retrieval_query_plan import QueryPlanReviewError, plan_retrieval, planning_messages
from knowledge.turn_dependencies import KINDS, parse_dependencies, query_segments

QUERY = "按刚才的两项分别核对，不改记录。"
PREMISE = "第一项核对本人已保存的周末阅读偏好及雨天例外。第二项只按本题材料计算：数量3、单价2.25元，没有其他费用。这是虚构假设，不是本人新事实。"


def test_full_original_query_and_complete_history_are_distinct_untrusted_data():
    history = [
        {"role": "user", "content": PREMISE},
        {"role": "assistant", "content": "尚未核对，不能把我的猜测当成事实。"},
    ]
    payload = json.loads(planning_messages(QUERY, history=history)[1]["content"])
    assert payload["history"] == history
    assert payload["query"] == QUERY
    assert "".join(item["text"] for item in payload["segments"]) == QUERY
    assert PREMISE not in "".join(item["text"] for item in payload["segments"])


def test_empty_history_preserves_current_only_payload():
    assert set(json.loads(planning_messages(QUERY)[1]["content"])) == {"query", "segments"}


def test_history_budget_keeps_complete_user_turn_without_orphan_assistant():
    history = [
        {"role": "user", "content": "旧请求"},
        {"role": "assistant", "content": "旧回答"},
        {"role": "user", "content": PREMISE},
        {"role": "assistant", "content": "等待追问"},
    ]
    payload = json.loads(
        planning_messages(QUERY, history=history, context_budget=ReviewContextBudget(8192, history_messages=3))[1][
            "content"
        ]
    )
    assert payload["history"] == history[-2:]
    assert payload["history_omitted_messages"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["latest_turn_count", "latest_turn_chars", "wire_tokens"])
async def test_incomplete_review_fails_before_any_model_call(monkeypatch, mode):
    monkeypatch.setenv("RAG_TASK_PLANNER_ENABLED", "true")
    history = [{"role": "user", "content": PREMISE}]
    budget = ReviewContextBudget(1024, history_messages=1)
    if mode == "latest_turn_count":
        history.append({"role": "assistant", "content": "不能单独留下这个回答。"})
    if mode == "latest_turn_chars":
        history[0]["content"] = PREMISE + "完整条件。" * 1000

    async def forbidden(_messages):
        pytest.fail("Incomplete history or mandatory input must not be sent")

    with pytest.raises(InputBudgetError):
        await plan_retrieval(QUERY, history=history, context_budget=budget, reviewer=forbidden)


@pytest.mark.asyncio
async def test_history_cannot_grant_search_spans_not_in_current_question(monkeypatch):
    monkeypatch.setenv("RAG_TASK_PLANNER_ENABLED", "true")

    async def unit_review(_messages):
        groups = {kind: [] for kind in KINDS}
        groups["public_knowledge"] = [0]
        return json.dumps({"search_views": [["周末阅读偏好"]], "dependencies": groups, "public_partitions": [dict(segment_id=0, cuts=[])]})

    with pytest.raises(QueryPlanReviewError, match="query_plan_invalid"):
        await plan_retrieval(QUERY, history=[{"role": "user", "content": PREMISE}], reviewer=unit_review)


@pytest.mark.asyncio
async def test_authority_fields_in_model_output_are_rejected(monkeypatch):
    monkeypatch.setenv("RAG_TASK_PLANNER_ENABLED", "true")

    async def unit_review(_messages):
        groups = {kind: [] for kind in KINDS}
        groups["private_memory"] = [0]
        return json.dumps({"dependencies": groups, "public_partitions": [], "source_grants": ["all_owners"]})

    with pytest.raises(QueryPlanReviewError, match="query_plan_invalid"):
        await plan_retrieval(QUERY, history=[{"role": "user", "content": PREMISE}], reviewer=unit_review)


@pytest.mark.asyncio
@pytest.mark.parametrize("history_source", ["prepared_database", "request_override"])
async def test_actual_generation_route_passes_the_same_effective_history(monkeypatch, history_source):
    from character.models import CompiledCharacterContext
    from db.schemas import MessageRequest
    from knowledge import retrieval_query_plan

    database_history = ({"role": "user", "content": PREMISE},)
    request_history = [
        {"role": "user", "content": "现场调用方完整前文：核对我的已有偏好，另计算材料4加7，未授权改写。"}
    ]
    expected = tuple(request_history) if history_source == "request_override" else database_history
    observed = []

    async def unit_planner(query, **kwargs):
        assert query == QUERY and tuple(kwargs["history"]) == expected
        observed.append("planner")
        groups = {kind: [] for kind in KINDS}
        groups["private_memory"] = list(range(len(query_segments(query))))
        return retrieval_query_plan.RetrievalQueryPlan(status="applied", dependencies=parse_dependencies(groups, query))

    async def forbidden_public(*_args, **_kwargs):
        pytest.fail("The explicit unit private dependency must preserve the private path")

    async def unit_main(**kwargs):
        contents = [message["content"] for message in kwargs["messages"]]
        assert all(row["content"] in contents for row in expected)
        observed.append("main")
        return "这是明确标记的单元测试控制回答，不是真实模型成功。"

    prepared = SimpleNamespace(compiled=CompiledCharacterContext("", "", ""), history=database_history)
    monkeypatch.setattr(retrieval_query_plan, "plan_retrieval", unit_planner)
    monkeypatch.setattr(chat_answer, 'retrieve_chat_knowledge', forbidden_public)
    monkeypatch.setattr(chat_answer, 'get_lora_system_prompt', lambda _: "单元角色规则")
    _, _, meta = await chat_answer.generate_chat_answer(
        MessageRequest(message=QUERY, history=request_history if history_source == "request_override" else []),
        None,
        prepared_character_turn=prepared,
        runtime_config={"useKnowledgeBase": True},
        model_generate=unit_main,
    )
    assert observed == ["planner", "main"] and meta["answerMode"] == "personal_context"
