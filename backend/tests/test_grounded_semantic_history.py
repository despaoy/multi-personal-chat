"""Source-blind history is semantic data, never a keyword or old-task append."""

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from inference.context_budget import ReviewContextBudget
from knowledge import dialogue_query, public_question_binding, retrieval_query_plan
from knowledge.grounded_answer.service import GroundedAnswerService
from knowledge.turn_dependencies import KINDS, query_segments

NAME = "虚构远岚档案馆"
HISTORY = [
    dict(
        role="user", content=f"请核对{NAME}的公共借还规则。未提供实际规程，不能假设周五可借，也不能从预约推断豁免原件。"
    ),
    dict(role="assistant", content="合成未验证猜测：可能全年开放；这些不是来源依据。"),
]

pytestmark = pytest.mark.usefixtures("no_live_context_review")


def controls(monkeypatch, query, history, *, names=None, kind="public_knowledge", corrupt=False):
    names = [NAME] if names is None else names
    budget = SimpleNamespace(window_tokens=65536, review=ReviewContextBudget(65536))
    monkeypatch.setattr(dialogue_query, "get_provider_context_budget", lambda: budget)
    exchanges = []

    async def planning(messages):
        payload = json.loads(messages[-1]["content"])
        assert payload["query"] == query and payload["history"] == history
        exchanges.append("plan")
        ids = list(range(len(query_segments(query))))
        return json.dumps(
            dict(
                dependencies={key: ids if key == kind else [] for key in KINDS},
                public_partitions=[dict(segment_id=i, cuts=[]) for i in ids] if kind == "public_knowledge" else [],
            )
        )

    async def objects(messages):
        payload = json.loads(messages[-1]["content"])
        assert payload["query"] == query and payload["reference_context"]["history"] == history
        exchanges.append("binding")
        return json.dumps(dict(scopes=[dict(task_id=task["id"], objects=names) for task in payload["public_tasks"]]))

    async def plan(*args, **kwargs):
        return await retrieval_query_plan.plan_retrieval(*args, reviewer=planning, **kwargs)

    async def resolve(*args, **kwargs):
        receipt = await public_question_binding.resolve_question_binding(*args, reviewer=objects, **kwargs)
        if corrupt:
            receipt["input_sha256"] = "0" * 64
        return receipt

    monkeypatch.setattr(dialogue_query, "plan_retrieval", plan)
    monkeypatch.setattr(dialogue_query, "resolve_question_binding", resolve)
    return exchanges


@pytest.mark.parametrize(
    "query", ["请给出原文依据。", "请核实上述机构现行借还规程。", "只核对周五借还，不涉及旧问题，预约也不豁免原件。"]
)
async def test_proven_object_supplements_complete_query_without_old_tasks(monkeypatch, query):
    history = deepcopy(HISTORY)
    before = deepcopy(history)
    exchanges = controls(monkeypatch, query, history)
    resolved = await dialogue_query.contextual_retrieval_query(query, history)
    assert resolved == NAME + "\n" + query
    assert "不能假设周五可借" not in resolved and "全年开放" not in resolved
    assert history == before and exchanges == ["plan", "binding"]


async def test_long_user_reference_keeps_its_full_tail_for_both_reviewers(monkeypatch):
    history = deepcopy(HISTORY)
    history[0]["content"] = "背景" * 1100 + history[0]["content"] + "最终限制：不得混用其他机构。"
    query = "核对它的现行规程，保留全部例外。"
    controls(monkeypatch, query, history)
    assert await dialogue_query.contextual_retrieval_query(query, history) == NAME + "\n" + query


async def test_new_explicit_object_does_not_inherit_a_previous_topic(monkeypatch):
    query = "请核对虚构临桥图书室的公开借阅期限。"
    controls(monkeypatch, query, HISTORY, names=["虚构临桥图书室"])
    assert await dialogue_query.contextual_retrieval_query(query, HISTORY) == query


async def test_empty_semantic_object_set_keeps_complete_literal_question(monkeypatch):
    query = "请解释资料中没有明确命名的那项业务。"
    controls(monkeypatch, query, HISTORY, names=[])
    assert await dialogue_query.contextual_retrieval_query(query, HISTORY) == query


async def test_no_history_needs_no_semantic_call(monkeypatch):
    review = AsyncMock(side_effect=AssertionError("A standalone literal query needs no history review"))
    monkeypatch.setattr(dialogue_query, "plan_retrieval", review)
    assert await dialogue_query.contextual_retrieval_query("  完整当前问题。  ", None) == "  完整当前问题。  "
    review.assert_not_awaited()


@pytest.mark.parametrize("failure", ["planner", "binding", "changed_proof", "assistant_only"])
@pytest.mark.parametrize("transport", ["normal", "stream"])
async def test_failed_review_cannot_retrieve_generate_or_publish_a_partial_stream(monkeypatch, failure, transport):
    query = "请给出原文依据。"
    history = deepcopy(HISTORY)
    if failure == "assistant_only":
        history = [dict(role="user", content="用户没有提供对象名称。"), dict(role="assistant", content=NAME)]
    exchanges = controls(monkeypatch, query, history, corrupt=failure == "changed_proof")
    error = RuntimeError("controlled unavailable review")
    if failure in {"planner", "binding"}:
        monkeypatch.setattr(
            dialogue_query,
            "plan_retrieval" if failure == "planner" else "resolve_question_binding",
            AsyncMock(side_effect=error),
        )
    retrieve, model = Mock(), AsyncMock()
    service = GroundedAnswerService(retriever=retrieve)
    emitted = []
    with pytest.raises((RuntimeError, ValueError)):
        if transport == "normal":
            await service.answer(query, history=history, generate=model)
        else:
            async for event in service.answer_stream(query, history=history, generate=model):
                emitted.append(event)
    retrieve.assert_not_called()
    model.assert_not_awaited()
    assert emitted == []
    assert exchanges == ([] if failure == "planner" else ["plan"] if failure == "binding" else ["plan", "binding"])


@pytest.mark.parametrize("kind", ["private_memory", "current_input"])
async def test_history_without_public_tasks_does_not_enter_public_retrieval(monkeypatch, kind):
    query = "请核对我前文所说的内容。"
    exchanges = controls(monkeypatch, query, HISTORY, kind=kind)
    retrieve, model = Mock(), AsyncMock()
    result = await GroundedAnswerService(retriever=retrieve).answer(query, history=HISTORY, generate=model)
    assert result.abstained and result.failure_kind == "no_domain"
    retrieve.assert_not_called()
    model.assert_not_awaited()
    assert exchanges == ["plan"]


@pytest.mark.parametrize("transport", ["normal", "stream"])
async def test_semantic_query_reaches_retrieval_while_original_input_reaches_answer(monkeypatch, transport):
    from knowledge.grounded_answer.cache import AnswerCache

    query = "请核实上述机构周五的借阅规则；预约是否免原件？"
    history = deepcopy(HISTORY)
    exchanges = controls(monkeypatch, query, history)
    source = f"{NAME}现行规则：周五不开放借阅；预约不豁免原件。"
    bundle = dict(
        domains=["synthetic"],
        confidence=0.95,
        abstained=False,
        query_analysis=dict(entities=[NAME], matched_domains=["synthetic"]),
        results=[
            dict(
                id="current-rules",
                document_type="fact",
                title=NAME,
                summary=source,
                content=source,
                domain_id="synthetic",
                index_version="v1",
                rerank_score=0.95,
                fused_score=0.95,
                vector_score=0.95,
                bm25_score=0.95,
                reality_status="objective",
                temporal_scope="current",
                content_scope="main_story",
            )
        ],
        citations=[dict(source_id="current-rules", source_title=NAME, domain_id="synthetic", index_version="v1")],
    )
    retrieve = Mock(return_value=bundle)

    async def generate(**kwargs):
        messages = kwargs["messages"]
        assert messages[1:-1] == history
        assert query in messages[-1]["content"] and source in messages[-1]["content"]
        return "周五不开放借阅；预约不豁免原件。[S1]"

    model = AsyncMock(side_effect=generate)
    service = GroundedAnswerService(
        retriever=retrieve,

        cache=AnswerCache(ttl_seconds=60),
        index_version_resolver=lambda _: "v1",
    )
    if transport == "normal":
        result = await service.answer(query, history=history, generate=model)
        assert result.citations and not result.abstained
        cached = await service.answer(query, history=history, generate=model)
        assert cached.cache_hit and not cached.model_invoked and cached.answer == result.answer
    else:
        events = [event async for event in service.answer_stream(query, history=history, generate=model)]
        assert [event.type for event in events] == ["meta", "delta", "citations", "done"]
        assert events[-1].data["model_invoked"] and not events[-1].data["abstained"]
    retrieve.assert_called_once_with(NAME + "\n" + query, top_k=3, filters=None, domain_id=None)
    model.assert_awaited_once()
    assert exchanges == ["plan", "binding"] and history == HISTORY


@pytest.mark.parametrize("transport", ["normal", "stream"])
async def test_unknown_source_keeps_original_retrieval_without_authorizing_old_objects(monkeypatch, transport):
    query = "请核对前文那项结论，目前尚未明确它的来源。"
    exchanges = controls(monkeypatch, query, HISTORY, kind="unresolved_source")
    retrieve, model = Mock(return_value=None), AsyncMock()
    service = GroundedAnswerService(retriever=retrieve)
    if transport == "normal":
        result = await service.answer(query, history=HISTORY, generate=model)
        assert result.abstained and not result.model_invoked
    else:
        events = [event async for event in service.answer_stream(query, history=HISTORY, generate=model)]
        assert events[-1].data["abstained"] and not events[-1].data["model_invoked"]
    retrieve.assert_called_once_with(query, top_k=3, filters=None, domain_id=None)
    model.assert_not_awaited()
    assert exchanges == ["plan"]
