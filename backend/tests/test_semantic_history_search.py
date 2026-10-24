"""History search anchors require the existing semantic object receipt."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from scope_fixtures import open_scope_response

from db.schemas import MessageRequest
from inference.context_budget import ReviewContextBudget
from knowledge.public_question_binding import validate_question_binding
from knowledge.retrieval_core.registry import KnowledgeDomainConfig
from knowledge.turn_dependencies import KINDS, query_segments

NAME = "虚构松檐馆"
HISTORY = [
    {"role": "user", "content": f"请核对{NAME}的公共借阅规程。未提供实际规程，不能确认是否允许归还，也不能假设预约豁免大风限制。"},
    {"role": "assistant", "content": "合成未验证猜测：可能全年开放，预约者也许不受限制；这些不能作为事实。"},
]

pytestmark = pytest.mark.usefixtures("no_live_context_review")


def install(monkeypatch, tmp_path, query, history, *, resolved=True, corrupt=None):
    from api import generate
    from knowledge import public_question_binding, retrieval_query_plan
    from knowledge.retrieval_core import registry

    budget = ReviewContextBudget(65536)
    config = KnowledgeDomainConfig("synthetic", tmp_path, lambda _: [], aliases={NAME: NAME})
    monkeypatch.setattr(registry, "get_default_registry", lambda: SimpleNamespace(list_domains=lambda: [config]))
    actual_plan = retrieval_query_plan.plan_retrieval
    actual_binding = public_question_binding.resolve_question_binding
    observed = []

    async def plan_review(messages):
        payload = json.loads(messages[-1]["content"])
        assert payload["query"] == query and payload["history"] == history
        observed.append("plan")
        ids = list(range(len(query_segments(query))))
        return json.dumps(dict(dependencies={kind: ids if kind == "public_knowledge" else [] for kind in KINDS},
                               public_partitions=[dict(segment_id=i, cuts=[]) for i in ids]))

    async def binding_review(messages):
        payload = json.loads(messages[-1]["content"])
        assert payload["query"] == query and payload["reference_context"]["history"] == history
        observed.append("binding")
        return open_scope_response(messages, [dict(task_id=task["id"], objects=[NAME] if resolved else []) for task in payload["public_tasks"]])

    async def planner(message, **kwargs):
        return await actual_plan(message, reviewer=plan_review, **kwargs)

    async def resolver(dependencies, message, **kwargs):
        receipt = await actual_binding(dependencies, message, reviewer=binding_review, **kwargs)
        if corrupt == "digest":
            receipt["input_sha256"] = "0" * 64
        elif corrupt == "history":
            receipt["input"]["reference_context"]["history"][0]["content"] += "篡改前文。"
        elif corrupt == "origin":
            receipt["scopes"]["objects"][0]["reference_origin"]["message_index"] = 1
        return receipt

    monkeypatch.setattr(retrieval_query_plan, "plan_retrieval", planner)
    monkeypatch.setattr(public_question_binding, "resolve_question_binding", resolver)
    monkeypatch.setattr(generate, "get_provider_context_budget", lambda: SimpleNamespace(review=budget, window_tokens=65536, evidence_max_chars=0))
    monkeypatch.setattr(generate, "_rag_retrieval_timeout", lambda: 30.0)
    model = AsyncMock(side_effect=AssertionError("Retrieval admission controls must not generate an answer"))
    return generate, observed, model


@pytest.mark.parametrize("case", ["prefix", "unlisted", "long", "unknown", "explicit"])
async def test_only_semantically_bound_names_extend_the_complete_question(monkeypatch, tmp_path, case):
    query = "请给出原文依据。" if case in {"prefix", "long", "unknown"} else "劳驾逐项援引可核对的来源。"
    if case == "explicit":
        query = f"请给出{NAME}的原文依据。"
    history = [dict(row) for row in HISTORY]
    if case == "long":
        history[0]["content"] += "这是完整虚构背景，不得丢失末尾条件。" * 45 + "但仍没有已验证的公共规程。"
    resolved = case != "unknown"
    generate, observed, model = install(monkeypatch, tmp_path, query, history, resolved=resolved)
    expected = NAME + "\n" + query if resolved and NAME not in query else query

    class ReachedRetrieval(BaseException):
        pass

    async def retrieve(message, top_k, filters, **kwargs):
        assert message == expected and top_k == 3 and filters is None
        receipt = kwargs["question_binding"]
        assert receipt["input"]["query"] == query
        assert validate_question_binding(receipt, receipt["input"]) == receipt["scopes"]
        assert "未验证猜测" not in message and "是否允许归还" not in message
        observed.append("retrieval")
        raise ReachedRetrieval()

    monkeypatch.setattr(generate, "retrieve_chat_knowledge", retrieve)
    with pytest.raises(ReachedRetrieval):
        await generate._generate_with_retrieval(MessageRequest(message=query, history=history), None,
            runtime_config={"useKnowledgeBase": True}, model_generate=model)
    assert observed == ["plan", "binding", "retrieval"]
    model.assert_not_awaited()


@pytest.mark.parametrize("corrupt", ["digest", "history", "origin"])
async def test_modified_binding_cannot_authorize_history_search(monkeypatch, tmp_path, corrupt):
    query = "请给出原文依据。"
    history = [dict(row) for row in HISTORY]
    generate, observed, model = install(monkeypatch, tmp_path, query, history, corrupt=corrupt)
    retrieve = AsyncMock()
    monkeypatch.setattr(generate, "retrieve_chat_knowledge", retrieve)
    with pytest.raises(HTTPException) as caught:
        await generate._generate_with_retrieval(MessageRequest(message=query, history=history), None,
            runtime_config={"useKnowledgeBase": True}, model_generate=model)
    assert caught.value.status_code == 503 and isinstance(caught.value.__cause__, ValueError)
    assert observed == ["plan", "binding"]
    retrieve.assert_not_awaited()
    model.assert_not_awaited()
