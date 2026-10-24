"""Semantic dependencies alone dispatch retrieval; no rule or classifier gate."""

import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from db.schemas import MessageRequest
from inference.context_budget import ReviewContextBudget
from knowledge.turn_dependencies import KINDS

CASES = [
    ("读取我保存的饮品偏好与全部例外；只读，不新增或删除记忆。", dict(private_memory=[0], control=[1]), False),
    ("按本题完整假设计费：票数3、每张2.25元，没有其他费用；只计算总额，不修改记录。", dict(current_input=[0], control=[1]), False),
    ("只用一句话回复，并且不要提问。", dict(control=[0]), False),
    ("核对虚构夏屿展览公开规程的费用、开放时段与雨天例外；资料缺失则保持未知。", dict(public_knowledge=[0], control=[1]), True),
    ("核对那个未说明出处的结论；缺少出处则说明无法确定。", dict(unresolved_source=[0], control=[1]), True),
    ("读取我保存的饮品偏好；同时核对虚构夏屿展览公开规程的费用与雨天限制；只读，不新增或删除记忆。", dict(private_memory=[0], public_knowledge=[1], control=[2]), True),
]


@pytest.mark.parametrize("query,groups,expected", CASES)
async def test_complete_semantic_plan_alone_controls_public_dispatch(monkeypatch, query, groups, expected):
    from api import generate
    from knowledge import intent_detector, public_question_binding, retrieval_query_plan

    monkeypatch.setattr(generate, "_rag_retrieval_timeout", lambda: 30.0)

    seen = []
    actual_plan = retrieval_query_plan.plan_retrieval
    budget = ReviewContextBudget(65536)
    history = ({"role": "user", "content": "这是完整虚构测试；没有额外已保存的偏好或公共规程。"},)

    def forbidden(*args, **kwargs):
        pytest.fail("Current chat must not call the rule engine or untrained classifier")

    async def review(messages):
        payload = json.loads(messages[-1]["content"])
        assert payload["query"] == query and payload["history"] == list(history)
        assert "".join(row["text"] for row in payload["segments"]) == query
        seen.append("plan")
        return json.dumps(dict(
            dependencies={kind: groups.get(kind, []) for kind in KINDS},
            public_partitions=[dict(segment_id=i, cuts=[]) for i in groups.get("public_knowledge", [])],
        ))

    async def planner(message, **kwargs):
        assert kwargs["context_budget"] is budget and kwargs["history"] == history
        return await actual_plan(message, reviewer=review, **kwargs)

    async def binding(*args, **kwargs):
        return None  # No object proof is granted by this dispatch control.

    async def retrieve(message, top_k, filters, **kwargs):
        assert message == query and filters is None
        seen.append("retrieve")
        return dict(results=[], citations=[], abstained=True)

    async def model(**kwargs):
        wire = "\n".join(message["content"] for message in kwargs["messages"])
        assert query in wire and history[0]["content"] in wire
        seen.append("answer")
        return "单元控制：当前未提供可核对的记录或规程，缺少依据保持未知。"

    monkeypatch.setenv("RAG_TASK_PLANNER_ENABLED", "false")
    monkeypatch.setenv("DYNAMIC_CONTEXT_SEMANTIC_REVIEW_ENABLED", "false")
    monkeypatch.setattr(intent_detector, "predict_rag_intent", forbidden)
    monkeypatch.setattr(intent_detector, "_load_ml_model", forbidden)
    monkeypatch.setattr(retrieval_query_plan, "plan_retrieval", planner)
    monkeypatch.setattr(public_question_binding, "resolve_question_binding", binding)
    monkeypatch.setattr(generate, "retrieve_chat_knowledge", retrieve)
    monkeypatch.setattr(generate, "_get_system_prompt", lambda _: "完整虚构角色，只依据已给出的内容回答。")
    monkeypatch.setattr(generate, "get_provider_context_budget", lambda: SimpleNamespace(
        review=budget, window_tokens=65536, evidence_max_chars=0,
    ))
    _, used, metadata = await generate._generate_with_retrieval(
        MessageRequest(message=query, history=list(history)), None,
        runtime_config=dict(useKnowledgeBase=True, maxTokens=256), model_generate=model,
    )
    assert seen == (["plan", "retrieve", "answer"] if expected else ["plan", "answer"])
    assert used is expected and bool(metadata.get("abstained")) is expected


@pytest.mark.parametrize("failure", ["provider", "invalid", "capacity"])
async def test_semantic_gate_failure_never_continues_with_rules_or_answer(monkeypatch, failure):
    from api import generate
    from knowledge import intent_detector, retrieval_query_plan

    query = CASES[3][0]
    actual_plan = retrieval_query_plan.plan_retrieval
    error = RuntimeError("Declared isolated semantic provider failure")

    async def review(messages):
        if failure == "provider":
            raise error
        if failure == "capacity":
            pytest.fail("Insufficient complete-input capacity must stop before the provider")
        return '{"search_views":[]}'

    async def planner(message, **kwargs):
        return await actual_plan(message, reviewer=review, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("Failed semantic planning cannot continue with rules, retrieval or answer")

    monkeypatch.setattr(intent_detector, "predict_rag_intent", forbidden)
    monkeypatch.setattr(retrieval_query_plan, "plan_retrieval", planner)
    monkeypatch.setattr(generate, "retrieve_chat_knowledge", forbidden)
    monkeypatch.setattr(generate, "_get_system_prompt", forbidden)
    monkeypatch.setattr(generate, "get_provider_context_budget", lambda: SimpleNamespace(
        review=ReviewContextBudget(1024 if failure == "capacity" else 65536),
    ))
    with pytest.raises(HTTPException) as caught:
        await generate._generate_with_retrieval(
            MessageRequest(message=query), None, runtime_config=dict(useKnowledgeBase=True), model_generate=forbidden,
        )
    assert caught.value.status_code == 503
    if failure == "provider":
        assert caught.value.__cause__.__cause__ is error


async def test_explicit_retrieval_off_does_not_pay_for_planning(monkeypatch):
    from api import generate
    from knowledge import retrieval_query_plan

    query = "只按本题完整假设计费：票数3、单价2.25元、没有其他费用。"

    def forbidden(*args, **kwargs):
        pytest.fail("Explicit retrieval-off must not plan or retrieve")

    async def model(**kwargs):
        assert query in "\n".join(message["content"] for message in kwargs["messages"])
        return "本题假设总额为6.75元。"

    monkeypatch.setattr(retrieval_query_plan, "plan_retrieval", forbidden)
    monkeypatch.setattr(generate, "retrieve_chat_knowledge", forbidden)
    monkeypatch.setattr(generate, "_get_system_prompt", lambda _: "只根据完整题设计算。")
    monkeypatch.setattr(generate, "get_provider_context_budget", lambda: SimpleNamespace(window_tokens=65536, evidence_max_chars=0))
    _, used, _ = await generate._generate_with_retrieval(
        MessageRequest(message=query), None, runtime_config=dict(useKnowledgeBase=False), model_generate=model,
    )
    assert not used
