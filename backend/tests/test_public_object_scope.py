"""Full fictional scope positives precede wrong-object and quote controls."""

import asyncio
import json
from copy import deepcopy
from dataclasses import replace
from html import unescape

import pytest
from tests.test_public_task_evidence_review import PRIVATE, bundle, request

from inference.generation_request import build_generation_request
from knowledge.public_obligations import parse_public_partitions
from knowledge.public_task_evidence import render_public_tasks, review_public_candidates
from knowledge.turn_dependencies import parse_dependencies

QUERY = "对象映射：前者为青桥延期，后者为蓝岸续租。核对前者的费用，另核对后者的时长。读取我保存的偏好。"


def dependencies():
    return parse_dependencies(
        dict(current_input=[0], public_knowledge=[1], private_memory=[2], control=[], unresolved_source=[]), QUERY
    )


def obligations():
    return parse_public_partitions([dict(segment_id=1, cuts=["另核对后者"])], dependencies(), QUERY)


def scopes():
    return dict(
        scopes=[dict(task_id="public:1:0", objects=["青桥延期"]), dict(task_id="public:1:1", objects=["蓝岸续租"])]
    )


def source_reply():
    return dict(
        decisions=[
            dict(
                source_id="doc_1",
                task_ids=["public:1:0"],
                object_evidence=[dict(object_id="query-object:0", source_quote="青桥延期费用为31元")],
            ),
            dict(source_id="doc_2", task_ids=[], object_evidence=[]),
        ]
    )


async def run(scope_value=None, source_value=None, *, window=65536, scope_error=None):
    calls = []

    async def resolve(messages):
        calls.append("scope")
        payload = json.loads(messages[-1]["content"])
        assert set(payload) == {"query", "public_tasks"} and payload["query"] == QUERY
        assert (
            bundle()["results"][0]["content"] not in messages[-1]["content"] and PRIVATE not in messages[-1]["content"]
        )
        if scope_error:
            raise scope_error
        return json.dumps(scopes() if scope_value is None else scope_value)

    async def review(messages):
        calls.append("source")
        payload = json.loads(messages[-1]["content"])
        assert payload["query"] == QUERY and payload["object_scopes"]["query"] == QUERY
        assert payload["sources"][0]["original_body"].endswith("本人携带完整登记材料才能办理。")
        return json.dumps(source_reply() if source_value is None else source_value)

    result = await review_public_candidates(
        bundle(),
        dependencies(),
        QUERY,
        window_tokens=window,
        public_obligations=obligations(),
        reviewer=review,
        scope_reviewer=resolve,
    )
    req = request(result)
    req = replace(
        req, message=QUERY, retrieval=replace(req.retrieval, public_task_query=QUERY, public_dependency_indices=(1,))
    )
    return result, req, calls


async def positive():
    result, req, calls = await run()
    built = build_generation_request(req)
    rows = render_public_tasks(built.retrieval)
    assert calls == ["scope", "source"] and [r["status"] for r in rows] == [
        "related_candidate_admitted",
        "no_related_evidence",
    ]
    assert rows[0]["object_scope"] == "source_object_evidence_verified"
    assert all(r["semantic_coverage"] == "unverified" for r in rows)
    assert PRIVATE in unescape(built.messages[-1]["content"])
    assert bundle()["original_source_packets"][0]["original_body"] in unescape(built.messages[-1]["content"])
    return result, req


@pytest.mark.asyncio
async def test_question_objects_resolved_before_any_source_is_visible():
    await positive()


@pytest.mark.asyncio
async def test_former_source_cannot_confirm_latter_even_if_reviewer_proposes_both():
    await positive()
    value = source_reply()
    value["decisions"][0]["task_ids"].append("public:1:1")
    result, req, _ = await run(source_value=value)
    rows = render_public_tasks(build_generation_request(req).retrieval)
    assert [r["status"] for r in rows] == ["related_candidate_admitted", "object_scope_unverified"]
    assert result["public_task_review"]["decisions"][0]["task_ids"] == ["public:1:0"]
    assert PRIVATE in unescape(build_generation_request(req).messages[-1]["content"])


@pytest.mark.asyncio
async def test_invented_source_quote_never_supplies_missing_object():
    await positive()
    value = source_reply()
    value["decisions"][0]["task_ids"].append("public:1:1")
    value["decisions"][0]["object_evidence"].append(dict(object_id="query-object:1", source_quote="蓝岸续租时长8天"))
    _, req, _ = await run(source_value=value)
    assert render_public_tasks(build_generation_request(req).retrieval)[1]["status"] == "object_scope_unverified"


@pytest.mark.asyncio
async def test_missing_literal_quote_is_unverified_not_semantic_irrelevance():
    await positive()
    value = source_reply()
    value["decisions"][0]["object_evidence"] = []
    _, req, _ = await run(source_value=value)
    assert render_public_tasks(build_generation_request(req).retrieval)[0]["status"] == "object_scope_unverified"


@pytest.mark.asyncio
@pytest.mark.parametrize("defect", ["invented_object", "private_task", "bool_task", "duplicate_task", "missing_task"])
async def test_untrusted_scope_output_never_invents_object_or_drops_task(defect):
    await positive()
    value = scopes()
    if defect == "invented_object":
        value["scopes"][1]["objects"] = ["未在原文的赤榕证照"]
    elif defect == "private_task":
        value["scopes"][1]["task_id"] = "public:2:0"
    elif defect == "bool_task":
        value["scopes"][1]["task_id"] = True
    elif defect == "duplicate_task":
        value["scopes"][1] = deepcopy(value["scopes"][0])
    elif defect == "missing_task":
        value["scopes"].pop()
    result, req, calls = await run(scope_value=value)
    assert (
        calls == ["scope"]
        and result["abstained"]
        and result["public_task_review"]["reason"] == "invalid_or_incomplete_review"
    )
    assert PRIVATE in unescape(build_generation_request(req).messages[-1]["content"])


@pytest.mark.asyncio
@pytest.mark.parametrize("defect", ["scope_binding", "accepted_links", "quote", "input_digest"])
async def test_canonical_generation_rechecks_object_proof_receipt(defect):
    _, req = await positive()
    receipt = deepcopy(req.retrieval.public_task_review)
    if defect == "scope_binding":
        receipt["object_scopes"]["objects"][0]["query_text"] = "蓝岸续租"
    elif defect == "accepted_links":
        receipt["decisions"][0]["task_ids"].append("public:1:1")
    elif defect == "quote":
        receipt["scoped_decisions"][0]["object_evidence"][0]["source_quote"] = "青桥延期未在实际正文的证明"
    elif defect == "input_digest":
        receipt["object_scope_input_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        build_generation_request(replace(req, retrieval=replace(req.retrieval, public_task_review=receipt)))


@pytest.mark.asyncio
async def test_unresolved_object_does_not_become_absent_rule():
    await positive()
    value = scopes()
    value["scopes"][1]["objects"] = []
    _, req, _ = await run(scope_value=value)
    assert render_public_tasks(build_generation_request(req).retrieval)[1]["status"] == "object_scope_unresolved"


@pytest.mark.asyncio
async def test_source_blind_failure_and_budget_do_not_retry_or_erase_private():
    await positive()
    result, req, calls = await run(scope_error=RuntimeError("unavailable"))
    assert calls == ["scope"] and result["public_task_review"]["reason"] == "provider_error"
    assert PRIVATE in unescape(build_generation_request(req).messages[-1]["content"])
    result, _, calls = await run(window=1)
    assert calls == [] and result["public_task_review"]["reason"] == "complete_scope_input_budget_exceeded"


@pytest.mark.asyncio
async def test_cancelled_question_scope_propagates():
    await positive()
    with pytest.raises(asyncio.CancelledError):
        await run(scope_error=asyncio.CancelledError())


@pytest.mark.asyncio
async def test_scope_row_order_does_not_change_bound_objects():
    await positive()
    value = scopes()
    value["scopes"].reverse()
    _, req, _ = await run(scope_value=value)
    assert [r["status"] for r in render_public_tasks(build_generation_request(req).retrieval)] == [
        "related_candidate_admitted",
        "no_related_evidence",
    ]


@pytest.mark.asyncio
async def test_actual_api_uses_verified_object_links_and_retains_private_answer(monkeypatch):
    from types import SimpleNamespace

    from api import generate
    from character.models import CompiledCharacterContext
    from db.schemas import MessageRequest
    from knowledge import intent_detector, public_task_evidence, retrieval_query_plan
    from knowledge.retrieval_query_plan import RetrievalQueryPlan

    await positive()
    calls = []

    async def planner(_query):
        return RetrievalQueryPlan(status="applied", dependencies=dependencies(), public_obligations=obligations())

    async def retrieve(*_args, **_kwargs):
        return deepcopy(bundle())

    async def review(messages):
        payload = json.loads(messages[-1]["content"])
        if set(payload) == {"query", "public_tasks"}:
            calls.append("scope")
            return json.dumps(scopes())
        calls.append("source")
        value = source_reply()
        # Reproduce the actual faulty proposal: one source proposes both objects.
        value["decisions"][0]["task_ids"].append("public:1:1")
        return json.dumps(value)

    async def model(**kwargs):
        assert kwargs["max_tokens"] == 2048
        wire = unescape(kwargs["messages"][-1]["content"])
        assert PRIVATE in wire and bundle()["original_source_packets"][0]["original_body"] in wire
        assert "object_scope_unverified" in wire
        return "已有私人偏好及前者费用可答，后者时长本次没有对应依据。"

    monkeypatch.setattr(retrieval_query_plan, "plan_retrieval_views", planner)
    monkeypatch.setattr(intent_detector, "needs_rag", lambda _: (True, "unit", None))
    monkeypatch.setattr(generate, "_retrieve_rag_bundle", retrieve)
    monkeypatch.setattr(public_task_evidence, "_review", review)
    monkeypatch.setattr(generate, "_get_system_prompt", lambda _: "人物规则")
    context = CompiledCharacterContext("", "", PRIVATE, used_memory_ids=("fiction-private",))
    prepared = SimpleNamespace(compiled=context, history=())
    _, used, meta = await generate._generate_with_retrieval(
        MessageRequest(message=QUERY),
        None,
        prepared_character_turn=prepared,
        runtime_config=dict(maxTokens=2048, useKnowledgeBase=True),
        model_generate=model,
    )
    assert calls == ["scope", "source"] and used and meta["abstained"] and meta["answerMode"] == "partial_answer"
    assert "partial_public_evidence" in meta["warnings"]


@pytest.mark.asyncio
async def test_complete_object_capacity_is_not_invalid_provider_output():
    await positive()
    names = [f"虚构对象{i}" for i in range(65)]
    query = "对象映射：" + "、".join(names) + "。" + "".join(f"核对第{i}组对象的公开规则。" for i in range(9))
    dep = parse_dependencies(
        dict(
            current_input=[0], public_knowledge=list(range(1, 10)), private_memory=[], control=[], unresolved_source=[]
        ),
        query,
    )
    parts = parse_public_partitions([dict(segment_id=i, cuts=[]) for i in range(1, 10)], dep, query)
    calls = []

    async def resolve(_messages):
        calls.append("scope")
        return json.dumps(
            dict(scopes=[dict(task_id=parts[i]["index"], objects=names[i * 8 : (i + 1) * 8]) for i in range(9)])
        )

    async def forbidden(_messages):
        pytest.fail("A graph capacity stop must not send incomplete source input")

    result = await review_public_candidates(
        bundle(), dep, query, window_tokens=65536, public_obligations=parts, reviewer=forbidden, scope_reviewer=resolve
    )
    assert (
        calls == ["scope"]
        and result["public_task_review"]["reason"] == "object_scope_capacity_exceeded"
        and result["abstained"]
    )
    assert len(names) == 65 and names[-1] in query
