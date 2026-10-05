"""Independent fictional controls; injected reviewers are unit controls only."""

import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from inference.context_budget import ReviewContextBudget
from knowledge.public_object_scope import parse_object_scopes
from knowledge.public_question_binding import (
    QuestionBindingCapacityError,
    QuestionBindingReviewError,
    bound_object_search_views,
    failed_question_binding_review,
    resolve_question_binding,
    validate_question_binding,
)
from knowledge.question_reference_context import build_reference_context
from knowledge.turn_dependencies import KINDS, parse_dependencies

QUERY = "查它们的费用与材料，保留预约不免原件的限制。"
HISTORY = [
    {"role": "user", "content": "前者是竹湾登记，后者是枫台续期；查询各自公共规定，只读，不能混用例外。"},
    {"role": "assistant", "content": "这是独立单元测试中的未验证助手文字，不是公共事实或新增授权。"},
]


def dependencies():
    groups = {kind: [] for kind in KINDS}
    groups["public_knowledge"] = [0]
    return parse_dependencies(groups, QUERY)


async def binding():
    async def unit_review(messages):
        payload = json.loads(messages[-1]["content"])
        assert payload["query"] == QUERY
        assert payload["reference_context"]["history"] == HISTORY
        assert payload["public_tasks"] == [{"id": 0, "text": QUERY}]
        return json.dumps({"scopes": [{"task_id": 0, "objects": ["竹湾登记", "枫台续期"]}]})

    return await resolve_question_binding(
        dependencies(),
        QUERY,
        window_tokens=65536,
        history=HISTORY,
        context_budget=ReviewContextBudget(65536),
        reviewer=unit_review,
    )


async def test_complete_original_user_history_preserves_object_reference_provenance():
    receipt = await binding()
    assert validate_question_binding(receipt, receipt["input"]) == receipt["scopes"]
    assert [r["query_text"] for r in receipt["scopes"]["objects"]] == ["竹湾登记", "枫台续期"]
    assert all(
        r["reference_origin"] == {"message_index": 0, "quote": r["query_text"]} for r in receipt["scopes"]["objects"]
    )
    assert bound_object_search_views(receipt, QUERY) == ("竹湾登记", "枫台续期")
    assert set(receipt["input"]) == {"query", "public_tasks", "reference_context"}
    assert receipt["scopes"]["query"] == QUERY


@pytest.mark.parametrize("source", ["assistant_only", "invented", "omitted_user"])
def test_assistant_guesses_and_absent_user_names_cannot_supply_objects(source):
    history = [
        {"role": "user", "content": "对象没有说清，请保留未知。"},
        {"role": "assistant", "content": "猜测对象为竹湾登记。"},
    ]
    context = build_reference_context(history)
    name = "竹湾登记" if source != "invented" else "未提供业务"
    if source == "omitted_user":
        context = build_reference_context(HISTORY + history, context_budget=ReviewContextBudget(65536, 2))
        assert context["omitted_messages"] == 2
    with pytest.raises(ValueError):
        parse_object_scopes(
            json.dumps({"scopes": [{"task_id": 0, "objects": [name]}]}), QUERY, [0], reference_context=context
        )


@pytest.mark.parametrize("changed", ["history", "origin", "digest"])
async def test_reference_or_provenance_tampering_cannot_reuse_actual_receipt(changed):
    receipt = await binding()
    expected = deepcopy(receipt["input"])
    if changed == "history":
        receipt["input"]["reference_context"]["history"][0]["content"] += "删除例外。"
    elif changed == "origin":
        receipt["scopes"]["objects"][0]["reference_origin"]["message_index"] = 1
    else:
        receipt["input_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validate_question_binding(receipt, expected)


@pytest.mark.parametrize("mode", ["whole_turn_count", "wire_capacity"])
async def test_incomplete_or_over_capacity_reference_input_is_rejected_before_model(mode):
    async def forbidden(_messages):
        pytest.fail("No cloud call may receive partially sliced reference history")

    with pytest.raises(QuestionBindingCapacityError):
        await resolve_question_binding(
            dependencies(),
            QUERY,
            window_tokens=128 if mode == "wire_capacity" else 65536,
            history=HISTORY,
            context_budget=ReviewContextBudget(65536, 1 if mode == "whole_turn_count" else 128),
            reviewer=forbidden,
        )


async def test_failure_diagnostic_keeps_complete_reference_input_without_granting_binding():
    async def invalid(_messages):
        return '{"scopes":[]}'

    with pytest.raises(QuestionBindingReviewError) as caught:
        await resolve_question_binding(dependencies(), QUERY, window_tokens=65536, history=HISTORY, reviewer=invalid)
    diagnostic = failed_question_binding_review(caught.value, dependencies(), QUERY, history=HISTORY)
    assert diagnostic["review_status"] == "unavailable" and diagnostic["decisions"] == []
    assert (
        json.loads(diagnostic["failure_diagnostic"]["input"][-1]["content"])["reference_context"]["history"] == HISTORY
    )
    with pytest.raises(ValueError):
        failed_question_binding_review(caught.value, dependencies(), QUERY)


async def test_only_exact_bound_reference_views_pass_retrieval_validation():
    from knowledge.rag_helper import RAGHelper

    receipt = await binding()
    assert RAGHelper._validated_task_views(QUERY, ("竹湾登记", "枫台续期"), question_binding=receipt)
    for query, views, proof in [
        (QUERY, ("竹湾登记",), None),
        (QUERY, ("其他业务",), receipt),
        ("查询另一个对象。", ("竹湾登记",), receipt),
    ]:
        with pytest.raises(ValueError):
            RAGHelper._validated_task_views(query, views, question_binding=proof)


async def test_reference_binding_is_reused_in_source_review_and_canonical_receipt(monkeypatch):
    from test_public_task_evidence_review import request

    from inference.generation_request import build_generation_request
    from knowledge.original_sources import attach_original_sources
    from knowledge.public_task_evidence import render_public_tasks, review_public_candidates

    bodies = ["竹湾登记费用12元，须登记表原件；预约不免原件。", "枫台续期费用18元，须确认函原件；预约不免原件。"]
    rows = [
        dict(
            id=f"doc_{i}_chunk_0",
            document_id=i,
            title=f"独立完整测试规则{i}",
            content=body,
            category="fiction",
            knowledge_base_id=7,
            score=0.9,
        )
        for i, body in enumerate(bodies, 1)
    ]
    original = attach_original_sources(
        dict(
            results=rows,
            citations=[],
            confidence=0.9,
            abstained=False,
            source_coverage=tuple(
                dict(
                    source_id=f"doc_{i}",
                    source_title=rows[i - 1]["title"],
                    indexed_document_ids=[rows[i - 1]["id"]],
                    retrieved_document_ids=[rows[i - 1]["id"]],
                )
                for i in [1, 2]
            ),
        ),
        lambda i: dict(rows[i - 1], id=i),
        source_budget_tokens=65536,
        authority_revision=3,
    )
    receipt = await binding()

    async def forbidden(_messages):
        pytest.fail("Object binding must be reused, not called again")

    async def unit_source(messages):
        payload = json.loads(messages[-1]["content"])
        assert payload["object_scopes"] == receipt["scopes"]
        assert [s["original_body"] for s in payload["sources"]] == bodies
        return json.dumps(
            {
                "decisions": [
                    dict(
                        source_id=f"doc_{i}",
                        task_ids=[0],
                        object_evidence=[dict(object_id=f"query-object:{i - 1}", source_quote=body)],
                    )
                    for i, body in enumerate(bodies, 1)
                ]
            }
        )

    result = await review_public_candidates(
        original,
        dependencies(),
        QUERY,
        window_tokens=65536,
        reviewer=unit_source,
        scope_reviewer=forbidden,
        question_binding=receipt,
    )
    assert result["public_task_review"]["review_status"] == "reviewed"
    req = request(result)
    req = replace(
        req,
        message=QUERY,
        history=tuple(HISTORY),
        retrieval=replace(req.retrieval, public_task_query=QUERY, public_dependency_indices=(0,)),
    )
    from knowledge import public_domains

    config = SimpleNamespace(domain_id="unit_reference_domain", story_titles=[], canonical_entity=lambda _: None)
    original_validate = public_domains.validate_domain_plan
    monkeypatch.setattr(
        public_domains, "validate_domain_plan", lambda candidate: original_validate(candidate, config=config)
    )
    plan = public_domains.domain_plan(receipt, config)
    req = replace(
        req,
        retrieval=replace(
            req.retrieval,
            public_domain_branches=dict(
                plan=plan, curated_status="unavailable", generic_status="retrieved", curated_packet_manifest=[]
            ),
        ),
    )
    built = build_generation_request(req)
    assert render_public_tasks(built.retrieval)[0]["object_scope"] == "source_object_evidence_verified"
    mutated = deepcopy(built.retrieval.public_task_review)
    mutated["object_scopes"]["reference_context"]["history"][0]["content"] += "新增权限"
    with pytest.raises(ValueError):
        build_generation_request(replace(req, retrieval=replace(req.retrieval, public_task_review=mutated)))


@pytest.mark.parametrize("history_source", ["prepared_database", "request_override"])
async def test_normal_api_binding_receives_same_effective_history_as_planning(monkeypatch, history_source):
    from api import generate
    from character.models import CompiledCharacterContext
    from db.schemas import MessageRequest
    from knowledge import intent_detector, public_question_binding, retrieval_query_plan

    expected = (
        HISTORY
        if history_source == "prepared_database"
        else [dict(role="user", content="本次用户前文：两项业务是竹湾登记和枫台续期，查公共规则，只读。")]
    )
    observed = []

    async def planner(query, **kwargs):
        assert query == QUERY and list(kwargs["history"]) == expected
        observed.append("planner")
        return retrieval_query_plan.RetrievalQueryPlan(status="applied", dependencies=dependencies())

    async def resolver(deps, query, **kwargs):
        assert deps == dependencies() and query == QUERY and list(kwargs["history"]) == expected
        observed.append("binding")
        return None

    class ExpectedStop(BaseException):
        pass

    async def retrieval(*_args, **_kwargs):
        observed.append("retrieval")
        raise ExpectedStop()

    monkeypatch.setattr(retrieval_query_plan, "plan_retrieval_views", planner)
    monkeypatch.setattr(public_question_binding, "resolve_question_binding", resolver)
    monkeypatch.setattr(intent_detector, "needs_rag", lambda _: (True, "unit", None))
    monkeypatch.setattr(generate, "_retrieve_rag_bundle", retrieval)
    prepared = SimpleNamespace(compiled=CompiledCharacterContext("", "", ""), history=tuple(HISTORY))
    with pytest.raises(ExpectedStop):
        await generate._generate_with_retrieval(
            MessageRequest(message=QUERY, history=expected if history_source == "request_override" else []),
            None,
            prepared_character_turn=prepared,
            runtime_config={"useKnowledgeBase": True},
        )
    assert observed == ["planner", "binding", "retrieval"]


async def test_history_binding_retains_independent_source_domain_permissions():
    from knowledge.public_domains import constrain_generic_sources, domain_plan, validate_generic_sources

    receipt = await binding()
    config = SimpleNamespace(domain_id="unit_reference_domain", story_titles=[], canonical_entity=lambda _: None)
    plan = domain_plan(receipt, config)
    import knowledge.public_domains as domains

    original = domains.validate_domain_plan
    domains.validate_domain_plan = lambda candidate: original(candidate, config=config)
    try:
        payload = dict(query=QUERY, public_tasks=[dict(id=0, text=QUERY)], sources=[dict(source_id="doc_1")])
        constrained = constrain_generic_sources(payload, plan)
        validate_generic_sources(constrained)
        assert constrained["sources"][0]["permitted_object_ids"] == ["query-object:0", "query-object:1"]
        constrained["public_tasks"][0]["text"] += "未批准的旧任务"
        with pytest.raises(ValueError):
            validate_generic_sources(constrained)
    finally:
        domains.validate_domain_plan = original


async def test_reference_proof_reaches_candidate_collection_and_final_visibility_without_scope_changes():
    from inference.task_evidence_coverage import render_task_coverage, validate_task_candidates
    from knowledge.task_retrieval import collect_task_candidates

    receipt = await binding()
    calls = []

    def retrieve(question, *, top_k, filters):
        assert top_k == 2 and filters == {"knowledge_base_id": 7}
        calls.append(question)
        filters["knowledge_base_id"] = 99
        return [
            dict(
                id="doc_1_chunk_0",
                document_id=1,
                title="独立完整规程",
                content="竹湾登记费用12元，不免原件。",
                score=0.9,
            )
        ]

    kwargs = dict(
        top_k=2,
        filters={"knowledge_base_id": 7},
        retrieve=retrieve,
        confidence=lambda rows: 0.9,
        stable_key=lambda row: row["id"],
        snapshot=lambda: ("unit", 4),
    )
    plan = collect_task_candidates(QUERY, ("竹湾登记", "枫台续期"), question_binding=receipt, **kwargs)
    assert calls == [QUERY, "竹湾登记", "枫台续期"]
    validated = validate_task_candidates(plan.coverage, query=QUERY)
    assert all(r["reference_binding"] == receipt for r in validated[1:])
    assert all("reference_binding" not in r for r in render_task_coverage(validated))
    assert all(r["semantic_coverage"] == "unverified" for r in validated)
    invalid = deepcopy(plan.coverage)
    invalid[1]["reference_binding"]["input_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validate_task_candidates(invalid, query=QUERY)
    before = len(calls)
    with pytest.raises(ValueError):
        collect_task_candidates(QUERY, ("不在输入的对象",), question_binding=receipt, **kwargs)
    assert len(calls) == before
