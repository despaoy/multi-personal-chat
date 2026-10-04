"""Complete multi-object controls retain independently supported public/private facts."""

import json
from copy import deepcopy
from dataclasses import replace
from html import unescape

import pytest
from tests.test_public_task_evidence_review import PRIVATE, request

from inference.generation_request import build_generation_request
from knowledge.original_sources import attach_original_sources
from knowledge.public_obligations import parse_public_partitions
from knowledge.public_task_evidence import render_public_tasks, review_public_candidates
from knowledge.turn_dependencies import parse_dependencies

QUERY = "同时核对青桥延期与蓝岸续租的费用和完整末尾限定，不得以一个对象的资料替代另一个。读取我保存的周末饮食偏好。"
BODIES = {
    1: "青桥延期费用为31元；完整末尾限定：本人携带完整登记材料，周三受理，特殊预约不得省去材料。",
    2: "蓝岸续租费用为46元；完整末尾限定：本人携带签约原件，周五受理，节假日暂停且不能线上代办。",
}


def dependencies():
    return parse_dependencies(
        dict(current_input=[], public_knowledge=[0], private_memory=[1], control=[], unresolved_source=[]), QUERY
    )


def obligations():
    return parse_public_partitions([dict(segment_id=0, cuts=[])], dependencies(), QUERY)


def bundle(*, missing=False, split_original=False):
    bodies = dict(BODIES)
    if missing:
        bodies[2] = "紫藤登山费用为82元，完全不适用于任何续租业务；末尾限定：必须取得山地许可。"
    rows = [
        dict(
            id=f"doc_{i}_chunk_0",
            document_id=i,
            title=f"虚构规程{i}",
            content=("独立规程导言，没有费用及对象依据。" if split_original and i == 2 else body),
            category="fiction",
            knowledge_base_id=7,
            score=0.9,
        )
        for i, body in bodies.items()
    ]
    raw = dict(
        results=rows,
        citations=[],
        confidence=0.9,
        abstained=False,
        source_coverage=tuple(
            dict(
                source_id=f"doc_{i}",
                source_title=f"虚构规程{i}",
                indexed_document_ids=[f"doc_{i}_chunk_0"],
                retrieved_document_ids=[f"doc_{i}_chunk_0"],
            )
            for i in bodies
        ),
    )
    return attach_original_sources(
        raw, lambda i: dict(rows[i - 1], id=i, content=bodies[i]), source_budget_tokens=65536, authority_revision=3
    )


def source_reply(*, missing=False, drop_relation=False):
    return dict(
        decisions=[
            dict(
                source_id="doc_1",
                task_ids=["public:0:0"],
                object_evidence=[dict(object_id="query-object:0", source_quote=BODIES[1])],
            ),
            dict(
                source_id="doc_2",
                task_ids=[] if drop_relation else ["public:0:0"],
                object_evidence=[] if missing else [dict(object_id="query-object:1", source_quote=BODIES[2])],
            ),
        ]
    )


async def run(*, missing=False, split_original=False, drop_relation=False):
    async def resolve(messages):
        payload = json.loads(messages[-1]["content"])
        assert payload == dict(query=QUERY, public_tasks=[dict(id="public:0:0", text=dependencies().segments[0])])
        return json.dumps(dict(scopes=[dict(task_id="public:0:0", objects=["青桥延期", "蓝岸续租"])]))

    async def review(messages):
        payload = json.loads(messages[-1]["content"])
        assert payload["query"] == QUERY and len(payload["sources"]) == 2
        assert payload["sources"][0]["original_body"] == BODIES[1]
        return json.dumps(source_reply(missing=missing, drop_relation=drop_relation))

    original = bundle(missing=missing, split_original=split_original)
    result = await review_public_candidates(
        original,
        dependencies(),
        QUERY,
        window_tokens=65536,
        public_obligations=obligations(),
        reviewer=review,
        scope_reviewer=resolve,
    )
    req = request(result)
    return result, replace(
        req, message=QUERY, retrieval=replace(req.retrieval, public_task_query=QUERY, public_dependency_indices=(0,))
    )


async def positive():
    result, req = await run()
    plan = build_generation_request(req)
    row = render_public_tasks(plan.retrieval)[0]
    assert row["status"] == "related_candidate_admitted" and row["semantic_coverage"] == "unverified"
    wire = unescape(plan.messages[-1]["content"])
    assert PRIVATE in wire and all(body in wire for body in BODIES.values()) and len(result["results"]) == 2
    return result, req, plan


@pytest.mark.asyncio
async def test_complementary_sources_can_cover_one_complete_multi_object_task():
    await positive()


@pytest.mark.asyncio
async def test_one_object_proof_cannot_mark_whole_multi_object_task_admitted():
    await positive()
    result, req = await run(missing=True)
    plan = build_generation_request(req)
    row = render_public_tasks(plan.retrieval)[0]
    assert row["status"] == "object_scope_partial"
    wire = unescape(plan.messages[-1]["content"])
    assert PRIVATE in wire and BODIES[1] in wire and len(result["results"]) == 1
    assert row["object_scope"] == "partial_source_object_evidence_verified"


@pytest.mark.asyncio
async def test_object_coverage_reports_independent_visible_and_missing_objects():
    _, _, full = await positive()
    assert [r["status"] for r in render_public_tasks(full.retrieval)[0]["object_coverage"]] == [
        "verified_object_evidence_admitted"
    ] * 2
    _, req = await run(missing=True)
    row = render_public_tasks(build_generation_request(req).retrieval)[0]
    assert [(r["query_text"], r["status"]) for r in row["object_coverage"]] == [
        ("青桥延期", "verified_object_evidence_admitted"),
        ("蓝岸续租", "no_verified_object_evidence"),
    ]
    assert row["object_coverage"][0]["source_ids"] == ["doc_1"] and row["object_coverage"][1]["source_ids"] == []


@pytest.mark.asyncio
async def test_post_budget_dropped_second_object_still_retains_first():
    _, req, _ = await positive()
    kept = tuple(
        p for p in req.retrieval.evidence_packets if all(not i.startswith("doc_2_") for i in p["document_ids"])
    )
    plan = build_generation_request(replace(req, retrieval=replace(req.retrieval, evidence_packets=kept)))
    row = render_public_tasks(plan.retrieval)[0]
    assert row["status"] == "object_scope_partial"
    assert row["object_coverage"][1]["status"] == "verified_object_evidence_not_admitted"
    wire = unescape(plan.messages[-1]["content"])
    assert BODIES[1] in wire and BODIES[2] not in wire and PRIVATE in wire


@pytest.mark.asyncio
async def test_visible_source_chunk_cannot_stand_in_for_unadmitted_object_quote():
    await positive()
    _, req = await run(split_original=True)
    kept = tuple(p for p in req.retrieval.evidence_packets if p.get("original_source_id") != "doc_2")
    plan = build_generation_request(replace(req, retrieval=replace(req.retrieval, evidence_packets=kept)))
    assert next(r for r in plan.retrieval.source_coverage if r["source_id"] == "doc_2")["admitted_chunk_count"] == 1
    row = render_public_tasks(plan.retrieval)[0]
    assert (
        row["status"] == "object_scope_partial"
        and row["object_coverage"][1]["status"] == "verified_object_evidence_not_admitted"
    )
    wire = unescape(plan.messages[-1]["content"])
    assert "独立规程导言，没有费用及对象依据。" in wire and BODIES[2] not in wire and BODIES[1] in wire


@pytest.mark.asyncio
async def test_object_mention_with_no_task_relation_does_not_cover_the_task():
    await positive()
    _, req = await run(drop_relation=True)
    row = render_public_tasks(build_generation_request(req).retrieval)[0]
    assert (
        row["status"] == "object_scope_partial" and row["object_coverage"][1]["status"] == "no_verified_object_evidence"
    )


@pytest.mark.asyncio
async def test_all_object_quotes_not_admitted_are_reported_separately():
    _, req, _ = await positive()
    plan = build_generation_request(
        replace(req, retrieval=replace(req.retrieval, evidence_packets=(), evidence="", status="character_abstention"))
    )
    row = render_public_tasks(plan.retrieval)[0]
    assert row["status"] == "related_candidate_not_admitted"
    assert all(r["status"] == "verified_object_evidence_not_admitted" for r in row["object_coverage"])


@pytest.mark.asyncio
async def test_same_object_quote_in_another_source_does_not_borrow_admission():
    _, req, _ = await positive()
    packets = [
        deepcopy(p)
        for p in req.retrieval.evidence_packets
        if all(not i.startswith("doc_2_") for i in p["document_ids"])
    ]
    # Unrelated visible packet text is not an admission of doc_2 evidence.
    packets.append(dict(kind="evidence", document_ids=["doc_3_chunk_0"], text=BODIES[2]))
    plan = build_generation_request(replace(req, retrieval=replace(req.retrieval, evidence_packets=tuple(packets))))
    row = render_public_tasks(plan.retrieval)[0]
    assert (
        row["status"] == "object_scope_partial"
        and row["object_coverage"][1]["status"] == "verified_object_evidence_not_admitted"
    )


@pytest.mark.asyncio
async def test_actual_api_marks_partial_multi_object_scope_without_erasing_private(monkeypatch):
    from types import SimpleNamespace

    from api import generate
    from character.models import CompiledCharacterContext
    from db.schemas import MessageRequest
    from knowledge import intent_detector, public_task_evidence, retrieval_query_plan
    from knowledge.retrieval_query_plan import RetrievalQueryPlan

    await positive()

    async def planner(_query):
        return RetrievalQueryPlan(status="applied", dependencies=dependencies(), public_obligations=obligations())

    async def retrieve(*_args, **_kwargs):
        return bundle(missing=True)

    async def review(messages):
        payload = json.loads(messages[-1]["content"])
        if set(payload) == {"query", "public_tasks"}:
            return json.dumps(dict(scopes=[dict(task_id="public:0:0", objects=["青桥延期", "蓝岸续租"])]))
        if "identity_review" not in payload:
            return json.dumps(dict(sources=[dict(source_id=s["source_id"], purpose="rules") for s in payload["sources"]], relations=[]))
        return json.dumps(source_reply(missing=True))

    async def model(**kwargs):
        wire = unescape(kwargs["messages"][-1]["content"])
        assert PRIVATE in wire and BODIES[1] in wire and "object_scope_partial" in wire and kwargs["max_tokens"] == 2048
        return "青桥费用及末尾限定有依据，蓝岸本次缺对应资料；你的私人偏好保留。"

    monkeypatch.setattr(retrieval_query_plan, "plan_retrieval_views", planner)
    monkeypatch.setattr(intent_detector, "needs_rag", lambda _: (True, "unit", None))
    monkeypatch.setattr(generate, "_retrieve_rag_bundle", retrieve)
    monkeypatch.setattr(public_task_evidence, "_review", review)
    monkeypatch.setattr(generate, "_get_system_prompt", lambda _: "人物规则")
    prepared = SimpleNamespace(
        compiled=CompiledCharacterContext("", "", PRIVATE, used_memory_ids=("fiction-private",)), history=()
    )
    _, used, meta = await generate._generate_with_retrieval(
        MessageRequest(message=QUERY),
        None,
        prepared_character_turn=prepared,
        runtime_config=dict(maxTokens=2048, useKnowledgeBase=True),
        model_generate=model,
    )
    assert (
        used
        and meta["abstained"]
        and meta["answerMode"] == "partial_answer"
        and "partial_public_evidence" in meta["warnings"]
    )


@pytest.mark.asyncio
async def test_model_facing_metadata_matches_final_full_object_coverage():
    _, _, plan = await positive()
    wire = unescape(plan.messages[-1]["content"])
    coverage = json.loads(
        wire.split("<retrieval_coverage source=", 1)[1].split(">", 1)[1].split("</retrieval_coverage>", 1)[0]
    )
    assert coverage["public_tasks"] == render_public_tasks(plan.retrieval)
    assert coverage["public_tasks"][0]["status"] == "related_candidate_admitted"
    assert tuple(plan.retrieval.admitted_evidence_packets) == tuple(plan.retrieval.evidence_packets)


@pytest.mark.asyncio
async def test_model_facing_metadata_matches_final_partial_object_coverage():
    await positive()
    _, req = await run(missing=True)
    plan = build_generation_request(req)
    wire = unescape(plan.messages[-1]["content"])
    coverage = json.loads(
        wire.split("<retrieval_coverage source=", 1)[1].split(">", 1)[1].split("</retrieval_coverage>", 1)[0]
    )
    assert coverage["public_tasks"] == render_public_tasks(plan.retrieval)
    assert coverage["public_tasks"][0]["status"] == "object_scope_partial"
    assert PRIVATE in wire and BODIES[1] in wire
