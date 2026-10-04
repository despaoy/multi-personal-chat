"""Complete facts precede missing, negated and unadmitted evidence controls."""

import asyncio
import json
from copy import deepcopy
from dataclasses import replace
from html import unescape

import pytest
from tests.test_public_task_evidence_review import PRIVATE, request

from inference.generation_request import build_generation_request
from knowledge.original_sources import attach_original_sources
from knowledge.public_fact_coverage import bounded_review, parse_fact_scope
from knowledge.public_object_scope import parse_object_scopes, scope_input_digest
from knowledge.public_obligations import parse_public_partitions
from knowledge.public_task_evidence import render_public_tasks, review_payload, review_public_candidates
from knowledge.turn_dependencies import parse_dependencies

QUERY = "核对柳港续租的费用、办理日、所需原件及末尾例外。读取我的周末饮食偏好。"
TASK = "public:0:0"
FIELDS = ["费用", "办理日", "所需原件", "末尾例外"]
FULL = "柳港续租收费68元，周二办理，必须携带签约原件；末尾例外：雨天暂停，预约不能免去原件。"
FEE = "柳港续租收费68元。本页未载办理日、所需原件及末尾例外；缺项不能套用其他业务规则。"
NEGATIVE = "柳港续租不收费，周二办理，必须携带签约原件；末尾例外：雨天暂停，预约不能免去原件。"
FIRST = "柳港续租收费68元，周二办理。本页仅载费用与办理日。"
SECOND = "柳港续租必须携带签约原件；末尾例外：雨天暂停，预约不能免去原件。本页仅载材料与例外。"


def deps():
    return parse_dependencies(
        dict(current_input=[], public_knowledge=[0], private_memory=[1], control=[], unresolved_source=[]), QUERY
    )


def obligations():
    return parse_public_partitions([dict(segment_id=0, cuts=[])], deps(), QUERY)


def bundle(mode="full", split_original=False):
    bodies = (
        [FULL] if mode == "full" else [FEE] if mode == "fee" else [NEGATIVE] if mode == "negative" else [FIRST, SECOND]
    )
    rows = [
        dict(
            id=f"doc_{i}_chunk_0",
            document_id=i,
            title=f"虚构规则{i}",
            content="柳港续租规则导言。" if split_original else body,
            knowledge_base_id=7,
            score=0.9,
        )
        for i, body in enumerate(bodies, 1)
    ]
    raw = dict(
        results=rows,
        citations=[],
        confidence=0.9,
        abstained=False,
        source_coverage=tuple(
            dict(
                source_id=f"doc_{i}",
                source_title=row["title"],
                indexed_document_ids=[row["id"]],
                retrieved_document_ids=[row["id"]],
            )
            for i, row in enumerate(rows, 1)
        ),
    )
    return attach_original_sources(
        raw, lambda i: dict(rows[i - 1], id=i, content=bodies[i - 1]), source_budget_tokens=65536, authority_revision=3
    )


def scope_reply():
    return dict(
        tasks=[dict(task_id=TASK, aspects=[dict(object_id="query-object:0", query_quote=field) for field in FIELDS])]
    )


def source_reply(mode="full", split_original=False):
    original = bundle(mode, split_original)
    return dict(
        decisions=[
            dict(
                source_id=f"doc_{r['document_id']}",
                task_ids=[TASK],
                object_evidence=[dict(object_id="query-object:0", source_quote=r["content"])],
            )
            for r in original["results"]
        ]
    )


def evidence_reply(mode="full"):
    values = [
        ("doc_1", "不收费" if mode == "negative" else "收费68元"),
        ("doc_1", "周二办理"),
        ("doc_2" if mode == "complement" else "doc_1", "必须携带签约原件"),
        ("doc_2" if mode == "complement" else "doc_1", "雨天暂停，预约不能免去原件"),
    ]
    return dict(
        assessments=[
            dict(
                aspect_id=f"fact-aspect:{i}",
                evidence=[]
                if mode == "fee" and i > 0
                else [
                    dict(
                        source_id=sid,
                        source_quote=quote,
                        assertion="negative" if mode == "negative" and i == 0 else "affirmative",
                    )
                ],
            )
            for i, (sid, quote) in enumerate(values)
        ]
    )


async def run(mode="full", *, split_original=False, scope_failure=False, bad_quote=False):
    async def object_review(messages):
        data = json.loads(messages[-1]["content"])
        assert set(data) == {"query", "public_tasks"}
        return json.dumps(dict(scopes=[dict(task_id=TASK, objects=["柳港续租"])]))

    async def fact_scope(messages):
        data = json.loads(messages[-1]["content"])
        assert set(data) == {"query", "public_tasks", "object_scopes"} and data["query"] == QUERY
        assert "sources" not in data and PRIVATE not in json.dumps(data)
        if scope_failure:
            raise RuntimeError("reviewer unavailable")
        return json.dumps(scope_reply())

    async def source_review(messages):
        data = json.loads(messages[-1]["content"])
        assert data["query"] == QUERY and "fact_scope_review" in data
        assert len(data["sources"]) == (2 if mode == "complement" else 1)
        return json.dumps(source_reply(mode, split_original))

    async def fact_review(messages):
        data = json.loads(messages[-1]["content"])
        assert (
            data["query"] == QUERY and data["approved_decisions"] and all("original_body" in s for s in data["sources"])
        )
        reply = evidence_reply(mode)
        if bad_quote:
            reply["assessments"][0]["evidence"][0]["source_quote"] = "费用不存在于正文的99元"
        return json.dumps(reply)

    result = await review_public_candidates(
        bundle(mode, split_original),
        deps(),
        QUERY,
        window_tokens=65536,
        public_obligations=obligations(),
        reviewer=source_review,
        scope_reviewer=object_review,
        fact_scope_reviewer=fact_scope,
        fact_reviewer=fact_review,
    )
    req = request(result)
    return result, replace(
        req, message=QUERY, retrieval=replace(req.retrieval, public_task_query=QUERY, public_dependency_indices=(0,))
    )


async def positive():
    result, req = await run()
    plan = build_generation_request(req)
    row = render_public_tasks(plan.retrieval)[0]
    wire = unescape(plan.messages[-1]["content"])
    assert (
        row["status"] == "related_candidate_admitted"
        and len(row["fact_coverage"]) == 4
        and all(r["status"] == "fact_evidence_admitted" for r in row["fact_coverage"])
    )
    assert row["semantic_coverage"] == "unverified" and FULL in wire and PRIVATE in wire
    return result, req, plan


@pytest.mark.asyncio
async def test_full_four_fact_control_preserves_whole_body_and_private():
    await positive()


@pytest.mark.asyncio
async def test_fee_only_is_partial_without_turning_missing_fields_into_negatives():
    await positive()
    _, req = await run("fee")
    plan = build_generation_request(req)
    row = render_public_tasks(plan.retrieval)[0]
    assert row["status"] == "partial_fact_evidence" and row["fact_coverage"][0]["status"] == "fact_evidence_admitted"
    assert all(
        r["status"] == "no_supporting_fact_evidence" and r["admitted_evidence"] == [] for r in row["fact_coverage"][1:]
    )
    assert FEE in unescape(plan.messages[-1]["content"]) and PRIVATE in unescape(plan.messages[-1]["content"])


@pytest.mark.asyncio
async def test_independent_sources_can_complement_four_fact_obligations():
    await positive()
    _, req = await run("complement")
    plan = build_generation_request(req)
    row = render_public_tasks(plan.retrieval)[0]
    assert row["status"] == "related_candidate_admitted" and [r["source_ids"] for r in row["fact_coverage"]] == [
        ["doc_1"],
        ["doc_1"],
        ["doc_2"],
        ["doc_2"],
    ]
    assert all(body in unescape(plan.messages[-1]["content"]) for body in (FIRST, SECOND, PRIVATE))


@pytest.mark.asyncio
async def test_explicit_negative_fact_is_supported_not_missing_information():
    await positive()
    _, req = await run("negative")
    row = render_public_tasks(build_generation_request(req).retrieval)[0]
    assert (
        row["status"] == "related_candidate_admitted"
        and row["fact_coverage"][0]["admitted_evidence"][0]["assertion"] == "negative"
    )
    assert row["fact_coverage"][0]["admitted_evidence"][0]["source_quote"] == "不收费"


@pytest.mark.asyncio
async def test_object_intro_cannot_admit_fact_quote_from_dropped_original():
    await positive()
    _, req = await run(split_original=True)
    packets = tuple(p for p in req.retrieval.evidence_packets if p.get("original_source_id") != "doc_1")
    plan = build_generation_request(replace(req, retrieval=replace(req.retrieval, evidence_packets=packets)))
    row = render_public_tasks(plan.retrieval)[0]
    assert row["object_scope"] == "source_object_evidence_verified" and row["status"] == "fact_evidence_unverified"
    assert all(r["status"] == "fact_evidence_not_admitted" and not r["admitted_evidence"] for r in row["fact_coverage"])
    wire = unescape(plan.messages[-1]["content"])
    assert PRIVATE in wire and FULL not in wire and "收费68元" not in wire


@pytest.mark.asyncio
async def test_dropped_complement_keeps_known_two_facts_without_quote_leak():
    await positive()
    _, req = await run("complement")
    packets = tuple(
        p for p in req.retrieval.evidence_packets if not any(i.startswith("doc_2_") for i in p["document_ids"])
    )
    plan = build_generation_request(replace(req, retrieval=replace(req.retrieval, evidence_packets=packets)))
    row = render_public_tasks(plan.retrieval)[0]
    assert (
        row["status"] == "partial_fact_evidence"
        and [r["status"] for r in row["fact_coverage"]]
        == ["fact_evidence_admitted"] * 2 + ["fact_evidence_not_admitted"] * 2
    )
    wire = unescape(plan.messages[-1]["content"])
    assert FIRST in wire and SECOND not in wire and "签约原件" not in wire and PRIVATE in wire


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["scope", "evidence"])
async def test_fact_stage_failure_keeps_source_body_private_and_honest_unknown(failure):
    await positive()
    result, req = await run(scope_failure=failure == "scope", bad_quote=failure == "evidence")
    plan = build_generation_request(req)
    row = render_public_tasks(plan.retrieval)[0]
    assert result["results"] and row["status"] == "fact_review_unavailable" and row["fact_coverage"] == []
    assert FULL in unescape(plan.messages[-1]["content"]) and PRIVATE in unescape(plan.messages[-1]["content"])


@pytest.mark.asyncio
async def test_actual_model_facing_fact_metadata_equals_final_partial_coverage():
    await positive()
    _, req = await run("fee")
    plan = build_generation_request(req)
    wire = unescape(plan.messages[-1]["content"])
    coverage = json.loads(
        wire.split("<retrieval_coverage source=", 1)[1].split(">", 1)[1].split("</retrieval_coverage>", 1)[0]
    )
    assert (
        coverage["public_tasks"] == render_public_tasks(plan.retrieval)
        and coverage["public_tasks"][0]["status"] == "partial_fact_evidence"
    )


@pytest.mark.asyncio
async def test_tampered_fact_scope_is_rejected_even_with_recomputed_input_hash():
    _, req, _ = await positive()
    receipt = deepcopy(req.retrieval.public_task_review)
    payload = receipt["object_scope_input"]
    payload["fact_scope_review"]["scope"]["aspects"][0]["query_quote"] = "臆造方面"
    receipt["object_scope_input_sha256"] = scope_input_digest(payload)
    with pytest.raises(ValueError):
        build_generation_request(replace(req, retrieval=replace(req.retrieval, public_task_review=receipt)))


@pytest.mark.parametrize(
    "mutation",
    ["private_task", "bool_task", "unknown_object", "invented_aspect", "lost_object", "duplicate", "incomplete_task"],
)
def test_fact_scope_rejects_lost_or_invented_question_bindings(mutation):
    data = review_payload(bundle(), deps(), QUERY, obligations())
    scopes = parse_object_scopes(json.dumps(dict(scopes=[dict(task_id=TASK, objects=["柳港续租"])])), QUERY, [TASK])
    good = scope_reply()
    assert len(parse_fact_scope(json.dumps(good), data, scopes)["aspects"]) == 4
    bad = deepcopy(good)
    row = bad["tasks"][0]
    if mutation == "private_task":
        row["task_id"] = 1
    elif mutation == "bool_task":
        row["task_id"] = True
    elif mutation == "unknown_object":
        row["aspects"][0]["object_id"] = "query-object:9"
    elif mutation == "invented_aspect":
        row["aspects"][0]["query_quote"] = "停车费"
    elif mutation == "lost_object":
        row["aspects"] = []
    elif mutation == "duplicate":
        row["aspects"].append(deepcopy(row["aspects"][0]))
    else:
        bad["tasks"] = []
    with pytest.raises(ValueError):
        parse_fact_scope(json.dumps(bad), data, scopes)


@pytest.mark.asyncio
async def test_fact_review_capacity_preserves_complete_input_without_calling_provider():
    await positive()
    messages = [dict(role="user", content="完整输入" * 1000)]

    async def forbidden(_messages):
        raise AssertionError("Should not call provider")

    raw, reason = await bounded_review(messages, forbidden, 100, 2048)
    assert raw is None and reason == "complete_fact_input_budget_exceeded" and len(messages[0]["content"]) == 4000


@pytest.mark.asyncio
async def test_fact_stage_cancellation_propagates():
    await positive()

    async def cancelled(_messages):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await bounded_review([dict(role="user", content="完整")], cancelled, 65536, 2048)




async def alias_plan():
    from tests import test_public_identity_dependencies as identity

    async def object_scope(_messages):
        return json.dumps(dict(scopes=[dict(task_id=identity.TASK, objects=["榆湾续办"])]))

    async def fact_scope(_messages):
        return json.dumps(
            dict(
                tasks=[
                    dict(
                        task_id=identity.TASK,
                        aspects=[
                            dict(object_id="query-object:0", query_quote=field)
                            for field in ["费用", "受理日", "原件", "例外"]
                        ],
                    )
                ]
            )
        )

    async def identify(_messages):
        return json.dumps(identity.identity())

    async def source(_messages):
        return json.dumps(
            dict(
                decisions=[
                    dict(source_id="doc_1", task_ids=[], object_evidence=[]),
                    dict(
                        source_id="doc_2",
                        task_ids=[identity.TASK],
                        object_evidence=[
                            dict(
                                object_id="query-object:0", source_quote=identity.RULE, identity_binding_id="identity:0"
                            )
                        ],
                    ),
                ]
            )
        )

    async def facts(_messages):
        return json.dumps(
            dict(
                assessments=[
                    dict(
                        aspect_id=f"fact-aspect:{i}",
                        evidence=[dict(source_id="doc_2", source_quote=identity.RULE, assertion="affirmative")],
                    )
                    for i in range(4)
                ]
            )
        )

    result = await review_public_candidates(
        identity.bundle(),
        identity.deps(),
        identity.QUERY,
        window_tokens=65536,
        public_obligations=identity.parts(),
        reviewer=source,
        scope_reviewer=object_scope,
        identity_reviewer=identify,
        fact_scope_reviewer=fact_scope,
        fact_reviewer=facts,
    )
    req = request(result)
    req = replace(
        req,
        message=identity.QUERY,
        retrieval=replace(req.retrieval, public_task_query=identity.QUERY, public_dependency_indices=(0,)),
    )
    plan = build_generation_request(req)
    row = render_public_tasks(plan.retrieval)[0]
    assert (
        row["status"] == "related_candidate_admitted"
        and len(row["fact_coverage"]) == 4
        and all(r["status"] == "fact_evidence_admitted" for r in row["fact_coverage"])
    )
    assert all(text in unescape(plan.messages[-1]["content"]) for text in (identity.BRIDGE, identity.RULE, PRIVATE))
    return req


@pytest.mark.asyncio
async def test_alias_fact_coverage_requires_registry_and_rule_together():
    await positive()
    await alias_plan()


@pytest.mark.asyncio
async def test_missing_registry_cannot_admit_any_alias_fact_quote():
    await positive()
    req = await alias_plan()
    packets = tuple(
        p for p in req.retrieval.evidence_packets if not any(i.startswith("doc_1_") for i in p["document_ids"])
    )
    plan = build_generation_request(replace(req, retrieval=replace(req.retrieval, evidence_packets=packets)))
    row = render_public_tasks(plan.retrieval)[0]
    assert row["status"] == "object_scope_not_admitted" and all(
        r["status"] == "fact_evidence_not_admitted" and not r["admitted_evidence"] for r in row["fact_coverage"]
    )
    assert PRIVATE in unescape(plan.messages[-1]["content"])


def test_complete_fact_graph_over_capacity_is_not_silently_trimmed():
    from knowledge.public_fact_coverage import FactScopeCapacityError

    query = "柳港续租：费用、办理日、所需原件、末尾例外、渠道、期限、预约、年龄、住址。"
    data = dict(query=query, public_task_ids=[TASK])
    scopes = parse_object_scopes(json.dumps(dict(scopes=[dict(task_id=TASK, objects=["柳港续租"])])), query, [TASK])
    fields = ["费用", "办理日", "所需原件", "末尾例外", "渠道", "期限", "预约", "年龄", "住址"]
    raw = json.dumps(
        dict(tasks=[dict(task_id=TASK, aspects=[dict(object_id="query-object:0", query_quote=f) for f in fields])])
    )
    with pytest.raises(FactScopeCapacityError):
        parse_fact_scope(raw, data, scopes)
