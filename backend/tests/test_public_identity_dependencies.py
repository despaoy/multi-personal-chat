"""Identity evidence is a dependency, never a substitute for rule evidence."""

import json
from copy import deepcopy
from dataclasses import replace
from html import unescape

import pytest
from tests.test_public_task_evidence_review import PRIVATE, request

from inference.generation_request import build_generation_request
from knowledge.original_sources import attach_original_sources
from knowledge.public_identity_dependencies import parse_identity_review
from knowledge.public_object_scope import parse_object_scopes
from knowledge.public_obligations import parse_public_partitions
from knowledge.public_task_evidence import render_public_tasks, review_payload, review_public_candidates
from knowledge.turn_dependencies import parse_dependencies

QUERY = "核对榆湾续办的费用、受理日、原件和例外。读取我的周末偏好。"
BRIDGE = "当前名称登记：榆湾续办与湾台延期是同一业务，登记号YW-18，不是榆湾迁出；本页仅登记名称，不载业务规则。"
RULE = "湾台延期费用57元，周四受理，必须携带签收原件；例外：雨天暂停，预约不得免除原件。"
DENIAL = "名称登记：榆湾续办不等于湾台延期，属于不同业务，不可互换费用及条件；本页仅登记名称关系。"
TASK = "public:0:0"


def deps():
    return parse_dependencies(
        dict(current_input=[], public_knowledge=[0], private_memory=[1], control=[], unresolved_source=[]), QUERY
    )


def parts():
    return parse_public_partitions([dict(segment_id=0, cuts=[])], deps(), QUERY)


def bundle(bridge=BRIDGE, kb=7):
    rows = [
        dict(
            id=f"doc_{i}_chunk_0",
            document_id=i,
            title=f"独立文档{i}",
            content=body,
            knowledge_base_id=7 if i == 1 else kb,
            score=0.9,
        )
        for i, body in enumerate([bridge, RULE], 1)
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
        raw, lambda i: dict(rows[i - 1], id=i), source_budget_tokens=65536, authority_revision=3
    )


def identity(bridge=BRIDGE, relation="same"):
    return dict(
        sources=[dict(source_id="doc_1", purpose="identity_only"), dict(source_id="doc_2", purpose="rules")],
        relations=[
            dict(
                object_id="query-object:0", alias="湾台延期", source_id="doc_1", source_quote=bridge, relation=relation
            )
        ],
    )


def payload():
    value = review_payload(bundle(), deps(), QUERY, parts())
    scope = parse_object_scopes(json.dumps(dict(scopes=[dict(task_id=TASK, objects=["榆湾续办"])])), QUERY, [TASK])
    return value, scope


async def run(*, bridge=BRIDGE, relation="same", kb=7, identity_only_claim=False):
    async def resolve(messages):
        data = json.loads(messages[-1]["content"])
        assert set(data) == {"query", "public_tasks"} and data["query"] == QUERY
        return json.dumps(dict(scopes=[dict(task_id=TASK, objects=["榆湾续办"])]))

    async def identify(messages):
        data = json.loads(messages[-1]["content"])
        assert data["query"] == QUERY and len(data["sources"]) == 2
        assert data["sources"][0]["original_body"] == bridge and data["sources"][1]["original_body"] == RULE
        assert PRIVATE not in json.dumps(data)
        return json.dumps(identity(bridge, relation))

    async def review(messages):
        data = json.loads(messages[-1]["content"])
        bindings = data["identity_review"]["bindings"]
        proof = dict(object_id="query-object:0", source_quote=RULE)
        if bindings:
            proof["identity_binding_id"] = bindings[0]["binding_id"]
        return json.dumps(
            dict(
                decisions=[
                    dict(
                        source_id="doc_1",
                        task_ids=[TASK] if identity_only_claim else [],
                        object_evidence=[dict(object_id="query-object:0", source_quote=bridge)],
                    ),
                    dict(source_id="doc_2", task_ids=[TASK], object_evidence=[proof]),
                ]
            )
        )

    result = await review_public_candidates(
        bundle(bridge, kb),
        deps(),
        QUERY,
        window_tokens=65536,
        public_obligations=parts(),
        reviewer=review,
        scope_reviewer=resolve,
        identity_reviewer=identify,
    )
    req = request(result)
    req = replace(
        req, message=QUERY, retrieval=replace(req.retrieval, public_task_query=QUERY, public_dependency_indices=(0,))
    )
    return result, req


async def positive():
    reviewed, req = await run()
    plan = build_generation_request(req)
    row = render_public_tasks(plan.retrieval)[0]
    wire = unescape(plan.messages[-1]["content"])
    assert row["status"] == "related_candidate_admitted" and row["semantic_coverage"] == "unverified"
    assert row["object_coverage"][0]["source_ids"] == ["doc_2"]
    assert all(s in wire for s in (BRIDGE, RULE, PRIVATE)) and len(reviewed["results"]) == 2
    assert reviewed["public_task_review"]["decisions"][0]["task_ids"] == []
    return reviewed, req, plan


@pytest.mark.asyncio
async def test_complete_alias_registry_and_rule_preserve_private_and_all_qualifiers():
    await positive()


@pytest.mark.asyncio
@pytest.mark.parametrize("relation", ["different", "uncertain"])
async def test_denial_and_unknown_do_not_create_identity_edges(relation):
    await positive()
    reviewed, req = await run(bridge=DENIAL, relation=relation)
    plan = build_generation_request(req)
    assert not reviewed["results"] and PRIVATE in unescape(plan.messages[-1]["content"])
    assert render_public_tasks(plan.retrieval)[0]["object_scope"] == "unverified"


@pytest.mark.asyncio
async def test_identity_only_source_cannot_certify_factual_task():
    await positive()
    reviewed, req = await run(identity_only_claim=True, kb=9)
    plan = build_generation_request(req)
    assert not reviewed["results"] and render_public_tasks(plan.retrieval)[0]["object_scope"] == "unverified"
    assert PRIVATE in unescape(plan.messages[-1]["content"])


@pytest.mark.asyncio
async def test_cross_knowledge_base_alias_rules_are_not_borrowed():
    await positive()
    reviewed, req = await run(kb=9)
    assert not reviewed["results"]
    assert render_public_tasks(build_generation_request(req).retrieval)[0]["status"] != "related_candidate_admitted"


@pytest.mark.asyncio
async def test_missing_admitted_registry_invalidates_alias_coverage():
    _, req, _ = await positive()
    packets = tuple(
        p for p in req.retrieval.evidence_packets if not any(i.startswith("doc_1_") for i in p["document_ids"])
    )
    plan = build_generation_request(replace(req, retrieval=replace(req.retrieval, evidence_packets=packets)))
    row = render_public_tasks(plan.retrieval)[0]
    assert row["status"] == "object_scope_not_admitted" and row["object_scope"] == "unverified"
    assert PRIVATE in unescape(plan.messages[-1]["content"])
    assert json.dumps(row, ensure_ascii=False) in unescape(plan.messages[-1]["content"])


@pytest.mark.asyncio
async def test_missing_rule_keeps_registry_but_not_factual_coverage():
    _, req, _ = await positive()
    packets = tuple(
        p for p in req.retrieval.evidence_packets if not any(i.startswith("doc_2_") for i in p["document_ids"])
    )
    plan = build_generation_request(replace(req, retrieval=replace(req.retrieval, evidence_packets=packets)))
    assert render_public_tasks(plan.retrieval)[0]["status"] != "related_candidate_admitted"
    assert PRIVATE in unescape(plan.messages[-1]["content"])


@pytest.mark.asyncio
async def test_tampered_identity_binding_is_rejected_at_final_render():
    _, req, _ = await positive()
    receipt = deepcopy(req.retrieval.public_task_review)
    receipt["object_scope_input"]["identity_review"]["bindings"][0]["alias"] = "别的业务"
    with pytest.raises(ValueError):
        build_generation_request(replace(req, retrieval=replace(req.retrieval, public_task_review=receipt)))


@pytest.mark.parametrize(
    "mutation",
    ["invented_quote", "title_only", "unknown_object", "unknown_source", "duplicate", "ambiguous", "unknown_role"],
)
def test_identity_protocol_rejects_unverifiable_relations(mutation):
    data, scope = payload()
    good = identity()
    assert len(parse_identity_review(json.dumps(good), data, scope)["bindings"]) == 1
    bad = deepcopy(good)
    row = bad["relations"][0]
    if mutation == "invented_quote":
        row["source_quote"] += "并可网上办理。"
    elif mutation == "title_only":
        data["sources"][0]["title"] = row["source_quote"]
        data["sources"][0]["original_body"] = "仅规则引言。"
        data["sources"][0]["indexed_chunks"][0]["content"] = "仅规则引言。"
    elif mutation == "unknown_object":
        row["object_id"] = "query-object:8"
    elif mutation == "unknown_source":
        row["source_id"] = "doc_8"
    elif mutation == "duplicate":
        bad["relations"].append(deepcopy(row))
    elif mutation == "ambiguous":
        bad["relations"].append(dict(row, relation="different"))
    else:
        bad["sources"][0]["purpose"] = "approved"
    with pytest.raises(ValueError):
        parse_identity_review(json.dumps(bad), data, scope)


@pytest.mark.asyncio
async def test_actual_api_production_identity_stage_keeps_registry_rules_and_private(monkeypatch):
    from types import SimpleNamespace

    from api import generate
    from character.models import CompiledCharacterContext
    from db.schemas import MessageRequest
    from knowledge import intent_detector, public_task_evidence, retrieval_query_plan
    from knowledge.retrieval_query_plan import RetrievalQueryPlan

    await positive()

    async def planner(_query):
        return RetrievalQueryPlan(status="applied", dependencies=deps(), public_obligations=parts())

    async def retrieve(*_args, **_kwargs):
        return bundle()

    async def review(messages):
        data = json.loads(messages[-1]["content"])
        if set(data) == {"query", "public_tasks"}:
            return json.dumps(dict(scopes=[dict(task_id=TASK, objects=["榆湾续办"])]))
        if "identity_review" not in data:
            return json.dumps(identity())
        return json.dumps(
            dict(
                decisions=[
                    dict(source_id="doc_1", task_ids=[], object_evidence=[]),
                    dict(
                        source_id="doc_2",
                        task_ids=[TASK],
                        object_evidence=[
                            dict(object_id="query-object:0", source_quote=RULE, identity_binding_id="identity:0")
                        ],
                    ),
                ]
            )
        )

    async def model(**kwargs):
        wire = unescape(kwargs["messages"][-1]["content"])
        assert all(text in wire for text in (BRIDGE, RULE, PRIVATE))
        assert "verified_object_evidence_admitted" in wire and kwargs["max_tokens"] == 2048
        return "续办费用与原件及例外依完整规则回答，周末偏好保留。"

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
    assert used and not meta["abstained"] and "partial_public_evidence" not in (meta["warnings"] or [])
