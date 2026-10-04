"""Complete fictional alias sources; rejected responses cannot authorize evidence."""

import asyncio
import json
from copy import deepcopy
from html import unescape

import pytest
from test_curated_task_evidence import BODY as CURATED_BODY
from test_curated_task_evidence import QUERY as MIXED_QUERY
from test_curated_task_evidence import bundle, plan_for, selection
from test_curated_task_evidence import corpus as corpus
from test_failed_fact_review_receipts import PRIVATE, private_context

from inference.generation_request import GenerationRequest, RetrievalResult, build_generation_request
from knowledge.curated_task_evidence import render_curated_tasks
from knowledge.evidence_packets import document_evidence_packets
from knowledge.original_sources import attach_original_sources
from knowledge.public_task_evidence import render_public_tasks, review_public_candidates
from knowledge.turn_dependencies import parse_dependencies

QUERY = "核对霁浦登记的费用、日期、材料和例外。核对竹原续签的费用。读取我档案核对的历史提醒限制。"
REGISTRY = "当前有效名称登记：霁浦登记与霁浦核验是同一业务，不属于竹原续签；本登记仅确认名称，不提供费用、日期、材料或例外规则。"
RULES = "霁浦核验不收费，每周三受理，必须携带核验原件；雨天暂停，预约不能豁免原件要求。"
OTHER = "竹原续签收费29元，仅适用于本业务，不证明霁浦登记的任何规则。"
MARKER = "DIAGNOSTIC_DO_NOT_AUTHORIZE_143"
STAGES = ("object", "identity", "source")


def complete_sources(bodies=(REGISTRY, RULES, OTHER), identities=(1, 2, 3)):
    rows = [
        dict(
            id=f"doc_{i}_chunk_0",
            document_id=i,
            title=f"独立完整虚构来源{i}",
            knowledge_base_id=23,
            content=body,
            score=0.9,
        )
        for i, body in zip(identities, bodies, strict=True)
    ]
    lookup = {row["document_id"]: dict(row, id=row["document_id"]) for row in rows}
    return attach_original_sources(
        dict(
            results=rows,
            abstained=False,
            citations=[],
            source_coverage=tuple(
                dict(
                    source_id=f"doc_{row['document_id']}",
                    source_title=row["title"],
                    indexed_document_ids=[row["id"]],
                    retrieved_document_ids=[row["id"]],
                )
                for row in rows
            ),
        ),
        lookup.get,
        source_budget_tokens=65536,
        authority_revision=4,
    )


def reply(defect, valid):
    if defect == "transport":
        raise RuntimeError("controlled transport failure")
    if defect == "timeout":
        raise asyncio.TimeoutError
    if defect == "json":
        return "{" + MARKER
    if defect == "non_text":
        return {"diagnostic": MARKER}
    return json.dumps(valid, ensure_ascii=False)


async def run_review(stage=None, defect=None):
    observed = {name: [] for name in (*STAGES, "fact_scope", "facts")}
    received = {}
    deps = parse_dependencies(
        dict(public_knowledge=[0, 1], private_memory=[2], current_input=[], control=[], unresolved_source=[]), QUERY
    )

    async def objects(messages):
        observed["object"].append(deepcopy(messages))
        payload = json.loads(messages[-1]["content"])
        assert payload == dict(query=QUERY, public_tasks=[dict(id=i, text=deps.segments[i]) for i in (0, 1)])
        value = dict(scopes=[dict(task_id=0, objects=["霁浦登记"]), dict(task_id=1, objects=["竹原续签"])])
        if stage == "object" and defect == "typed_id":
            value["scopes"][0]["task_id"] = "0"
        received["object"] = reply(defect if stage == "object" else None, value)
        return received["object"]

    async def fact_scope(messages):
        observed["fact_scope"].append(deepcopy(messages))
        assert json.loads(messages[-1]["content"])["query"] == QUERY
        return json.dumps(
            dict(
                tasks=[
                    dict(
                        task_id=0,
                        aspects=[
                            dict(object_id="query-object:0", query_quote=field)
                            for field in ["费用", "日期", "材料", "例外"]
                        ],
                    ),
                    dict(task_id=1, aspects=[dict(object_id="query-object:1", query_quote="费用")]),
                ]
            )
        )

    async def identities(messages):
        observed["identity"].append(deepcopy(messages))
        data = json.loads(messages[-1]["content"])
        assert data["query"] == QUERY and [s["original_body"] for s in data["sources"]] == [REGISTRY, RULES, OTHER]
        value = dict(
            sources=[dict(source_id=f"doc_{i}", purpose="identity_only" if i == 1 else "rules") for i in (1, 2, 3)],
            relations=[
                dict(
                    object_id="query-object:0",
                    alias="霁浦核验",
                    source_id="doc_1",
                    source_quote=REGISTRY,
                    relation="same",
                )
            ],
        )
        if stage == "identity" and defect == "forged_quote":
            value["relations"][0]["source_quote"] = "霁浦登记就是霁浦核验" + MARKER
        received["identity"] = reply(defect if stage == "identity" else None, value)
        return received["identity"]

    async def sources(messages):
        observed["source"].append(deepcopy(messages))
        data = json.loads(messages[-1]["content"])
        assert data["query"] == QUERY and [s["original_body"] for s in data["sources"]] == [REGISTRY, RULES, OTHER]
        assert len(data["identity_review"]["bindings"]) == 1 and MARKER not in json.dumps(messages)
        decisions = [dict(source_id="doc_1", task_ids=[], object_evidence=[])]
        for sid, task in [("doc_2", 0), ("doc_3", 1)]:
            proof = dict(
                object_id=f"query-object:{task}",
                source_span_id=next(s["span_id"] for s in data["source_span_catalog"] if s["source_id"] == sid),
            )
            if task == 0:
                proof["identity_binding_id"] = "identity:0"
            decisions.append(dict(source_id=sid, task_ids=[task], object_evidence=[proof]))
        if stage == "source" and defect == "foreign_span":
            decisions[1]["object_evidence"][0]["source_span_id"] = decisions[2]["object_evidence"][0]["source_span_id"]
        received["source"] = reply(defect if stage == "source" else None, dict(decisions=decisions))
        return received["source"]

    async def facts(messages):
        observed["facts"].append(deepcopy(messages))
        data = json.loads(messages[-1]["content"])
        assert data["query"] == QUERY and [s["original_body"] for s in data["sources"]] == [REGISTRY, RULES, OTHER]
        assert MARKER not in json.dumps(messages)
        evidence = [
            ("doc_2", "不收费", "negative"),
            ("doc_2", "每周三受理", "affirmative"),
            ("doc_2", "必须携带核验原件", "affirmative"),
            ("doc_2", "雨天暂停，预约不能豁免原件要求", "negative"),
            ("doc_3", "收费29元", "affirmative"),
        ]
        return json.dumps(
            dict(
                assessments=[
                    dict(
                        aspect_id=f"fact-aspect:{i}", evidence=[dict(source_id=sid, source_quote=quote, assertion=kind)]
                    )
                    for i, (sid, quote, kind) in enumerate(evidence)
                ]
            )
        )

    result = await review_public_candidates(
        complete_sources(),
        deps,
        QUERY,
        window_tokens=65536,
        scope_reviewer=objects,
        fact_scope_reviewer=fact_scope,
        identity_reviewer=identities,
        reviewer=sources,
        fact_reviewer=facts,
        span_references=True,
    )
    return result, received, observed


async def build(result, tmp_path, query=QUERY, dependencies=(0, 1)):
    packets = tuple(result.get("evidence_packets", ())) or (
        document_evidence_packets(result["results"]) + result["original_source_packets"]
    )
    retrieval = RetrievalResult(
        status="ok" if packets else "character_abstention",
        evidence="\n".join(p["text"] for p in packets),
        evidence_packets=packets,
        source_coverage=tuple(result.get("source_coverage", ())),
        public_task_review=result["public_task_review"],
        public_task_query=query,
        public_dependency_indices=dependencies,
        public_curated_review=result.get("public_curated_review", {}),
        public_domain_branches=result.get("public_domain_branches", {}),
    )
    planned = build_generation_request(
        GenerationRequest(
            message=query,
            retrieval=retrieval,
            character_context=await private_context(tmp_path),
            context_window_tokens=65536,
            max_tokens=2048,
            evidence_max_chars=0,
        )
    )
    wire = unescape("\n".join(m["content"] for m in planned.messages))
    assert PRIVATE in wire and query in wire and MARKER not in wire and planned.should_generate
    return planned, wire


async def test_complete_alias_negation_and_five_facts_positive(tmp_path):
    result, received, observed = await run_review()
    assert result["public_task_review"]["review_status"] == "reviewed" and len(result["results"]) == 3
    assert all(len(calls) == 1 for calls in observed.values())
    assert "failure_diagnostic" not in result["public_task_review"]
    assert result["public_task_review"]["object_span_review"]["raw"] == received["source"]
    plan, wire = await build(result, tmp_path)
    assert all(body in wire for body in (REGISTRY, RULES, OTHER))
    facts = [f for row in render_public_tasks(plan.retrieval) for f in row["fact_coverage"]]
    assert len(facts) == 5 and all(f["status"] == "fact_evidence_admitted" for f in facts)
    assert sum(e["assertion"] == "negative" for f in facts for e in f["admitted_evidence"]) == 2


@pytest.mark.parametrize(
    "stage,defect",
    [
        ("object", "json"),
        ("object", "typed_id"),
        ("object", "non_text"),
        ("object", "transport"),
        ("identity", "json"),
        ("identity", "forged_quote"),
        ("identity", "non_text"),
        ("identity", "transport"),
        ("source", "json"),
        ("source", "foreign_span"),
        ("source", "non_text"),
        ("source", "transport"),
        ("source", "timeout"),
    ],
)
async def test_failed_stage_keeps_exact_diagnostic_without_authorizing_sources(tmp_path, stage, defect):
    result, received, observed = await run_review(stage, defect)
    receipt = result["public_task_review"]
    assert receipt["review_status"] == "unavailable" and receipt["decisions"] == [] and result["results"] == []
    assert (
        not result["original_source_packets"] and not result["source_coverage"] and "object_span_review" not in receipt
    )
    diagnostic = receipt.get("failure_diagnostic")
    assert diagnostic == dict(
        stage=stage, input=observed[stage][0], raw=received.get(stage) if isinstance(received.get(stage), str) else None
    )
    assert all(not observed[later] for later in STAGES[STAGES.index(stage) + 1 :]) and not observed["facts"]
    plan, wire = await build(result, tmp_path)
    assert all(body not in wire for body in (REGISTRY, RULES, OTHER))
    assert all(row["status"] == "review_unavailable" for row in render_public_tasks(plan.retrieval))
    # Even a valid response saved as a diagnostic cannot upgrade unavailable evidence.
    changed = deepcopy(receipt)
    changed["failure_diagnostic"]["raw"] = json.dumps(dict(decisions=[]))
    result["public_task_review"] = changed
    plan, _ = await build(result, tmp_path / "changed")
    assert all(row["status"] == "review_unavailable" for row in render_public_tasks(plan.retrieval))


@pytest.mark.parametrize("failure", [None, "identity", "source"])
async def test_mixed_authority_keeps_curated_original_and_private_on_generic_failure(corpus, tmp_path, failure):
    plan, deps = await plan_for(corpus)
    registry = "当前名称登记：晨露街续借与晨露借还为同一业务，仅确认名称关系，不载明业务条款。"
    rules = "晨露借还42元，假日不办理，预约不能豁免原件。"
    generic = complete_sources((registry, rules), (8, 9))
    container = dict(
        plan=plan,
        branches=dict(
            curated_character=dict(status="retrieved", bundle=bundle(corpus)),
            generic_knowledge=dict(status="retrieved", bundle=generic),
        ),
    )
    received = {}
    observed = {}
    obj = next(o["object_id"] for o in plan["objects"] if o["authority"] == "generic_knowledge")

    async def identity(messages):
        data = json.loads(messages[-1]["content"])
        assert data["query"] == MIXED_QUERY and [s["original_body"] for s in data["sources"]] == [registry, rules]
        observed["identity"] = deepcopy(messages)
        received["identity"] = reply(
            "json" if failure == "identity" else None,
            dict(
                sources=[dict(source_id="doc_8", purpose="identity_only"), dict(source_id="doc_9", purpose="rules")],
                relations=[
                    dict(object_id=obj, alias="晨露借还", source_id="doc_8", source_quote=registry, relation="same")
                ],
            ),
        )
        return received["identity"]

    async def source(messages):
        observed["source"] = deepcopy(messages)
        data = json.loads(messages[-1]["content"])
        received["source"] = reply(
            "json" if failure == "source" else None,
            dict(
                decisions=[
                    dict(source_id="doc_8", task_ids=[], object_evidence=[]),
                    dict(
                        source_id="doc_9",
                        task_ids=[2],
                        object_evidence=[
                            dict(
                                object_id=obj,
                                source_span_id=next(
                                    s["span_id"] for s in data["source_span_catalog"] if s["source_id"] == "doc_9"
                                ),
                                identity_binding_id="identity:0",
                            )
                        ],
                    ),
                ]
            ),
        )
        return received["source"]

    async def curated(messages):
        data = json.loads(messages[-1]["content"])
        assert data["query"] == MIXED_QUERY and data["sources"][0]["excerpt"]["text"] == CURATED_BODY
        assert MARKER not in json.dumps(messages)
        return json.dumps(selection(data))

    result = await review_public_candidates(
        dict(independent_domains=container),
        deps,
        MIXED_QUERY,
        window_tokens=65536,
        question_binding=plan["binding"],
        identity_reviewer=identity,
        reviewer=source,
        span_references=True,
        curated_reviewer=curated,
    )
    built, wire = await build(result, tmp_path / "private", MIXED_QUERY, (1, 2))
    assert CURATED_BODY in wire and render_curated_tasks(built.retrieval)[0]["status"] == "related_candidate_admitted"
    if failure is None:
        assert registry in wire and rules in wire and len(result["results"]) == 2
    else:
        assert registry not in wire and rules not in wire and result["results"] == []
        assert result["public_task_review"]["failure_diagnostic"] == dict(
            stage=failure, input=observed[failure], raw=received[failure]
        )
        statuses = {row["task_id"]: row["status"] for row in render_public_tasks(built.retrieval)}
        assert statuses == {1: "handled_by_independent_curated_domain", 2: "review_unavailable"}
