"""Literal selection keeps source identity, permissions and final packet scope."""

import json
from copy import deepcopy
from dataclasses import replace
from html import unescape

import pytest
from tests.test_public_identity_dependencies import (
    BRIDGE,
    PRIVATE,
    QUERY,
    RULE,
    TASK,
    bundle,
    deps,
    identity,
    parts,
    payload,
)
from tests.test_public_task_evidence_review import request

from inference.generation_request import build_generation_request
from knowledge.public_identity_dependencies import parse_identity_review
from knowledge.public_object_scope import parse_scoped_decisions
from knowledge.public_source_spans import SourceSpanCapacityError, build_source_spans, expand_span_decisions
from knowledge.public_task_evidence import render_public_tasks, review_public_candidates


def prepared(*, kb=7, relation="same"):
    value, scope = payload()
    value["sources"][1]["knowledge_base_id"] = kb
    raw = json.dumps(identity(relation=relation))
    value["object_scopes"] = scope
    value["identity_review"] = dict(raw=raw, **parse_identity_review(raw, value, scope))
    value["source_span_catalog"] = build_source_spans(value, scope)
    return value, scope


def response(value):
    selected = next(
        row for row in value["source_span_catalog"] if row["source_id"] == "doc_2" and "湾台延期" in row["source_quote"]
    )
    return dict(
        decisions=[
            dict(source_id="doc_1", task_ids=[], object_evidence=[]),
            dict(
                source_id="doc_2",
                task_ids=[TASK],
                object_evidence=[
                    dict(
                        object_id="query-object:0", source_span_id=selected["span_id"], identity_binding_id="identity:0"
                    )
                ],
            ),
        ]
    )


def test_complete_alias_rule_pointer_expands_to_literal_original_not_question_name():
    value, scope = prepared()
    expanded = expand_span_decisions(json.dumps(response(value)), value, scope)
    parsed = parse_scoped_decisions(expanded, value, scope)
    proof = parsed["scoped_decisions"][1]["object_evidence"][0]
    assert proof["source_quote"] == RULE and "榆湾续办" not in proof["source_quote"]
    assert parsed["decisions"][1]["task_ids"] == [TASK]
    assert value["sources"][0]["original_body"] == BRIDGE and value["sources"][1]["original_body"] == RULE


@pytest.mark.parametrize(
    "defect", ["rewritten_quote", "cross_source", "unknown_ref", "catalogue_quote", "catalogue_source"]
)
def test_model_rewriting_and_changed_reference_catalogue_are_rejected(defect):
    value, scope = prepared()
    actual = response(value)
    proof = actual["decisions"][1]["object_evidence"][0]
    if defect == "rewritten_quote":
        proof["source_quote"] = RULE.replace("湾台延期", "榆湾续办")
    elif defect == "cross_source":
        proof["source_span_id"] = next(
            row["span_id"] for row in value["source_span_catalog"] if row["source_id"] == "doc_1"
        )
    elif defect == "unknown_ref":
        proof["source_span_id"] = "source-span:999"
    elif defect == "catalogue_quote":
        next(row for row in value["source_span_catalog"] if row["source_id"] == "doc_2")["source_quote"] = (
            "榆湾续办免费且不用原件。"
        )
    else:
        next(row for row in value["source_span_catalog"] if row["source_id"] == "doc_2")["source_id"] = "doc_1"
    with pytest.raises(ValueError):
        expand_span_decisions(json.dumps(actual), value, scope)


@pytest.mark.parametrize("options", [dict(kb=9), dict(relation="different"), dict(relation="uncertain")])
def test_denial_unknown_and_other_knowledge_base_do_not_create_rule_references(options):
    value, _ = prepared(**options)
    assert not any(row["source_id"] == "doc_2" for row in value["source_span_catalog"])


def test_complete_reference_catalogue_capacity_is_explicit_without_dropping_spans():
    value, scope = prepared()
    value["sources"][0]["indexed_chunks"] = [
        dict(id=f"doc_1_chunk_{i}", content=f"独立段{i}：榆湾续办实际材料" + "完整限制" * 140) for i in range(260)
    ]
    with pytest.raises(SourceSpanCapacityError):
        build_source_spans(value, scope)


async def selected_review():
    calls = []

    async def resolve(messages):
        return json.dumps(dict(scopes=[dict(task_id=TASK, objects=["榆湾续办"])]))

    async def identify(messages):
        return json.dumps(identity())

    async def review(messages):
        calls.append(messages)
        value = json.loads(messages[-1]["content"])
        assert value["sources"][0]["original_body"] == BRIDGE and value["sources"][1]["original_body"] == RULE
        assert PRIVATE not in json.dumps(value)
        return json.dumps(response(value))

    reviewed = await review_public_candidates(
        bundle(),
        deps(),
        QUERY,
        window_tokens=65536,
        public_obligations=parts(),
        reviewer=review,
        scope_reviewer=resolve,
        identity_reviewer=identify,
        span_references=True,
    )
    req = request(reviewed)
    req = replace(
        req, message=QUERY, retrieval=replace(req.retrieval, public_task_query=QUERY, public_dependency_indices=(0,))
    )
    assert len(calls) == 1
    return reviewed, req


async def test_real_final_packet_reference_recheck_retains_registry_rule_private_and_negation():
    reviewed, req = await selected_review()
    plan = build_generation_request(req)
    wire = unescape(plan.messages[-1]["content"])
    assert all(text in wire for text in (BRIDGE, RULE, PRIVATE))
    assert render_public_tasks(plan.retrieval)[0]["status"] == "related_candidate_admitted"
    assert reviewed["public_task_review"]["scoped_decisions"][1]["object_evidence"][0]["source_quote"] == RULE


async def test_final_render_rejects_changed_actual_reference_response():
    _, req = await selected_review()
    receipt = deepcopy(req.retrieval.public_task_review)
    raw = json.loads(receipt["object_span_review"]["raw"])
    raw["decisions"][1]["object_evidence"][0]["source_span_id"] = "source-span:999"
    receipt["object_span_review"]["raw"] = json.dumps(raw)
    with pytest.raises(ValueError):
        build_generation_request(replace(req, retrieval=replace(req.retrieval, public_task_review=receipt)))


async def test_missing_final_registry_does_not_certify_selected_alias_rule():
    _, req = await selected_review()
    packets = tuple(
        p for p in req.retrieval.evidence_packets if not any(i.startswith("doc_1_") for i in p["document_ids"])
    )
    plan = build_generation_request(replace(req, retrieval=replace(req.retrieval, evidence_packets=packets)))
    assert render_public_tasks(plan.retrieval)[0]["status"] == "object_scope_not_admitted"
    assert PRIVATE in unescape(plan.messages[-1]["content"])
