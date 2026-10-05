"""Declared fictional unit reviewers; no cloud, real-auth, or stored-source claim."""

import json
from copy import deepcopy

import pytest
from tests.test_public_fact_coverage import QUERY, alias_plan, run

from knowledge.public_fact_coverage import fact_input, parse_fact_evidence, review_fact_evidence


def inputs(receipt):
    return receipt["object_scope_input"], receipt["object_scopes"], receipt["scoped_decisions"], receipt["decisions"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,expected", [("full", ["doc_1"]), ("complement", ["doc_1", "doc_2"])])
async def test_each_aspect_lists_only_verified_task_object_sources(mode, expected):
    reviewed, _ = await run(mode)
    payload, _, scoped, decisions = inputs(reviewed["public_task_review"])
    data = fact_input(payload, scoped, decisions)
    assert data["query"] == QUERY and data["sources"] == payload["sources"]
    assert data["aspect_allowed_sources"] == [dict(aspect_id=f"fact-aspect:{i}", source_ids=expected) for i in range(4)]


@pytest.mark.asyncio
@pytest.mark.parametrize("defect", ["task", "object_quote"])
async def test_allowlist_and_parser_share_denied_task_or_object_scope(defect):
    reviewed, _ = await run()
    receipt = reviewed["public_task_review"]
    payload, scopes, scoped, decisions = deepcopy(inputs(receipt))
    if defect == "task":
        decisions[0]["task_ids"] = []
    else:
        scoped[0]["object_evidence"][0]["source_quote"] = "不在来源原文中的虚构引用"
    assert all(not row["source_ids"] for row in fact_input(payload, scoped, decisions)["aspect_allowed_sources"])
    with pytest.raises(ValueError, match="accepted source-object-task scope"):
        parse_fact_evidence(receipt["fact_review"]["raw"], payload, scopes, scoped, decisions)


@pytest.mark.asyncio
async def test_alias_registry_is_identity_evidence_not_allowed_business_source():
    req = await alias_plan()
    payload, _, scoped, decisions = inputs(req.retrieval.public_task_review)
    assert all(
        row["source_ids"] == ["doc_2"] for row in fact_input(payload, scoped, decisions)["aspect_allowed_sources"]
    )


@pytest.mark.asyncio
async def test_missing_identity_binding_never_grants_alias_fact_source():
    req = await alias_plan()
    receipt = req.retrieval.public_task_review
    payload, scopes, scoped, decisions = deepcopy(inputs(receipt))
    scoped[1]["object_evidence"][0]["identity_binding_id"] = "identity:missing"
    assert all(not row["source_ids"] for row in fact_input(payload, scoped, decisions)["aspect_allowed_sources"])
    with pytest.raises(ValueError, match="accepted source-object-task scope"):
        parse_fact_evidence(receipt["fact_review"]["raw"], payload, scopes, scoped, decisions)


@pytest.mark.asyncio
async def test_failed_scope_contains_no_asserted_aspect_allowlist():
    reviewed, _ = await run(scope_failure=True)
    payload, _, scoped, decisions = inputs(reviewed["public_task_review"])
    assert fact_input(payload, scoped, decisions)["aspect_allowed_sources"] == []


@pytest.mark.asyncio
async def test_transmitted_allowlist_cannot_promote_denied_source():
    reviewed, _ = await run()
    receipt = reviewed["public_task_review"]
    payload, scopes, scoped, decisions = deepcopy(inputs(receipt))
    decisions[0]["task_ids"] = []
    payload["aspect_allowed_sources"] = [dict(aspect_id=f"fact-aspect:{i}", source_ids=["doc_1"]) for i in range(4)]
    assert all(not row["source_ids"] for row in fact_input(payload, scoped, decisions)["aspect_allowed_sources"])
    with pytest.raises(ValueError, match="accepted source-object-task scope"):
        parse_fact_evidence(receipt["fact_review"]["raw"], payload, scopes, scoped, decisions)


@pytest.mark.asyncio
async def test_reviewer_receives_full_sources_and_explicit_empty_scope_without_fact_claim():
    reviewed, _ = await run()
    payload, scopes, scoped, decisions = deepcopy(inputs(reviewed["public_task_review"]))
    decisions[0]["task_ids"] = []
    received = []

    async def declared_unit_reviewer(messages):
        data = json.loads(messages[-1]["content"])
        received.append(data)
        assert data["query"] == QUERY and data["sources"] == payload["sources"]
        assert all(not row["source_ids"] for row in data["aspect_allowed_sources"])
        return json.dumps(
            dict(assessments=[dict(aspect_id=r["aspect_id"], evidence=[]) for r in data["aspect_allowed_sources"]])
        )

    result = await review_fact_evidence(payload, scopes, scoped, decisions, declared_unit_reviewer, 65536)
    assert len(received) == 1 and result["review_status"] == "reviewed"
    assert all(not row["evidence"] for row in result["assessments"])
