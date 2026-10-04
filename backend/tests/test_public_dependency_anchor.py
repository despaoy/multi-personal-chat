"""Complete public scope positives precede receipt/anchor omission controls."""

from copy import deepcopy
from dataclasses import replace
from html import unescape

import pytest
from tests.test_public_task_evidence_review import PRIVATE, QUERY, bundle, dependencies, request

from inference.generation_request import build_generation_request
from knowledge.public_obligations import parse_public_partitions
from knowledge.public_task_evidence import render_public_tasks, review_public_candidates


async def positive(*, coarse=False, empty=False):
    parts = (
        ()
        if coarse
        else parse_public_partitions([dict(segment_id=0, cuts=[]), dict(segment_id=1, cuts=[])], dependencies(), QUERY)
    )

    async def review(_messages):
        return (
            '{"decisions":[{"source_id":"doc_1","task_ids":[0]},{"source_id":"doc_2","task_ids":[]}]}'
            if coarse
            else '{"decisions":[{"source_id":"doc_1","task_ids":["public:0:0"]},{"source_id":"doc_2","task_ids":[]}]}'
        )

    async def forbidden(_messages):
        pytest.fail("Empty candidates must not add a provider call")

    result = await review_public_candidates(
        dict(results=[], abstained=True) if empty else bundle(),
        dependencies(),
        QUERY,
        window_tokens=65536,
        reviewer=forbidden if empty else review,
        public_obligations=parts,
    )
    req = request(result)
    req = replace(req, retrieval=replace(req.retrieval, public_dependency_indices=(0, 1)))
    built = build_generation_request(req)
    rows = render_public_tasks(built.retrieval)
    assert len(rows) == 2 and all(r["dependency_coverage"] == "verified_expected_segments" for r in rows)
    assert all(r["semantic_coverage"] == "unverified" for r in rows)
    assert PRIVATE in unescape(built.messages[-1]["content"])
    assert result["public_task_review"]["public_segment_indices"] == [0, 1]
    if not empty:
        assert [r["status"] for r in rows] == ["related_candidate_admitted", "no_related_evidence"]
        assert bundle()["original_source_packets"][0]["original_body"] in unescape(built.messages[-1]["content"])
    return req


@pytest.mark.asyncio
async def test_complete_original_public_scope_reaches_canonical_generation():
    await positive()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "defect",
    ["tasks_only", "tasks_and_declared", "missing_declared", "duplicate", "bool", "private_parent", "legacy_downgrade"],
)
async def test_whole_parent_loss_or_false_declaration_cannot_hide_missing_task(defect):
    req = await positive()
    receipt = deepcopy(req.retrieval.public_task_review)
    if defect == "tasks_only":
        receipt["tasks"].pop()
    elif defect == "tasks_and_declared":
        receipt["tasks"].pop()
        receipt["public_segment_indices"] = [0]
    elif defect == "missing_declared":
        receipt.pop("public_segment_indices")
    elif defect == "duplicate":
        receipt["public_segment_indices"] = [0, 1, 1]
    elif defect == "bool":
        receipt["public_segment_indices"] = [False, 1]
    elif defect == "private_parent":
        receipt["public_segment_indices"] = [0, 1, 2]
    elif defect == "legacy_downgrade":
        receipt["tasks"].pop()
        receipt.pop("public_segment_indices")
        receipt.pop("receipt_schema_version")
    with pytest.raises(ValueError):
        build_generation_request(replace(req, retrieval=replace(req.retrieval, public_task_review=receipt)))


@pytest.mark.asyncio
@pytest.mark.parametrize("anchor", [(False, 1), (0, 0), (0, 3), [0, 1]])
async def test_independent_anchor_must_be_actual_unique_parent_indices(anchor):
    req = await positive()
    with pytest.raises(ValueError):
        build_generation_request(replace(req, retrieval=replace(req.retrieval, public_dependency_indices=anchor)))


@pytest.mark.asyncio
async def test_absent_independent_anchor_never_claims_planned_scope_verified():
    req = await positive()
    built = build_generation_request(replace(req, retrieval=replace(req.retrieval, public_dependency_indices=())))
    assert all(r["dependency_coverage"] == "unverified" for r in render_public_tasks(built.retrieval))


@pytest.mark.asyncio
async def test_no_candidate_receipt_still_preserves_all_public_parents():
    await positive()
    req = await positive(empty=True)
    receipt = deepcopy(req.retrieval.public_task_review)
    receipt["tasks"].pop()
    with pytest.raises(ValueError):
        build_generation_request(replace(req, retrieval=replace(req.retrieval, public_task_review=receipt)))


@pytest.mark.asyncio
async def test_valid_coarse_protocol_keeps_scope_without_claiming_atomic_tasks():
    await positive()
    req = await positive(coarse=True)
    rows = render_public_tasks(build_generation_request(req).retrieval)
    assert all(r["task_granularity"] == "segment_unverified" for r in rows)
