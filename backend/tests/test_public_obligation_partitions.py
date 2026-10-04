"""Full fictional positives precede partition/identity negative controls."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest
from tests.test_public_task_evidence_review import PRIVATE, bundle, request

from inference.generation_request import build_generation_request
from knowledge.public_obligations import parse_public_partitions, validate_public_obligations
from knowledge.public_task_evidence import render_public_tasks, review_public_candidates
from knowledge.retrieval_query_plan import plan_retrieval_views
from knowledge.turn_dependencies import parse_dependencies

QUERY = "读取我的梅糕偏好。核对青桥延期规定的费用，另核对蓝岸续租规定的时长。两部分分别保留所有限定，缺对应资料则未知。"


def dependencies():
    return parse_dependencies(
        dict(private_memory=[0], public_knowledge=[1], current_input=[], control=[2], unresolved_source=[]), QUERY
    )


def proposal():
    return [dict(segment_id=1, cuts=["另核对蓝岸"])]


def partitions():
    tasks = parse_public_partitions(proposal(), dependencies(), QUERY)
    assert [task["index"] for task in tasks] == ["public:1:0", "public:1:1"]
    assert "".join(task["query"] for task in tasks) == dependencies().segments[1]
    assert tasks[0]["start"] == len(dependencies().segments[0])
    assert validate_public_obligations(tasks, QUERY, public_indices=[1]) == tasks
    return tasks


async def positive():
    tasks = partitions()

    async def review(messages):
        payload = json.loads(messages[-1]["content"])
        assert payload["query"] == QUERY
        assert payload["public_tasks"] == [dict(id=t["index"], text=t["query"]) for t in tasks]
        assert payload["sources"][0]["original_body"].endswith("本人携带完整登记材料才能办理。")
        assert payload["public_task_ids"] == ["public:1:0", "public:1:1"]
        return '{"decisions":[{"source_id":"doc_1","task_ids":["public:1:0"]},{"source_id":"doc_2","task_ids":[]}]}'

    reviewed = await review_public_candidates(
        bundle(), dependencies(), QUERY, window_tokens=65536, reviewer=review, public_obligations=tasks
    )
    req = request(reviewed)
    req = replace(req, message=QUERY, retrieval=replace(req.retrieval, public_task_query=QUERY))
    built = build_generation_request(req)
    rows = render_public_tasks(built.retrieval)
    assert [r["status"] for r in rows] == ["related_candidate_admitted", "no_related_evidence"]
    assert [r["task_id"] for r in rows] == ["public:1:0", "public:1:1"]
    assert all(r["task_granularity"] == "literal_partition" and r["semantic_coverage"] == "unverified" for r in rows)
    assert PRIVATE in built.messages[-1]["content"]
    return req


@pytest.mark.asyncio
async def test_same_sentence_independent_businesses_do_not_share_source_confirmation():
    await positive()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "defect",
    [
        "missing",
        "private_parent",
        "bool_parent",
        "duplicate_parent",
        "invented_marker",
        "ambiguous_marker",
        "zero_cut",
        "order",
        "too_many",
    ],
)
async def test_invalid_cuts_never_rewrite_or_drop_public_query(defect):
    await positive()
    value = proposal()
    if defect == "missing":
        value = []
    elif defect == "private_parent":
        value[0]["segment_id"] = 0
    elif defect == "bool_parent":
        value[0]["segment_id"] = True
    elif defect == "duplicate_parent":
        value.append(deepcopy(value[0]))
    elif defect == "invented_marker":
        value[0]["cuts"] = ["未在原文的业务"]
    elif defect == "ambiguous_marker":
        value[0]["cuts"] = ["规定"]
    elif defect == "zero_cut":
        value[0]["cuts"] = ["核对青桥"]
    elif defect == "order":
        value[0]["cuts"] = ["另核对蓝岸", "费用"]
    elif defect == "too_many":
        value[0]["cuts"] = ["另核对蓝岸"] * 16
    with pytest.raises(ValueError):
        parse_public_partitions(value, dependencies(), QUERY)


@pytest.mark.asyncio
@pytest.mark.parametrize("defect", ["gap", "drop_suffix", "wrong_identity", "foreign_query"])
async def test_canonical_generation_rejects_tampered_partition_binding(defect):
    req = await positive()
    receipt = deepcopy(req.retrieval.public_task_review)
    if defect == "gap":
        receipt["tasks"][1]["start"] += 1
    elif defect == "drop_suffix":
        receipt["tasks"].pop()
    elif defect == "wrong_identity":
        receipt["tasks"][0]["index"] = "public:0:0"
    elif defect == "foreign_query":
        receipt["query"] = QUERY.replace("青桥", "黑桥")
    with pytest.raises(ValueError):
        build_generation_request(replace(req, retrieval=replace(req.retrieval, public_task_review=receipt)))


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_ids", [[1], [True], ["public:0:0"]])
async def test_review_cannot_return_parent_private_or_boolean_identity(bad_ids):
    await positive()

    async def review(_messages):
        return json.dumps(
            dict(decisions=[dict(source_id="doc_1", task_ids=bad_ids), dict(source_id="doc_2", task_ids=[])])
        )

    result = await review_public_candidates(
        bundle(), dependencies(), QUERY, window_tokens=65536, reviewer=review, public_obligations=partitions()
    )
    assert result["abstained"] and result["public_task_review"]["reason"] == "invalid_or_incomplete_review"


@pytest.mark.asyncio
async def test_planner_keeps_valid_dependency_when_partition_invalid(monkeypatch):
    await positive()
    monkeypatch.setenv("RAG_TASK_PLANNER_ENABLED", "true")
    value = dict(
        search_views=[["青桥", "延期规定"]],
        dependencies={k: list(v) for k, v in dependencies().groups},
        public_partitions=proposal(),
    )

    async def review(_messages):
        return json.dumps(value)

    plan = await plan_retrieval_views(QUERY, reviewer=review)
    assert plan.public_obligations == partitions() and not plan.private_context_only
    value["public_partitions"][0]["cuts"] = ["另一个不存在的业务"]
    coarse = await plan_retrieval_views(QUERY, reviewer=review)
    assert coarse.status == "applied_coarse_invalid_partitions" and coarse.dependencies == dependencies()
    assert not coarse.public_obligations and not coarse.private_context_only
