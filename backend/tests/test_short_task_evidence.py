"""Complete fictional tasks establish positives before short/capacity controls."""

import json
from html import unescape

import pytest

from character.models import CompiledCharacterContext
from inference.generation_request import GenerationRequest, RetrievalResult, build_generation_request
from knowledge.evidence_packets import document_evidence_packets
from knowledge.original_sources import attach_original_sources
from knowledge.public_task_evidence import render_public_tasks, review_public_candidates
from knowledge.retrieval_query_plan import plan_retrieval_views
from knowledge.turn_dependencies import query_segments

QUERY = "核对紫岑借阅规定的费用与登记材料。核对白榆续借规定的费用。读取我保存的梅糕偏好。"
PRIVATE = '{"memory_id":"fiction-private","content":"我喜欢梅糕，但仅在假日，且完成个人复核时。"}'


def fixture(count=2):
    rows = [
        dict(
            id=f"doc_{i}_chunk_0",
            document_id=i,
            title=f"虚构规程{i}",
            content="紫岑借阅费用19元，完整登记材料为借阅证和确认书。末尾限定：本人携带原件。"
            if i == 1
            else "红柏游泳规则收费22元，与续借和借阅无关。",
            category="fiction",
            knowledge_base_id=6,
            score=0.95,
        )
        for i in range(1, count + 1)
    ]
    raw = dict(
        results=rows,
        confidence=0.95,
        abstained=False,
        citations=[],
        source_coverage=tuple(
            dict(
                source_id=f"doc_{i}",
                source_title=f"虚构规程{i}",
                indexed_document_ids=[f"doc_{i}_chunk_0"],
                retrieved_document_ids=[f"doc_{i}_chunk_0"],
            )
            for i in range(1, count + 1)
        ),
    )
    return attach_original_sources(
        raw, lambda i: dict(rows[i - 1], id=i), source_budget_tokens=65536, authority_revision=3
    )


async def valid_plan(monkeypatch, query=QUERY):
    monkeypatch.setenv("RAG_TASK_PLANNER_ENABLED", "true")
    monkeypatch.setenv("CORRECTIVE_RAG_ENABLED", "false")
    seen = []

    async def reviewer(messages):
        payload = json.loads(messages[-1]["content"])
        assert payload["query"] == query
        segments = query_segments(query)
        assert payload["segments"] == [dict(id=i, text=s) for i, s in enumerate(segments)]
        seen.append(True)
        groups = dict(
            private_memory=[2],
            public_knowledge=[0, 1],
            current_input=[],
            control=list(range(3, len(segments))),
            unresolved_source=[],
        )
        return json.dumps(dict(search_views=[["紫岑", "借阅规定"], ["白榆", "续借规定"]], dependencies=groups))

    result = await plan_retrieval_views(query, reviewer=reviewer)
    assert seen == [True] and result.status == "applied" and not result.private_context_only
    assert result.dependencies.segments == query_segments(query)
    return result


async def positive(monkeypatch):
    plan = await valid_plan(monkeypatch)

    async def review(messages):
        payload = json.loads(messages[-1]["content"])
        assert payload["query"] == QUERY and payload["public_task_ids"] == [0, 1]
        assert payload["sources"][0]["original_body"].endswith("本人携带原件。")
        return '{"decisions":[{"source_id":"doc_1","task_ids":[0]},{"source_id":"doc_2","task_ids":[]}]}'

    result = await review_public_candidates(fixture(), plan.dependencies, QUERY, window_tokens=65536, reviewer=review)
    packets = document_evidence_packets(result["results"]) + result["original_source_packets"]
    request = GenerationRequest(
        message=QUERY,
        retrieval=RetrievalResult(
            status="ok",
            evidence="\n".join(p["text"] for p in packets),
            evidence_packets=packets,
            source_coverage=result["source_coverage"],
            public_task_review=result["public_task_review"],
            public_task_query=QUERY,
        ),
        character_context=CompiledCharacterContext("", "", PRIVATE, used_memory_ids=("fiction-private",)),
        context_window_tokens=65536,
        max_tokens=2048,
        evidence_max_chars=0,
    )
    built = build_generation_request(request)
    assert PRIVATE in unescape(built.messages[-1]["content"])
    assert fixture()["results"][0]["content"] in unescape(built.messages[-1]["content"])
    assert [r["status"] for r in render_public_tasks(built.retrieval)] == [
        "related_candidate_admitted",
        "no_related_evidence",
    ]
    assert all(r["semantic_coverage"] == "unverified" for r in render_public_tasks(built.retrieval))
    return plan


@pytest.mark.asyncio
async def test_short_mixed_request_has_independent_public_and_private_evidence(monkeypatch):
    assert len(QUERY) < 512
    await positive(monkeypatch)


@pytest.mark.asyncio
@pytest.mark.parametrize("length", [511, 512])
async def test_same_task_semantics_across_previous_character_boundary(monkeypatch, length):
    await positive(monkeypatch)
    query = QUERY + " " * (length - len(QUERY))
    assert len(query) == length
    result = await valid_plan(monkeypatch, query)
    assert dict(result.dependencies.groups)["public_knowledge"] == (0, 1)


@pytest.mark.asyncio
async def test_short_private_plan_requires_complete_valid_protocol(monkeypatch):
    await positive(monkeypatch)
    query = "读取我的梅糕偏好及完整例外。不要新增或删除记忆。"

    async def private(messages):
        assert json.loads(messages[-1]["content"])["query"] == query
        return json.dumps(
            dict(
                search_views=[],
                dependencies=dict(
                    private_memory=[0], control=[1], current_input=[], public_knowledge=[], unresolved_source=[]
                ),
            )
        )

    plan = await plan_retrieval_views(query, reviewer=private)
    assert plan.private_context_only and not plan.views

    async def invalid(messages):
        value = json.loads(await private(messages))
        value["search_views"] = [["梅糕"]]
        return json.dumps(value)

    bad = await plan_retrieval_views(query, reviewer=invalid)
    assert bad.status == "invalid" and not bad.private_context_only


@pytest.mark.asyncio
async def test_complete_input_budget_exclusion_is_not_invalid_model_output(monkeypatch):
    plan = await positive(monkeypatch)

    async def forbidden(_messages):
        pytest.fail("No call with incomplete provider capacity")

    result = await review_public_candidates(fixture(), plan.dependencies, QUERY, window_tokens=1, reviewer=forbidden)
    assert result["abstained"] and not result["results"]
    assert result["public_task_review"]["reason"] == "complete_input_budget_exceeded"
    assert result["public_task_review"]["review_status"] == "unavailable"
    assert fixture()["results"][0]["content"].endswith("本人携带原件。")


@pytest.mark.asyncio
async def test_whole_source_capacity_is_not_invalid_model_output(monkeypatch):
    plan = await positive(monkeypatch)
    large = fixture(25)
    assert len(large["results"]) == 25

    async def forbidden(_messages):
        pytest.fail("Capacity exclusions do not call the model")

    result = await review_public_candidates(large, plan.dependencies, QUERY, window_tokens=65536, reviewer=forbidden)
    assert result["abstained"] and result["public_task_review"]["reason"] == "source_capacity_exceeded"
    assert len(large["results"]) == 25 and large["results"][-1]["content"].endswith("无关。")


@pytest.mark.asyncio
async def test_invalid_paid_review_remains_protocol_failure(monkeypatch):
    plan = await positive(monkeypatch)
    calls = []

    async def invalid(_messages):
        calls.append(True)
        return '{"decisions":[]}'

    result = await review_public_candidates(fixture(), plan.dependencies, QUERY, window_tokens=65536, reviewer=invalid)
    assert calls == [True] and result["public_task_review"]["reason"] == "invalid_or_incomplete_review"


@pytest.mark.asyncio
async def test_empty_short_input_never_calls_planner(monkeypatch):
    await positive(monkeypatch)

    async def forbidden(_messages):
        pytest.fail("Empty input is not a representable task")

    result = await plan_retrieval_views("", reviewer=forbidden)
    assert result.status == "invalid" and result.dependencies is None
