"""Complete fictional positives precede public relevance/admission negative controls."""

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import replace
from html import unescape

import pytest

from character.models import CompiledCharacterContext
from inference.generation_request import GenerationRequest, RetrievalResult, build_generation_request
from knowledge.original_sources import attach_original_sources
from knowledge.public_task_evidence import render_public_tasks, review_public_candidates
from knowledge.retrieval_query_plan import RetrievalQueryPlan, plan_retrieval_views
from knowledge.turn_dependencies import parse_dependencies

QUERY = "核对青桥延期规定的费用。核对蓝岸续租规定的时长。读取我的已有偏好。"
PRIVATE = '{"memory_id":"private-a","content":"用户喜欢布丁，但只在周末食用。"}'


def dependencies():
    return parse_dependencies(
        dict(private_memory=[2], public_knowledge=[0, 1], current_input=[], control=[], unresolved_source=[]), QUERY
    )


def bundle():
    bodies = {
        1: "青桥延期费用为31元，不办理续租。末尾限定：本人携带完整登记材料才能办理。",
        2: "这是完全无关的紫藤登山规定，时长8小时，不能作为蓝岸续租规定。",
    }
    rows = [
        dict(
            id=f"doc_{i}_chunk_0",
            document_id=i,
            title=f"独立规程{i}",
            content=body,
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
                source_title=f"独立规程{i}",
                indexed_document_ids=[f"doc_{i}_chunk_0"],
                retrieved_document_ids=[f"doc_{i}_chunk_0"],
            )
            for i in bodies
        ),
    )
    return attach_original_sources(
        raw, lambda i: dict(rows[i - 1], id=i), source_budget_tokens=65536, authority_revision=3
    )


async def accept(messages):
    payload = json.loads(messages[-1]["content"])
    assert payload["query"] == QUERY
    assert payload["segments"] == [
        {"id": 0, "text": dependencies().segments[0]},
        {"id": 1, "text": dependencies().segments[1]},
    ]
    assert payload["query"] == QUERY and payload["public_task_ids"] == [0, 1]
    assert payload["required_source_ids"] == ["doc_1", "doc_2"]
    assert payload["sources"][0]["original_body"].endswith("本人携带完整登记材料才能办理。")
    assert PRIVATE not in messages[0]["content"]
    return '{"decisions":[{"source_id":"doc_1","task_ids":[0]},{"source_id":"doc_2","task_ids":[]}]}'


def request(reviewed, huge=False):
    from knowledge.evidence_packets import document_evidence_packets

    packets = document_evidence_packets(reviewed["results"]) + reviewed["original_source_packets"]
    coverage = deepcopy(reviewed["source_coverage"])
    if huge:
        packets = tuple(dict(p, text="完整独立条款。" * 50000 + p["text"]) for p in packets)
        for row in coverage:
            original = next(p for p in packets if p.get("original_source_id") == row["source_id"])
            row["original_source_receipt"]["original_packet_sha256"] = hashlib.sha256(
                original["text"].encode()
            ).hexdigest()
    retrieval = RetrievalResult(
        status="ok" if reviewed["results"] else "character_abstention",
        evidence="\n".join(p["text"] for p in packets),
        evidence_packets=packets,
        source_coverage=coverage,
        public_task_review=reviewed["public_task_review"],
        public_task_query=QUERY,
    )
    return GenerationRequest(
        message=QUERY,
        retrieval=retrieval,
        character_context=CompiledCharacterContext("", "", PRIVATE, used_memory_ids=("private-a",)),
        context_window_tokens=65536,
        max_tokens=2048,
        evidence_max_chars=0,
    )


async def positive():
    original = bundle()
    result = await review_public_candidates(original, dependencies(), QUERY, window_tokens=65536, reviewer=accept)
    assert len(result["results"]) == 1 and len(original["results"]) == 2
    plan = build_generation_request(request(result))
    rows = render_public_tasks(plan.retrieval)
    assert rows[0]["status"] == "related_candidate_admitted" and rows[1]["status"] == "no_related_evidence"
    wire = unescape(plan.messages[-1]["content"])
    assert bundle()["original_source_packets"][0]["original_body"] in wire and PRIVATE in wire
    return result, plan


async def test_review_preserves_complete_matched_source_and_excludes_other_task_vote():
    result, plan = await positive()
    assert all(row["semantic_coverage"] == "unverified" for row in render_public_tasks(plan.retrieval))
    assert bundle()["original_source_packets"][1]["original_body"] not in unescape(plan.messages[-1]["content"])
    assert plan.generation["max_tokens"] == 2048 and plan.retrieval.public_task_query == QUERY


@pytest.mark.parametrize(
    "bad",
    [
        '{"decisions":[{"source_id":"invented","task_ids":[0]},{"source_id":"doc_2","task_ids":[]}]}',
        '{"decisions":[{"source_id":"doc_1","task_ids":[2]},{"source_id":"doc_2","task_ids":[]}]}',
        '{"decisions":[{"source_id":"doc_1","task_ids":[true]},{"source_id":"doc_2","task_ids":[]}]}',
        '{"decisions":[{"source_id":"doc_1","task_ids":[0,0]},{"source_id":"doc_2","task_ids":[]}]}',
        '{"decisions":[{"source_id":"doc_1","task_ids":[0]}]}',
        '{"decisions":[{"source_id":"doc_1","task_ids":[0]},{"source_id":"doc_1","task_ids":[]}]}',
        '{"decisions":[],"facts":{"price":0}}',
        '{"decisions":[],"decisions":[]}',
    ],
)
async def test_invalid_review_never_retains_candidates_or_erases_private_context(bad):
    await positive()

    async def review(_messages):
        return bad

    result = await review_public_candidates(bundle(), dependencies(), QUERY, window_tokens=65536, reviewer=review)
    assert not result["results"] and result["abstained"] and not result["original_source_packets"]
    plan = build_generation_request(request(result))
    assert PRIVATE in unescape(plan.messages[-1]["content"])
    assert all(row["status"] == "review_unavailable" for row in render_public_tasks(plan.retrieval))


async def test_provider_failure_rejects_public_candidates_without_retry():
    await positive()
    calls = []

    async def fail(_messages):
        calls.append(True)
        raise RuntimeError("provider unavailable")

    result = await review_public_candidates(bundle(), dependencies(), QUERY, window_tokens=65536, reviewer=fail)
    assert calls == [True] and result["abstained"] and result["public_task_review"]["reason"] == "provider_error"


async def test_cancelled_review_propagates():
    import asyncio

    await positive()

    async def cancel(_messages):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await review_public_candidates(bundle(), dependencies(), QUERY, window_tokens=65536, reviewer=cancel)


async def test_complete_review_budget_stops_before_call_without_prefix_clipping():
    await positive()

    async def fail(_messages):
        pytest.fail("Incomplete input must not be paid for")

    result = await review_public_candidates(bundle(), dependencies(), QUERY, window_tokens=1, reviewer=fail)
    assert not result["results"] and result["abstained"]


async def test_related_source_is_not_claimed_admitted_after_actual_budget_exclusion():
    result, _ = await positive()
    # Preserve complete source provenance but give the packet a genuinely huge
    # representation. It must be omitted whole, not repaired into visible text.
    changed = request(result, huge=True)
    plan = build_generation_request(changed)
    assert render_public_tasks(plan.retrieval)[0]["status"] == "related_candidate_not_admitted"
    assert PRIVATE in unescape(plan.messages[-1]["content"])


async def test_other_query_cannot_reuse_public_review_receipt():
    result, _ = await positive()
    with pytest.raises(ValueError, match="original message"):
        build_generation_request(replace(request(result), message="另一个任务"))


async def test_invalid_views_cannot_erase_valid_public_dependencies(monkeypatch):
    await positive()
    monkeypatch.setenv("RAG_TASK_PLANNER_ENABLED", "true")
    query = QUERY + "只核对已有依据。" * 80
    segments = re.findall(r"[^。]+。", query)
    dep = dict(
        private_memory=[2],
        public_knowledge=[0, 1],
        current_input=[],
        control=list(range(3, len(segments))),
        unresolved_source=[],
    )

    async def review(_messages):
        return json.dumps(dict(search_views=[["改写而非原文的检索词"]], dependencies=dep))

    result = await plan_retrieval_views(query, reviewer=review)
    assert result.status == "dependencies_only_invalid_views" and result.views == ()
    assert result.dependencies is not None and not result.private_context_only


async def test_empty_candidate_path_adds_no_review_call():
    await positive()

    async def fail(_messages):
        pytest.fail("No candidate needs no review")

    result = await review_public_candidates(
        dict(results=[], abstained=True), dependencies(), QUERY, window_tokens=65536, reviewer=fail
    )
    assert result["public_task_review"]["review_status"] == "no_candidates" and result["abstained"]


async def test_actual_api_marks_partial_answer_only_for_visible_private_context(monkeypatch):
    from types import SimpleNamespace

    from api import generate
    from db.schemas import MessageRequest
    from knowledge import intent_detector, public_task_evidence, retrieval_query_plan

    await positive()

    async def planner(_query):
        return RetrievalQueryPlan(status="applied", dependencies=dependencies())

    async def retrieve(*_args, **_kwargs):
        return deepcopy(bundle())

    async def reject(messages):
        payload = json.loads(messages[-1]["content"])
        if "sources" not in payload:
            if "object_scopes" not in payload:
                return json.dumps(dict(scopes=[
                    dict(task_id=0, objects=["青桥延期"]),
                    dict(task_id=1, objects=["蓝岸续租"]),
                ]))
            return json.dumps(dict(tasks=[
                dict(task_id=0, aspects=[dict(object_id="query-object:0", query_quote="费用")]),
                dict(task_id=1, aspects=[dict(object_id="query-object:1", query_quote="时长")]),
            ]))
        return '{"decisions":[{"source_id":"doc_1","task_ids":[]},{"source_id":"doc_2","task_ids":[]}]}'

    async def model(**kwargs):
        assert kwargs["max_tokens"] == 2048
        return "本人偏好有记录，公共规定本次无法确认。"

    monkeypatch.setattr(retrieval_query_plan, "plan_retrieval_views", planner)
    monkeypatch.setattr(intent_detector, "needs_rag", lambda _: (True, "unit", None))
    monkeypatch.setattr(generate, "_retrieve_rag_bundle", retrieve)
    monkeypatch.setattr(public_task_evidence, "_review", reject)
    monkeypatch.setattr(generate, "_get_system_prompt", lambda _: "人物规则")
    for reference, expected in [(PRIVATE, "partial_answer"), ("", "abstention")]:
        context = CompiledCharacterContext("", "", reference, used_memory_ids=("private-a",))
        prepared = SimpleNamespace(compiled=context, history=())
        _, used, meta = await generate._generate_with_retrieval(
            MessageRequest(message=QUERY),
            None,
            prepared_character_turn=prepared,
            runtime_config=dict(maxTokens=2048, useKnowledgeBase=True),
            model_generate=model,
        )
        assert used and meta["abstained"] and meta["answerMode"] == expected
        assert "partial_public_evidence" in meta["warnings"]


async def test_abstention_model_failure_propagates(monkeypatch):
    from types import SimpleNamespace

    from api import generate
    from db.schemas import MessageRequest
    from knowledge import intent_detector, public_task_evidence, retrieval_query_plan

    await positive()

    async def planner(_query):
        return RetrievalQueryPlan(status="applied", dependencies=dependencies())

    async def retrieve(*_args, **_kwargs):
        return deepcopy(bundle())

    async def reject(messages):
        payload = json.loads(messages[-1]["content"])
        if "sources" not in payload:
            if "object_scopes" not in payload:
                return json.dumps(dict(scopes=[
                    dict(task_id=0, objects=["青桥延期"]),
                    dict(task_id=1, objects=["蓝岸续租"]),
                ]))
            return json.dumps(dict(tasks=[
                dict(task_id=0, aspects=[dict(object_id="query-object:0", query_quote="费用")]),
                dict(task_id=1, aspects=[dict(object_id="query-object:1", query_quote="时长")]),
            ]))
        return '{"decisions":[{"source_id":"doc_1","task_ids":[]},{"source_id":"doc_2","task_ids":[]}]}'

    async def failure(**_kwargs):
        raise RuntimeError("generation failed")

    monkeypatch.setattr(retrieval_query_plan, "plan_retrieval_views", planner)
    monkeypatch.setattr(intent_detector, "needs_rag", lambda _: (True, "unit", None))
    monkeypatch.setattr(generate, "_retrieve_rag_bundle", retrieve)
    monkeypatch.setattr(public_task_evidence, "_review", reject)
    monkeypatch.setattr(generate, "_get_system_prompt", lambda _: "人物规则")
    context = CompiledCharacterContext("", "", PRIVATE, used_memory_ids=("private-a",))
    prepared = SimpleNamespace(compiled=context, history=())
    with pytest.raises(RuntimeError, match="generation failed"):
        await generate._generate_with_retrieval(
            MessageRequest(message=QUERY),
            None,
            prepared_character_turn=prepared,
            runtime_config=dict(maxTokens=2048, useKnowledgeBase=True),
            model_generate=failure,
        )
