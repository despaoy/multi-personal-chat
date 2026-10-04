"""Full source-blind resolver input; raw response must survive caller fallback."""

import asyncio
import json
from copy import deepcopy

import pytest

from knowledge.public_question_binding import resolve_question_binding, validate_question_binding
from knowledge.turn_dependencies import parse_dependencies

QUERY = "核对霁浦登记的费用、日期、材料和末尾例外。核对竹原续签费用，不要借另一业务的规则补齐。读取我档案核对的历史提醒限制；保留明确否定，未知字段保持未知。"
MARKER = "BINDING_FAILURE_DIAGNOSTIC_144"


def deps():
    return parse_dependencies(
        dict(public_knowledge=[0, 1], private_memory=[2], current_input=[], control=[3], unresolved_source=[]), QUERY
    )


async def exercise(defect=None):
    received = {}

    async def reviewer(messages):
        received["input"] = deepcopy(messages)
        value = json.loads(messages[-1]["content"])
        assert value == dict(query=QUERY, public_tasks=[dict(id=i, text=deps().segments[i]) for i in (0, 1)])
        assert set(value) == {"query", "public_tasks"} and MARKER not in json.dumps(messages)
        if defect == "transport":
            raise RuntimeError("controlled missing response")
        if defect == "timeout":
            raise asyncio.TimeoutError
        value = dict(scopes=[dict(task_id=0, objects=["霁浦登记"]), dict(task_id=1, objects=["竹原续签"])])
        if defect == "typed_id":
            value["scopes"][0]["task_id"] = "0"
        raw = json.dumps(value, ensure_ascii=False)
        if defect == "json":
            raw = "{" + MARKER
        elif defect == "fence":
            raw = "```json\n" + raw + "\n```\n" + MARKER
        elif defect == "non_text":
            raw = {"marker": MARKER}
        received["raw"] = raw
        return raw

    try:
        binding = await resolve_question_binding(deps(), QUERY, window_tokens=65536, reviewer=reviewer)
    except Exception as exc:
        return None, exc, received
    return binding, None, received


async def test_complete_source_blind_positive_preserves_exact_input_and_response():
    binding, error, received = await exercise()
    assert error is None and binding["raw"] == received["raw"]
    assert binding["input"] == json.loads(received["input"][-1]["content"])
    assert validate_question_binding(binding, binding["input"]) == binding["scopes"]


@pytest.mark.parametrize("defect", ["json", "typed_id", "fence", "non_text", "transport", "timeout"])
async def test_failed_binding_exposes_internal_diagnostic_without_binding(defect):
    binding, error, received = await exercise(defect)
    assert binding is None and error is not None
    diagnostic = getattr(error, "diagnostic", None)
    assert diagnostic == dict(
        stage="object",
        input=received["input"],
        raw=received.get("raw") if isinstance(received.get("raw"), str) else None,
    )
    assert not hasattr(error, "binding")


async def run_api(tmp_path, monkeypatch, defect=None, *, private=True, fee_only=False, private_large=False):
    from html import unescape
    from types import SimpleNamespace

    from test_failed_fact_review_receipts import (
        FIRST,
        PRIVATE,
        SECOND,
        complete_sources,
        fact_value,
        private_context,
        scope_value,
    )

    from api import generate
    from character.models import CompiledCharacterContext
    from db.schemas import MessageRequest
    from knowledge import intent_detector, public_fact_coverage, public_task_evidence, retrieval_query_plan
    from knowledge.retrieval_query_plan import RetrievalQueryPlan

    observed = dict(binding=[], retrieval_calls=0, model_calls=0)
    context = (
        await budget_pressure_private(tmp_path)
        if private_large
        else await private_context(tmp_path)
        if private
        else CompiledCharacterContext("独立测试画像", "", "", memory_status="no_match")
    )
    assert not context.memory_packets and not context.used_memory_ids
    first_body = "霁浦登记不收费；当前完整公开页未载日期、材料、例外，缺项保持未知。" if fee_only else FIRST
    if private_large:
        monkeypatch.setattr(
            generate, "get_provider_context_budget", lambda: SimpleNamespace(window_tokens=8192, evidence_max_chars=0)
        )

    async def planner(query):
        assert query == QUERY
        return RetrievalQueryPlan(status="applied", dependencies=deps())

    async def retrieve(*args, **kwargs):
        observed["retrieval_calls"] += 1
        assert args[0] == QUERY and kwargs["question_binding"]["input"]["query"] == QUERY
        if fee_only:
            from test_failed_object_identity_source_reviews import complete_sources as whole_sources

            return whole_sources((first_body, SECOND), (1, 2))
        return complete_sources()

    async def review(messages):
        data = json.loads(messages[-1]["content"])
        assert data["query"] == QUERY
        if set(data) == {"query", "public_tasks"}:
            observed["binding"].append(deepcopy(messages))
            if defect == "transport":
                raise RuntimeError("controlled missing response")
            if defect == "timeout":
                raise asyncio.TimeoutError
            value = dict(scopes=[dict(task_id=0, objects=["霁浦登记"]), dict(task_id=1, objects=["竹原续签"])])
            if defect == "typed_id":
                value["scopes"][0]["task_id"] = "0"
            raw = json.dumps(value, ensure_ascii=False)
            if defect == "json":
                raw = "{" + MARKER
            elif defect == "fence":
                raw = "```json\n" + raw + "\n```\n" + MARKER
            elif defect == "non_text":
                raw = {"marker": MARKER}
            observed["raw"] = raw
            return raw
        if set(data) == {"query", "public_tasks", "object_scopes"}:
            return json.dumps(scope_value())
        assert [source["original_body"] for source in data["sources"]] == [first_body, SECOND]
        return json.dumps(
            dict(
                decisions=[
                    dict(
                        source_id=f"doc_{i}",
                        task_ids=[i - 1],
                        object_evidence=[
                            dict(
                                object_id=f"query-object:{i - 1}",
                                source_span_id=next(
                                    span["span_id"]
                                    for span in data["source_span_catalog"]
                                    if span["source_id"] == f"doc_{i}"
                                ),
                            )
                        ],
                    )
                    for i in (1, 2)
                ]
            )
        )

    async def facts(messages):
        data = json.loads(messages[-1]["content"])
        assert data["query"] == QUERY and [source["original_body"] for source in data["sources"]] == [
            first_body,
            SECOND,
        ]
        value = fact_value()
        if fee_only:
            for aspect in value["assessments"][1:4]:
                aspect["evidence"] = []
        return json.dumps(value)

    actual_generate = generate.generate_character_response

    async def capture(request, model):
        observed["initial_retrieval"] = request.retrieval
        result = await actual_generate(request, model)
        observed["plan"] = result.plan
        return result

    async def model(**kwargs):
        observed["model_calls"] += 1
        wire = unescape("\n".join(message["content"] for message in kwargs["messages"]))
        assert QUERY in wire and MARKER not in wire and kwargs["max_tokens"] == 2048
        assert (PRIVATE in wire) is (private and not private_large)
        if defect is None:
            assert first_body in wire and SECOND in wire and "fact_evidence_admitted" in wire
            if fee_only:
                assert "partial_fact_evidence" in wire
        else:
            assert first_body not in wire and SECOND not in wire
        return (
            "已有私人提醒保留，公共规则按可见依据核对，缺失保持未知。"
            if private
            else "没有可接纳的公共依据，规则保持未知。"
        )

    monkeypatch.setattr(retrieval_query_plan, "plan_retrieval_views", planner)
    monkeypatch.setattr(intent_detector, "needs_rag", lambda _: (True, "declared-unit", None))
    monkeypatch.setattr(generate, "_retrieve_rag_bundle", retrieve)
    monkeypatch.setattr(public_task_evidence, "_review", review)
    monkeypatch.setattr(public_fact_coverage, "_review_facts", facts)
    monkeypatch.setattr(generate, "_get_system_prompt", lambda _: "独立测试人物规则")
    monkeypatch.setattr(generate, "generate_character_response", capture)
    _, used, meta = await generate._generate_with_retrieval(
        MessageRequest(message=QUERY),
        None,
        prepared_character_turn=SimpleNamespace(compiled=context, history=()),
        runtime_config=dict(maxTokens=2048, useKnowledgeBase=True),
        model_generate=model,
    )
    assert used and observed["model_calls"] == 1 and len(observed["binding"]) == 1
    assert MARKER not in json.dumps(meta)
    return observed, meta


async def test_actual_generation_positive_keeps_five_facts_and_normal_private(tmp_path, monkeypatch):
    from knowledge.public_task_evidence import render_public_tasks

    observed, meta = await run_api(tmp_path, monkeypatch)
    assert observed["retrieval_calls"] == 1 and meta["answerMode"] == "grounded_answer" and not meta["abstained"]
    assert sum(len(row["fact_coverage"]) for row in render_public_tasks(observed["plan"].retrieval)) == 5
    assert "failure_diagnostic" not in observed["initial_retrieval"].public_task_review


@pytest.mark.parametrize("defect", ["json", "typed_id", "fence", "non_text", "transport", "timeout"])
async def test_actual_generation_failed_binding_keeps_diagnostic_and_private_partial_answer(
    tmp_path, monkeypatch, defect, caplog
):
    from knowledge.public_task_evidence import render_public_tasks

    observed, meta = await run_api(tmp_path, monkeypatch, defect)
    assert observed["retrieval_calls"] == 0
    assert MARKER not in caplog.text
    receipt = observed["initial_retrieval"].public_task_review
    assert receipt.get("failure_diagnostic") == dict(
        stage="object",
        input=observed["binding"][0],
        raw=observed.get("raw") if isinstance(observed.get("raw"), str) else None,
    )
    rows = render_public_tasks(observed["plan"].retrieval)
    assert [row["task_id"] for row in rows] == [0, 1] and all(row["status"] == "review_unavailable" for row in rows)
    assert (
        meta["abstained"] and meta["answerMode"] == "partial_answer" and "partial_public_evidence" in meta["warnings"]
    )


async def test_actual_generation_failed_binding_without_private_stays_abstention(tmp_path, monkeypatch):
    from knowledge.public_task_evidence import render_public_tasks

    observed, meta = await run_api(tmp_path, monkeypatch, "json", private=False)
    assert observed["retrieval_calls"] == 0 and meta["answerMode"] == "abstention" and meta["abstained"]
    assert all(row["status"] == "review_unavailable" for row in render_public_tasks(observed["plan"].retrieval))
    assert observed["initial_retrieval"].public_task_review.get("failure_diagnostic")


async def test_cancelled_binding_propagates_without_failure_receipt():
    async def reviewer(_messages):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await resolve_question_binding(deps(), QUERY, window_tokens=65536, reviewer=reviewer)


async def test_capacity_failure_does_not_claim_a_received_exchange():
    from knowledge.public_question_binding import QuestionBindingCapacityError

    calls = []

    async def reviewer(messages):
        calls.append(messages)
        return "not received"

    with pytest.raises(QuestionBindingCapacityError) as captured:
        await resolve_question_binding(deps(), QUERY, window_tokens=100, reviewer=reviewer)
    assert not calls and not hasattr(captured.value, "diagnostic")


async def budget_pressure_private(tmp_path):
    from datetime import datetime, timezone

    from repositories.character_memory import DatabaseCharacterMemoryRepository
    from test_failed_fact_review_receipts import PRIVATE

    from character.models import CompiledCharacterContext, UserScope
    from character.source_memory import SourceMemoryService, attach_sources
    from db.database import SQLiteDB

    db = SQLiteDB(tmp_path / "whole-budget-source.sqlite")
    scope = UserScope("web", "binding-budget", "owner", "owner", "private")
    body = PRIVATE + "\n" + "普通背景不代表费用或材料依据。" * 2000
    assert (
        db.capture_memory_source(
            character_id="fiction-role",
            platform=scope.platform,
            adapter=scope.adapter,
            sender_id=scope.sender_id,
            conversation_type=scope.conversation_type,
            conversation_id=scope.conversation_id,
            source_message_id="complete-budget-pressure",
            body=body,
            observed_at=datetime.now(timezone.utc),
        )
        == "recorded"
    )
    repo = DatabaseCharacterMemoryRepository(db)
    recalled = await SourceMemoryService(repo, defer_budget=True).recall("fiction-role", scope, QUERY)
    context = attach_sources(CompiledCharacterContext("独立测试画像", "", "", memory_status="no_match"), recalled)
    assert (
        json.loads(context.episodic_reference_context or context.source_candidate_context)["records"][0]["text"] == body
    )
    assert not await repo.list_memory_records("fiction-role", scope, limit=None)
    return context


async def test_current_protocol_fee_only_and_private_original_is_partial(tmp_path, monkeypatch):
    from knowledge.public_task_evidence import render_public_tasks

    observed, meta = await run_api(tmp_path, monkeypatch, fee_only=True)
    rows = render_public_tasks(observed["plan"].retrieval)
    assert rows[0]["status"] == "partial_fact_evidence"
    assert [fact["status"] for fact in rows[0]["fact_coverage"]] == ["fact_evidence_admitted"] + [
        "no_supporting_fact_evidence"
    ] * 3
    assert meta["answerMode"] == "partial_answer" and meta["abstained"] and not meta.get("generationError")


async def test_unadmitted_whole_private_source_does_not_mark_partial_answer(tmp_path, monkeypatch):
    observed, meta = await run_api(tmp_path, monkeypatch, "json", private_large=True)
    context = observed["plan"].character_context
    assert context.memory_source_status == "budget_omitted" and not context.episodic_reference_context
    assert meta["answerMode"] == "abstention" and meta["abstained"]


async def test_diagnostic_valid_json_cannot_grant_binding_or_hide_changed_input():
    from knowledge.public_question_binding import failed_question_binding_review

    _, error, _ = await exercise("json")
    error.diagnostic["raw"] = json.dumps(
        dict(scopes=[dict(task_id=0, objects=["霁浦登记"]), dict(task_id=1, objects=["竹原续签"])])
    )
    receipt = failed_question_binding_review(error, deps(), QUERY)
    assert receipt["review_status"] == "unavailable" and receipt["decisions"] == [] and "object_scopes" not in receipt
    with pytest.raises(ValueError):
        validate_question_binding(error.diagnostic, json.loads(error.diagnostic["input"][-1]["content"]))
    error.diagnostic["input"][-1]["content"] = json.dumps(dict(query=QUERY + "其他问题", public_tasks=[]))
    with pytest.raises(ValueError, match="complete original input"):
        failed_question_binding_review(error, deps(), QUERY)
