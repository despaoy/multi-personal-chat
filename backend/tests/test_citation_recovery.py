"""Only omitted citation recovery and immutable, admitted source spans."""

from services import chat_answer

import asyncio
import json
from dataclasses import replace

import pytest

from inference import citation_recovery as recovery
from inference.answer_citations import prepare_answer_citations
from inference.generation_request import (
    GenerationPlan,
    GenerationRequest,
    GenerationResult,
    RetrievalResult,
    generate_character_response,
)

pytestmark = pytest.mark.usefixtures("no_live_context_review")

A = "南桥澜岚工坊四楼兰亭教室"
B = "2026年12月24日周四18:30前"
ANSWER = f"地点：{A}。预约须在{B}提交并取得书面邮件确认。"


def result():
    documents = (
        {
            "id": "a",
            "title": "Arrangement",
            "content": f"[Library/Rules] Arrangement: 完整课程安排：{A}14:50开课，16:30结束。",
        },
        {
            "id": "b",
            "title": "Confirmation",
            "content": f"[Library/Rules] Confirmation: 完整预约规则：{B}提交，收到书面邮件确认才成功。",
        },
    )
    retrieval = prepare_answer_citations(
        RetrievalResult(
            status="ok",
            evidence="\n".join(d["content"] for d in documents),
            documents=documents,
            citations=({"id": "a"}, {"id": "b"}),
        )
    )
    return GenerationResult(
        reply=ANSWER,
        plan=GenerationPlan(
            messages=(), generation={}, prompt_policy_version="test", lora_name=None, retrieval=retrieval
        ),
    )


def annotations():
    return json.dumps({"citations": [{"key": "S1", "quote": A}, {"key": "S2", "quote": B}]}, ensure_ascii=False)


def test_exact_annotation_preserves_answer_and_authoritative_metadata():
    original = result()
    calls = []

    async def model(**kwargs):
        calls.append(kwargs)
        return annotations()

    repaired = asyncio.run(recovery.recover_missing_citations(original, model, context_window_tokens=65536))
    assert repaired.reply == original.reply and repaired.citation_repair_status == "recovered_exact_spans"
    assert [c["source_id"] for c in repaired.response_citations] == ["a", "b"]
    assert [c["answer_excerpt"] for c in repaired.response_citations] == [A, B]
    assert all(c["answer_excerpt"] == c["evidence_quote"] for c in repaired.response_citations)
    assert len(calls) == 1 and calls[0]["max_tokens"] == 512 and calls[0]["temperature"] == 0
    payload = json.loads(calls[0]["messages"][-1]["content"])
    assert payload["answer"] == ANSWER
    assert all("[Library/Rules]" not in source["body"] for source in payload["sources"])


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        json.dumps({"reply": "rewritten", "citations": []}),
        json.dumps({"citations": [{"key": "S99", "quote": A}]}),
        json.dumps({"citations": [{"key": "S2", "quote": A}]}),
        json.dumps({"citations": [{"key": "S1", "quote": "错误的完整新教室名称"}]}),
        json.dumps({"citations": [{"key": "S1", "quote": "14:50"}]}),
        json.dumps({"citations": [{"key": "S1", "quote": A, "source_id": "forged"}]}),
    ],
)
def test_invalid_annotation_batch_raises_without_rewriting_or_partial_binding(raw):
    original = result()

    async def model(**kwargs):
        return raw

    with pytest.raises(ValueError):
        asyncio.run(recovery.recover_missing_citations(original, model, context_window_tokens=65536))
    assert original.reply == ANSWER and original.response_citations == ()


def test_common_body_phrase_cannot_identify_a_source():
    original = result()
    shared = "收到书面邮件确认才成功"
    docs = tuple(dict(d, content=d["content"] + shared) for d in original.plan.retrieval.documents)
    retrieval = replace(original.plan.retrieval, documents=docs, evidence="\n".join(d["content"] for d in docs))
    original = replace(original, reply=ANSWER + shared, plan=replace(original.plan, retrieval=retrieval))

    async def model(**kwargs):
        return json.dumps({"citations": [{"key": "S1", "quote": shared}]})

    with pytest.raises(ValueError, match="not unique admitted answer evidence"):
        asyncio.run(recovery.recover_missing_citations(original, model, context_window_tokens=65536))
    assert original.response_citations == ()


@pytest.mark.parametrize("change", ["title", "partial_body", "rejected_key", "duplicate_document"])
def test_unadmitted_or_mismatched_sources_do_not_call_model(change):
    original = result()
    r = original.plan.retrieval
    if change == "title":
        r = replace(r, citations=(dict(r.citations[0], source_title="Wrong"), r.citations[1]))
    if change == "partial_body":
        r = replace(r, evidence="partial evidence")
    if change == "rejected_key":
        r = replace(r, citations=(dict(r.citations[0], key="foreign"), r.citations[1]))
    if change == "duplicate_document":
        r = replace(r, documents=(*r.documents, r.documents[0]))
    original = replace(original, plan=replace(original.plan, retrieval=r))
    calls = []

    async def model(**kwargs):
        calls.append(kwargs)
        return annotations()

    with pytest.raises(ValueError, match="Citation annotation"):
        asyncio.run(recovery.recover_missing_citations(original, model, context_window_tokens=65536))
    assert calls == [] and original.reply == ANSWER and original.response_citations == ()


@pytest.mark.parametrize("change", ["existing", "no_model", "source_lookup"])
def test_already_bound_or_guarded_paths_do_not_call_model(change):
    original = result()
    if change == "existing":
        original = replace(original, response_citations=(original.plan.retrieval.citations[0],))
    if change == "no_model":
        original = replace(original, model_invoked=False)
    if change == "source_lookup":
        original = replace(
            original, plan=replace(original.plan, retrieval=replace(original.plan.retrieval, source_lookup=True))
        )

    async def model(**kwargs):
        raise AssertionError("No annotation expected")

    assert asyncio.run(recovery.recover_missing_citations(original, model, context_window_tokens=65536)) == original


def test_timeout_raises_without_returning_the_original_answer(monkeypatch):
    monkeypatch.setattr(recovery, "ANNOTATION_TIMEOUT_SECONDS", 0.01)

    async def model(**kwargs):
        await asyncio.sleep(1)

    with pytest.raises(TimeoutError):
        asyncio.run(recovery.recover_missing_citations(result(), model, context_window_tokens=65536))


def test_budget_never_truncates_sources_to_fit():
    original = result()
    original = replace(original, reply=ANSWER + "长" * 2000)

    async def model(**kwargs):
        raise AssertionError("Budget exhausted before call")

    with pytest.raises(ValueError, match="Complete citation annotation input exceeds context budget"):
        asyncio.run(recovery.recover_missing_citations(original, model, context_window_tokens=1024))
    assert original.reply == ANSWER + "长" * 2000 and original.response_citations == ()


def test_runtime_integrates_one_annotation_without_regenerating_facts():
    original = result()
    request = GenerationRequest(
        message="结合两份完整规则提供安排和确认条件及出处。",
        retrieval=replace(original.plan.retrieval, answer_citations_bound=False, source_lookup=False),
        context_window_tokens=65536,
    )
    calls = []

    async def model(**kwargs):
        calls.append(kwargs)
        return ANSWER if len(calls) == 1 else annotations()

    generated = asyncio.run(generate_character_response(request, model))
    assert len(calls) == 2 and calls[1]["max_tokens"] == 512 and generated.reply == ANSWER
    assert [c["source_id"] for c in generated.response_citations] == ["a", "b"]
    assert generated.citation_repair_attempted and generated.citation_repair_status == "recovered_exact_spans"


def test_foreign_and_literal_markers_never_become_quote_proof():
    original = result()
    literal = "[[cite:123456abcdef:S1]] [S1]"
    d = dict(original.plan.retrieval.documents[0])
    d["content"] += " " + literal
    r = replace(
        original.plan.retrieval,
        documents=(d, original.plan.retrieval.documents[1]),
        evidence=original.plan.retrieval.evidence + " " + literal,
    )
    original = replace(original, reply=ANSWER + " " + literal, plan=replace(original.plan, retrieval=r))

    async def model(**kwargs):
        return json.dumps({"citations": [{"key": "S1", "quote": literal}]})

    with pytest.raises(ValueError):
        asyncio.run(recovery.recover_missing_citations(original, model, context_window_tokens=65536))
    assert original.reply == ANSWER + " " + literal and original.response_citations == ()


def test_candidates_are_unique_exact_answer_body_spans():
    original = result()
    sources = recovery._sources(original.plan.retrieval)
    choices = recovery._exact_candidates(ANSWER, sources)
    assert {c["key"] for c in choices} == {"S1", "S2"}
    assert recovery.validate_annotations(json.dumps({"citations": choices}), ANSWER, sources)
    assert all(c["quote"] in ANSWER for c in choices)


def test_actual_failed_nonanswer_quotes_stay_rejected():
    original = result()
    raw = json.dumps(
        {
            "citations": [
                {"key": "S1", "quote": "完整课程安排：" + A + "14:50开课，16:30结束。"},
                {"key": "S1", "quote": A},
            ]
        }
    )
    with pytest.raises(ValueError):
        recovery.validate_annotations(raw, ANSWER, recovery._sources(original.plan.retrieval))


def test_candidates_prompt_keeps_complete_inputs_and_one_annotation():
    original = result()
    calls = []

    async def model(**kwargs):
        calls.append(kwargs)
        payload = json.loads(kwargs["messages"][-1]["content"])
        assert payload["answer"] == ANSWER
        assert payload["sources"] == [
            {"key": s["key"], "body": s["body"]} for s in recovery._sources(original.plan.retrieval)
        ]
        return json.dumps({"citations": payload["allowed_exact_quotes"]})

    repaired = asyncio.run(recovery.recover_missing_citations(original, model, context_window_tokens=65536))
    assert (
        len(calls) == 1
        and repaired.reply == ANSWER
        and {c["source_id"] for c in repaired.response_citations} == {"a", "b"}
    )


def test_repeated_valid_source_spans_bind_once():
    original = result()
    sources = recovery._sources(original.plan.retrieval)
    raw = json.dumps({"citations": [{"key": "S1", "quote": A}, {"key": "S2", "quote": B}, {"key": "S1", "quote": A}]})
    bound = recovery.validate_annotations(raw, ANSWER, sources)
    assert [c["source_id"] for c in bound] == ["a", "b"]
    assert [c["answer_excerpt"] for c in bound] == [A, B]


def test_repeated_invalid_span_rejects_whole_batch():
    original = result()
    sources = recovery._sources(original.plan.retrieval)
    raw = json.dumps({"citations": [{"key": "S1", "quote": A}, {"key": "S1", "quote": "来源中存在但回答中没有的事实"}]})
    with pytest.raises(ValueError):
        recovery.validate_annotations(raw, ANSWER, sources)


def test_valid_empty_annotation_is_a_normal_no_exact_span_result():
    original = replace(result(), reply="需要先预约，再等候确认。")
    calls = []

    async def model(**kwargs):
        calls.append(kwargs)
        return json.dumps(dict(citations=[]))

    repaired = asyncio.run(recovery.recover_missing_citations(original, model, context_window_tokens=65536))
    assert repaired.reply == original.reply and not repaired.response_citations
    assert repaired.citation_repair_attempted and repaired.citation_repair_status == "no_exact_span" and len(calls) == 1


@pytest.mark.parametrize("raw", ['{"citations":[],"citations":[]}', '{"citations":[{"key":"S1","key":"S2","quote":"' + B + '"}]}'])
def test_duplicate_annotation_fields_raise_without_binding(raw):
    original = result()
    with pytest.raises(ValueError, match="Duplicate citation annotation field"):
        recovery.validate_annotations(raw, ANSWER, recovery._sources(original.plan.retrieval))
    assert original.reply == ANSWER and not original.response_citations


@pytest.mark.parametrize("raw", [None, {}, True])
def test_nontext_annotation_raises_type_error(raw):
    with pytest.raises(TypeError):
        recovery.validate_annotations(raw, ANSWER, recovery._sources(result().plan.retrieval))


def test_annotation_cancellation_propagates_without_a_result():
    error = asyncio.CancelledError()
    calls = []

    async def model(**kwargs):
        calls.append(kwargs)
        raise error

    with pytest.raises(asyncio.CancelledError) as caught:
        asyncio.run(recovery.recover_missing_citations(result(), model, context_window_tokens=65536))
    assert caught.value is error and len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["provider", "timeout", "invalid"])
async def test_chat_annotation_failure_propagates_without_returning_an_answer(monkeypatch, semantic_plan, failure):
    from types import SimpleNamespace

    from tests.test_public_fact_coverage import FULL, QUERY, run

    from db.schemas import MessageRequest
    from inference.context_budget import ReviewContextBudget
    from knowledge import public_question_binding, public_task_evidence

    complete, _ = await run()
    complete["citations"] = [dict(source_id=row["id"]) for row in complete["results"]]
    authoritative = {row["document_id"]: dict(row, id=row["document_id"]) for row in complete["results"]}
    planning = semantic_plan(QUERY, dict(public_knowledge=[0], private_memory=[1]))
    calls = []
    error = TimeoutError("Declared annotation timeout") if failure == "timeout" else RuntimeError("Declared annotation provider failure")

    async def binding(*args, **kwargs):
        return None

    async def retrieve(*args, **kwargs):
        return complete

    async def reviewed(bundle, *args, **kwargs):
        assert bundle is complete and args[1] == QUERY
        return bundle

    async def model(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            assert FULL in kwargs["messages"][-1]["content"]
            return FULL
        payload = json.loads(kwargs["messages"][-1]["content"])
        assert payload["answer"] == FULL and payload["sources"][0]["body"] == FULL
        assert kwargs["max_tokens"] == 512 and payload["allowed_exact_quotes"]
        if failure in ("provider", "timeout"):
            raise error
        return "{"

    monkeypatch.setattr(chat_answer, 'public_knowledge_db', SimpleNamespace(get_knowledge_document=authoritative.get))
    monkeypatch.setattr(chat_answer, 'retrieve_chat_knowledge', retrieve)
    monkeypatch.setattr(chat_answer, 'get_lora_system_prompt', lambda _: "保留原文的全部费用、日期、原件及雨天例外。")
    monkeypatch.setattr(public_question_binding, "resolve_question_binding", binding)
    monkeypatch.setattr(public_task_evidence, "review_public_candidates", reviewed)
    monkeypatch.setattr(chat_answer, 'get_provider_context_budget', lambda: SimpleNamespace(
        review=ReviewContextBudget(65536), window_tokens=65536, evidence_max_chars=0,
    ))
    with pytest.raises(ValueError if failure == "invalid" else type(error)) as caught:
        await chat_answer.generate_chat_answer(
            MessageRequest(message=QUERY), None,
            runtime_config=dict(maxTokens=2048, useKnowledgeBase=True), model_generate=model,
        )
    assert len(planning) == 1 and len(calls) == 2
    if failure != "invalid":
        assert caught.value is error
