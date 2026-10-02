"""Only omitted citation recovery and immutable, admitted source spans."""

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
def test_invalid_annotation_batch_never_rewrites_or_partially_binds(raw):
    original = result()

    async def model(**kwargs):
        return raw

    repaired = asyncio.run(recovery.recover_missing_citations(original, model, context_window_tokens=65536))
    assert (
        repaired.reply == ANSWER
        and repaired.response_citations == ()
        and repaired.citation_repair_status == "invalid_annotation"
    )


def test_common_body_phrase_cannot_identify_a_source():
    original = result()
    shared = "收到书面邮件确认才成功"
    docs = tuple(dict(d, content=d["content"] + shared) for d in original.plan.retrieval.documents)
    retrieval = replace(original.plan.retrieval, documents=docs, evidence="\n".join(d["content"] for d in docs))
    original = replace(original, reply=ANSWER + shared, plan=replace(original.plan, retrieval=retrieval))

    async def model(**kwargs):
        return json.dumps({"citations": [{"key": "S1", "quote": shared}]})

    repaired = asyncio.run(recovery.recover_missing_citations(original, model, context_window_tokens=65536))
    assert repaired.citation_repair_status == "invalid_annotation" and repaired.response_citations == ()


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

    repaired = asyncio.run(recovery.recover_missing_citations(original, model, context_window_tokens=65536))
    assert calls == [] and repaired.reply == ANSWER and repaired.response_citations == ()


@pytest.mark.parametrize("change", ["existing", "guard_fallback", "no_model", "source_lookup"])
def test_already_bound_or_guarded_paths_do_not_call_model(change):
    original = result()
    if change == "existing":
        original = replace(original, response_citations=(original.plan.retrieval.citations[0],))
    if change == "guard_fallback":
        original = replace(original, guard_fallback="safe")
    if change == "no_model":
        original = replace(original, model_invoked=False)
    if change == "source_lookup":
        original = replace(
            original, plan=replace(original.plan, retrieval=replace(original.plan.retrieval, source_lookup=True))
        )

    async def model(**kwargs):
        raise AssertionError("No annotation expected")

    assert asyncio.run(recovery.recover_missing_citations(original, model, context_window_tokens=65536)) == original


def test_timeout_keeps_original_answer(monkeypatch):
    monkeypatch.setattr(recovery, "ANNOTATION_TIMEOUT_SECONDS", 0.01)

    async def model(**kwargs):
        await asyncio.sleep(1)

    repaired = asyncio.run(recovery.recover_missing_citations(result(), model, context_window_tokens=65536))
    assert (
        repaired.reply == ANSWER
        and repaired.citation_repair_status == "annotation_failed"
        and repaired.response_citations == ()
    )


def test_budget_never_truncates_sources_to_fit():
    original = result()
    original = replace(original, reply=ANSWER + "长" * 2000)

    async def model(**kwargs):
        raise AssertionError("Budget exhausted before call")

    repaired = asyncio.run(recovery.recover_missing_citations(original, model, context_window_tokens=1024))
    assert repaired.reply == original.reply and repaired.citation_repair_status == "annotation_budget_exhausted"


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

    repaired = asyncio.run(recovery.recover_missing_citations(original, model, context_window_tokens=65536))
    assert repaired.reply == original.reply and repaired.response_citations == ()


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
    assert recovery.validate_annotations(raw, ANSWER, recovery._sources(original.plan.retrieval)) is None


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
    assert recovery.validate_annotations(raw, ANSWER, sources) is None
