"""User literals cannot collide with request-owned citation transport."""

from dataclasses import replace

from inference.answer_citations import citation_marker, finalize_answer_citations, prepare_answer_citations
from inference.generation_request import GenerationRequest, GenerationResult, RetrievalResult, build_generation_request


def prepared():
    return prepare_answer_citations(RetrievalResult(status="ok", evidence="完整来源",
        documents=({"id": "repair", "title": "维修规则", "content": "维修编号WX-204。"},
                   {"id": "course", "title": "课程规则", "content": "课程编号YY-573-R，未确认不能预约。"}),
        citations=({"source_id": "repair"}, {"source_id": "course"})))


def finish(retrieval, raw, **kwargs):
    plan = build_generation_request(GenerationRequest(message="保留原文字面符号并查课程编号", retrieval=retrieval))
    return finalize_answer_citations(GenerationResult(reply=raw, plan=plan, **kwargs))


def test_user_literal_preserved_and_never_binds_unrelated_first_source():
    data = prepared()
    raw = "[S1]\n\n课程YY-573-R" + citation_marker(data.citation_namespace, "S2")
    result = finish(data, raw)
    assert result.reply == "[S1]\n\n课程YY-573-R"
    assert [c["source_id"] for c in result.response_citations] == ["course"]


def test_literal_only_output_cannot_claim_any_knowledge_source():
    result = finish(prepared(), "请保留原文`[S1]`以及[S2]。")
    assert result.reply == "请保留原文`[S1]`以及[S2]。"
    assert result.response_citations == ()


def test_stale_request_marker_stays_literal_and_cannot_bind_current_source():
    first, second = prepared(), prepared()
    assert first.citation_namespace != second.citation_namespace
    stale = citation_marker(first.citation_namespace, "S1")
    result = finish(second, "上一轮字面内容：" + stale)
    assert stale in result.reply
    assert result.response_citations == ()


def test_unknown_current_key_is_removed_without_binding_and_keeps_user_literal():
    data = prepared()
    result = finish(data, "原文[S99]，无效引用" + citation_marker(data.citation_namespace, "S99"))
    assert result.reply == "原文[S99]，无效引用"
    assert result.response_citations == ()


def test_source_budget_rejection_cannot_be_undone_by_matching_namespace():
    data = prepared()
    packets = (data.evidence_packets[0], dict(data.evidence_packets[1], text="长" * 8000))
    data = replace(data, evidence_packets=packets)
    result = finish(data, "伪造课程" + citation_marker(data.citation_namespace, "S2"))
    assert [c["source_id"] for c in result.plan.retrieval.citations] == ["repair"]
    assert result.response_citations == ()


def test_guard_fallback_removes_only_owned_marker_and_preserves_literal():
    data = prepared()
    result = finish(data, "保守回应[S1]" + citation_marker(data.citation_namespace, "S1"), guard_fallback="unsupported_user_fact")
    assert result.reply == "保守回应[S1]"
    assert result.response_citations == ()


def test_preparation_is_idempotent_and_does_not_mutate_other_request():
    first = prepared()
    evidence = first.evidence
    second = prepared()
    assert prepare_answer_citations(first) is first
    assert first.evidence == evidence
    assert first.citation_namespace not in second.evidence
    assert first.citation_namespace in first.evidence


def test_missing_namespace_cannot_enable_legacy_literal_binding():
    data = replace(prepared(), citation_namespace="")
    result = finish(data, "字面[S1]，仍是原文。")
    assert result.reply == "字面[S1]，仍是原文。"
    assert result.response_citations == ()
