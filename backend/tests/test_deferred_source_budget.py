import asyncio
import json
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from character.models import CompiledCharacterContext, UserScope
from character.source_memory import SourceMemoryService, attach_sources, compile_sources
from db.database import SQLiteDB
from inference.context_budget import CONTEXT_SAFETY_MARGIN_TOKENS, estimated_tokens
from inference.generation_request import (
    GenerationRequest,
    RetrievalResult,
    build_generation_request,
    generate_character_response,
)
from inference.provider_context import get_provider_context_budget
from repositories.character_memory import DatabaseCharacterMemoryRepository

BODY = (
    "Draft candidates are not permission.\n"
    + ("Complete synthetic stock record with no project authority.\n" * 400)
    + "末尾最终限制：只能离线推理，禁止联网、训练和微调。"
)
ROW = dict(source_message_id="full", observed_at="2026-10-02T01:00:00+00:00", body=BODY)
FULL = compile_sources([ROW], max_chars=None).context


def context(candidate=FULL, **kw):
    return CompiledCharacterContext(
        "profile",
        "",
        "",
        memory_status="no_match",
        memory_source_status="budget_omitted",
        source_candidate_context=candidate,
        **kw,
    )


def source_text(plan):
    import re
    from html import unescape

    wire = "\n".join(m["content"] for m in plan.messages)
    match = re.search(r"<dialogue_evidence[^>]*>\n(.*?)\n</dialogue_evidence>", wire, re.S)
    return json.loads(unescape(match[1])) if match else {}


def test_complete_long_packet_recovered_by_actual_fixed_request_budget():
    assert len(BODY) > 16384 and compile_sources([ROW], max_chars=16384).context == ""
    request = GenerationRequest(
        message="按最终条件安排验收。", character_context=context(), context_window_tokens=65536, max_tokens=1024
    )
    plan = build_generation_request(request)
    assert source_text(plan)["records"][0]["text"] == BODY
    assert (
        plan.character_context.memory_source_status == "available"
        and plan.character_context.source_candidate_context == ""
    )
    assert sum(estimated_tokens(m["content"]) + 4 for m in plan.messages) + 1024 + CONTEXT_SAFETY_MARGIN_TOKENS <= 65536
    assert plan.character_context.memory_packets == () and request.character_context.source_candidate_context == FULL


def test_too_large_source_omitted_whole_and_preserves_current_message_and_typed_fact():
    huge = compile_sources([ROW | dict(body="完整限制。" * 5000)], max_chars=None).context
    ctx = replace(
        context(huge),
        reference_context="independently valid fact",
        used_memory_ids=("known",),
        memory_field_presence=(("known", True), ("unresolved", False)),
    )
    plan = build_generation_request(
        GenerationRequest(message="当前完整任务", character_context=ctx, context_window_tokens=8192, max_tokens=1024)
    )
    assert source_text(plan) == {} and "当前完整任务" in plan.messages[-1]["content"]
    assert (
        plan.character_context.reference_context == "independently valid fact"
        and plan.character_context.used_memory_ids == ("known",)
    )
    assert (
        plan.character_context.memory_source_status == "budget_omitted"
        and plan.character_context.source_candidate_context == ""
    )
    assert plan.character_context.memory_field_presence == (("known", True), ("unresolved", None))


@pytest.mark.parametrize("corrupt", ["speaker", "duplicate", "inferred_current"])
def test_invalid_deferred_packet_is_not_injected_or_absence_proof(corrupt):
    p = json.loads(FULL)
    if corrupt == "speaker":
        p["speaker_role"] = "assistant"
    elif corrupt == "duplicate":
        p["records"].append(p["records"][0])
    else:
        p["current_validity"] = "confirmed_current"
    plan = build_generation_request(
        GenerationRequest(
            message="验收",
            character_context=context(json.dumps(p), memory_field_presence=(("field", False),)),
            context_window_tokens=65536,
        )
    )
    assert source_text(plan) == {} and plan.character_context.memory_source_status == "retrieval_error"
    assert plan.character_context.memory_field_presence == (("field", None),)


def test_public_packet_attempts_cannot_change_private_source_decision():
    retrieval = RetrievalResult(
        status="ok",
        evidence="pending",
        evidence_packets=(
            dict(kind="foreground", text="长公共证据。" * 2000, document_ids=["too-large"]),
            dict(kind="foreground", text="Complete small public evidence.", document_ids=["small"]),
        ),
        citations=({"id": "too-large"}, {"id": "small"}),
    )
    plan = build_generation_request(
        GenerationRequest(
            message="验收并核对公共材料",
            character_context=context(),
            retrieval=retrieval,
            context_window_tokens=12000,
            max_tokens=1024,
            evidence_max_chars=20000,
        )
    )
    assert source_text(plan)["records"][0]["text"] == BODY
    assert plan.retrieval.has_evidence and plan.retrieval.evidence == "Complete small public evidence."
    assert [c["id"] for c in plan.retrieval.citations] == ["small"]


def test_fresh_granted_source_is_retained_only_as_deferred_candidate(tmp_path):
    db = SQLiteDB(tmp_path / "source.sqlite")
    fields = dict(
        character_id="role",
        platform="web",
        adapter="deferred",
        sender_id="owner",
        conversation_type="private",
        conversation_id="owner",
    )
    assert (
        db.capture_memory_source(**fields, source_message_id="full", body=BODY, observed_at=datetime.now(timezone.utc))
        == "recorded"
    )
    repo = DatabaseCharacterMemoryRepository(db)
    scope = UserScope("web", "deferred", "owner", "owner", "private")
    result = asyncio.run(
        SourceMemoryService(repo, max_chars=16384, defer_budget=True).recall("role", scope, "synthetic stock")
    )
    assert result.context == "" and result.diagnostics["status"] == "budget_omitted" and result.candidate_context
    assert json.loads(result.candidate_context)["records"][0]["text"] == BODY
    compiled = attach_sources(CompiledCharacterContext("p", "", "", memory_field_presence=(("field", False),)), result)
    assert (
        compiled.episodic_reference_context == ""
        and compiled.source_candidate_context == result.candidate_context
        and compiled.memory_packets == ()
    )
    assert asyncio.run(repo.list_memory_records("role", scope, limit=None)) == []


def test_revoked_candidate_not_retained_and_local_defer_disabled_by_default():
    class Repository:
        async def linked_sources(self, *a, **kw):
            return []

        async def search_sources(self, *a, **kw):
            return [ROW]

        async def list_sources(self, *a, **kw):
            return []

    result = asyncio.run(
        SourceMemoryService(Repository(), max_chars=16384, defer_budget=True).recall(
            "role", UserScope("web", "deferred", "owner", "owner", "private"), "query"
        )
    )
    assert result.context == result.candidate_context == "" and result.diagnostics["status"] == "no_match"
    cloud = get_provider_context_budget(
        SimpleNamespace(_current_provider=SimpleNamespace(value="openai_compat")),
        env={"OPENAI_COMPAT_CONTEXT_WINDOW_TOKENS": "65536"},
    )
    local = get_provider_context_budget(
        SimpleNamespace(_current_provider=SimpleNamespace(value="vllm")), env={"VLLM_MAX_MODEL_LEN": "8192"}
    )
    assert cloud.defer_source_budget and not local.defer_source_budget and local.source_max_chars == 2400


def test_model_and_post_plan_logic_use_same_admitted_context(monkeypatch):
    import inference.memory_response as responses

    observed = []

    def inspect(message, ctx, **kw):
        observed.append(ctx)
        return None

    monkeypatch.setattr(responses, "render_memory_response", inspect)

    async def generate(**kw):
        assert source_text(SimpleNamespace(messages=kw["messages"]))["records"][0]["text"] == BODY
        return "仅整理验收清单。"

    result = asyncio.run(
        generate_character_response(
            GenerationRequest(
                message="整理验收", character_context=context(), context_window_tokens=65536, max_tokens=1024
            ),
            generate,
        )
    )
    assert result.model_invoked and observed and observed[0] == result.plan.character_context
    assert observed[0].memory_source_status == "available" and observed[0].source_candidate_context == ""
