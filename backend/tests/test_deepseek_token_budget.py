"""Real numeric-request calibration and complete-input serving boundaries."""

import json
import sys
from copy import deepcopy
from dataclasses import replace
from html import unescape
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from character.models import CompiledCharacterContext
from character.output_guard import ReplyGuard
from character.source_memory import compile_sources
from inference import token_counting as counter
from inference.context_budget import CONTEXT_SAFETY_MARGIN_TOKENS, ReviewContextBudget, estimated_tokens
from inference.generation_request import (
    GenerationRequest,
    RetrievalResult,
    _trim_history_to_budget,
    build_generation_request,
    generate_character_response,
)

FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN = json.loads((FIXTURES / "deepseek_numeric_history_budget_case.json").read_text())
CASE = json.loads((FIXTURES / "deepseek_mixed_subject_history_case.json").read_text())


@pytest.fixture
def deepseek(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "openai_compat")
    monkeypatch.setenv("OPENAI_COMPAT_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "https://api.deepseek.com")
    counter._deepseek_tokenizer.cache_clear()
    counter._deepseek_text_tokens.cache_clear()
    yield
    counter._deepseek_tokenizer.cache_clear()
    counter._deepseek_text_tokens.cache_clear()


def independent_cost(messages):
    tokenizer = Tokenizer.from_file(str(counter._TOKENIZER_PATH))
    return sum(len(tokenizer.encode(m["content"], add_special_tokens=False).ids) + 4 for m in messages)


def assert_budget(messages, output, window):
    assert independent_cost(messages) + output + CONTEXT_SAFETY_MARGIN_TOKENS <= window


def test_complete_recorded_provider_request_detects_real_numeric_underestimate(deepseek):
    assert GOLDEN["synthetic"] and GOLDEN["usage"]["prompt_tokens"] == 68116
    count = sum(estimated_tokens(m["content"]) + 4 for m in GOLDEN["request"]["messages"])
    assert count == 68152 == GOLDEN["measured_counting"]["official_body_plus_four_per_message"]
    assert count > GOLDEN["measured_counting"]["old_estimate"] == 41478
    assert count + GOLDEN["request"]["max_tokens"] + CONTEXT_SAFETY_MARGIN_TOKENS > 65536
    assert counter.token_counter_info()["mode"] == "deepseek_v4_pro_bpe"


def test_real_history_budget_keeps_full_recent_turns_and_all_mandatory_wire_evidence(deepseek):
    messages = GOLDEN["request"]["messages"]
    before = deepcopy(messages)
    fixed = [messages[0], messages[-1]]
    history = messages[1:-1]
    kept = _trim_history_to_budget(history, fixed_messages=fixed, context_window_tokens=65536, max_output_tokens=2048)
    assert 0 < len(kept) < len(history)
    assert kept == history[-len(kept) :] and kept[0]["role"] == "user"
    assert all(kept[i]["role"] == "user" and kept[i + 1]["role"] == "assistant" for i in range(0, len(kept), 2))
    wire = [fixed[0], *kept, fixed[-1]]
    assert_budget(wire, 2048, 65536)
    whole = unescape(wire[-1]["content"])
    assert CASE["question"] in whole and CASE["source_message"] in whole
    assert all(doc["content"] in whole for doc in CASE["documents"])
    assert messages == before


def test_review_budget_uses_same_real_numeric_count(deepseek):
    budget = ReviewContextBudget(65536)
    assert not budget.fits(GOLDEN["request"]["messages"], 2048)
    messages = GOLDEN["request"]["messages"]
    kept = _trim_history_to_budget(
        messages[1:-1], fixed_messages=[messages[0], messages[-1]], context_window_tokens=65536, max_output_tokens=2048
    )
    assert budget.fits([messages[0], *kept, messages[-1]], 2048)


@pytest.mark.parametrize(
    "change",
    [
        {"MODEL_PROVIDER": "vllm"},
        {"OPENAI_COMPAT_MODEL": "other-model"},
        {"OPENAI_COMPAT_BASE_URL": "https://example.invalid"},
        {"OPENAI_COMPAT_BASE_URL": ""},
        {"MODEL_PROVIDER": ""},
    ],
)
def test_other_provider_or_model_keeps_existing_estimate(deepseek, monkeypatch, change):
    for name, value in change.items():
        monkeypatch.setenv(name, value)
    assert estimated_tokens("1234567890") == 3
    assert estimated_tokens("身份核验失败") == 6
    assert counter.token_counter_info() == {"mode": "generic_character_estimate"}


@pytest.mark.parametrize("asset", ["missing", "corrupt"])
def test_missing_or_changed_data_uses_byte_bound_not_unsafe_average(deepseek, monkeypatch, tmp_path, asset):
    path = tmp_path / "tokenizer.json"
    if asset == "corrupt":
        path.write_text("{}")
    monkeypatch.setattr(counter, "_TOKENIZER_PATH", path)
    text = CASE["bridges"][0]["message"] + "身份核验失败且材料不全。"
    assert estimated_tokens(text) == len(text.encode("utf-8"))
    assert estimated_tokens(text) > GOLDEN["measured_counting"]["old_estimate"] // 8
    assert counter.token_counter_info()["mode"] == "utf8_byte_upper_bound"


def test_missing_optional_library_does_not_restore_character_average(deepseek, monkeypatch):
    monkeypatch.setitem(sys.modules, "tokenizers", None)
    text = CASE["source_message"] + CASE["bridges"][0]["message"]
    assert estimated_tokens(text) == len(text.encode("utf-8"))
    assert counter.token_counter_info()["unavailable_reason"] == "ModuleNotFoundError"


def complete_request(**changes):
    source = compile_sources(
        [dict(source_message_id="user-source", observed_at="2026-10-03T00:00:00+00:00", body=CASE["source_message"])],
        max_chars=None,
    ).context
    ctx = CompiledCharacterContext("", "", "", source_candidate_context=source)
    packets = tuple(
        dict(kind="evidence", document_ids=[str(i)], text=doc["content"]) for i, doc in enumerate(CASE["documents"])
    )
    return GenerationRequest(
        message=CASE["question"],
        character_context=ctx,
        context_window_tokens=8192,
        max_tokens=1024,
        retrieval=RetrievalResult(status="ok", evidence="pending", evidence_packets=packets),
        **changes,
    )


def test_complete_small_originals_and_negated_conjunctive_source_survive_precise_count(deepseek):
    request = complete_request()
    plan = build_generation_request(request)
    text = unescape(plan.messages[-1]["content"])
    assert all(d["content"] in text for d in CASE["documents"])
    assert CASE["source_message"] in text and CASE["question"] in text
    assert len(plan.retrieval.evidence_packets) == 4
    assert plan.character_context.memory_packets == ()  # No fabricated semantic memories.
    assert_budget(plan.messages, 1024, 8192)
    assert request.character_context.source_candidate_context


def test_nonfitting_numeric_packet_omitted_whole_keeps_complete_required_small_sources(deepseek):
    request = complete_request()
    numeric = "\n".join(b["message"].strip() for b in CASE["bridges"]) + "\n完整虚构目录终点；不描述用户事实。"
    extra = dict(kind="evidence", document_ids=["too-large"], text=numeric)
    request = replace(
        request, retrieval=replace(request.retrieval, evidence_packets=(extra, *request.retrieval.evidence_packets))
    )
    plan = build_generation_request(request)
    assert extra not in plan.retrieval.evidence_packets
    assert len(plan.retrieval.evidence_packets) == 4
    text = unescape(plan.messages[-1]["content"])
    assert all(d["content"] in text for d in CASE["documents"])
    assert CASE["source_message"] in text and CASE["question"] in text
    assert_budget(plan.messages, 1024, 8192)


def retry_request():
    # Complete recorded structured evidence is a supplied unit-test document,
    # not a fabricated MemoryItem or a claim of native retrieval qualification.
    return GenerationRequest(
        message=GOLDEN["request"]["messages"][-1]["content"] + "\n请只核对给定完整资料，不要提供建议。",
        persona_prompt="核对给定完整资料。",
        apply_prompt_policy=False,
        history=GOLDEN["request"]["messages"][-5:-1],
        max_tokens=128,
        context_window_tokens=65536,
        reply_guard=ReplyGuard(forbid_advice=True),
    )


async def test_guard_retry_rebudgets_new_fixed_instruction_without_splitting_turns(deepseek, monkeypatch):
    from character import output_guard

    request = retry_request()
    fixed = build_generation_request(replace(request, history=())).messages
    window = independent_cost(fixed) + independent_cost(request.history) + 128 + CONTEXT_SAFETY_MARGIN_TOKENS
    request = replace(request, context_window_tokens=window)
    instruction = "请依据本轮已取得的完整证据复核。" * 100
    monkeypatch.setattr(output_guard, "retry_instruction", lambda violations: instruction)
    calls = []

    async def generate(**kwargs):
        calls.append(kwargs)
        assert_budget(kwargs["messages"], 128, window)
        return "建议你去散步。" if len(calls) == 1 else "已核对给定资料。"

    result = await generate_character_response(request, generate)
    assert len(calls) == 2 and result.guard_retried
    initial, retry = [c["messages"] for c in calls]
    assert initial[-1] == retry[-1]
    assert len(retry) < len(initial) and retry[1]["role"] == "user"
    assert instruction in retry[0]["content"]
    assert retry[1:-1] == initial[1:-1][-len(retry[1:-1]) :]
    assert "建议你去散步。" not in str(retry)


async def test_retry_fixed_overflow_stops_before_second_model_request(deepseek, monkeypatch):
    from character import output_guard

    request = retry_request()
    fixed = build_generation_request(replace(request, history=())).messages
    window = independent_cost(fixed) + independent_cost(request.history) + 128 + CONTEXT_SAFETY_MARGIN_TOKENS
    request = replace(request, context_window_tokens=window)
    instruction = "\n".join(x["message"] for x in CASE["bridges"])
    monkeypatch.setattr(output_guard, "retry_instruction", lambda violations: instruction)
    calls = []

    async def generate(**kwargs):
        calls.append(kwargs)
        return "建议你去散步。"

    with pytest.raises(ValueError, match="serving context budget"):
        await generate_character_response(request, generate)
    assert len(calls) == 1
    assert CASE["question"] in unescape(calls[0]["messages"][-1]["content"])
