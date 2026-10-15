"""Budget failures are typed; broken question authority is not a capacity miss.

The explicit character-count counter controls budget arithmetic only; no model,
tokenizer asset, database or network is used. Existing packet/guard tests retain
the production admission and retry behavior checks.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from character.evidence_selector import InputBudgetError
from character.models import CompiledCharacterContext
from inference.context_budget import ReviewContextBudget
from inference.generation_request import GenerationRequest, build_generation_request
from inference.request_context import InferenceRequestContext, use_request_context
from knowledge.public_obligations import parse_public_partitions
from knowledge.public_question_binding import (
    QuestionBindingCapacityError,
    question_scope_input,
    resolve_question_binding,
)
from knowledge.turn_dependencies import KINDS, parse_dependencies

QUERY = "查虚构青屿档案馆2026年10月周二的开放时段，保留雨天只关闭室外展区及周一不开放的例外。"
HISTORY = (
    {"role": "user", "content": "完整虚构资料：青屿档案馆2026年10月周二09:00至17:00开放；周一不开放，雨天只关闭室外展区。"},
    {"role": "assistant", "content": "这是一份虚构对照资料；保留日期、否定和雨天范围，不当作真实公共事实。"},
)
FULL_SOURCE = "\n".join(
    f"虚构完整资料第{index}条：2026年10月周二09:00至17:00开放；仅预约观众可入馆，但预约不免除原件核验。"
    for index in range(40)
) + "\n末尾范围：周一不开放；雨天只关闭室外展区，暴雨全馆暂停且已预约也不能例外入馆。"


@pytest.fixture(autouse=True)
def isolated_counter():
    with use_request_context(InferenceRequestContext("vllm", None, len)):
        yield


def complete_input():
    groups = {kind: [] for kind in KINDS}
    groups["public_knowledge"] = [0]
    dependencies = parse_dependencies(groups, QUERY)
    obligations = parse_public_partitions([{"segment_id": 0, "cuts": []}], dependencies, QUERY)
    payload = question_scope_input(dependencies, QUERY, obligations)
    assert payload == {"query": QUERY, "public_tasks": [{"id": "public:0:0", "text": QUERY}]}
    return dependencies, obligations


async def resolve_invalid_input(kind):
    dependencies, obligations = complete_input()
    query = QUERY
    if kind == "query":
        query += "还须核对暴雨时全馆暂停的范围。"
    elif kind == "obligations":
        obligations = ({**obligations[0], "query": obligations[0]["query"] + "未经原输入确认的增补"},)
    else:
        raise AssertionError("unknown isolated integrity case")
    calls = []

    async def forbidden(messages):
        calls.append(messages)
        pytest.fail("Invalid question authority must fail before a reviewer call")

    try:
        await resolve_question_binding(
            dependencies, query, public_obligations=obligations, window_tokens=65536, reviewer=forbidden,
        )
    finally:
        assert calls == []


@pytest.mark.parametrize("kind, message", [
    ("query", "Question binding changed the original query"),
    ("obligations", "Public obligation is not exact original query data"),
])
async def test_question_integrity_errors_are_not_relabelled_as_capacity(kind, message):
    with pytest.raises(ValueError, match=message) as caught:
        await resolve_invalid_input(kind)
    assert type(caught.value) is ValueError
    assert not isinstance(caught.value, QuestionBindingCapacityError)


async def test_whole_reference_turn_capacity_retains_typed_budget_cause_without_review():
    dependencies, obligations = complete_input()
    calls = []

    async def forbidden(messages):
        calls.append(messages)
        pytest.fail("A complete two-message turn cannot be sliced to meet a one-message budget")

    with pytest.raises(QuestionBindingCapacityError) as caught:
        await resolve_question_binding(
            dependencies, QUERY, public_obligations=obligations, history=HISTORY,
            context_budget=ReviewContextBudget(65536, history_messages=1),
            window_tokens=65536, reviewer=forbidden,
        )
    assert isinstance(caught.value.__cause__, InputBudgetError)
    assert str(caught.value.__cause__) == "latest turn exceeds history message budget"
    assert calls == []


@pytest.mark.parametrize("branch", [False, True])
def test_complete_fixed_generation_overflow_uses_budget_type_and_preserves_original_message(branch):
    request = GenerationRequest(
        message=QUERY if branch else QUERY + "\n完整虚构资料：\n" + FULL_SOURCE,
        persona_prompt="只根据完整输入说明条件，不擅自补齐缺少的事实。",
        max_tokens=64, context_window_tokens=1024, apply_prompt_policy=False,
    )
    if branch:
        request = replace(request, character_context=CompiledCharacterContext("", "", "", branch_context=FULL_SOURCE))
    original_message = request.message
    expected = (
        "Branch evidence exceeds context budget; no facts were silently dropped"
        if branch else "Current message and evidence exceed the serving context budget; shorten the message or evidence"
    )
    with pytest.raises(InputBudgetError) as caught:
        build_generation_request(request)
    assert str(caught.value) == expected
    assert request.message == original_message
    if branch:
        assert request.character_context.branch_context == FULL_SOURCE
    else:
        assert request.message.endswith(FULL_SOURCE)
