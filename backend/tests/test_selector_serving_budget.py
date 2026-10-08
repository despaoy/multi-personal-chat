"""Serving-budget boundaries: whole turns yield to complete authorized evidence."""

import json

import pytest

from character.evidence_selector import ContextualEvidenceSelector, InputBudgetError, selection_messages
from character.models import MemoryItem
from inference.context_budget import ReviewContextBudget


def item(source="只有当天不下雨、护目镜完好、收到书面确认才出门；尚未预约或参加。"):
    return MemoryItem("plan", "shared_event", "完整个人条件计划", evidence=(source,))


def test_trim_old_turn_and_keep_full_latest_correction_and_candidate():
    budget = ReviewContextBudget(8192)
    history = [{"role": "user", "content": "旧背景" * 2000},
               {"role": "assistant", "content": "仅理解为背景。"},
               {"role": "user", "content": "说明" * 200 + "此前不是我的经历，护目镜已损坏。"},
               {"role": "assistant", "content": "已收到这一更正。"}]
    candidate = item("完整条件" * 200 + "没有书面确认就不出门。")
    query = "请依据个人完整计划核对条件，不要使用公开课程规则代替。"
    messages = selection_messages(query, [candidate], history=history, context_budget=budget)
    payload = json.loads(messages[1]["content"])
    assert payload["history"] == history[2:]
    assert payload["history_omitted_messages"] == 2
    assert payload["query"] == query
    assert payload["required_ids"] == ["plan"]
    assert payload["candidates"][0]["evidence"] == list(candidate.evidence)
    assert budget.fits(messages, 2048)


def test_small_fitting_history_serialization_is_unchanged():
    history = [{"role": "user", "content": "护目镜损坏。"}, {"role": "assistant", "content": "收到。"}]
    assert selection_messages("条件呢", [item()], history=history, context_budget=ReviewContextBudget(8192)) == selection_messages("条件呢", [item()], history=history)


@pytest.mark.parametrize("overflow", ["latest_turn", "candidate", "query", "empty_history_fixed"])
async def test_mandatory_overflow_fails_without_provider_call(overflow):
    async def reviewer(_messages):
        pytest.fail("Incomplete mandatory evidence must never reach the provider")

    history = [{"role": "user", "content": "旧" * 1000}, {"role": "assistant", "content": "旧回答"},
               {"role": "user", "content": "最新纠正"}]
    candidate = item()
    query = "核对条件"
    if overflow == "latest_turn":
        history[-1]["content"] = "新" * 9000 + "末尾撤回之前全部计划。"
    elif overflow == "candidate":
        candidate = item("完整来源" * 3000 + "末尾否定。")
    else:
        query = "完整问题" * 3000
        if overflow == "empty_history_fixed":
            history = []
    with pytest.raises(InputBudgetError):
        await ContextualEvidenceSelector(reviewer, context_budget=ReviewContextBudget(8192)).select(query, [candidate], history=history)


async def test_reviewer_receives_complete_source_and_selects_original_object():
    source = item()
    history = [{"role": "user", "content": "旧" * 6000}, {"role": "assistant", "content": "仅虚构背景"},
               {"role": "user", "content": "护目镜完好，但没有书面确认。"}]

    async def reviewer(messages):
        data = json.loads(messages[1]["content"])
        assert data["history"] == history[2:]
        assert data["candidates"][0]["evidence"] == list(source.evidence)
        return '{"decisions":[{"id":"plan","label":"use"}]}'

    result = await ContextualEvidenceSelector(reviewer, context_budget=ReviewContextBudget(8192)).select("我的出行条件", [source], history=history)
    assert result.status == "selected"
    assert result.memories[0] is source


def test_leading_assistant_background_is_not_kept_as_orphan_after_trim():
    history = [{"role": "assistant", "content": "旧" * 6000},
               {"role": "user", "content": "以下是我的最新更正。"}, {"role": "assistant", "content": "收到。"}]
    messages = selection_messages("核对条件", [item()], history=history, context_budget=ReviewContextBudget(8192))
    assert json.loads(messages[1]["content"])["history"] == history[1:]


def test_legacy_history_character_guard_remains_fail_closed():
    with pytest.raises(InputBudgetError):
        selection_messages("核对条件", [item()], history=[{"role": "user", "content": "长" * 6001}])
