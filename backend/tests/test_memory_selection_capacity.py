"""Complete multi-field evidence survives selection; real budgets still apply."""

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from character.context_builder import compile_reference_context
from character.evidence_selector import MAX_CANDIDATES, ContextualEvidenceSelector
from character.models import MemoryItem
from inference.context_budget import ReviewContextBudget

FACTS = (
    ("user_name", "用户说自己叫晏禾", "我叫晏禾。"),
    ("user_origin", "用户说自己来自宣城", "我来自宣城。"),
    ("user_residence", "用户说自己居住在焦作", "我现在住在焦作。"),
    ("user_major", "用户说自己的专业是材料科学", "我的专业是材料科学。"),
    ("user_workplace", "用户说自己在澄川研究院工作", "我在澄川研究院工作。"),
    ("user_study_stage", "用户说自己是大二学生", "我是大二学生。"),
)
QUESTION = "请核对我的当前个人资料：姓名、籍贯、现居地、专业、工作单位、年级各是什么。只写我本人的当前资料，不做建议。"


def facts():
    return tuple(
        MemoryItem(str(i), "user_fact", content, evidence=(source,), memory_key=key)
        for i, (key, content, source) in enumerate(FACTS)
    )


async def all_use(messages):
    payload = json.loads(messages[1]["content"])
    return json.dumps({"decisions": [{"id": key, "label": "use"} for key in payload["required_ids"]]})


async def test_complete_six_field_profile_survives_selector_and_compiler():
    original = facts()
    result = await ContextualEvidenceSelector(all_use, context_budget=ReviewContextBudget(65536)).select(
        QUESTION, original
    )
    stats = {}
    context, ids = compile_reference_context(result.memories, complete_evidence=True, diagnostics=stats)
    assert result.memories == original
    assert all(actual is expected for actual, expected in zip(result.memories, original, strict=True))
    assert ids == tuple(item.memory_id for item in original)
    assert all(source in context for _, _, source in FACTS)
    assert stats["count_skipped"] == stats["budget_skipped"] == 0
    assert stats["used_chars"] < stats["budget_chars"]


async def test_selection_keeps_candidate_bound_and_explicit_caller_limits():
    original = tuple(
        MemoryItem(str(i), "user_fact", f"用户的完整资料第{i}项", evidence=(f"这是第{i}项完整原话。",))
        for i in range(26)
    )
    selector = ContextualEvidenceSelector(all_use, context_budget=ReviewContextBudget(65536))
    result = await selector.select("核对所有提供的本人资料", original)
    assert result.candidate_count == MAX_CANDIDATES and result.memories == original[:MAX_CANDIDATES]
    for limit in (0, 1, 5):
        limited = await selector.select(QUESTION, facts(), max_items=limit)
        assert limited.memories == facts()[:limit]


def test_complete_mode_uses_budget_beyond_six_and_legacy_mode_still_limits_count():
    original = tuple(MemoryItem(str(i), "user_fact", f"用户的完整资料第{i}项") for i in range(24))
    stats = {}
    context, ids = compile_reference_context(original, complete_evidence=True, max_chars=50000, diagnostics=stats)
    assert len(ids) == 24 and stats["count_skipped"] == 0
    assert "第23项" in context
    _, legacy_ids = compile_reference_context(original, max_chars=50000)
    assert len(legacy_ids) == 5


@pytest.mark.parametrize(
    "invalid",
    [
        {"status": "erased"},
        {"status": "pending"},
        {"status": "superseded"},
        {"confidence": 0.1},
        {"valid_to": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()},
        {"valid_from": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()},
    ],
)
def test_complete_mode_never_bypasses_lifecycle_or_confidence(invalid):
    original = facts()
    forbidden = replace(original[-1], memory_id="forbidden", **invalid)
    context, ids = compile_reference_context((*original, forbidden), complete_evidence=True)
    assert len(ids) == 6 and "forbidden" not in ids
    assert '"memory_id":"forbidden"' not in context


def test_real_character_budget_omits_whole_packet_and_keeps_later_fitting_fact():
    original = facts()
    huge = replace(original[0], memory_id="huge", evidence=("完整原文开始" + "证据" * 7000 + "完整原文末尾",))
    stats = {}
    context, ids = compile_reference_context(
        (*original[:5], huge, original[-1]), complete_evidence=True, diagnostics=stats
    )
    assert ids == tuple(item.memory_id for item in original)
    assert stats["budget_skipped"] == 1 and stats["count_skipped"] == 0
    assert stats["used_chars"] <= stats["budget_chars"]
    assert "完整原文开始" not in context and "完整原文末尾" not in context
    assert FACTS[-1][2] in context


async def test_non_use_decisions_do_not_gain_admission_when_count_limit_removed():
    original = facts()

    async def reviewer(messages):
        payload = json.loads(messages[1]["content"])
        return json.dumps(
            {
                "decisions": [
                    {"id": key, "label": "use" if key != original[-1].memory_id else "wrong_subject"}
                    for key in payload["required_ids"]
                ]
            }
        )

    result = await ContextualEvidenceSelector(reviewer).select(QUESTION, original)
    assert result.memories == original[:-1]
