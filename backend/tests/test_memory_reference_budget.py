"""A reference budget covers its complete emitted text, including fixed lines."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from character.context_builder import (
    MEMORY_REFERENCE_DISCLAIMER,
    MIXED_REFERENCE_DISCLAIMER,
    MemoryContextBudgetError,
    compile_reference_context,
)
from character.models import MemoryItem

ITEM = MemoryItem(
    "synthetic-rule", "user_fact", "只有周二预约且持原件时办理；周末不受理，预约不能免除原件。",
    evidence=("我只在周二预约，且必须带原件；周末不办理，预约不能代替原件。",),
    source_message_ids=("synthetic-source",),
)


@pytest.mark.parametrize("complete", [False, True])
def test_exact_full_reference_budget_includes_address_disclaimer_and_newlines(complete):
    expected, _ = compile_reference_context((ITEM,), preferred_address="青汐", complete_evidence=complete, max_chars=10000)
    stats = {}
    text, identities = compile_reference_context(
        (ITEM,), preferred_address="青汐", complete_evidence=complete, max_chars=len(expected), diagnostics=stats,
    )
    assert text == expected and identities == (ITEM.memory_id,)
    assert text.startswith(MEMORY_REFERENCE_DISCLAIMER + "\n")
    assert ITEM.content in text and ITEM.evidence[0] in text
    assert stats["used_chars"] == len(text) == stats["budget_chars"]


@pytest.mark.parametrize("address", ["", "青汐"])
def test_complete_selected_packet_one_character_over_budget_fails(address):
    expected, _ = compile_reference_context((ITEM,), preferred_address=address, complete_evidence=True, max_chars=10000)
    with pytest.raises(MemoryContextBudgetError, match="reference context budget"):
        compile_reference_context((ITEM,), preferred_address=address, complete_evidence=True, max_chars=len(expected) - 1)


@pytest.mark.parametrize("complete", [False, True])
def test_address_only_still_requires_space_for_its_full_reference(complete):
    text, identities = compile_reference_context((), preferred_address="青汐", complete_evidence=complete)
    assert text and identities == ()
    stats = {}
    assert compile_reference_context(
        (), preferred_address="青汐", complete_evidence=complete, max_chars=len(text), diagnostics=stats,
    ) == (text, ())
    assert stats["used_chars"] == len(text)
    with pytest.raises(MemoryContextBudgetError):
        compile_reference_context((), preferred_address="青汐", complete_evidence=complete, max_chars=len(text) - 1)


@pytest.mark.parametrize("status", ["active", "erased"])
def test_only_admitted_observations_choose_the_actual_budgeted_disclaimer(status):
    observed = MemoryItem(
        "synthetic-observation", "user_fact", "同事说只有周六登记并持原件才办理；这不是我的既有事实。",
        source_observation=True, temporal_mode="observation", status=status,
        observed_at=(datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
    )
    expected, identities = compile_reference_context(
        (ITEM, observed), complete_evidence=True, observation_semantics=True, max_chars=10000,
    )
    stats = {}
    actual = compile_reference_context(
        (ITEM, observed), complete_evidence=True, observation_semantics=True, max_chars=len(expected), diagnostics=stats,
    )
    assert actual == (expected, identities) and stats["used_chars"] == len(expected)
    if status == "active":
        assert expected.startswith(MIXED_REFERENCE_DISCLAIMER + "\n")
        assert observed.memory_id in identities and '"subject_scope":"not_resolved"' in expected
    else:
        assert expected.startswith(MEMORY_REFERENCE_DISCLAIMER + "\n")
        assert observed.memory_id not in identities and observed.content not in expected
        assert stats["filtered_count"] == 1


def test_empty_or_filtered_reference_emits_nothing_and_uses_zero_characters():
    stats = {}
    assert compile_reference_context(
        (replace(ITEM, status="erased"),), complete_evidence=True, max_chars=1, diagnostics=stats,
    ) == ("", ())
    assert stats["used_chars"] == 0 and stats["filtered_count"] == 1
