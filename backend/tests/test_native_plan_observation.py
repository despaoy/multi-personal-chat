import json
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from character.context_builder import compile_reference_context
from character.memory_llm import parse_llm_proposals
from character.models import MemoryItem

PLAN = (
    "我计划每周日上午九点半到城东青苔工坊练习木刻。"
    + "登记材料的背景说明。" * 650
    + "只有当天不下雨才去；下雨就取消。我还没有开始实施。"
)
QUOTE = "我计划每周日上午九点半到城东青苔工坊练习木刻。"


def proposal(evidence=QUOTE):
    return json.dumps(
        {
            "memories": [
                {
                    "kind": "shared_event",
                    "value": "城东青苔工坊练习木刻",
                    "evidence": evidence,
                    "operation": "ADD",
                    "confidence": 0.97,
                    "attributed_to": "user",
                    "qualifiers": {},
                }
            ]
        },
        ensure_ascii=False,
    )


def observation():
    validated = parse_llm_proposals(proposal(), source_message=PLAN)
    assert len(validated) == 1
    item = validated[0]
    assert item.source_observation and item.evidence == PLAN
    return MemoryItem(
        memory_id="plan-1",
        memory_type="shared_event",
        content=item.memory.content,
        evidence=(item.evidence,),
        confidence=0.97,
        status="active",
        relation_type="ADD",
        source_message_ids=("plan-source",),
        source_observation=True,
        temporal_mode="observation",
        observed_at=datetime.now(timezone.utc).isoformat(),
    )


def test_long_plan_is_a_complete_source_observation_not_a_completed_action():
    item = observation()
    assert "完整内容见证据" in item.content
    assert item.evidence[0].endswith("我还没有开始实施。")


@pytest.mark.parametrize(
    "evidence", ["我计划每周日上午九点半到城东青苔工坊练习木刻。只有当天不下雨才去", "我计划每天十点到虚构工坊练习木刻"]
)
def test_source_observations_still_reject_stitched_or_invented_quotes(evidence):
    assert parse_llm_proposals(proposal(evidence), source_message=PLAN) == []


def test_cloud_reference_allowance_preserves_full_plan_and_late_conditions():
    item = observation()
    # Actual temporal read views retain the original quotation in content too.
    item = replace(item, content="用户原话记录：" + json.dumps(list(item.evidence), ensure_ascii=False))
    reference, ids = compile_reference_context(
        (item,), complete_evidence=True, max_chars=32768, observation_semantics=True
    )
    assert ids == ("plan-1",)
    packet = json.loads(reference.split("\n", 1)[1][2:])
    assert packet["evidence"] == [PLAN] and packet["subject_scope"] == "not_resolved"
    assert packet["content_semantics"] == "quoted_source" and packet["speaker_role"] == "user"


def test_small_reference_budget_omits_whole_packet_without_losing_late_condition():
    diagnostics = {}
    reference, ids = compile_reference_context(
        (observation(),), complete_evidence=True, max_chars=2000, diagnostics=diagnostics
    )
    assert reference == "" and ids == ()
    assert diagnostics["budget_skipped"] == 1 and diagnostics["budget_chars"] == 2000


def test_native_cloud_service_propagates_reference_budget_and_observation_semantics(monkeypatch):
    from inference import model_manager
    from services.character_context import build_character_context_service

    monkeypatch.setattr(
        model_manager,
        "get_model_manager",
        lambda: SimpleNamespace(_current_provider=SimpleNamespace(value="openai_compat")),
    )
    monkeypatch.setenv("OPENAI_COMPAT_CONTEXT_WINDOW_TOKENS", "65536")
    service = build_character_context_service(object())
    assert service._reference_max_chars == 32768 and service._reference_observation_semantics is True


def test_writer_declares_numeric_schema_bounds_and_plan_observation_route():
    from character.memory_llm import build_memory_llm_messages

    messages = build_memory_llm_messages(PLAN, (), (), (), 2000, 0.85, context_window_tokens=65536)
    payload = json.loads(messages[1]["content"])
    contract = payload["proposal_constraints"]
    assert contract["max_value_chars"] == 48 and contract["max_evidence_chars"] == 120
    assert "status" not in contract["qualifier_keys"]
    assert "shared_event" in messages[0]["content"] and "原话观察" in messages[0]["content"]


def test_plan_observation_accepts_exact_late_source_condition():
    raw = json.loads(proposal())
    raw["memories"][0]["qualifiers"] = {"condition": "只有当天不下雨才去；下雨就取消"}
    result = parse_llm_proposals(json.dumps(raw, ensure_ascii=False), source_message=PLAN)
    assert len(result) == 1 and result[0].source_observation
    assert result[0].evidence == PLAN
    assert dict(result[0].qualifiers) == {"condition": "只有当天不下雨才去；下雨就取消"}


@pytest.mark.parametrize("qualifiers", [{"condition": "下雨也一定去"}, {"status": "已经完成"}])
def test_plan_observation_still_rejects_invented_or_unknown_late_qualifiers(qualifiers):
    raw = json.loads(proposal())
    raw["memories"][0]["qualifiers"] = qualifiers
    assert parse_llm_proposals(json.dumps(raw, ensure_ascii=False), source_message=PLAN) == []


def test_semantic_fact_cannot_borrow_an_unquoted_late_qualifier():
    raw = {
        "memories": [
            {"attributed_to": "user",
                "kind": "like",
                "value": "红茶",
                "evidence": "我喜欢红茶。",
                "operation": "ADD",
                "confidence": 0.97,
                "qualifiers": {"condition": "仅周末喝"},
            }
        ]
    }
    assert (
        parse_llm_proposals(
            json.dumps(raw, ensure_ascii=False), source_message="我喜欢红茶。" + "背景文字。" * 100 + "仅周末喝"
        )
        == []
    )
