import json

import pytest
from scripts.evaluate_contextual_replies import (
    ARMS,
    blind_packet,
    check_environment,
    evaluate,
    markdown_packet,
    validate_cases,
)


def test_blinding_is_order_independent_complete_and_separate_from_key():
    cases = [{"id": "a", "query": "你好"}, {"id": "b", "query": "再见"}]
    report = {
        "cases": [
            {"id": case["id"], "arm": arm, "status": "generated", "reply": "同一回答"} for case in cases for arm in ARMS
        ]
    }
    packet, key = blind_packet(cases, report)
    reversed_packet, reversed_key = blind_packet(list(reversed(cases)), report)
    assert packet == list(reversed(reversed_packet))
    assert {tuple(sorted(row.items())) for row in key} == {tuple(sorted(row.items())) for row in reversed_key}
    assert len(key) == 6
    assert all(set(case["outputs"]) == {"A", "B", "C"} for case in packet)
    assert "arm" not in json.dumps(packet)
    assert "semantic_all" not in markdown_packet(packet)
    report["cases"].pop()
    with pytest.raises(ValueError, match="each case"):
        blind_packet(cases, report)


@pytest.mark.parametrize("flag", ["CONTEXTUAL_MEMORY_SELECTION_ENABLED", "CONTEXTUAL_DECISION_POLICY_ENABLED"])
def test_ambient_production_opt_in_is_rejected(monkeypatch, flag):
    monkeypatch.setenv(flag, "true")
    with pytest.raises(ValueError, match="offline isolation"):
        check_environment()


@pytest.mark.parametrize("cases", [[], [{"id": "a", "query": ""}], [{"id": "a", "query": "q", "history": [{}]}]])
def test_fixture_validation(cases):
    with pytest.raises(ValueError):
        validate_cases(cases)


async def test_real_context_and_guard_pipeline_has_no_gold_leakage(monkeypatch):
    monkeypatch.setenv("CONTEXTUAL_MEMORY_SELECTION_ENABLED", "false")
    monkeypatch.setenv("CONTEXTUAL_DECISION_POLICY_ENABLED", "false")
    calls = []

    async def reviewer(messages):
        calls.append(messages)
        assert "secret_gold_marker" not in json.dumps(messages)
        # Invalid semantic review is recorded and falls back to the actual rule
        # state; the fixture cannot secretly inject its gold into that state.
        if "state 必须包含" in messages[0]["content"]:
            return "invalid"
        if "allowed_strategies" in messages[-1]["content"]:
            return '{"strategy_ids":["reflect_content"]}'
        return "嗯，今天过得如何？"

    report = await evaluate([{"id": "a", "query": "你好", "gold": "secret_gold_marker"}], reviewer)
    assert len(report["cases"]) == 3
    assert report["summary"]["generation_failures"] == 0
    assert report["summary"]["human_reviews_completed"] == 0
    assert report["cases"][0]["diagnostics"]["state_status"] == "disabled"
    assert report["cases"][1]["diagnostics"]["state_status"] == "fallback"
    assert report["cases"][2]["diagnostics"]["policy_status"] == "applied"
    assert all(row["diagnostics"]["used_memory_ids"] == [] for row in report["cases"])
    assert len(calls) >= 6


async def test_generation_errors_are_preserved_not_replaced(monkeypatch):
    monkeypatch.setenv("CONTEXTUAL_MEMORY_SELECTION_ENABLED", "false")
    monkeypatch.setenv("CONTEXTUAL_DECISION_POLICY_ENABLED", "false")

    async def reviewer(messages):
        raise TimeoutError("private provider text must not appear in report")

    cases = [{"id": "a", "query": "你好"}]
    report = await evaluate(cases, reviewer)
    assert report["summary"]["generation_failures"] == 3
    assert "private provider text" not in json.dumps(report)
    packet, _ = blind_packet(cases, report)
    assert all("生成失败" in value for value in packet[0]["outputs"].values())


@pytest.mark.parametrize("stage", ["stranger", "familiar"])
async def test_relationship_fixture_uses_actual_repository_schema(monkeypatch, stage):
    monkeypatch.setenv("CONTEXTUAL_MEMORY_SELECTION_ENABLED", "false")
    monkeypatch.setenv("CONTEXTUAL_DECISION_POLICY_ENABLED", "false")

    async def reviewer(messages):
        return "你好。"

    report = await evaluate([{"id": "a", "query": "你好"}], reviewer, arms=("rules",), relationship_stage=stage)
    assert report["relationship_stage"] == stage
    assert report["reply_guard_mode"] == "strict"
    assert report["cases"][0]["diagnostics"]["relationship"]["stage"] == stage
    assert "model_reply_attempts" not in report["cases"][0]


async def test_optional_attempts_survive_a_later_guard_contract_failure(monkeypatch):
    monkeypatch.setenv("CONTEXTUAL_MEMORY_SELECTION_ENABLED", "false")
    monkeypatch.setenv("CONTEXTUAL_DECISION_POLICY_ENABLED", "false")

    async def reviewer(messages):
        return "模型原始回答。"

    async def guarded_failure(request, adapter):
        await adapter(
            messages=[{"role": "user", "content": "你好"}],
            temperature=0.0,
            max_tokens=768,
            lora_name=None,
            enable_thinking=False,
        )
        raise RuntimeError("deterministic closed guard fallback did not close the violation")

    monkeypatch.setattr("scripts.evaluate_contextual_replies.generate_character_response", guarded_failure)
    report = await evaluate([{"id": "a", "query": "你好"}], reviewer, arms=("rules",), capture_reply_attempts=True)
    row = report["cases"][0]
    assert row["status"] == "error"
    assert row["local_contract_error"] == "guard_fallback_invalid"
    assert row["error_stage"] == "generate_response"
    assert row["generation_attempt_count"] == 1
    assert row["model_reply_attempts"][0]["reply"] == "模型原始回答。"
    assert row["reply"] == ""
