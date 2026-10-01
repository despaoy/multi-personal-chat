import json
from pathlib import Path

import pytest

from character.memory_llm import (
    build_memory_llm_messages,
    is_memory_erasure_request,
    parse_llm_proposals,
)
from character.source_erasure_selection import selected_ids


def original():
    return json.loads((Path(__file__).parent / "fixtures/deepseek_memory_archive_retention_cases.json").read_text())[
        "cases"
    ][-1]["message"]


@pytest.mark.parametrize(
    "message",
    [
        "请从长期记忆中删除我的猫名，不要求删除聊天历史。",
        "请忘掉我提供的住处。不要删除聊天记录。",
        "请把你记住的工作地点删除；也不用移除对话历史。",
        "不要清除对话记录，请从记忆中删除那条信息。",
        "请从长期记忆中删除这项偏好，同时不希望删掉聊天历史本身。",
        "请从记忆中删除那条资料。聊天历史里仍会保留原话。",
        "请从长期记忆中删除这项偏好，也不要求取消现实课程预约。",
    ],
)
def test_memory_authorization_is_independent_of_explicit_archive_retention(message):
    assert is_memory_erasure_request(message)


def test_original_complete_failure_case_is_authorized():
    source = original()
    assert len(source) > 200 and "不要求删除聊天历史" in source
    assert is_memory_erasure_request(source)


@pytest.mark.parametrize(
    "message",
    [
        "不要求删除聊天历史，也不要从长期记忆中删除我的猫名。",
        "不要删除聊天历史里保存的猫名。",
        "不要删除聊天历史里的那条记忆。",
        "不要求删除聊天历史中的工作地点，请忘掉我说的住处。",
        "不要删除聊天历史和长期记忆，请忘掉我的猫名。",
        "不要求删除聊天历史或任何记忆，请忘掉住处。",
        "如果要从长期记忆中删除猫名，不要求删除聊天历史，该怎么办？",
        "朋友说请从长期记忆中删除我的猫名。不要删除聊天历史。",
        "你说过请忘掉我的住处。不要求删除聊天历史。",
        "“请从长期记忆中删除我的猫名，不要求删除聊天历史。”是示例。",
        "不要删除聊天历史，“请忘掉我的住处”只是引用。",
        "不要删除聊天历史。",
    ],
)
def test_memory_negatives_ambiguous_archive_targets_and_untrusted_intent_stay_denied(message):
    assert not is_memory_erasure_request(message)


def records():
    return (
        dict(
            id=7,
            memory_key="fact_course",
            memory_type="user_fact",
            content="用户预约海庭鹤林工坊纸版压印课，回执MB-764-C",
            status="active",
        ),
    )


def proposal(source):
    return json.dumps(
        {
            "memories": [
                dict(
                    kind="other_user_fact",
                    value="",
                    content="",
                    evidence=source[:70],
                    operation="ERASE",
                    target_memory_id="7",
                    target_memory_key="fact_course",
                    confidence=0.99,
                )
            ]
        }
    )


def test_original_authorization_reaches_existing_whitelist_without_rewriting_evidence():
    source = original()
    parsed = parse_llm_proposals(proposal(source), source_message=source, existing_memories=records())
    assert len(parsed) == 1 and parsed[0].operation == "ERASE"
    assert parsed[0].target_memory_id == "7" and parsed[0].target_memory_key == "fact_course"
    wrong = json.loads(proposal(source))
    wrong["memories"][0]["target_memory_id"] = "8"
    assert not parse_llm_proposals(json.dumps(wrong), source_message=source, existing_memories=records())


def test_archive_retention_does_not_allow_model_to_override_memory_denial():
    source = "不要求删除聊天历史，也不要从长期记忆中删除我的课程预约。"
    assert not parse_llm_proposals(proposal(source), source_message=source, existing_memories=records())
    with pytest.raises(ValueError, match="unauthorized"):
        selected_ids(
            {"erase_source_ids": ["owner-source"]}, {"owner-source"}, authorized=is_memory_erasure_request(source)
        )


def test_original_source_and_archive_negation_reach_memory_model_unchanged():
    source = original()
    messages = build_memory_llm_messages(source, (), (), records(), 10000, 0.8)
    payload = json.loads(messages[-1]["content"])
    assert payload["current_user_message"] == source
    assert "不要求删除聊天历史" in payload["current_user_message"]
    assert payload["existing_memories"][0]["memory_id"] == "7"
    assert selected_ids(
        {"erase_source_ids": ["owner-source"]}, {"owner-source"}, authorized=is_memory_erasure_request(source)
    ) == ("owner-source",)


def test_complete_archive_retention_request_cannot_bypass_sensitive_source_gate():
    source = original() + "我的密码是ExampleSecret123456。"
    assert not parse_llm_proposals(proposal(source), source_message=source, existing_memories=records())


@pytest.mark.parametrize(
    "reply, expected",
    [
        ("目前还没有执行，不能声称已经删掉或一定会成功。", False),
        ("交付确认后才处理。我已经删除这两项记忆。", True),
        ("不得声称已删除。", False),
        ("我无法确认已经删除。", False),
        ("不能声称已经删掉，但我已经删除两项记忆。", True),
        ("我已经删掉，后面不会再删除。", True),
        ("删除已经完成。", True),
        ("已彻底清除你的资料。", True),
    ],
)
def test_completed_claim_audit_distinguishes_negation_from_actual_assertion(reply, expected):
    from evaluation.memory_operation_pending_probe import _claims_completed_erasure

    assert _claims_completed_erasure(reply) is expected
