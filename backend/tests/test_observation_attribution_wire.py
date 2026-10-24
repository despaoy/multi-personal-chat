"""Authorization/speaker identity cannot establish a fact's subject.

Default factory, actual profile loading, SQLite source/claim authority,
selection parser, service, API and final message compilation are real.
Only selector labels/final answers are synthetic; no paid model calls.
"""

import json
import re
from datetime import datetime, timedelta, timezone
from html import unescape

import pytest
import test_character_provider_budget as provider_fixture
from repositories.character_memory import DatabaseCharacterMemoryRepository

from api import generate
from character.models import MemoryItem
from db.schemas import MessageRequest
from inference import generation_request
from inference.request_context import get_request_context

SISTER = (
    "我妹妹计划本周四14:00至15:00排练，尚未报名；"
    "只有取得书面确认且当天不下雨才参加，预约不豁免雨天限制；这不是我本人的排练安排。"
)
SELF = (
    "我本人不喜欢薄荷水；仅周日12:30至13:00、吃过午饭且没有牙痛时喝温水，"
    "牙痛时不喝，预约不豁免这些限制。这是我本人的饮水偏好，不是我妹妹的偏好。"
)
QUERIES = {
    "observation": "核对我保存的妹妹排练引文，说明描述对象、时间、否定与完整条件；不要把她的经历归给我，不推断已经报名。",
    "mixed": "分别核对我保存的妹妹排练引文和本人饮水偏好，保留各自描述对象、时间、否定及全部条件；不要混淆两人的经历。",
    "self": "核对我保存的本人饮水偏好，说明时间、否定与全部条件；保留本人归属，不把它说成角色或妹妹的经历。",
}
REPLIES = {
    "observation": "记录描述的是你妹妹：周四14:00至15:00计划排练，尚未报名；只在获书面确认且不下雨时参加，预约不豁免雨天限制。这不是你的排练安排。",
    "mixed": "你妹妹周四14:00至15:00计划排练，尚未报名，须获书面确认且不下雨。你本人不喜欢薄荷水，仅周日12:30至13:00午饭后且不牙痛时喝温水，牙痛时不喝。两人的条件都不因预约豁免。",
    "self": "你本人不喜欢薄荷水；仅周日12:30至13:00午饭后且没有牙痛时喝温水，牙痛时不喝，预约不豁免这些条件。这是你的偏好。",
}


async def arrange_observation_attribution(tmp_path, monkeypatch, mode):
    if mode not in QUERIES:
        raise ValueError("Unknown attribution case")
    monkeypatch.setattr(provider_fixture, "BODY", SISTER)
    monkeypatch.setattr(provider_fixture, "QUERY", QUERIES[mode])
    rig = await provider_fixture.arrange_provider_budget(tmp_path, monkeypatch)
    rig.query, rig.mode = QUERIES[mode], mode
    repository = DatabaseCharacterMemoryRepository(rig.database)
    original, = await repository.list_memory_records(provider_fixture.ROLE, rig.scope, limit=None)
    rig.observation_id = str(original["id"])
    rig.self_id = None
    if mode in {"mixed", "self"}:
        stamp = datetime.now(timezone.utc) - timedelta(days=1)
        assert await repository.capture_source(
            provider_fixture.ROLE, rig.scope, source_message_id="fiction-self-drink", body=SELF, observed_at=stamp,
        ) == "recorded"
        stored = await repository.append_claim(
            provider_fixture.ROLE, rig.scope,
            MemoryItem("", "user_fact", "用户本人的完整饮水偏好：" + SELF, .9),
            memory_key="preference_温水", confidence=.99, evidence=(SELF,),
            source_message_id="fiction-self-drink", observed_at=stamp.isoformat(), metadata={"origin": "rule_v2"},
        )
        rig.self_id = str(stored["id"])
    if mode == "self":
        assert await repository.erase_memory(provider_fixture.ROLE, rig.scope, memory_id=int(rig.observation_id)) == 1
    rig.expected = ({rig.observation_id: SISTER} if mode == "observation" else
                    {rig.self_id: SELF} if mode == "self" else
                    {rig.observation_id: SISTER, rig.self_id: SELF})

    async def completion(messages, **params):
        try:
            payload = json.loads(messages[-1]["content"])
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and "required_ids" in payload:
            assert payload["query"] == rig.query and set(payload["required_ids"]) == set(rig.expected)
            assert len(payload["candidates"]) == len(rig.expected)
            for candidate in payload["candidates"]:
                assert candidate["evidence"] == [rig.expected[candidate["id"]]]
                assert candidate["content_complete"] is True
                assert candidate["source_observation"] is (candidate["id"] == rig.observation_id)
            rig.reviews.append(payload)
            return json.dumps({"decisions": [{"id": identity, "label": "use"} for identity in payload["required_ids"]]}), 0.0
        visible = unescape("\n".join(message["content"] for message in messages))
        assert rig.query in visible and all(body in visible for body in rig.expected.values())
        rig.answers.append({"messages": [dict(message) for message in messages], "context": get_request_context()})
        return REPLIES[mode], 0.0

    rig.manager._current_provider = provider_fixture.model_manager.ModelProvider.OPENAI_COMPAT
    monkeypatch.setattr(rig.manager.get_current_provider(), "async_complete", completion)
    return rig


async def final_wire(rig):
    result = await generate._generate_reply_impl(
        MessageRequest(
            message=rig.query, characterId=provider_fixture.ROLE, loraName="default", platform="web", adapter="web-character",
            senderId=provider_fixture.OWNER, userId=provider_fixture.OWNER, conversationId=provider_fixture.OWNER,
            conversationType="private", sessionId="fiction-attribution", sourceMessageId="fiction-attribution-read",
        ),
        persist_message=False, enable_rag=False, record_invocation=False,
        message_db=generate.db,
    )
    assert result.reply == REPLIES[rig.mode] and result.modelInvoked is True
    assert len(rig.reviews) == len(rig.answers) == 1
    messages = rig.answers[0]["messages"]
    system = "\n".join(message["content"] for message in messages if message["role"] == "system")
    user = unescape(messages[-1]["content"])
    contract, = re.findall(r'<memory_response_contract source="application">\n(.*?)\n</memory_response_contract>', user, re.S)
    reference, = re.findall(r'<character_memory trust="untrusted" purpose="historical_reference">\n(.*?)\n</character_memory>', user, re.S)
    packets = [json.loads(line[2:]) for line in reference.splitlines() if line.startswith("- {")]
    return system, user, contract, packets


async def baseline_preflight(rig):
    """Return scope/identity and conflict booleans; no private/raw output."""
    system, user, contract, packets = await final_wire(rig)
    return {
        "mode": rig.mode, "packet_count": len(packets), "selector_calls": len(rig.reviews), "answer_calls": len(rig.answers),
        "subject_scopes": sorted({packet["subject_scope"] for packet in packets}),
        "observation_count": sum(packet["source_observation"] is True for packet in packets),
        "old_universal_subject_assertion": "这些记忆的主体是当前对话者" in contract,
        "system_and_user_contract_equal": contract.strip() == generation_request.MEMORY_ATTRIBUTION_POLICY.strip(),
        "shared_policy_in_system": generation_request.MEMORY_ATTRIBUTION_POLICY in system,
        "complete_sources_in_user": all(body in user for body in rig.expected.values()),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["observation", "mixed", "self"])
async def test_final_wire_respects_each_memory_subject_without_promoting_owner_or_speaker(tmp_path, monkeypatch, mode):
    rig = await arrange_observation_attribution(tmp_path, monkeypatch, mode)
    try:
        system, user, contract, packets = await final_wire(rig)
        assert contract.strip() == generation_request.MEMORY_ATTRIBUTION_POLICY.strip()
        assert generation_request.MEMORY_ATTRIBUTION_POLICY in system
        assert "这些记忆的主体是当前对话者" not in contract
        assert "subject_scope" in contract and "not_resolved" in contract
        assert "speaker_role" in contract  # The common contract must explicitly distinguish speaker from subject.
        assert "你" in contract and "角色" in contract  # Preserve confirmed self facts' second-person attribution.
        assert all(body in user and body not in system for body in rig.expected.values())
        assert {packet["id"] for packet in packets} == set(rig.expected)
        for packet in packets:
            assert packet["evidence"] == [rig.expected[packet["id"]]]
            if packet["id"] == rig.observation_id:
                assert packet["source_observation"] is True
                assert packet["speaker_role"] == "user" and packet["subject_scope"] == "not_resolved"
                assert packet["content_semantics"] == "quoted_source" and SISTER in user
            else:
                assert packet["id"] == rig.self_id and packet["source_observation"] is False
                assert packet["subject_scope"] == "current_user_not_character" and SELF in user
    finally:
        await rig.registry.clear()
        rig.database.close_connection()
        rig.custom_db.close_connection()
