"""Tests for asynchronous LLM-assisted character memory."""

import json

import pytest

from character.memory_extractor import ExtractedMemory
from character.memory_llm import (
    MemoryEnrichmentScheduler,
    MemoryLlmConfig,
    _search_existing_memories,
    parse_llm_memories,
)
from character.models import UserScope


def _response(*memories):
    return json.dumps({"memories": list(memories)}, ensure_ascii=False)


def test_llm_candidate_requires_exact_evidence_and_builds_local_content():
    result = parse_llm_memories(
        _response(
            {
                "kind": "goal",
                "value": "保研面试",
                "evidence": "最近主要在准备保研面试",
                "confidence": 0.94,
            }
        ),
        source_message="最近主要在准备保研面试，所以回复可能慢一点。",
    )

    assert len(result) == 1
    assert result[0].memory_key == "goal_保研面试"
    assert result[0].content == "用户正在进行或准备：保研面试"


def test_llm_candidate_cannot_invent_evidence_or_value():
    invented_evidence = parse_llm_memories(
        _response(
            {
                "kind": "like",
                "value": "咖啡",
                "evidence": "我喜欢咖啡",
                "confidence": 0.99,
            }
        ),
        source_message="我喜欢红茶。",
    )
    invented_value = parse_llm_memories(
        _response(
            {
                "kind": "like",
                "value": "咖啡",
                "evidence": "我喜欢红茶",
                "confidence": 0.99,
            }
        ),
        source_message="我喜欢红茶。",
    )

    assert invented_evidence == []
    assert invented_value == []


def test_llm_path_respects_opt_out_sensitive_and_uncertain_policy():
    candidate = _response(
        {
            "kind": "like",
            "value": "咖啡",
            "evidence": "我喜欢咖啡",
            "confidence": 0.99,
        }
    )

    assert parse_llm_memories(candidate, source_message="不要记住我喜欢咖啡") == []
    assert parse_llm_memories(candidate, source_message="我的支付密码是123456") == []
    uncertain = _response(
        {
            "kind": "like",
            "value": "咖啡",
            "evidence": "我可能喜欢咖啡",
            "confidence": 0.99,
        }
    )
    assert parse_llm_memories(uncertain, source_message="我可能喜欢咖啡") == []


class _Completion:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = []
        self.closed = False

    async def complete(self, messages):
        self.calls.append(messages)
        return self.response

    async def close(self):
        self.closed = True


class _Repository:
    def __init__(self) -> None:
        self.writes = []

    async def list_memory_records(self, character_id, user_scope, limit=30):
        return []

    async def add_or_update_memory(
        self,
        character_id,
        user_scope,
        memory,
        *,
        memory_key,
        source_message_id=None,
    ):
        self.writes.append((character_id, user_scope, memory, memory_key, source_message_id))


def test_write_search_recalls_old_fact_beyond_recent_window():
    records = tuple(
        {"id": str(index), "content": "用户参加了周末的登山活动", "memory_key": f"event_{index}"}
        for index in range(600)
    ) + ({"id": "old", "content": "用户喜欢咖啡", "memory_key": "preference_咖啡"},)
    selected = _search_existing_memories(records, "我现在不喝咖啡了", (), ())
    assert [record["id"] for record in selected] == ["old"]


def test_write_search_uses_key_for_changed_value_and_keeps_feedback_first():
    records = (
        {"id": "name", "content": "小明", "memory_key": "user_name"},
        {"id": "feedback", "content": "用户喜欢红茶", "memory_key": "preference_红茶"},
        {"id": "inactive", "content": "用户叫小宇", "status": "superseded"},
    )
    hints = (ExtractedMemory("user_fact", "user_name", "用户叫小宇", 0.9),)
    selected = _search_existing_memories(records, "以后我叫小宇", hints, ("feedback",))
    assert [record["id"] for record in selected] == ["feedback", "name"]


def test_write_search_limits_topk_and_does_not_fill_with_unrelated_records():
    records = tuple({"id": str(index), "content": "用户喜欢咖啡"} for index in range(30))
    assert len(_search_existing_memories(records, "咖啡", (), ())) == 10
    assert _search_existing_memories(records, "明天去看展", (), ()) == ()


@pytest.mark.asyncio
async def test_scheduler_passes_old_relevant_memory_to_llm():
    class Repository(_Repository):
        async def list_memory_records(self, character_id, user_scope, limit=30):
            records = [{"id": str(index), "content": "用户参加周末登山活动"} for index in range(600)] + [
                {"id": "old", "content": "用户喜欢咖啡", "memory_key": "preference_咖啡"}
            ]
            return records[:limit]

    completion = _Completion(_response())
    scheduler = MemoryEnrichmentScheduler(
        config=MemoryLlmConfig(enabled=True, base_url="http://127.0.0.1", model="test"),
        completion=completion,
    )
    scheduler.schedule(
        repository=Repository(),
        character_id="kisaki",
        user_scope=UserScope("qq", "astrbot", "user-1", "user-1", "private"),
        message="请记住我现在不喝咖啡了",
        rule_hints=[],
    )
    await scheduler.shutdown(timeout=2.0)
    assert len(completion.calls) == 1
    payload = json.loads(completion.calls[0][1]["content"])
    assert [record["memory_id"] for record in payload["existing_memories"]] == ["old"]


@pytest.mark.asyncio
async def test_scheduler_processes_memory_after_enqueue_and_closes_cleanly():
    completion = _Completion(
        _response(
            {
                "kind": "study_stage",
                "value": "大三",
                "evidence": "今年刚升大三",
                "confidence": 0.96,
            }
        )
    )
    repository = _Repository()
    scheduler = MemoryEnrichmentScheduler(
        config=MemoryLlmConfig(
            enabled=True,
            base_url="http://127.0.0.1:8001",
            model="test-model",
            queue_size=4,
        ),
        completion=completion,
    )
    scope = UserScope(
        platform="qq",
        adapter="astrbot",
        sender_id="user-1",
        conversation_id="user-1",
        conversation_type="private",
    )

    scheduled = scheduler.schedule(
        repository=repository,
        character_id="kisaki",
        user_scope=scope,
        message="今年刚升大三",
        rule_hints=[],
        source_message_id="msg-1",
    )
    await scheduler.shutdown(timeout=1.0)

    assert scheduled is True
    assert len(completion.calls) == 1
    assert completion.closed is True
    assert len(repository.writes) == 1
    assert repository.writes[0][3] == "user_study_stage"
    assert repository.writes[0][4] == "msg-1"
