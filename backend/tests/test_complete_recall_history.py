"""Complete available topic context before memory ranking, no assistant promotion."""

from datetime import datetime, timezone

import pytest

from character.memory_service import CharacterMemoryService
from character.models import UserScope
from services.character_context import compile_user_recall_context


def test_head_and_tail_of_long_user_message_are_preserved():
    text = "榆桐项目" + ("完整无关背景。" * 400) + "后置离线限定"
    assert compile_user_recall_context([{"role": "user", "content": text}]) == text


def test_available_topic_older_than_six_messages_is_not_dropped():
    rows = [{"role": "user", "content": "当前主题是榆桐项目"}] + [
        {"role": "user", "content": "完整气象记录" + str(i)} for i in range(8)
    ]
    assert compile_user_recall_context(rows) == "\n".join(r["content"] for r in rows)


def test_assistant_guesses_and_system_messages_do_not_expand_recall():
    rows = [
        {"role": "user", "content": "榆桐项目"},
        {"role": "assistant", "content": "猜测24GB和云端API"},
        {"role": "system", "content": "其他指令"},
    ]
    assert compile_user_recall_context(rows) == "榆桐项目"


def test_missing_or_nontext_user_content_is_ignored():
    rows = [
        {"role": "user"},
        {"role": "user", "content": None},
        {"role": "user", "content": 7},
        {"role": "user", "content": "完整用户原话"},
    ]
    assert compile_user_recall_context(rows) == "完整用户原话"


@pytest.mark.asyncio
async def test_full_context_reaches_semantic_and_lexical_recall(monkeypatch):
    class Repository:
        async def list_memory_records(self, *args, **kwargs):
            return [
                {
                    "id": 1,
                    "memory_type": "user_fact",
                    "memory_key": "project",
                    "content": "我的榆桐项目只允许离线推理。",
                    "confidence": 1,
                    "importance": 0.5,
                    "status": "active",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "metadata": {},
                }
            ]

    service = CharacterMemoryService(Repository(), semantic_enabled=True)
    seen = []

    def scores(query, records):
        seen.append(query)
        return {0: 1.0}

    monkeypatch.setattr(service, "_semantic_similarities", scores)
    context = "榆桐项目" + ("完整背景。" * 400) + "末尾只允许离线"
    items, count, trace = await service.recall_with_diagnostics(
        "role",
        UserScope("web", "test", "user", "room", "private"),
        "那接下来如何？",
        for_contextual_selection=True,
        retrieval_context=context,
    )
    assert len(seen) == 1 and seen[0].endswith(context) and "榆桐项目" in seen[0] and items
