"""Successive writes need scoped prior versions without new mutation authority."""

import asyncio
import json
from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest
from test_writer_existing_context import CURRENT, QUALIFIERS, SOURCE
from test_writer_existing_context import stored as stored

from character.memory_llm import (
    MemoryEnrichmentScheduler,
    MemoryLlmConfig,
    MemoryResponseError,
    build_memory_llm_messages,
    parse_llm_proposals,
)
from character.models import MemoryItem
from inference.context_budget import CONTEXT_SAFETY_MARGIN_TOKENS

NEXT = "请记住：出差的周日早晨也喝咖啡，只增加这个例外，其余日常限制不变。"
LATEST = "请记住：出差的周一晚上也喝咖啡，只增加这一项例外，其余限制不变。"


class SyntheticEmbedding:
    def embed_texts(self, texts):
        return np.ones((len(texts), 4), dtype=np.float32)


class Completion:
    def __init__(self, inspect=None):
        self.requests = []
        self.inspect = inspect

    async def complete(self, messages):
        self.requests.append(messages)
        if self.inspect:
            self.inspect(json.loads(messages[1]["content"]))
        return '{"memories":[]}'

    async def close(self):
        pass


@pytest.fixture
async def versions(stored):
    repository, scope, records, observed = stored
    (old,) = records
    previous = old
    for index, source in enumerate([CURRENT, NEXT], start=1):
        sid = f"supplement-{index}"
        stamp = observed + timedelta(hours=index)
        assert (
            await repository.capture_source("role", scope, source_message_id=sid, body=source, observed_at=stamp)
            == "recorded"
        )
        item = MemoryItem("", "user_fact", f"用户补充咖啡例外{index}", evidence=(source,), source_message_ids=(sid,))
        result = await repository.append_claim(
            "role",
            scope,
            item,
            memory_key=old["memory_key"],
            relation_type="MERGE",
            parent_memory_id=previous["id"],
            supersedes_memory_id=previous["id"],
            evidence=(source,),
            source_message_id=sid,
            source_message_ids=(sid,),
            observed_at=stamp.isoformat(),
            metadata={"fact_kind": "other_user_fact", "qualifiers": {"condition": source}},
        )
        assert result["persisted"] is True
        previous = result
    rows = await repository.list_memory_records("role", scope, limit=None, include_inactive=True)
    assert len(rows) == 3
    return repository, scope, rows, observed + timedelta(hours=3)


async def run(repository, scope, observed, monkeypatch, completion, window=65536):
    monkeypatch.setattr("character.memory_llm.text_token_counter", lambda *_: len)
    scheduler = MemoryEnrichmentScheduler(
        config=MemoryLlmConfig(enabled=True, base_url="http://127.0.0.1:1", model="test", context_window_tokens=window),
        completion=completion,
        embedding_provider=SyntheticEmbedding(),
    )
    try:
        receipt = asyncio.get_running_loop().create_future()
        assert scheduler.schedule(
            repository=repository,
            character_id="role",
            user_scope=scope,
            message=LATEST,
            rule_hints=[],
            source_message_id="last-supplement",
            immediate=True,
            observed_at=observed,
            receipt=receipt,
        )
        return await asyncio.wait_for(receipt, timeout=3)
    finally:
        await scheduler.shutdown(timeout=3)


async def test_successive_merge_keeps_transitive_conditions_without_old_target_authority(versions, monkeypatch):
    repository, scope, rows, observed = versions
    active = tuple(row for row in rows if row["status"] == "active")
    old = min(rows, key=lambda row: row["id"])
    middle = next(row for row in rows if row["id"] != old["id"] and row["status"] == "superseded")

    def inspect(payload):
        (current,) = payload["existing_memories"]
        assert current["memory_id"] == str(active[0]["id"])
        assert current["parent_memory_id"] == middle["id"]
        assert current["fact_kind"] == "other_user_fact"
        history = {row["memory_id"]: row for row in payload["memory_version_context"]}
        assert set(history) == {str(old["id"]), str(middle["id"])}
        assert history[str(old["id"])]["evidence"] == [SOURCE]
        assert history[str(old["id"])]["qualifiers"] == QUALIFIERS
        assert history[str(middle["id"])]["evidence"] == [CURRENT]
        assert all(row["status"] == "superseded" for row in history.values())
        assert all(row["fact_kind"] == "other_user_fact" for row in history.values())
        assert all(row["was_injected_in_last_reply"] is False for row in history.values())
        proposal = json.dumps(
            {
                "memories": [
                    {
                        "operation": "ERASE",
                        "target_memory_id": str(old["id"]),
                        "target_memory_key": old["memory_key"],
                        "evidence": LATEST,
                        "confidence": 0.99,
                        "attributed_to": "user",
                    }
                ]
            }
        )
        with pytest.raises(MemoryResponseError, match="白名单"):
            parse_llm_proposals(proposal, source_message=LATEST, existing_memories=active, confidence_threshold=0.85)

    completion = Completion(inspect)
    result = await run(repository, scope, observed, monkeypatch, completion)
    assert result["status"] == "no_change" and result["accepted"] == 0
    assert len(completion.requests) == 1
    assert await repository.list_memory_records("role", scope, limit=None, include_inactive=True) == rows


async def test_complete_history_must_fit_before_provider_request(versions, monkeypatch):
    repository, scope, rows, observed = versions
    active = tuple(row for row in rows if row["status"] == "active")
    request = build_memory_llm_messages(
        LATEST, (), (), active, 2000, 0.85, observed_at=observed, context_window_tokens=65536, count_tokens=len
    )
    window = sum(len(item["content"]) + 4 for item in request) + 768 + CONTEXT_SAFETY_MARGIN_TOKENS + 64
    completion = Completion()
    result = await run(repository, scope, observed, monkeypatch, completion, window)
    assert result["status"] == "failed" and result["error"] == "InputBudgetError"
    assert result["stage"] == "proposal_generation" and completion.requests == []
    assert await repository.list_memory_records("role", scope, limit=None, include_inactive=True) == rows


async def test_inaccessible_parent_is_an_error_without_foreign_disclosure(stored, monkeypatch):
    repository, scope, _, observed = stored
    foreign = replace(scope, sender_id="other-owner")
    secret = "另一用户只在其私有条件下喝咖啡。"
    assert (
        await repository.capture_source(
            "role", foreign, source_message_id="foreign-source", body=secret, observed_at=observed
        )
        == "recorded"
    )
    old = await repository.append_claim(
        "role",
        foreign,
        MemoryItem("", "user_fact", secret),
        memory_key="fact_private",
        evidence=(secret,),
        source_message_id="foreign-source",
        source_message_ids=("foreign-source",),
        observed_at=observed.isoformat(),
    )
    # An existing link cannot grant read access to a different user's record.
    await repository.append_claim(
        "role",
        scope,
        MemoryItem("", "user_fact", "用户补充咖啡例外"),
        memory_key="fact_visible",
        relation_type="COEXIST",
        parent_memory_id=old["id"],
        observed_at=observed.isoformat(),
    )
    rows = await repository.list_memory_records("role", scope, limit=None, include_inactive=True)
    foreign_rows = await repository.list_memory_records("role", foreign, limit=None, include_inactive=True)
    completion = Completion()
    result = await run(repository, scope, observed + timedelta(hours=1), monkeypatch, completion)
    assert result["status"] == "failed" and result["stage"] == "memory_history"
    assert result["error"] == "RuntimeError" and completion.requests == []
    assert secret not in json.dumps(result, ensure_ascii=False)
    assert await repository.list_memory_records("role", scope, limit=None, include_inactive=True) == rows
    assert await repository.list_memory_records("role", foreign, limit=None, include_inactive=True) == foreign_rows
