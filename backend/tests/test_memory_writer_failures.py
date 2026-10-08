"""Writer preparation fails before semantic mutations, with explicit receipts."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from repositories.character_memory import DatabaseCharacterMemoryRepository

from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig, _search_existing_memories
from character.models import MemoryItem, UserScope
from character.source_erasure_selection import candidates
from db.database import SQLiteDB


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), 1e30, 0.0, "empty", "wrong_rows"])
def test_invalid_writer_vectors_cannot_fall_back_to_lexical_or_perfect_scores(bad):
    def encode(texts):
        if bad == "empty":
            return np.zeros((len(texts), 0), dtype=np.float32)
        if bad == "wrong_rows":
            return np.ones((len(texts) + 1, 2), dtype=np.float32)
        return np.full((len(texts), 2), bad, dtype=np.float32)
    with pytest.raises(ValueError):
        _search_existing_memories((dict(id="1", content="用户喜欢咖啡"),), "请记住我喜欢咖啡", (), (),
                                  SimpleNamespace(embed_texts=encode))


def test_empty_active_collection_does_not_require_embedding():
    provider = SimpleNamespace(embed_texts=Mock(side_effect=RuntimeError("must not run")))
    assert _search_existing_memories((dict(id="1", status="erased", content="旧偏好"),), "请记住我喜欢茶", (), (), provider) == ()
    provider.embed_texts.assert_not_called()


async def test_missing_source_reader_cannot_report_empty_coverage():
    with pytest.raises(AttributeError, match="search_sources"):
        await candidates(object(), "role", UserScope("web", "test", "a", "a", "private"), "删除", (), context_window_tokens=8192)


@pytest.mark.parametrize("stage", ["memory_read", "memory_search", "source_candidates"])
async def test_failed_preparation_never_calls_model_or_mutates_claims(tmp_path, monkeypatch, caplog, stage):
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / "writer.sqlite"))
    scope = UserScope("web", "strict-writer", "fiction", "fiction", "private")
    await repo.append_claim("role", scope, MemoryItem("", "user_fact", "用户喜欢咖啡", .9),
        memory_key="preference_coffee", confidence=.99, evidence=("我喜欢咖啡。",), source_message_id="old")
    original_read = repo.list_memory_records
    before = await original_read("role", scope)
    private = "synthetic-private-provider-detail"
    failure = OSError(private)
    provider = SimpleNamespace(embed_texts=Mock(side_effect=lambda texts: np.ones((len(texts), 2), dtype=np.float32)))
    if stage == "memory_read":
        monkeypatch.setattr(repo, "list_memory_records", AsyncMock(side_effect=failure))
    elif stage == "memory_search":
        provider.embed_texts.side_effect = failure
    else:
        monkeypatch.setattr(repo, "search_sources", AsyncMock(side_effect=failure))
    completion = SimpleNamespace(complete=AsyncMock(), close=AsyncMock())
    worker = MemoryEnrichmentScheduler(config=MemoryLlmConfig(True, "unused", "fixture"),
        completion=completion, embedding_provider=provider)
    message = "请从记忆里删除我喜欢咖啡的记录。" if stage == "source_candidates" else "请记住，我现在喜欢红茶。"
    try:
        result = await worker.schedule_and_wait(repository=repo, character_id="role", user_scope=scope,
            message=message, rule_hints=(), source_message_id="new")
        assert result["status"] == "failed" and result["stage"] == stage
        assert result["persisted"] == result["accepted"] == 0
        assert result["error"] == "OSError" and worker.status.failed == 1
        completion.complete.assert_not_awaited()
        assert await original_read("role", scope) == before
        assert private not in json.dumps(result) and private not in caplog.text
        assert private not in worker.status.last_error
        if stage != "source_candidates":
            assert result["source_capture"] == "recorded"
            assert (await repo.list_sources("role", scope, source_message_ids=("new",)))[0]["body"] == message
    finally:
        await worker.shutdown(timeout=1)
