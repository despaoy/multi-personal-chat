import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig
from character.models import UserScope
from db.database import SQLiteDB
from repositories.character_memory import DatabaseCharacterMemoryRepository

SCOPE = UserScope("web", "capture-test", "alice", "room", "private")
FIELDS = dict(character_id="role", platform="web", adapter="capture-test", sender_id="alice",
              conversation_type="private", conversation_id="room")


class Completion:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    async def complete(self, messages):
        self.calls += 1
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    async def close(self):
        pass


async def submit(repo, completion, *, message, source_id="source", observed=None, source_type="user"):
    scheduler = MemoryEnrichmentScheduler(
        config=MemoryLlmConfig(enabled=True, base_url="http://unused", model="stub"), completion=completion)
    admitted = scheduler.schedule(repository=repo, character_id="role", user_scope=SCOPE,
                                   message=message, rule_hints=[], source_message_id=source_id,
                                   observed_at=observed, source_type=source_type)
    await scheduler.shutdown(timeout=3)
    return admitted, scheduler.status


@pytest.mark.asyncio
@pytest.mark.parametrize("response", ['{"memories":[]}', 'not valid JSON', RuntimeError("model unavailable"),
    json.dumps({"memories": [{"operation": "SUPERSEDE", "target_memory_id": "user_workplace",
                              "kind": "workplace", "value": "图书馆", "confidence": .95,
                              "evidence": "去年我在图书馆工作，后来离职了。"}]}, ensure_ascii=False)])
async def test_source_survives_empty_invalid_and_failed_interpretation(tmp_path, response):
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / "capture.sqlite"))
    source = "去年我在图书馆工作，后来离职了。"
    at = datetime(2026, 1, 2, tzinfo=timezone.utc)
    completion = Completion(response)
    admitted, status = await submit(repo, completion, message=source, observed=at)
    assert admitted and completion.calls == 1
    assert await repo.list_memory_records("role", SCOPE) == []
    rows = await repo.list_sources("role", SCOPE)
    assert rows == [dict(source_message_id="source", body=source, observed_at=at.isoformat(timespec="microseconds"))]
    assert status.recent_results[-1]["source_capture"] == "recorded"


@pytest.mark.asyncio
async def test_complete_source_is_not_truncated_to_model_input_limit(tmp_path):
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / "long.sqlite"))
    source = "我今天搬家了。" + "这里是在转述小说。" * 600 + "以上不是我的真实经历。"
    admitted, _ = await submit(repo, Completion('{"memories":[]}'), message=source)
    assert admitted
    assert (await repo.list_sources("role", SCOPE))[0]["body"] == source


@pytest.mark.asyncio
@pytest.mark.parametrize("message,source_type", [("不要记住，我喜欢烘焙。", "user"),
    ("我喜欢烘焙。", "assistant"), ("我喜欢烘焙。", "tool")])
async def test_existing_source_admission_gates_apply_before_capture(tmp_path, message, source_type):
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / "gate.sqlite"))
    completion = Completion('{"memories":[]}')
    admitted, _ = await submit(repo, completion, message=message, source_type=source_type)
    assert not admitted and completion.calls == 0
    assert await repo.list_sources("role", SCOPE) == []


@pytest.mark.asyncio
async def test_erase_request_itself_does_not_enter_source_pool(tmp_path):
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / "erase.sqlite"))
    admitted, status = await submit(repo, Completion('{"memories":[]}'), message="忘掉我的住址。")
    assert admitted
    assert status.recent_results[-1]["source_capture"] == "erase_request"
    assert await repo.list_sources("role", SCOPE) == []


@pytest.mark.asyncio
async def test_queued_old_work_skips_model_and_claim_mutation(tmp_path):
    db = SQLiteDB(tmp_path / "queued.sqlite")
    repo = DatabaseCharacterMemoryRepository(db)
    before = datetime.now(timezone.utc) - timedelta(seconds=2)
    db.clear_character_memories(**FIELDS)
    completion = Completion('{"memories":[]}')
    admitted, status = await submit(repo, completion, message="我目前在外地轮岗。", observed=before)
    assert admitted and completion.calls == 0
    assert status.recent_results[-1]["reason"] == "source_stale"
    assert await repo.list_sources("role", SCOPE) == []


@pytest.mark.parametrize("legacy", [False, True])
def test_unlinked_repeat_and_other_claims_cannot_restore_erased_raw_information(tmp_path, legacy):
    db = SQLiteDB(tmp_path / "unlinked.sqlite")
    at = datetime.now(timezone.utc) - timedelta(seconds=1)
    for source in ("first", "empty-output", "only-name"):
        db.capture_memory_source(**FIELDS, source_message_id=source, body="我叫阿黎，住在海棠路。", observed_at=at)
    writer = db.add_or_update_character_memory if legacy else db.append_character_memory_claim
    target = writer(**FIELDS, memory_type="user_fact", memory_key="address", content="住海棠路",
                    source_message_id="first")
    sibling = db.append_character_memory_claim(**FIELDS, memory_type="user_fact", memory_key="name",
                                               content="叫阿黎", source_message_id="only-name")
    assert db.erase_character_memories(**FIELDS, memory_id=target["id"]) == 1
    assert db.list_memory_sources(**FIELDS) == []
    assert db.list_memory_sources(**FIELDS, source_message_ids=["empty-output", "only-name"]) == []
    assert [row["id"] for row in db.list_character_memory_claims(**FIELDS)] == [sibling["id"]]


@pytest.mark.asyncio
async def test_inflight_model_cannot_restore_claim_after_clear(tmp_path):
    db = SQLiteDB(tmp_path / "inflight.sqlite")
    repo = DatabaseCharacterMemoryRepository(db)
    started, resume = asyncio.Event(), asyncio.Event()

    class Delayed(Completion):
        async def complete(self, messages):
            started.set()
            await resume.wait()
            return json.dumps({"memories": [{"operation": "ADD", "kind": "name", "value": "阿黎", "evidence": "我叫阿黎",
                                               "confidence": .96}]}, ensure_ascii=False)

    pending = asyncio.create_task(submit(repo, Delayed(""), message="我叫阿黎",
                                         observed=datetime.now(timezone.utc) - timedelta(seconds=1)))
    await asyncio.wait_for(started.wait(), 3)
    try:
        await asyncio.to_thread(db.clear_character_memories, **FIELDS)
    finally:
        resume.set()
    _, status = await pending
    assert await repo.list_memory_records("role", SCOPE) == []
    assert await repo.list_sources("role", SCOPE) == []
    assert status.recent_results[-1]["status"] == "skipped"
