"""Account consent, scheduling and idempotent in-app delivery."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.character_chat import repository, router
from app.dependencies import get_current_user
from character.library import CharacterSettings
from db.database import SQLiteDB
from db.errors import RevisionConflictError
from repositories.character_chat_proactive import CharacterChatProactiveRepository
from services.character_chat import CharacterChatService


@pytest.fixture
def proactive(tmp_path):
    db = SQLiteDB(tmp_path / "proactive.db")
    repo = CharacterChatProactiveRepository(db)
    character = repo.chats.characters.create("owner", CharacterSettings(name="旅人", persona="坦率", greeting="你好"))
    session = repo.chats.create("owner", character["id"], "新会话")
    try:
        yield repo, session
    finally:
        db.close_connection()


def policy(**values):
    return {
        "enabled": True, "cadence_hours": 24, "timezone": "Asia/Shanghai", "quiet_start": 22 * 60, "quiet_end": 8 * 60,
        **values,
    }


def test_app_delivery_respects_quiet_hours_frequency_and_is_replay_safe(proactive):
    repo, session = proactive
    identity = session["id"]
    assert repo.view("owner", identity)["policy"]["enabled"] is False
    repo.set_policy("owner", identity, 0, **policy())
    midnight = datetime(2026, 11, 20, 16, tzinfo=UTC).timestamp()
    morning = midnight + 9 * 3600
    due = midnight + 60
    args = ("owner", identity, 1, "first", "event", "面试", due)
    assert repo.schedule(*args, now=midnight)["status"] == "pending"
    assert repo.schedule(*args, now=midnight)["id"] == "first"
    with pytest.raises(ValueError, match="已安排"):
        repo.schedule("owner", identity, 1, "duplicate", "event", "面试", due, now=midnight)
    repo.schedule("owner", identity, 1, "second", "fiction", "窗外的雪", due, now=midnight)
    assert repo.deliver_due(now=due) == 0
    with ThreadPoolExecutor(2) as pool:
        assert sum(pool.map(lambda _: repo.deliver_due(now=morning), range(2))) == 1
    assert repo.deliver_due(now=morning + 60) == 0
    assert repo.deliver_due(now=morning + 86400) == 1
    assert repo.deliver_due(now=morning + 172800) == 0
    messages = {job["id"]: job for job in repo.view("owner", identity)["jobs"]}
    assert "面试" in messages["first"]["content"]
    assert messages["second"]["content"].startswith("【角色虚构日常】")
    assert messages["first"]["status"] == messages["second"]["status"] == "sent"


def test_revoke_blocks_inflight_old_job_followed_by_reenable(proactive, monkeypatch):
    repo, session = proactive
    identity = session["id"]
    repo.set_policy("owner", identity, 0, **policy(quiet_start=None, quiet_end=None))
    repo.schedule("owner", identity, 1, "old", "follow_up", "旧话题", 101, now=100)
    guard = repo._session_guard
    other = CharacterChatProactiveRepository(repo.db)

    def revoke_after_read(value):
        other.set_policy("owner", identity, 1, **policy(enabled=False))
        return guard(value)

    monkeypatch.setattr(repo, "_session_guard", revoke_after_read)
    assert repo.deliver_due(now=102) == 0
    monkeypatch.setattr(repo, "_session_guard", guard)
    repo.set_policy("owner", identity, 2, **policy())
    assert repo.view("owner", identity)["jobs"] == []
    assert repo.deliver_due(now=103) == 0
    with pytest.raises(RevisionConflictError):
        repo.schedule("owner", identity, 1, "old", "follow_up", "旧话题", 202, now=200)
    repo.chats.delete_sessions("owner", [{"id": identity, "revision": session["revision"]}])
    assert not repo.db.execute_sql("SELECT * FROM character_chat_proactive_jobs")
    assert not repo.db.execute_sql("SELECT * FROM character_chat_proactive_policies")


def test_proactive_api_owner_and_policy_bounds(proactive):
    repo, session = proactive
    owner = "owner"
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user] = lambda: {"user_id": owner}
    app.dependency_overrides[repository] = lambda: repo.chats
    path = f"/api/character-chats/{session['id']}/proactive"
    with TestClient(app) as client:
        assert client.get(path).json()["policy"]["enabled"] is False
        assert client.put(path, json={"revision": 0, **policy(timezone="Invalid/Zone")}).status_code == 422
        assert client.put(path, json={"revision": 0, **policy(quiet_end=None)}).status_code == 422
        assert client.put(path, json={"revision": 0, **policy(cadence_hours=1)}).status_code == 422
        assert client.put(path, json={"revision": 0, **policy()}).json()["revision"] == 1
        assert client.put(path, json={"revision": 0, **policy(enabled=False)}).status_code == 409
        owner = "other"
        assert client.get(path).status_code == 404
        assert client.put(path, json={"revision": 1, **policy(enabled=False)}).status_code == 404
        payload = {"revision": 1, "request_id": "foreign", "kind": "event", "topic": "失败", "due_at": 2000000000}
        assert client.post(path + "/jobs", json=payload).status_code == 404


def test_outreach_context_is_available_only_to_original_ordinary_session(proactive):
    repo, session = proactive
    identity = session["id"]
    repo.set_policy("owner", identity, 0, **policy(quiet_start=None, quiet_end=None))
    repo.schedule("owner", identity, 1, "event", "event", "PRIVATE-OUTREACH", 101, now=100)
    assert repo.deliver_due(now=101) == 1
    assert repo.recent_sent("other", identity) == []

    async def complete(request, target, request_id):
        assert ("PRIVATE-OUTREACH" in request.character_context.conversation_reference_context) == (target["id"] == identity)
        return "会话回复", "test"

    client = CharacterChatService(repo.db, complete, proactive_history=repo.recent_sent)
    turn = asyncio.run(client.generate("owner", identity, revision=1, request_id="answer", message="继续聊"))
    branch = repo.chats.create("owner", session["character_id"], "假想分支", source_session_id=identity,
                               source_revision=2, source_turn_id=turn["id"])
    asyncio.run(client.generate("owner", branch["id"], revision=1, request_id="branch-answer", message="继续聊"))
