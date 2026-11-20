"""Account consent, scheduling and idempotent in-app delivery."""

import asyncio
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.character_chat import repository, router
from app.dependencies import get_current_user
from character.library import CharacterSettings
from character.proactive_candidates import outreach_candidates
from db.database import SQLiteDB
from db.errors import RevisionConflictError
from repositories.character_chat_proactive import CharacterChatProactiveRepository
from services.character_chat import CharacterChatService
from services.temporal_proactive import ChatProactiveActivities, WebPushConfig


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


def test_candidates_use_source_time_and_do_not_guess_uncertain_dates():
    created = "2026-11-20T17:00:00+00:00"
    turns = [
        {"id": "new-cancellation", "message": "考试取消了。", "created_at": created},
        {"id": "old-event", "message": "我下周有考试。", "created_at": created},
        {"id": "current", "message": "明天我有面试。下周我要答辩。我们下次再聊这份方案。", "created_at": created},
        {"id": "generic", "message": "比赛怎么报名？如果我参加演讲。", "created_at": created},
    ]
    now = datetime.fromisoformat(created).timestamp()
    candidates = outreach_candidates(turns, "Asia/Shanghai", now)
    assert [(item["kind"], item["topic"], item["suggested_at"]) for item in candidates] == [
        ("follow_up", "我们下次再聊这份方案", None),
        ("event", "下周我要答辩", None),
        ("event", "明天我有面试", datetime(2026, 11, 23, 1, tzinfo=UTC).timestamp()),
    ]


@pytest.mark.parametrize(("message", "reminder"), [
    ("我下周三参加面试", "2026-11-26T01:00:00+00:00"),
    ("本星期日我有考试", "2026-11-23T01:00:00+00:00"),
    ("下下礼拜一我有答辩", "2026-12-01T01:00:00+00:00"),
    ("12月3号我有复试", "2026-12-04T01:00:00+00:00"),
    ("2027-01-03我有演讲", "2027-01-04T01:00:00+00:00"),
    ("周三我有面试", None),
    ("下周三或周四我有面试", None),
    ("下周三至五我有面试", None),
    ("下周一、三我有面试", None),
    ("12月3号到5号我有面试", None),
    ("明年12月3日我有面试", None),
    ("本周一我有面试", None),
    ("2月30号我有面试", None),
])
def test_candidate_reminders_use_only_unambiguous_calendar_dates(message, reminder):
    created = "2026-11-20T17:00:00+00:00"  # Saturday in the configured timezone.
    now = datetime.fromisoformat(created).timestamp()
    candidate, = outreach_candidates([{"id": "source", "message": message, "created_at": created}], "Asia/Shanghai", now)
    assert candidate["topic"] == message
    assert candidate["suggested_at"] == (datetime.fromisoformat(reminder).timestamp() if reminder else None)


def test_sourced_job_requires_confirmation_and_disappears_when_source_is_retracted(proactive):
    repo, session = proactive
    owner, identity = "owner", session["id"]
    sentence = "下周三我有面试"
    turn = repo.chats.commit_turn(session, "first", sentence + "。", "我记得了。", "fixture")
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user] = lambda: {"user_id": owner}
    app.dependency_overrides[repository] = lambda: repo.chats
    path = f"/api/character-chats/{identity}/proactive"
    with TestClient(app) as client:
        assert client.get(path + "/candidates").json() == {"candidates": []}
        repo.set_policy(owner, identity, 0, **policy(quiet_start=None, quiet_end=None))
        candidate, = client.get(path + "/candidates").json()["candidates"]
        assert candidate["source_turn_id"] == turn["id"]
        assert candidate["topic"] == sentence and candidate["suggested_at"]
        assert not repo.view(owner, identity)["jobs"]
        job = {"revision": 1, "request_id": "first", "kind": "event", "topic": sentence,
               "source_turn_id": turn["id"], "due_at": candidate["suggested_at"]}
        assert client.post(path + "/jobs", json={**job, "topic": "伪造原话"}).status_code == 422
        assert client.post(path + "/jobs", json=job).status_code == 201
        assert client.post(path + "/jobs", json=job).status_code == 201
        assert client.post(path + "/jobs", json={**job, "request_id": "duplicate", "due_at": job["due_at"] + 60}).status_code == 422
        assert client.get(path + "/candidates").json() == {"candidates": []}
        owner = "other"
        assert client.get(path + "/candidates").status_code == 404
        owner = "owner"
        assert repo.deliver_due(now=job["due_at"]) == 1
        repo.chats.retract(owner, identity, turn["id"], repo.chats.get(owner, identity)["revision"])
        assert repo.inbox(owner) == []
        assert repo.recent_sent(owner, identity) == []
        assert repo.view(owner, identity)["jobs"] == []
        assert client.post(path + "/jobs", json={**job, "request_id": "stale"}).status_code == 422


def test_edit_cancels_sourced_pending_job(proactive):
    repo, session = proactive
    owner, identity = "owner", session["id"]
    sentence = "我们下次再聊这份方案"
    turn = repo.chats.commit_turn(session, "first", sentence, "好", "fixture")
    repo.set_policy(owner, identity, 0, **policy(quiet_start=None, quiet_end=None))
    repo.schedule(owner, identity, 1, "later", "follow_up", sentence, 101, now=100, source_turn_id=turn["id"])
    current = repo.chats.get(owner, identity)
    repo.chats.commit_turn(current, "edit", "你好", "你好", "fixture", replace_turn_id=turn["id"])
    assert repo.deliver_due(now=101) == 0
    assert repo.view(owner, identity)["jobs"] == []


def test_fiction_history_is_scoped_paginated_and_preserved_after_revocation(proactive):
    repo, session = proactive
    owner, identity = "owner", session["id"]
    settings = policy(quiet_start=None, quiet_end=None)
    repo.set_policy(owner, identity, 0, **settings)
    scheduled = [
        ("one", "fiction", "初雪"),
        ("event", "event", "面试"),
        ("two", "fiction", "雨声"),
        ("pending", "fiction", "晴天"),
    ]
    for number, (request_id, kind, topic) in enumerate(scheduled):
        repo.schedule(owner, identity, 1, request_id, kind, topic, 101 + number * 86400, now=100)
    for number in range(3):
        assert repo.deliver_due(now=101 + number * 86400) == 1
    repo.set_policy(owner, identity, 1, **{**settings, "enabled": False})
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user] = lambda: {"user_id": owner}
    app.dependency_overrides[repository] = lambda: repo.chats
    path = f"/api/character-chats/{identity}/proactive/fiction"
    with TestClient(app) as client:
        first = client.get(path, params={"limit": 1}).json()
        assert first["next_before"] == "two"
        assert first["entries"][0]["id"] == "two"
        assert "雨声" in first["entries"][0]["content"]
        second = client.get(path, params={"before": first["next_before"]}).json()
        assert [entry["id"] for entry in second["entries"]] == ["one"]
        assert second["next_before"] is None
        assert client.get(path, params={"before": "event"}).status_code == 404
        owner = "other"
        assert client.get(path).status_code == 404


def test_pending_schedule_changes_are_scoped_replayable_and_fence_delivery(proactive, monkeypatch):
    from repositories import character_chat_proactive

    repo, session = proactive
    identity = session["id"]
    repo.set_policy("owner", identity, 0, **policy(quiet_start=None, quiet_end=None))
    repo.schedule("owner", identity, 1, "first", "event", "面试", 101, now=100)
    repo.schedule("owner", identity, 1, "second", "event", "面试", 201, now=100)
    stale, = repo._due_jobs(101)
    monkeypatch.setattr(character_chat_proactive, "time", lambda: 100)
    app = FastAPI()
    app.include_router(router)
    owner = "other"
    app.dependency_overrides[get_current_user] = lambda: {"user_id": owner}
    app.dependency_overrides[repository] = lambda: repo.chats
    path = f"/api/character-chats/{identity}/proactive/jobs/"
    with TestClient(app) as client:
        move = {"expected_due_at": 101, "due_at": 201}
        assert client.patch(path + "first", json=move).status_code == 404
        owner = "owner"
        assert client.patch(path + "first", json=move).status_code == 409
        for _ in range(2):
            cancelled = client.patch(path + "second", json={"expected_due_at": 201, "due_at": None})
            assert cancelled.status_code == 200 and cancelled.json() == {"status": "cancelled"}
        for _ in range(2):
            moved = client.patch(path + "first", json=move)
            assert moved.status_code == 200 and moved.json() == {"status": "pending"}
        assert client.patch(path + "first", json={"expected_due_at": 101, "due_at": None}).status_code == 409
        with pytest.raises(RevisionConflictError):
            repo._deliver(stale, 101)
        assert repo.deliver_due(now=101) == 0
        assert repo.deliver_due(now=201) == 1
        assert repo.deliver_due(now=202) == 0
        assert client.patch(path + "first", json={"expected_due_at": 201, "due_at": None}).status_code == 409
    only, = repo.view("owner", identity)["jobs"]
    assert only["id"] == "first" and only["status"] == "sent" and only["sent_at"] == 201


def test_reschedule_at_previous_fiction_lease_fences_the_draft(proactive):
    repo, session = proactive
    identity = session["id"]
    repo.set_policy("owner", identity, 0, **policy(quiet_start=None, quiet_end=None))
    repo.schedule("owner", identity, 1, "story", "fiction", "意外的旅程", 101, now=100)
    claim = repo.claim_fiction(now=101)
    deadline = claim[0]["next_attempt_at"]
    repo.change_schedule("owner", identity, "story", 101, deadline, now=102)
    assert not repo.finish_fiction(claim, "旧草稿", now=102)
    assert not repo.fiction_history("owner", identity)["entries"]
    new_claim = repo.claim_fiction(now=deadline)
    assert new_claim is not None and repo.finish_fiction(new_claim, "新草稿", now=deadline)
    assert repo.fiction_history("owner", identity)["entries"][0]["content"] == "【角色虚构日常】新草稿"


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


def test_idle_greeting_is_opt_in_and_delivered_once_per_activity(proactive, monkeypatch):
    from repositories import character_chat

    repo, session = proactive
    identity = session["id"]
    clock = [2000000000]
    monkeypatch.setattr(character_chat, "timestamp", lambda: datetime.fromtimestamp(clock[0], UTC).isoformat())
    waiting = 72 * 3600
    rules = policy(quiet_start=None, quiet_end=None)
    assert repo.set_policy("owner", identity, 0, **rules)["inactivity_hours"] is None
    first = repo.chats.commit_turn(session, "first", "你好", "你好", "test")
    assert repo.deliver_due(now=clock[0] + waiting) == 0
    repo.set_policy("owner", identity, 1, inactivity_hours=72, now=clock[0], **rules)
    old_due = clock[0] + waiting
    assert repo.deliver_due(now=old_due - 1) == 0
    clock[0] = old_due - 2
    session = repo.chats.get("owner", identity)
    repo.chats.commit_turn(session, "returned", "我回来了", "你好", "test")
    assert repo.deliver_due(now=old_due) == 0
    due = clock[0] + waiting
    from repositories.character_chat_generation import CharacterChatGenerationRepository

    generations = CharacterChatGenerationRepository(repo.db)
    token = generations.claim(repo.chats.get("owner", identity), "active", "正在聊天", lease_seconds=10**9)
    assert repo.deliver_due(now=due) == 0
    generations.fail(identity, "active", token, "fixture")
    with ThreadPoolExecutor(2) as pool:
        assert sum(pool.map(lambda _: repo.deliver_due(now=due), range(2))) == 1
    repo.set_policy("owner", identity, 2, inactivity_hours=72, now=due, **policy(enabled=False))
    repo.set_policy("owner", identity, 3, inactivity_hours=72, now=due + 1, **rules)
    assert repo.deliver_due(now=due + waiting + 1) == 0
    clock[0] = due + waiting + 1
    repo.chats.commit_turn(repo.chats.get("owner", identity), "another-return", "又回来了", "你好", "test")
    assert repo.has_due(clock[0] + waiting)
    session = repo.chats.get("owner", identity)
    repo.chats.retract("owner", identity, first["id"], session["revision"])
    assert repo.deliver_due(now=due + 2 * waiting) == 0
    assert not repo.has_due(clock[0] + waiting)
    assert [job["kind"] for job in repo.view("owner", identity)["jobs"]] == ["inactivity"]


def test_running_generations_do_not_fill_the_inactivity_batch(proactive):
    from repositories.character_chat_generation import CharacterChatGenerationRepository

    repo, original = proactive
    generations = CharacterChatGenerationRepository(repo.db)
    rules = policy(quiet_start=None, quiet_end=None, inactivity_hours=72)
    begin = 100
    due = begin + 72 * 3600
    for number in range(100):
        busy = repo.chats.create("owner", original["character_id"], f"busy-{number}")
        repo.chats.commit_turn(busy, "first", "你好", "你好", "test")
        repo.set_policy("owner", busy["id"], 0, now=begin, **rules)
        generations.claim(repo.chats.get("owner", busy["id"]), "running", "已经回来了")
    assert not repo.has_due(due)
    assert repo.deliver_due(now=due) == 0
    repo.chats.commit_turn(original, "first", "你好", "你好", "test")
    repo.set_policy("owner", original["id"], 0, now=begin + 1, **rules)
    assert repo.has_due(due + 1)
    assert repo.deliver_due(now=due + 1) == 1
    assert [item["session_id"] for item in repo.inbox("owner")] == [original["id"]]
    assert not repo.has_due(due + 1)
    expired_at = repo.db.execute_sql(
        "SELECT MAX(lease_until) AS expired_at FROM character_chat_generations",
    )[0]["expired_at"]
    assert repo.has_due(expired_at)
    assert repo.deliver_due(now=expired_at) == 100
    assert not repo.has_due(expired_at)


def test_inactivity_rechecks_a_generation_claim_after_scan(proactive, monkeypatch):
    from repositories.character_chat_generation import CharacterChatGenerationRepository

    repo, session = proactive
    identity = session["id"]
    repo.chats.commit_turn(session, "first", "你好", "你好", "test")
    repo.set_policy("owner", identity, 0, now=100,
                    **policy(quiet_start=None, quiet_end=None, inactivity_hours=72))
    due = 100 + 72 * 3600
    generations = CharacterChatGenerationRepository(repo.db)
    deliver = repo._deliver_inactivity
    token = None

    def claim_before_delivery(entry, now):
        nonlocal token
        token = generations.claim(repo.chats.get("owner", identity), "running", "我回来了")
        return deliver(entry, now)

    monkeypatch.setattr(repo, "_deliver_inactivity", claim_before_delivery)
    assert repo.deliver_due(now=due) == 0
    assert token is not None and repo.inbox("owner") == []
    generations.fail(identity, "running", token, "fixture")
    monkeypatch.setattr(repo, "_deliver_inactivity", deliver)
    assert repo.deliver_due(now=due) == 1


def test_cross_session_inbox_is_owner_scoped_and_uses_latest_message(proactive):
    repo, first = proactive
    second = repo.chats.create("owner", first["character_id"], "另一个会话")
    character = repo.chats.characters.create("other", CharacterSettings(name="其他", persona="坦率", greeting="你好"))
    foreign = repo.chats.create("other", character["id"], "其他账号会话")
    for owner, session in (("owner", first), ("owner", second), ("other", foreign)):
        identity = session["id"]
        repo.set_policy(owner, identity, 0, **policy(quiet_start=None, quiet_end=None))
        repo.schedule(owner, identity, 1, "first", "event", "旧话题", 101, now=100)
        repo.schedule(owner, identity, 1, "second", "event", owner + identity, 86502, now=100)
    assert repo.deliver_due(now=101) == 3
    assert repo.deliver_due(now=86502) == 3
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user] = lambda: {"user_id": "owner"}
    app.dependency_overrides[repository] = lambda: repo.chats
    with TestClient(app) as client:
        response = client.get("/api/character-chats/proactive-inbox")
    assert response.status_code == 200
    messages = response.json()["messages"]
    assert {item["session_id"] for item in messages} == {first["id"], second["id"]}
    assert all(item["id"] == "second" for item in messages)


def test_sqlite_startup_upgrades_existing_manual_policy_without_enabling_inactivity(tmp_path):
    path = tmp_path / "older-policy.db"
    db = SQLiteDB(path)
    repo = CharacterChatProactiveRepository(db)
    character = repo.chats.characters.create("owner", CharacterSettings(name="角色", persona="坦率", greeting="你好"))
    session = repo.chats.create("owner", character["id"], "原有会话")
    try:
        repo.set_policy("owner", session["id"], 0, **policy(cadence_hours=96))
    finally:
        db.close_connection()
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("DROP INDEX ix_chat_proactive_inactive_due")
        for name in ("inactivity_hours", "inactivity_due_at", "inactivity_turn_id", "inactivity_last_sent_turn_id"):
            connection.execute(f"ALTER TABLE character_chat_proactive_policies DROP COLUMN {name}")

    db = SQLiteDB(path)
    try:
        repo = CharacterChatProactiveRepository(db)
        upgraded = repo.view("owner", session["id"])["policy"]
        assert upgraded["enabled"] and upgraded["cadence_hours"] == 96
        assert upgraded["inactivity_hours"] is None
        assert repo.deliver_due() == 0
    finally:
        db.close_connection()


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


def test_worker_writes_role_specific_fiction_and_keeps_prior_story(proactive, monkeypatch):
    from repositories import character_chat_proactive

    repo, session = proactive
    identity = session["id"]
    settings = policy(quiet_start=None, quiet_end=None)
    repo.set_policy("owner", identity, 0, **settings)
    repo.schedule("owner", identity, 1, "old", "fiction", "旧地图", 101, now=100)
    assert repo.deliver_due(now=101) == 1
    repo.schedule("owner", identity, 1, "new", "fiction", "起雾的码头", 86502, now=100)
    monkeypatch.setattr(character_chat_proactive, "time", lambda: 86502)
    monkeypatch.setattr(WebPushConfig, "from_env", staticmethod(lambda: None))

    async def complete(request, target, request_id):
        assert target["id"] == identity
        assert request_id == "proactive-fiction:new"
        assert "角色名：旅人" in request.persona_prompt and "坦率" in request.persona_prompt
        assert "起雾的码头" in request.message and "旧地图" in request.message
        assert request.memory_write_enabled is False
        return SimpleNamespace(reply="【角色虚构日常】我把旧地图摊在码头，雾里竟传来风铃声。")

    worker = ChatProactiveActivities(repo.db, complete)
    assert asyncio.run(worker.deliver()) == 1
    assert asyncio.run(worker.deliver()) == 0
    latest, previous = repo.fiction_history("owner", identity)["entries"]
    assert latest["content"] == "【角色虚构日常】我把旧地图摊在码头，雾里竟传来风铃声。"
    assert "旧地图" in previous["content"]


def test_fiction_claim_recovers_and_cannot_publish_after_consent_changes(proactive):
    repo, session = proactive
    identity = session["id"]
    settings = policy(quiet_start=None, quiet_end=None)
    repo.set_policy("owner", identity, 0, **settings)
    repo.schedule("owner", identity, 1, "fiction", "fiction", "写一段虚构日常", 101, now=100)
    with ThreadPoolExecutor(2) as workers:
        first, = [claim for claim in workers.map(lambda _: repo.claim_fiction(now=101), range(2)) if claim]
    assert repo.claim_fiction(now=190) is None
    second = repo.claim_fiction(now=191)
    assert second is not None
    assert not repo.finish_fiction(first, "旧 worker 返回的正文", now=192)
    repo.set_policy("owner", identity, 1, **{**settings, "enabled": False})
    assert not repo.finish_fiction(second, "已撤销的正文", now=192)
    assert not repo.fiction_history("owner", identity)["entries"]


def test_fiction_falls_back_if_model_is_unavailable(proactive, monkeypatch):
    from repositories import character_chat_proactive

    repo, session = proactive
    identity = session["id"]
    repo.set_policy("owner", identity, 0, **policy(quiet_start=None, quiet_end=None))
    repo.schedule("owner", identity, 1, "fiction", "fiction", "旧地图", 101, now=100)
    monkeypatch.setattr(character_chat_proactive, "time", lambda: 101)
    monkeypatch.setattr(WebPushConfig, "from_env", staticmethod(lambda: None))

    async def unavailable(*args):
        raise TimeoutError()

    assert asyncio.run(ChatProactiveActivities(repo.db, unavailable).deliver()) == 1
    entry, = repo.fiction_history("owner", identity)["entries"]
    assert entry["content"].startswith("【角色虚构日常】") and "旧地图" in entry["content"]


def test_fiction_result_is_deferred_if_quiet_hours_start_during_generation(proactive):
    repo, session = proactive
    identity = session["id"]
    repo.set_policy("owner", identity, 0, **policy(timezone="UTC"))
    quiet_begins = 22 * 3600
    repo.schedule("owner", identity, 1, "fiction", "fiction", "窗外下雪", quiet_begins - 1, now=100)
    claim = repo.claim_fiction(now=quiet_begins - 1)
    assert claim is not None
    assert not repo.finish_fiction(claim, "初雪到了。", now=quiet_begins + 1)
    assert not repo.fiction_history("owner", identity)["entries"]
    later = 32 * 3600
    fresh_claim = repo.claim_fiction(now=later)
    assert fresh_claim is not None and repo.finish_fiction(fresh_claim, "雪已经落了半夜。", now=later)
