"""Generation retries, expired writers and transactional result persistence."""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest

from character.library import CharacterSettings
from db.database import SQLiteDB
from db.errors import RevisionConflictError
from repositories.character_chat import CharacterChatRepository
from repositories.character_chat_generation import CharacterChatGenerationRepository, GenerationInProgressError
from repositories.character_library import CharacterNotFoundError
from services.character_chat import CharacterChatService
from services.character_chat_tasks import CharacterChatTasks


@pytest.fixture
def setup(tmp_path):
    db = SQLiteDB(tmp_path / "generation.db")
    repo = CharacterChatRepository(db)
    char = repo.characters.create("1", CharacterSettings(name="旅人", persona="谨慎", greeting="你好"))
    session = repo.create("1", char["id"], "初识")
    yield db, repo, session, CharacterChatGenerationRepository(db)
    db.close_connection()


@pytest.mark.parametrize("changed", [False, True])
def test_terminal_enrichment_failure_is_idempotent_and_preserves_user_changes(setup, changed):
    from repositories.character_chat_enrichment import CharacterChatEnrichmentRepository

    db, repo, session, _ = setup
    state = dict(session["state"], memory_processing="pending", relationship_processing="pending")
    turn = repo.commit_turn(session, "request", "你好", "已保存", "test",
                            state=state, enqueue_enrichment=True)
    worker = CharacterChatEnrichmentRepository(db)
    worker.submitted(turn["id"], 2)
    if changed:
        repo.update_relationship("1", session["id"], 2, "close", "用户决定")
        before = repo.get("1", session["id"])
        assert worker.fail(turn["id"]) == "inactive"
        assert repo.get("1", session["id"]) == before
    else:
        assert worker.fail(turn["id"]) == "failed"
        current = repo.get("1", session["id"])
        assert current["state"]["memory_processing"] == "failed"
        assert current["state"]["relationship_processing"] == "failed"
        assert current["revision"] == 3
        assert worker.fail(turn["id"]) == "inactive"
        assert repo.get("1", session["id"]) == current
        assert repo.turns("1", session["id"])[0]["reply"] == "已保存"


@pytest.mark.parametrize("outcome", ["completed", "failed", "manual"])
def test_next_turn_waits_for_enrichment_without_invoking_model(setup, outcome):
    from repositories.character_chat_enrichment import CharacterChatEnrichmentRepository
    from repositories.character_chat_generation import EnrichmentInProgressError

    db, repo, session, jobs = setup
    turn = repo.commit_turn(session, "first", "你好", "已保存", "test", enqueue_enrichment=True)
    worker = CharacterChatEnrichmentRepository(db)
    current = repo.get("1", session["id"])
    assert worker.is_pending(current)
    with pytest.raises(EnrichmentInProgressError):
        jobs.claim(current, "second", "继续")
    assert jobs.get("1", session["id"], "second") is None
    if outcome == "completed":
        loaded, source = worker.load(turn["id"])
        assert worker.commit(loaded, source, loaded["state"])
    elif outcome == "failed":
        assert worker.fail(turn["id"]) == "failed"
    else:
        repo.update_relationship("1", session["id"], 2, "close", "用户决定")
    current = repo.get("1", session["id"])
    assert not worker.is_pending(current)
    assert jobs.claim(current, "second", "继续")


def test_enrichment_retry_fences_old_workflow_and_is_owner_scoped(setup):
    from repositories.character_chat_enrichment import CharacterChatEnrichmentRepository

    db, repo, session, _ = setup
    turn = repo.commit_turn(session, "request", "你好", "原回复", "test", enqueue_enrichment=True)
    worker = CharacterChatEnrichmentRepository(db)
    old_session, old_turn = worker.load(turn["id"], 2)
    assert worker.fail(turn["id"], 2) == "failed"
    assert worker.retryable(repo.get("1", session["id"]))
    with pytest.raises(CharacterNotFoundError):
        worker.retry("other", session["id"], 3)
    assert worker.retry("1", session["id"], 3) == {"queued": True}
    with pytest.raises(RevisionConflictError):
        worker.retry("1", session["id"], 3)
    assert worker.load(turn["id"], 2) is None
    assert worker.fail(turn["id"], 2) == "inactive"
    worker.submitted(turn["id"], 2)
    assert not worker.commit(old_session, old_turn, old_session["state"])
    assert worker.pending() == [{"turn_id": turn["id"], "expected_revision": 4}]
    current, source = worker.load(turn["id"], 4)
    assert worker.commit(current, source, current["state"])
    assert len(repo.turns("1", session["id"])) == 1
    assert repo.turns("1", session["id"])[0]["reply"] == "原回复"


def test_expired_attempt_is_fenced_and_result_is_atomic(setup):
    db, repo, session, jobs = setup
    first = jobs.claim(session, "request", "你好", lease_seconds=-1)
    assert jobs.get("1", session["id"], "request")["status"] == "interrupted"
    second = jobs.claim(session, "request", "你好")
    assert second != first
    with pytest.raises(RevisionConflictError):
        repo.commit_turn(session, "request", "你好", "旧回复", "test", generation_token=first)
    assert repo.get("1", session["id"])["revision"] == 1
    assert repo.turns("1", session["id"]) == []
    jobs.fail(session["id"], "request", first, "late_failure")
    assert jobs.get("1", session["id"], "request")["status"] == "running"
    turn = repo.commit_turn(session, "request", "你好", "新回复", "test", generation_token=second)
    status = jobs.get("1", session["id"], "request")
    assert status["status"] == "completed" and status["turn_id"] == turn["id"] and status["attempt"] == 2
    with pytest.raises(CharacterNotFoundError):
        jobs.get("other", session["id"], "request")
    repo.delete_sessions("1", [{"id": session["id"], "revision": 2}])
    assert db.execute_sql("SELECT * FROM character_chat_generations") == []


def test_concurrent_claims_allow_only_one_provider_attempt(setup):
    _, _, session, jobs = setup

    def claim(request):
        try:
            return jobs.claim(session, request, "你好")
        except (RevisionConflictError, GenerationInProgressError):
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(claim, ["first", "second"]))
    assert sum(item is not None for item in outcomes) == 1


def test_failure_retries_same_identity_and_replay_skips_provider(setup):
    db, repo, session, jobs = setup
    calls = 0

    async def complete(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("provider unavailable")
        return "你好", "test"

    service = CharacterChatService(db, complete)

    def generate(message="你好"):
        return asyncio.run(service.generate("1", session["id"], revision=1, request_id="request", message=message))

    with pytest.raises(RuntimeError):
        generate()
    assert jobs.get("1", session["id"], "request")["status"] == "failed"
    with pytest.raises(RevisionConflictError):
        generate("偷偷替换消息")
    first = generate()
    assert generate()["id"] == first["id"]
    assert calls == 2
    assert len(repo.turns("1", session["id"])) == 1


def test_cancelled_generation_can_retry_and_no_result_is_fabricated(setup):
    db, repo, session, jobs = setup

    async def exercise():
        started = asyncio.Event()

        async def complete(*args):
            started.set()
            await asyncio.Event().wait()

        service = CharacterChatService(db, complete)
        task = asyncio.create_task(service.generate("1", session["id"], revision=1, request_id="request", message="你好"))
        await started.wait()
        with pytest.raises(GenerationInProgressError):
            await service.generate("1", session["id"], revision=1, request_id="request", message="你好")
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())
    status = jobs.get("1", session["id"], "request")
    assert status["status"] == "failed" and status["error_code"] == "interrupted"
    assert repo.turns("1", session["id"]) == []


def test_user_change_invalidates_running_generation(setup):
    _, repo, session, jobs = setup
    token = jobs.claim(session, "request", "你好")
    repo.update_relationship("1", session["id"], 1, "familiar", "手动更正")
    with pytest.raises(RevisionConflictError):
        repo.commit_turn(session, "request", "你好", "旧回复", "test", generation_token=token)
    assert repo.turns("1", session["id"]) == []
    assert repo.get("1", session["id"])["state"]["relationship"]["summary"] == "手动更正"


def test_pending_request_recovery_is_owner_and_revision_scoped(setup):
    _, repo, session, jobs = setup
    assert jobs.pending("1", session["id"]) is None
    token = jobs.claim(session, "request", "需要恢复的消息")
    recovered = jobs.pending("1", session["id"])
    assert recovered["body"] == {
        "revision": 1, "request_id": "request", "message": "需要恢复的消息", "replace_turn_id": None,
    }
    assert "lease_token" not in recovered
    with pytest.raises(CharacterNotFoundError):
        jobs.pending("other", session["id"])
    jobs.fail(session["id"], "request", token, "interrupted")
    assert jobs.pending("1", session["id"])["status"] == "failed"
    repo.update_relationship("1", session["id"], 1, "familiar", "新状态")
    assert jobs.pending("1", session["id"]) is None
    stale = jobs.get("1", session["id"], "request")
    assert stale["status"] == "superseded" and "body" not in stale


def test_enrichment_dispatch_is_atomic_with_reply_and_removed_with_session(setup):
    db, repo, session, _ = setup
    turn = repo.commit_turn(session, "request", "你好", "回复", "test", enqueue_enrichment=True)
    jobs = db.execute_sql("SELECT * FROM character_chat_enrichment_jobs")
    assert len(jobs) == 1 and jobs[0]["turn_id"] == turn["id"]
    assert jobs[0]["expected_revision"] == 2 and jobs[0]["status"] == "pending"
    assert set(jobs[0]) == {"turn_id", "session_id", "expected_revision", "status", "created_at"}
    with pytest.raises(RevisionConflictError):
        repo.commit_turn(session, "stale", "旧消息", "回复", "test", enqueue_enrichment=True)
    assert len(db.execute_sql("SELECT * FROM character_chat_enrichment_jobs")) == 1
    repo.delete_sessions("1", [{"id": session["id"], "revision": 2}])
    assert db.execute_sql("SELECT * FROM character_chat_enrichment_jobs") == []


def test_background_mode_saves_reply_without_calling_auxiliary_models(setup):
    db, repo, session, _ = setup

    async def complete(*args):
        return "已回复", "test"

    async def forbidden(*args):
        raise AssertionError("Auxiliary work must not delay reply")

    service = CharacterChatService(db, complete, forbidden, forbidden, background=True)
    turn = asyncio.run(service.generate("1", session["id"], revision=1, request_id="background", message="你好"))
    assert turn["reply"] == "已回复"
    assert repo.get("1", session["id"])["state"]["memory_processing"] == "pending"
    assert db.execute_sql("SELECT turn_id FROM character_chat_enrichment_jobs") == [{"turn_id": turn["id"]}]


def test_temporal_dispatch_retries_same_id_after_acknowledgement_loss(setup):
    from temporalio.exceptions import WorkflowAlreadyStartedError

    from services.temporal_chat import dispatch_enrichment

    db, repo, session, _ = setup
    turn = repo.commit_turn(session, "request", "private message", "private reply", "test", enqueue_enrichment=True)

    class Client:
        calls = []

        async def start_workflow(self, workflow, argument, **kwargs):
            self.calls.append((argument, kwargs))
            if len(self.calls) == 1:
                raise TimeoutError("acknowledgement lost")
            raise WorkflowAlreadyStartedError(kwargs["id"], "ChatEnrichmentWorkflow")

    client = Client()
    with pytest.raises(TimeoutError):
        asyncio.run(dispatch_enrichment(db, client, "queue"))
    assert db.execute_sql("SELECT status FROM character_chat_enrichment_jobs")[0]["status"] == "pending"
    assert asyncio.run(dispatch_enrichment(db, client, "queue")) == 1
    assert client.calls[0] == client.calls[1]
    assert client.calls[0][0] == {"turn_id": turn["id"], "expected_revision": 2}
    assert "private" not in str(client.calls)
    assert db.execute_sql("SELECT status FROM character_chat_enrichment_jobs")[0]["status"] == "submitted"


@pytest.mark.parametrize("change", ["none", "manual", "retract", "delete"])
def test_background_enrichment_cannot_overwrite_later_user_actions(setup, change):
    from repositories.character_chat_enrichment import CharacterChatEnrichmentRepository

    db, repo, session, _ = setup
    turn = repo.commit_turn(session, "request", "喜欢茶", "知道了", "test", enqueue_enrichment=True)
    worker = CharacterChatEnrichmentRepository(db)
    assert worker.pending() == [{"turn_id": turn["id"], "expected_revision": 2}]
    worker.submitted(turn["id"], 2)
    current, source = worker.load(turn["id"])
    state = current["state"]
    state["memories"].append({"id": "memory", "status": "confirmed", "content": "喜欢茶",
                              "source_request_id": "request", "source_session_id": session["id"]})
    if change == "manual":
        repo.update_relationship("1", session["id"], 2, "close", "手动设定")
    elif change == "retract":
        repo.retract("1", session["id"], turn["id"], 2)
    elif change == "delete":
        repo.delete_sessions("1", [{"id": session["id"], "revision": 2}])
    assert worker.commit(current, source, state) is (change == "none")
    assert worker.load(turn["id"]) is None
    if change == "none":
        assert repo.get("1", session["id"])["state"]["memories"][0]["source_turn_id"] == turn["id"]
        assert not worker.commit(current, source, state)
        assert repo.get("1", session["id"])["revision"] == 3
    elif change != "delete":
        assert repo.get("1", session["id"])["state"]["memories"] == []


def test_draft_is_fenced_resettable_and_never_a_saved_turn(setup):
    _, repo, session, jobs = setup
    old = jobs.claim(session, "request", "你好", lease_seconds=-1)
    token = jobs.claim(session, "request", "你好")
    with pytest.raises(RevisionConflictError):
        jobs.publish(session, "request", old, {"type": "draft_delta", "text": "过期"})
    jobs.publish(session, "request", token, {"type": "draft_delta", "text": "草稿"})
    assert jobs.get("1", session["id"], "request")["draft_text"] == "草稿"
    assert repo.turns("1", session["id"]) == []
    jobs.publish(session, "request", token, {"type": "draft_reset"})
    assert jobs.get("1", session["id"], "request")["draft_text"] == ""
    repo.update_relationship("1", session["id"], 1, "familiar", "更新")
    with pytest.raises(RevisionConflictError):
        jobs.publish(session, "request", token, {"type": "draft_delta", "text": "失效"})
    assert "draft_text" not in jobs.get("1", session["id"], "request")


def test_sse_reconnect_reads_snapshot_and_only_commit_reports_complete(setup):
    import json

    from services.character_chat_events import generation_events

    _, repo, session, jobs = setup
    token = jobs.claim(session, "request", "你好")
    jobs.publish(session, "request", token, {"type": "draft_delta", "text": "草稿"})

    async def exercise():
        stream = generation_events(jobs, "1", session["id"], "request", interval=0)
        first = await anext(stream)
        assert json.loads(first.split("data: ")[1])["draft_text"] == "草稿"
        await stream.aclose()
        stream = generation_events(jobs, "1", session["id"], "request", interval=0)
        assert await anext(stream) == first
        turn = repo.commit_turn(session, "request", "你好", "最终回答", "test", generation_token=token)
        final = json.loads((await anext(stream)).split("data: ")[1])
        assert final["status"] == "completed" and final["turn_id"] == turn["id"]
        assert "draft_text" not in final
        with pytest.raises(StopAsyncIteration):
            await anext(stream)
        repo.delete_sessions("1", [{"id": session["id"], "revision": 2}])
        revoked = generation_events(jobs, "1", session["id"], "request")
        assert "event: revoked" in await anext(revoked)
        await revoked.aclose()

    asyncio.run(exercise())


def test_draft_migration_preserves_existing_request(tmp_path):
    import importlib.util
    from pathlib import Path

    import sqlalchemy as sa
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    engine = sa.create_engine("sqlite:///" + str(tmp_path / "upgrade.db"))
    try:
        with engine.begin() as connection:
            modules = []
            for name in ["021_character_chat_generation", "022_character_chat_draft"]:
                spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "alembic" / "versions" / (name + ".py"))
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                module.op = Operations(MigrationContext.configure(connection))
                modules.append(module)
            modules[0].upgrade()
            connection.execute(sa.text("INSERT INTO character_chat_generations "
                "(session_id,request_id,payload_json,status,lease_token,lease_until,attempt) "
                "VALUES ('session','request','[]','failed','token',0,1)"))
            modules[1].upgrade()
            modules[1].upgrade()
            row = connection.execute(sa.text("SELECT * FROM character_chat_generations")).mappings().one()
            assert row["request_id"] == "request" and row["draft_text"] == "" and row["draft_revision"] == 0
            modules[1].downgrade()
            assert connection.execute(sa.text("SELECT request_id FROM character_chat_generations")).scalar_one() == "request"
    finally:
        engine.dispose()


def test_generation_state_survives_database_reopen(setup):
    db, repo, session, jobs = setup
    token = jobs.claim(session, "request", "你好")
    reopened = SQLiteDB(db.db_path)
    try:
        recovered = CharacterChatGenerationRepository(reopened)
        assert recovered.get("1", session["id"], "request")["status"] == "running"
        with pytest.raises(GenerationInProgressError):
            recovered.claim(session, "request", "你好")
        turn = repo.commit_turn(session, "request", "你好", "回复", "test", generation_token=token)
        assert recovered.get("1", session["id"], "request")["turn_id"] == turn["id"]
        repo.retract("1", session["id"], turn["id"], 2)
        assert recovered.get("1", session["id"], "request")["status"] == "superseded"
    finally:
        reopened.close_connection()


def test_disconnected_waiter_does_not_cancel_generation(setup):
    db, repo, session, jobs = setup

    async def exercise():
        owner = CharacterChatTasks()
        started, release = asyncio.Event(), asyncio.Event()
        calls = 0

        async def complete(*args):
            nonlocal calls
            calls += 1
            started.set()
            await release.wait()
            return "保存的回复", "test"

        service = CharacterChatService(db, complete)

        def factory():
            return service.generate("1", session["id"], revision=1, request_id="request", message="你好")

        first = asyncio.create_task(owner.run("key", "same", factory))
        await started.wait()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert jobs.get("1", session["id"], "request")["status"] == "running"
        with pytest.raises(RevisionConflictError):
            await owner.run("key", "different", factory)
        second = asyncio.create_task(owner.run("key", "same", factory))
        release.set()
        result = await second
        assert result["reply"] == "保存的回复"
        assert calls == 1
        assert jobs.get("1", session["id"], "request")["status"] == "completed"
        await owner.shutdown()

    asyncio.run(exercise())
    assert len(repo.turns("1", session["id"])) == 1


def test_application_shutdown_records_interrupted_generation(setup):
    db, _, session, jobs = setup

    async def exercise():
        owner = CharacterChatTasks()
        started = asyncio.Event()

        async def complete(*args):
            started.set()
            await asyncio.Event().wait()

        service = CharacterChatService(db, complete)
        task = asyncio.create_task(owner.run("key", "same", lambda: service.generate(
            "1", session["id"], revision=1, request_id="request", message="你好"
        )))
        await started.wait()
        await owner.shutdown()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not owner.pending

    asyncio.run(exercise())
    assert jobs.get("1", session["id"], "request")["error_code"] == "interrupted"
