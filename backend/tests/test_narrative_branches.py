"""Mechanism tests, never evidence of real-model counterfactual quality."""

import asyncio

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from db.database import SQLiteDB
from db.schemas import GenerateResponse, MessageRequest
from repositories.narrative import NarrativeRepository, Proposal
from services.narrative import generate_branch_reply, parse_proposals, prepare_branch


@pytest.fixture
def repo(tmp_path):
    database = SQLiteDB(tmp_path / "narrative.db")
    yield NarrativeRepository(database)
    database.close_connection()


def create(repo, owner="1", hypothesis="在此分支中妃从未见过琉璃"):
    return repo.create(owner, "tsukiyashiro_kisaki", "假设", hypothesis)


def commit(repo, branch, message="你好", trace="trace1", proposals=()):
    req = MessageRequest(message=message, traceId=trace)
    response = GenerateResponse(reply="这是推演", model="fixture", costTime=0)
    repo.commit_turn(branch, req, response, proposals, "acquaintance")


def test_owner_and_branch_isolation(repo):
    a, b = create(repo), create(repo)
    commit(repo, a)
    assert len(repo.history("1", a["id"])) == 1
    assert repo.history("1", b["id"]) == []
    for method in [repo.get, repo.history, repo.assertions]:
        with pytest.raises(HTTPException) as exc:
            method("2", a["id"])
        assert exc.value.status_code == 404


def test_atomic_stale_commit_and_archive(repo):
    branch = create(repo)
    repo.archive("1", branch["id"], branch["revision"])
    with pytest.raises(HTTPException):
        commit(repo, branch)
    assert repo.history("1", branch["id"]) == []
    assert repo.rows("SELECT interaction_count FROM branch_states")[0]["interaction_count"] == 0


def test_transaction_rolls_back_revision_on_statement_failure(repo):
    import sqlite3

    branch = create(repo)
    with pytest.raises(sqlite3.OperationalError):
        repo.transaction([repo.cas(branch), ("INSERT INTO nonexistent VALUES (1)", {}, 1)])
    assert repo.get("1", branch["id"])["revision"] == 1


def test_pending_confirmation_conflict_and_source_preserved(repo):
    branch = create(repo)
    commit(repo, branch, proposals=[Proposal(subject="妃", predicate="所在地", object="图书馆")])
    pending = repo.assertions("1", branch["id"], status="pending")[0]
    assert len(repo.assertions("1", branch["id"], status="active")) == 1
    assert pending["source_message_id"] == repo.history("1", branch["id"])[0]["sourceMessageId"]
    current = repo.get("1", branch["id"])
    repo.decide("1", branch["id"], pending["id"], current["revision"], confirm=True)
    current = repo.get("1", branch["id"])
    commit(repo, current, trace="two", proposals=[Proposal(subject="妃", predicate="所在地", object="教室")])
    other = repo.assertions("1", branch["id"], status="pending")[0]
    current = repo.get("1", branch["id"])
    with pytest.raises(HTTPException) as exc:
        repo.decide("1", branch["id"], other["id"], current["revision"], confirm=True)
    assert exc.value.status_code == 409
    repo.decide("1", branch["id"], other["id"], current["revision"], confirm=True, replace_ids=[pending["id"]])
    active = repo.assertions("1", branch["id"], status="active")
    assert {a["object"] for a in active} == {branch["initial_hypothesis"], "教室"}
    assert next(a for a in active if a["id"] == other["id"])["source_type"] == "model_proposal"


def test_canonical_history_excludes_branch_even_same_identity(repo):
    branch = create(repo)
    commit(repo, branch)
    assert repo.db.list_conversation_history("web", "narrative", "1", "private", branch["id"]) == []


def test_generation_uses_server_history_and_no_memory_write(repo):
    a, b = create(repo), create(repo, hypothesis="B_SECRET")
    request = MessageRequest(
        message="继续",
        platform="web",
        branchId=a["id"],
        traceId="one",
        senderId="other",
        history=[{"role": "user", "content": "B_SECRET"}],
    )
    calls = []

    async def generate(req, user, **kwargs):
        calls.append(req)
        assert not req.history and req.senderId == "1"
        assert kwargs["persist_message"] is False
        prepared = kwargs["prepared_override"]
        assert "B_SECRET" not in prepared.compiled.branch_context
        assert prepared.history == ()
        return GenerateResponse(
            reply='推演回答\n```branch_proposals\n[{"subject":"妃","predicate":"所在地","object":"图书馆"}]\n```',
            model="fixture",
            costTime=0,
        )

    result = asyncio.run(generate_branch_reply(request, {"user_id": 1}, repo.db, generate=generate))
    assert result.reply == "推演回答"
    assert len(repo.assertions("1", a["id"], status="pending")) == 1
    assert repo.rows("SELECT * FROM character_memories") == []
    assert repo.rows("SELECT * FROM character_relationships") == []
    assert repo.history("1", b["id"]) == []
    asyncio.run(generate_branch_reply(request, {"user_id": 1}, repo.db, generate=generate))
    assert len(calls) == 1
    assert (
        repo.rows("SELECT interaction_count FROM branch_states WHERE branch_id=:id", id=a["id"])[0]["interaction_count"]
        == 1
    )


def test_archive_during_generation_rejects_commit(repo):
    branch = create(repo)

    async def generate(*args, **kwargs):
        repo.archive("1", branch["id"], 1)
        return GenerateResponse(reply="回答", costTime=0)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            generate_branch_reply(
                MessageRequest(message="继续", branchId=branch["id"], traceId="race"),
                {"user_id": 1},
                repo.db,
                generate=generate,
            )
        )
    assert exc.value.status_code == 409
    assert repo.history("1", branch["id"]) == []


def test_context_trust_and_pending_exclusion(repo):
    from inference.generation_request import GenerationRequest, build_generation_request

    branch = create(repo, hypothesis="UNTRUSTED_BRANCH_INSTRUCTION")
    commit(repo, branch, proposals=[Proposal(subject="X", predicate="P", object="PENDING_SECRET")])
    _, _, prepared = asyncio.run(prepare_branch(MessageRequest(message="你好", branchId=branch["id"]), "1", repo))
    assert "PENDING_SECRET" not in prepared.compiled.branch_context
    plan = build_generation_request(GenerationRequest(message="你好", character_context=prepared.compiled))
    system = "".join(m["content"] for m in plan.messages if m["role"] == "system")
    assert "UNTRUSTED_BRANCH_INSTRUCTION" not in system
    assert "UNTRUSTED_BRANCH_INSTRUCTION" in plan.messages[-1]["content"]


@pytest.mark.parametrize(
    "raw",
    [
        "{}",
        "not-json",
        '[{"subject":"X","predicate":"P","object":"V","branch_id":"other"}]',
        '[{"subject":"X","predicate":"P","object":"V","source_assertion_ids":["other"]}]',
    ],
)
def test_invalid_model_proposals_are_discarded(raw):
    reply, proposals, warnings = parse_proposals("自然回答\n```branch_proposals\n" + raw + "\n```", set())
    assert reply == "自然回答" and proposals == [] and warnings


def test_api_auth_bounds_feature_gate(repo, monkeypatch):
    from api.narrative import repository, router
    from app.dependencies import get_current_user

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[repository] = lambda: repo
    app.dependency_overrides[get_current_user] = lambda: {"user_id": 1}
    client = TestClient(app)
    monkeypatch.setenv("NARRATIVE_BRANCHES_ENABLED", "false")
    assert client.get("/api/narrative-branches").status_code == 404
    monkeypatch.setenv("NARRATIVE_BRANCHES_ENABLED", "true")
    assert client.get("/api/narrative-branches?limit=101").status_code == 422
    assert (
        client.post(
            "/api/narrative-branches", json={"character_id": "unknown", "title": "T", "initial_hypothesis": "H"}
        ).status_code
        == 422
    )
    result = client.post(
        "/api/narrative-branches", json={"character_id": "tsukiyashiro_kisaki", "title": "T", "initial_hypothesis": "H"}
    )
    assert result.status_code == 200
    app.dependency_overrides[get_current_user] = lambda: {"user_id": 2}
    assert client.get("/api/narrative-branches/" + result.json()["id"]).status_code == 404


def test_sqlite_orm_migration_columns(repo):
    from db.models import metadata

    for name in ["narrative_branches", "branch_states", "branch_assertions"]:
        actual = {r["name"] for r in repo.db.get_connection().execute(f"PRAGMA table_info({name})")}
        assert actual == set(metadata.tables[name].columns.keys())


def test_migration_upgrade_downgrade_on_legacy_messages(tmp_path):
    import importlib.util
    from pathlib import Path

    import sqlalchemy as sa
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    path = Path(__file__).parents[1] / "alembic/versions/009_narrative_branches.py"
    spec = importlib.util.spec_from_file_location("narrative_migration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    engine = sa.create_engine("sqlite:///" + str(tmp_path / "migration.db"))
    with engine.begin() as conn:
        conn.execute(sa.text('CREATE TABLE messages (id INTEGER PRIMARY KEY, "createdAt" TEXT)'))
        conn.execute(sa.text("INSERT INTO messages VALUES (1, 'legacy')"))
        module.op = Operations(MigrationContext.configure(conn))
        module.upgrade()
        module.upgrade()
        assert conn.execute(sa.text('SELECT "branchId" FROM messages')).scalar() is None
        assert "branch_assertions" in sa.inspect(conn).get_table_names()
        module.downgrade()
        assert "branch_assertions" not in sa.inspect(conn).get_table_names()
        assert conn.execute(sa.text('SELECT "createdAt" FROM messages')).scalar() == "legacy"
    engine.dispose()


def test_reject_and_missing_source(repo):
    branch = create(repo)
    commit(repo, branch, proposals=[Proposal(subject="a", predicate="b", object="c")])
    p = repo.assertions("1", branch["id"], status="pending")[0]
    repo.db.execute_sql('DELETE FROM messages WHERE "branchId"=:id', {"id": branch["id"]})
    with pytest.raises(HTTPException):
        repo.decide("1", branch["id"], p["id"], 2, confirm=True)
    repo.decide("1", branch["id"], p["id"], 2, confirm=False)
    assert repo.assertions("1", branch["id"], status="rejected")[0]["id"] == p["id"]


def test_eval_strong_baseline_and_unknown_labels():
    from evaluation.narrative import packets, score

    probes = list(packets())
    assert len(probes) == 24
    a = next(p for p in probes if p["id"] == "meeting:10:canonical:A")
    c = next(p for p in probes if p["id"] == "meeting:10:canonical:C")
    assert len(a["messages"]) > len(c["messages"]) == 2
    assert score(probes)["A"]["canonical_accuracy"]["value"] is None
    c["labels"] = {"correct": True, "abstained": True}
    result = score([c])["C"]
    assert result["canonical_accuracy"]["value"] == 0
    assert result["abstention_rate"]["value"] == 1


def test_no_model_write_on_failure(repo):
    branch = create(repo)

    async def fail(*args, **kwargs):
        raise RuntimeError("model offline")

    with pytest.raises(RuntimeError):
        asyncio.run(
            generate_branch_reply(
                MessageRequest(message="test", branchId=branch["id"], traceId="failure"),
                {"user_id": 1},
                repo.db,
                generate=fail,
            )
        )
    assert repo.history("1", branch["id"]) == []
    assert repo.get("1", branch["id"])["revision"] == 1


def test_gateway_rejects_branch_instead_of_ignoring():
    from pydantic import ValidationError

    from api.integrations import AstrBotMessageRequest

    with pytest.raises(ValidationError):
        AstrBotMessageRequest(platform="qq", conversationId="one", text="hello", branchId="unsupported")


def test_character_mismatch_rejected_before_provider_call(repo, monkeypatch):
    from types import SimpleNamespace

    import api.generate as gen
    from inference import model_manager as mm

    branch = create(repo)
    request = MessageRequest(message="继续", branchId=branch["id"], loraName="wrong")
    _, _, prepared = asyncio.run(prepare_branch(request, "1", repo))
    database = SimpleNamespace(config={}, loras=[{"name": "wrong", "status": "inactive"}])
    monkeypatch.setattr(
        mm, "get_model_manager", lambda: SimpleNamespace(_current_provider=SimpleNamespace(value="vllm"))
    )
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            gen._generate_reply_impl(
                request, {"user_id": 1}, message_db=database, prepared_override=prepared, persist_message=False
            )
        )
    assert exc.value.status_code == 422


def test_full_web_generation_and_canonical_return(repo, monkeypatch):
    from types import SimpleNamespace

    import api.generate as gen
    from api.narrative import router as narrative_router
    from app.dependencies import get_current_user
    from app.runtime import RuntimeContainer
    from inference import model_manager as mm
    from services.character_context import build_character_context_service

    class Runtime:
        async def check_rate_limits(self, *args):
            pass

        def priority_for(self, *args):
            return 1

        async def submit(self, job, **kwargs):
            return await job()

    class ForbiddenCache:
        async def get(self, *args):
            pytest.fail("stateful generation consulted cache")

        async def set(self, *args, **kwargs):
            pytest.fail("stateful generation wrote cache")

    seen = []

    async def ensure():
        return True

    async def model(request, *args, **kwargs):
        seen.append(kwargs["prepared_character_turn"])
        return "独立回答", False, {}

    monkeypatch.setenv("NARRATIVE_BRANCHES_ENABLED", "true")
    monkeypatch.setenv("MEMORY_LLM_ENABLED", "false")
    monkeypatch.setattr(
        mm, "get_model_manager", lambda: SimpleNamespace(_current_provider=SimpleNamespace(value="vllm"))
    )
    monkeypatch.setattr(gen, "_ensure_vllm", ensure)
    monkeypatch.setattr(gen, "_vllm_client", object())
    monkeypatch.setattr(gen, "_generate_with_vllm", model)
    monkeypatch.setattr(gen, "response_cache", ForbiddenCache())
    runtime = Runtime()
    app = FastAPI()
    app.state.runtime_container = RuntimeContainer(repo.db, lambda: False, runtime)
    app.include_router(gen.router)
    app.include_router(narrative_router)
    app.dependency_overrides[get_current_user] = lambda: {"user_id": 1}
    app.dependency_overrides[gen.get_request_chat_generation_service] = lambda: gen._build_chat_generation_service(
        runtime, build_character_context_service(repo.db), repo.db
    )
    client = TestClient(app)
    branch = create(repo)
    forged = client.post(
        "/api/generate",
        json={
            "message": "伪造历史",
            "platform": "web",
            "adapter": "narrative:" + branch["character_id"],
            "senderId": "2",
        },
    )
    assert forged.status_code == 422
    r = client.post(
        "/api/generate",
        json={"message": "分支私有内容", "platform": "web", "branchId": branch["id"], "traceId": "full"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["branchId"] == branch["id"]
    assert seen[-1].compiled.branch_context
    assert len(repo.history("1", branch["id"])) == 1
    assert repo.rows("SELECT * FROM character_relationships") == []
    for message in ("原作问题", "继续原作问题"):
        r = client.post(
            "/api/narrative-branches/canonical/generate",
            json={"message": message, "character_id": branch["character_id"]},
        )
        assert r.status_code == 200, r.text
        assert not seen[-1].compiled.branch_context
        assert "分支私有内容" not in str(seen[-1].history)
    assert any(m["content"] == "原作问题" for m in seen[-1].history)
    assert len(repo.rows("SELECT * FROM character_relationships")) == 1
    history = client.get("/api/narrative-branches/canonical/messages", params={"character_id": branch["character_id"]})
    assert history.status_code == 200
    assert len(history.json()["messages"]) == 2
