"""Versioned conversation mechanics through real storage and HTTP contracts."""

import asyncio
import json
import sqlite3
from contextlib import closing

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.character_chat import repository, router, service
from app.dependencies import get_current_user
from app.runtime import RuntimeContainer
from character.library import CharacterSettings
from db.database import SQLiteDB
from db.errors import RevisionConflictError
from repositories.character_chat import CharacterChatRepository
from repositories.character_library import CharacterNotFoundError
from services.character_chat import CharacterChatService


def test_enrichment_retry_http_ownership_and_revision(repo):
    from repositories.character_chat_enrichment import CharacterChatEnrichmentRepository

    entry = character(repo)
    session = repo.create("1", entry["id"], "后台重试")
    turn = repo.commit_turn(session, "request", "消息", "回复", "test", enqueue_enrichment=True)
    jobs = CharacterChatEnrichmentRepository(repo.database)
    assert jobs.fail(turn["id"], 2) == "failed"
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user] = lambda: {"user_id": "1"}
    app.dependency_overrides[repository] = lambda: repo
    url = f"/api/character-chats/{session['id']}"
    with TestClient(app) as client:
        assert client.get(url).json()["enrichment_retryable"]
        app.dependency_overrides[get_current_user] = lambda: {"user_id": "2"}
        assert client.post(url + "/enrichment/retry", json={"revision": 3}).status_code == 404
        app.dependency_overrides[get_current_user] = lambda: {"user_id": "1"}
        assert client.post(url + "/enrichment/retry", json={"revision": 2}).status_code == 409
        assert client.post(url + "/enrichment/retry", json={"revision": 3}).status_code == 200
        assert client.post(url + "/enrichment/retry", json={"revision": 3}).status_code == 409
        current = client.get(url).json()
        assert current["enrichment_pending"] and not current["enrichment_retryable"]
        assert len(current["turns"]) == 1


def test_provider_path_compiles_prompt_and_records_usage(repo, monkeypatch):
    from functools import partial
    from types import SimpleNamespace

    from inference import model_manager
    from inference.usage import begin_completion
    from services.character_chat import complete_with_provider

    calls = []

    class Provider:
        name = "openai_compat"
        base_url = "https://controlled-test.invalid/v1"
        model = "controlled-test"
        api_key = "synthetic"

        def snapshot(self):
            return self

        def get_status(self):
            return {"modelName": self.model}

        async def async_complete(
            self,
            messages,
            *,
            temperature,
            max_tokens,
            top_p,
            frequency_penalty,
            repetition_penalty,
            enable_thinking,
            lora_name,
        ):
            calls.append(messages)
            begin_completion().read({"usage": {"prompt_tokens": 15, "completion_tokens": 5, "total_tokens": 20}})
            return "我们从这里开始。", 0.1

    provider = Provider()
    monkeypatch.setattr(
        model_manager, "get_model_manager", lambda **kwargs: SimpleNamespace(get_current_provider=lambda: provider)
    )
    entry = character(repo)
    session = repo.create("1", entry["id"], "模型接线")
    complete = partial(complete_with_provider, database=repo.database, http_client_pool=None)
    chat = CharacterChatService(repo.database, complete)
    result = asyncio.run(chat.generate("1", session["id"], revision=1, request_id="provider-path", message="你好"))
    assert result["reply"] == "我们从这里开始。"
    assert "谨慎" in calls[0][0]["content"]
    assert "只复述已有对话、设定和核实资料明确支持的细节" in calls[0][0]["content"]
    records = repo.database.execute_sql(
        'SELECT * FROM model_invocations WHERE "traceId"=:trace', {"trace": "provider-path"}
    )
    assert len(records) == 1 and records[0]["totalTokens"] == 20


@pytest.fixture
def repo(tmp_path):
    db = SQLiteDB(tmp_path / "chat.db")
    yield CharacterChatRepository(db)
    db.close_connection()


def character(repo):
    return repo.characters.create("1", CharacterSettings(name="旅人", persona="谨慎", greeting="你好"))


@pytest.mark.parametrize("delete", [False, True])
def test_snapshot_rejects_session_changed_during_turn_read(repo, monkeypatch, delete):
    entry = character(repo)
    session = repo.create("1", entry["id"], "并发读取")
    repo.commit_turn(session, "previous", "原消息", "原回复", "test")
    session = repo.get("1", session["id"])
    assert repo._active_turns("2", session["id"]) == []
    active_turns = repo._active_turns

    def read_then_change(owner, session_id):
        turns = active_turns(owner, session_id)
        if delete:
            repo.delete_sessions(owner, [{"id": session_id, "revision": session["revision"]}])
        else:
            repo.commit_turn(session, "concurrent", "新消息", "新回复", "test")
        return turns

    monkeypatch.setattr(repo, "_active_turns", read_then_change)
    error = CharacterNotFoundError if delete else RevisionConflictError
    with pytest.raises(error):
        repo.snapshot("1", session["id"])


def test_selected_deletion_erases_inactive_history_and_shares_only_in_scope(repo):
    entry = character(repo)
    source = repo.create("1", entry["id"], "删除来源")
    source = repo.manage_memory("1", source["id"], 1, content="喜欢茶")
    source, first = turn(repo, source)
    source = repo.share_memory("1", source["id"], source["state"]["memories"][0]["id"], source["revision"])
    shared_id = repo.shared_memories("1")[0]["id"]
    copied = repo.create("1", entry["id"], "共享副本", shared_memory_ids=[shared_id])
    branch = repo.create(
        "1",
        entry["id"],
        "保留分支",
        source_session_id=source["id"],
        source_revision=source["revision"],
        source_turn_id=first["id"],
        hypothesis="海边",
    )
    source = repo.retract("1", source["id"], first["id"], source["revision"])
    assert repo.database.execute_sql(
        "SELECT id FROM character_chat_turns WHERE session_id=:id AND active=0", {"id": source["id"]}
    )
    result = repo.delete_sessions("1", [{"id": source["id"], "revision": source["revision"]}])
    assert result["deleted_session_ids"] == [source["id"]]
    for table, column in [
        ("character_chat_sessions", "id"),
        ("character_chat_turns", "session_id"),
        ("character_shared_memories", "source_session_id"),
    ]:
        assert not repo.database.execute_sql(f"SELECT * FROM {table} WHERE {column}=:id", {"id": source["id"]})
    assert repo.get("1", branch["id"])["baseline"]["history"][0]["message"] == "你好"
    assert repo.get("1", copied["id"])["state"]["memories"][0]["content"] == "喜欢茶"
    with pytest.raises(RevisionConflictError):
        repo.commit_turn(source, "late", "在途", "回复", "test")
    with pytest.raises(CharacterNotFoundError):
        repo.create("1", entry["id"], "迟到分支", source_session_id=source["id"], source_revision=source["revision"])
    repo.delete_sessions(
        "1",
        [{"id": branch["id"], "revision": branch["revision"]}, {"id": copied["id"], "revision": copied["revision"]}],
    )
    assert repo.list("1") == []
    assert repo.characters.get("1", entry["id"])
    assert repo.database.execute_sql("SELECT * FROM character_chat_history_path") == []
    assert repo.database.execute_sql("SELECT * FROM character_chat_history_nodes") == []


def test_migrates_legacy_inherited_history_on_sqlite_reopen(tmp_path):
    path = tmp_path / "old-history.sqlite"
    db = SQLiteDB(path)
    try:
        repo = CharacterChatRepository(db)
        entry = character(repo)
        source = repo.create("1", entry["id"], "来源")
        source, first = turn(repo, source, "第一条")
        source, second = turn(repo, source, "第二条")
        branch = repo.create("1", entry["id"], "旧分支", source_session_id=source["id"],
                             source_revision=source["revision"], source_turn_id=second["id"])
        baseline = branch["baseline"]
        baseline["history"][0].pop("state_after")
        baseline["history"][0].pop("character_revision")
        baseline["history"][1]["evidence"] = {"sources": [{"title": "旧出处", "excerpt": "旧出处正文"}]}
        source_baseline = source["baseline"]
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.executemany("UPDATE character_chat_sessions SET baseline_json=? WHERE id=?", [
                (json.dumps(baseline, ensure_ascii=False), branch["id"]),
                (json.dumps(source_baseline, ensure_ascii=False), source["id"]),
            ])
            connection.execute("DROP TABLE character_chat_history_path")
            connection.execute("DROP TABLE character_chat_history_nodes")
    finally:
        db.close_connection()

    for _ in range(2):
        db = SQLiteDB(path)
        try:
            repo = CharacterChatRepository(db)
            assert repo.get("1", branch["id"])["baseline"] == baseline
            assert repo.get("1", source["id"])["baseline"] == source_baseline
            rows = db.execute_sql("SELECT baseline_json FROM character_chat_sessions WHERE id=:id", {"id": branch["id"]})
            assert "旧出处正文" not in rows[0]["baseline_json"]
            assert len(db.execute_sql("SELECT * FROM character_chat_history_path")) == 2
            with pytest.raises(ValueError, match="未保存逐轮状态"):
                repo.create("1", entry["id"], "缺少旧状态", source_session_id=branch["id"],
                            source_revision=branch["revision"], source_turn_id=first["id"])
        finally:
            db.close_connection()


def test_delete_selected_http_ownership_and_atomic_conflicts(repo):
    entry = character(repo)
    first = repo.create("1", entry["id"], "一")
    second = repo.create("1", entry["id"], "二")
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user] = lambda: {"user_id": "1"}
    app.dependency_overrides[repository] = lambda: repo
    selections = [{"id": first["id"], "revision": 1}, {"id": second["id"], "revision": 2}]
    with TestClient(app) as client:
        assert client.post("/api/character-chats/delete-selected", json={"sessions": selections}).status_code == 409
        assert repo.get("1", first["id"])["revision"] == 1
        assert repo.get("1", second["id"])["revision"] == 1
        assert (
            client.post("/api/character-chats/delete-selected", json={"sessions": [selections[0]] * 2}).status_code
            == 422
        )
        app.dependency_overrides[get_current_user] = lambda: {"user_id": "2"}
        assert (
            client.post("/api/character-chats/delete-selected", json={"sessions": [selections[0]]}).status_code == 404
        )
        assert len(repo.list("1")) == 2
        app.dependency_overrides[get_current_user] = lambda: {"user_id": "1"}
        selections[1]["revision"] = 1
        assert client.post("/api/character-chats/delete-selected", json={"sessions": selections}).status_code == 200


def test_deletion_preview_finds_transitive_branches_and_unshared_cross_character_copies(repo):
    entry = character(repo)
    other_entry = repo.characters.create("1", CharacterSettings(name="朋友", persona="热情", greeting="你好"))
    source = repo.create("1", entry["id"], "来源")
    source = repo.manage_memory("1", source["id"], 1, content="只供验证的私密记忆")
    source, first = turn(repo, source, text="只供验证的私密原话")
    source = repo.share_memory("1", source["id"], source["state"]["memories"][0]["id"], source["revision"])
    shared_id = repo.shared_memories("1")[0]["id"]
    copied = repo.create("1", other_entry["id"], "其他角色副本", shared_memory_ids=[shared_id])
    repo.unshare_memory("1", shared_id)
    continuation = repo.create("1", entry["id"], "延续", source_session_id=source["id"],
                               source_revision=source["revision"])
    nested = repo.create("1", entry["id"], "分支", source_session_id=continuation["id"],
                         source_revision=continuation["revision"], source_turn_id=first["id"])
    unrelated = repo.create("1", entry["id"], "独立会话")
    other_account_entry = repo.characters.create("2", CharacterSettings(name="其他账号", persona="谨慎", greeting="你好"))
    other_account = repo.create("2", other_account_entry["id"], "其他账号会话")

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user] = lambda: {"user_id": "1"}
    app.dependency_overrides[repository] = lambda: repo
    picks = [{"id": source["id"], "revision": source["revision"]}]
    with TestClient(app) as client:
        url = "/api/character-chats/delete-preview"
        response = client.post(url, json={"sessions": picks})
        assert response.status_code == 200
        result = response.json()
        assert [item["id"] for item in result["selected"]] == [source["id"]]
        assert {item["id"] for item in result["related"]} == {continuation["id"], nested["id"], copied["id"]}
        assert all(set(item) == {"id", "character_id", "title", "revision", "mode"}
                   for item in result["selected"] + result["related"])
        assert "只供验证" not in response.text
        assert client.post(url, json={"sessions": [{"id": source["id"], "revision": 1}]}).status_code == 409
        assert client.post(url, json={"sessions": [{"id": other_account["id"], "revision": 1}]}).status_code == 404
        assert client.post(url, json={"sessions": picks * 2}).status_code == 422
        assert len(repo.list("1")) == 5

        copied_pick = {key: copied[key] for key in ("id", "revision")}
        assert client.post("/api/character-chats/delete-selected", json={"sessions": [*picks, copied_pick]}).status_code == 200
        assert {item["id"] for item in repo.list("1")} == {unrelated["id"], continuation["id"], nested["id"]}


def turn(repo, session, text="你好", state=None):
    result = repo.commit_turn(session, str(session["revision"]), text, "收到", "controlled-test", state=state)
    return repo.get("1", session["id"]), result


def test_initial_and_explicit_inheritance_versions(repo):
    entry = character(repo)
    original = repo.create("1", entry["id"], "初识")
    state = {"relationship": {"stage": "familiar", "summary": "一起旅行"}, "memories": [{"content": "喜欢茶"}]}
    original, _ = turn(repo, original, state=state)
    fresh = repo.create("1", entry["id"], "重新认识")
    assert fresh["baseline"]["history"] == []
    assert fresh["state"]["memories"] == []
    inherited = repo.create("1", entry["id"], "继续", source_session_id=original["id"], source_revision=2)
    assert inherited["state"] == state
    assert len(inherited["baseline"]["history"]) == 1
    updated = CharacterSettings(name="新旅人", persona="更开朗", greeting="又见面了")
    repo.characters.update("1", entry["id"], 1, updated)
    assert repo.get("1", original["id"])["character"]["settings"]["name"] == "旅人"
    with pytest.raises(ValueError, match="新版本"):
        repo.create("1", entry["id"], "冲突", source_session_id=original["id"], source_revision=2)
    accepted = repo.create(
        "1",
        entry["id"],
        "确认",
        source_session_id=original["id"],
        source_revision=2,
        accept_settings_change=True,
        expected_character_revision=2,
    )
    assert accepted["character_revision"] == 2
    assert accepted["state"] == state
    with pytest.raises(CharacterNotFoundError):
        repo.get("2", original["id"])


def test_retract_restores_prior_state_and_preserves_branch(repo):
    entry = character(repo)
    session = repo.create("1", entry["id"], "对话")
    first_state = {"relationship": {"stage": "acquaintance", "summary": "初次见面"}, "memories": []}
    session, first = turn(repo, session, state=first_state)
    session, second = turn(repo, session, "一起出发", state={"relationship": {"stage": "close"}, "memories": ["约定"]})
    branch = repo.create(
        "1",
        entry["id"],
        "如果留下",
        source_session_id=session["id"],
        source_revision=3,
        source_turn_id=first["id"],
        hypothesis="留在这里",
    )
    assert branch["state"] == first_state
    assert len(branch["baseline"]["history"]) == 1
    with pytest.raises(ValueError, match="假想分支"):
        repo.create("1", entry["id"], "普通会话", source_session_id=branch["id"], source_revision=1)
    retracted = repo.retract("1", session["id"], second["id"], 3)
    assert retracted["state"] == first_state
    repo.retract("1", session["id"], first["id"], retracted["revision"])
    assert repo.turns("1", session["id"]) == []
    assert repo.get("1", branch["id"])["baseline"]["history"][0]["id"] == first["id"]
    with pytest.raises(CharacterNotFoundError):
        repo.create(
            "1", entry["id"], "失效起点", source_session_id=session["id"], source_revision=5, source_turn_id=first["id"]
        )


def test_generation_edits_idempotency_and_inflight_conflict(repo):
    entry = character(repo)
    session = repo.create("1", entry["id"], "聊天")
    captured = []

    async def complete(request, session, request_id):
        captured.append(request)
        return "回答 " + request.message, "controlled-test"

    chat = CharacterChatService(repo.database, complete)

    async def scenario():
        first = await chat.generate("1", session["id"], revision=1, request_id="one", message="同意")
        again = await chat.generate("1", session["id"], revision=1, request_id="one", message="同意")
        assert again["id"] == first["id"] and len(captured) == 1
        await chat.generate("1", session["id"], revision=2, request_id="two", message="明天见")
        await chat.generate(
            "1", session["id"], revision=3, request_id="edit", message="不同意", replace_turn_id=first["id"]
        )
        assert not any(item["content"] in {"同意", "明天见"} for item in captured[-1].history)
        assert [row["message"] for row in repo.turns("1", session["id"])] == ["不同意"]
        with pytest.raises(RevisionConflictError):
            await chat.generate("1", session["id"], revision=1, request_id="one", message="同意")

        async def mutating_completion(request, snapshot, request_id):
            active = repo.turns("1", session["id"])[0]
            repo.retract("1", session["id"], active["id"], snapshot["revision"])
            return "过期回复", "controlled-test"

        chat.complete = mutating_completion
        with pytest.raises(RevisionConflictError):
            await chat.generate("1", session["id"], revision=4, request_id="race", message="继续")
        assert repo.turns("1", session["id"]) == []
        assert repo.prior("1", session["id"], "race") is None

    asyncio.run(scenario())


def test_http_session_generation_and_scope(repo):
    class Runtime:
        async def check_rate_limits(self, *args):
            pass

        def priority_for(self, *args):
            return 1

        async def submit(self, factory, **kwargs):
            return await factory()

    async def complete(request, session, request_id):
        assert "谨慎" in request.persona_prompt
        return "你好，旅行者。", "controlled-test"

    app = FastAPI()
    app.include_router(router)
    app.state.runtime_container = RuntimeContainer(
        db=repo.database, is_pg_mode=lambda: False, inference_runtime=Runtime()
    )
    app.dependency_overrides[get_current_user] = lambda: {"user_id": "1"}
    app.dependency_overrides[repository] = lambda: repo
    app.dependency_overrides[service] = lambda: CharacterChatService(repo.database, complete)
    entry = character(repo)
    with TestClient(app) as client:
        response = client.post("/api/character-chats", json={"character_id": entry["id"], "title": "开始"})
        assert response.status_code == 201
        session = response.json()
        url = "/api/character-chats/" + session["id"]
        reply = client.post(url + "/messages", json={"revision": 1, "request_id": "test", "message": "你好"})
        assert reply.status_code == 200, reply.text
        assert reply.json()["reply"] == "你好，旅行者。"
        assert len(client.get(url).json()["turns"]) == 1
        added = client.post(url + "/memories", json={"revision": 2, "content": "喜欢茶"})
        assert added.status_code == 201, added.text
        memory_id = added.json()["state"]["memories"][0]["id"]
        assert client.put(url + "/memories/" + memory_id, json={"revision": 2, "content": "咖啡"}).status_code == 409
        assert client.post(url + "/memories", json={"revision": 3, "content": "  "}).status_code == 422
        app.dependency_overrides[get_current_user] = lambda: {"user_id": "2"}
        assert client.get(url).status_code == 404
        assert client.request("DELETE", url + "/memories/" + memory_id, json={"revision": 3}).status_code == 404
        assert (
            client.post(url + "/messages", json={"revision": 2, "request_id": "hack", "message": "hi"}).status_code
            == 404
        )


def test_memory_correction_deletion_and_branch_snapshots(repo):
    import json

    entry = character(repo)
    session = repo.create("1", entry["id"], "记忆")
    session = repo.manage_memory("1", session["id"], 1, content="喜欢茶")
    memory_id = session["state"]["memories"][0]["id"]
    session, first = turn(repo, session)
    session, second = turn(repo, session, "继续")
    branch = repo.create(
        "1",
        entry["id"],
        "另一种走向",
        source_session_id=session["id"],
        source_revision=session["revision"],
        source_turn_id=second["id"],
        hypothesis="去海边",
    )
    session = repo.manage_memory("1", session["id"], session["revision"], memory_id=memory_id, content="喜欢咖啡")
    session = repo.retract("1", session["id"], second["id"], session["revision"])
    assert session["state"]["memories"][0]["content"] == "喜欢咖啡"
    assert repo.get("1", branch["id"])["state"]["memories"][0]["content"] == "喜欢茶"
    session = repo.manage_memory("1", session["id"], session["revision"], memory_id=memory_id)
    session = repo.retract("1", session["id"], first["id"], session["revision"])
    assert session["state"]["memories"] == []
    assert session["baseline"]["state"]["memories"] == []
    for row in repo.database.execute_sql(
        "SELECT state_after_json FROM character_chat_turns WHERE session_id=:id", {"id": session["id"]}
    ):
        assert json.loads(row["state_after_json"])["memories"] == []
    assert repo.get("1", branch["id"])["state"]["memories"]


def test_manual_memory_anchor_and_prompt_admission(repo):
    from services.character_chat import prepare_request

    entry = character(repo)
    session = repo.create("1", entry["id"], "记忆起点")
    session, first = turn(repo, session)
    stale = session
    session = repo.manage_memory("1", session["id"], session["revision"], content="喜欢茶")
    with pytest.raises(RevisionConflictError):
        repo.commit_turn(stale, "stale-generation", "旧请求", "旧回答", "test")
    memory = session["state"]["memories"][0]
    assert memory["source_turn_id"] == first["id"]
    session["state"]["memories"].append({"id": "pending", "content": "不确定的偏好", "status": "pending"})
    request = prepare_request(session, [], "你好")
    assert "喜欢茶" in request.character_context.conversation_reference_context
    assert "不确定的偏好" not in request.character_context.conversation_reference_context
    session = repo.retract("1", session["id"], first["id"], session["revision"])
    assert session["state"]["memories"] == []


def test_branch_from_inherited_point_preserves_original_version_and_state(repo):
    entry = character(repo)
    original = repo.create("1", entry["id"], "旧版本")
    original = repo.manage_memory("1", original["id"], 1, content="喜欢茶")
    original, first = turn(repo, original, "第一天")
    original, second = turn(repo, original, "第二天")
    repo.characters.update("1", entry["id"], 1, CharacterSettings(name="新旅人", persona="开朗", greeting="你好"))
    continuation = repo.create(
        "1",
        entry["id"],
        "新版本延续",
        source_session_id=original["id"],
        source_revision=original["revision"],
        accept_settings_change=True,
        expected_character_revision=2,
    )
    # The source can be retracted later; inherited checkpoints are independent.
    repo.retract("1", original["id"], first["id"], original["revision"])
    memory_id = continuation["state"]["memories"][0]["id"]
    continuation = repo.manage_memory("1", continuation["id"], 1, memory_id=memory_id, content="喜欢咖啡")
    branch = repo.create(
        "1",
        entry["id"],
        "回到第一天",
        source_session_id=continuation["id"],
        source_revision=continuation["revision"],
        source_turn_id=first["id"],
        hypothesis="下雨了",
    )
    assert branch["character_revision"] == 1
    assert branch["state"]["memories"][0]["content"] == "喜欢咖啡"
    assert [item["id"] for item in branch["baseline"]["history"]] == [first["id"]]
    path_nodes = repo.database.execute_sql(
        "SELECT session_id,node_key FROM character_chat_history_path WHERE position=0"
    )
    assert {row["session_id"] for row in path_nodes} == {continuation["id"], branch["id"]}
    assert len({row["node_key"] for row in path_nodes}) == 1
    branch, local = turn(repo, branch, "避雨")
    nested = repo.create(
        "1",
        entry["id"],
        "子分支",
        source_session_id=branch["id"],
        source_revision=branch["revision"],
        source_turn_id=local["id"],
        hypothesis="雨停了",
    )
    assert nested["hypothesis"] == "下雨了\n后续调整：雨停了"
    assert second["id"] not in [item["id"] for item in nested["baseline"]["history"]]
    continuation = repo.manage_memory("1", continuation["id"], continuation["revision"], memory_id=memory_id)
    erased = repo.create(
        "1",
        entry["id"],
        "已删记忆后分支",
        source_session_id=continuation["id"],
        source_revision=continuation["revision"],
        source_turn_id=first["id"],
    )
    assert erased["state"]["memories"] == []
    assert repo.get("1", branch["id"])["state"]["memories"]


def test_continuation_preview_and_version_confirmation(repo):
    entry = character(repo)
    source = repo.create("1", entry["id"], "来源")
    repo.characters.update("1", entry["id"], 1, CharacterSettings(name="旅人", persona="开朗", greeting="你好"))
    preview = repo.continuation_preview("1", source["id"])
    assert preview["changes"] == {"persona": {"previous": "谨慎", "next": "开朗"}}
    assert preview["source_revision"] == 1 and preview["character_revision"] == 2
    with pytest.raises(CharacterNotFoundError):
        repo.continuation_preview("2", source["id"])
    with pytest.raises(ValueError, match="指定确认"):
        repo.create(
            "1", entry["id"], "缺少版本", source_session_id=source["id"], source_revision=1, accept_settings_change=True
        )
    repo.characters.update("1", entry["id"], 2, CharacterSettings(name="旅人", persona="冷淡", greeting="你好"))
    with pytest.raises(RevisionConflictError):
        repo.create(
            "1",
            entry["id"],
            "过期确认",
            source_session_id=source["id"],
            source_revision=1,
            accept_settings_change=True,
            expected_character_revision=2,
        )
    assert len(repo.list("1")) == 1
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user] = lambda: {"user_id": "1"}
    app.dependency_overrides[repository] = lambda: repo
    with TestClient(app) as client:
        result = client.get(f"/api/character-chats/{source['id']}/continuation-preview")
        assert result.status_code == 200 and result.json()["character_revision"] == 3


def test_continuation_choices_isolate_memory_relationship_and_raw_history(repo):
    from services.character_chat import prepare_request

    entry = character(repo)
    source = repo.create("1", entry["id"], "来源")
    source = repo.manage_memory("1", source["id"], 1, content="喜欢咖啡")
    excluded_id = source["state"]["memories"][0]["id"]
    source = repo.manage_memory("1", source["id"], source["revision"], content="喜欢旅行")
    state = source["state"]
    state["relationship"] = {"stage": "close", "summary": "多年好友"}
    source, first = turn(repo, source, "我们以前一起喝咖啡", state=state)
    choices = dict(
        source_session_id=source["id"],
        source_revision=source["revision"],
        excluded_memory_ids=[excluded_id],
        reset_relationship=True,
    )
    kept_history = repo.create("1", entry["id"], "筛选", **choices)
    assert kept_history["state"]["relationship"]["stage"] == "stranger"
    assert [item["content"] for item in kept_history["state"]["memories"]] == ["喜欢旅行"]
    branch = repo.create(
        "1",
        entry["id"],
        "筛选后分支",
        source_session_id=kept_history["id"],
        source_revision=1,
        source_turn_id=first["id"],
    )
    assert branch["state"] == kept_history["state"]
    clean = repo.create("1", entry["id"], "不带原文", inherit_history=False, **choices)
    request = prepare_request(clean, [], "你好")
    assert clean["baseline"]["history"] == []
    assert "咖啡" not in str(request.history)
    assert "咖啡" not in request.character_context.conversation_reference_context
    clean, local = turn(repo, clean)
    clean = repo.retract("1", clean["id"], local["id"], clean["revision"])
    assert clean["state"] == kept_history["state"]
    unchanged = repo.get("1", source["id"])
    assert unchanged["state"]["relationship"]["stage"] == "close"
    assert len(unchanged["state"]["memories"]) == 2
    with pytest.raises(ValueError, match="所选记忆"):
        repo.create(
            "1",
            entry["id"],
            "错误记忆",
            source_session_id=source["id"],
            source_revision=source["revision"],
            excluded_memory_ids=["missing"],
        )
    with pytest.raises(ValueError, match="仅适用于"):
        repo.create("1", entry["id"], "初识不能调整继承", reset_relationship=True)


@pytest.mark.parametrize(
    ("message", "evidence", "kind", "status"),
    [
        ("今天不想喝咖啡", "今天不想喝咖啡", "dislike", "pending"),
        ("我好像喜欢咖啡", "喜欢咖啡", "like", "pending"),
        ("假设我喜欢咖啡", "我喜欢咖啡", "like", "pending"),
        ("我喜欢咖啡吗？", "我喜欢咖啡", "like", "pending"),
        ("我喜欢咖啡", "我喜欢咖啡", "like", "confirmed"),
        ("记住，我平时不喝咖啡", "我平时不喝咖啡", "other_user_fact", "confirmed"),
    ],
)
def test_web_memory_auto_confirmation_requires_user_assertion(repo, message, evidence, kind, status):
    import json

    from character.memory_llm import MemoryLlmConfig
    from services.character_chat import prepare_request
    from services.character_chat_memory import enrich_state

    operation = "ADD"

    class Writer:
        async def complete(self, messages):
            return json.dumps({"memories": [{
                "operation": operation, "kind": kind, "value": "咖啡", "evidence": evidence,
                "confidence": 0.99, "attributed_to": "user",
            }]})

    entry = character(repo)
    session = repo.create("1", entry["id"], "偏好边界")
    config = MemoryLlmConfig(enabled=True, base_url="https://test.invalid", model="test")
    state = asyncio.run(enrich_state(message, session, "memory", config=config, completion=Writer()))
    assert state["memory_processing"] == "completed"
    candidate, = state["memories"]
    assert candidate["status"] == status
    assert candidate["evidence"] == evidence
    context = prepare_request({**session, "state": state}, [], "我们继续聊").character_context
    assert ("咖啡" in context.conversation_reference_context) == (status == "confirmed")
    if kind == "other_user_fact":
        assert candidate["content"] == "用户平时不喝咖啡"
        evidence, operation = "我平时喝咖啡", "PENDING"
        updated = asyncio.run(enrich_state(
            evidence, {**session, "state": state}, "change", config=config, completion=Writer(),
        ))
        first, pending = updated["memories"]
        assert first == candidate
        assert pending["status"] == "pending" and pending["memory_key"] == candidate["memory_key"]


def test_semantic_memory_is_committed_with_turn_and_conflicts_need_confirmation(repo):
    import json
    from functools import partial

    from character.memory_llm import MemoryLlmConfig
    from services.character_chat_memory import enrich_state

    class Writer:
        async def complete(self, messages):
            source = json.loads(messages[-1]["content"])["current_user_message"]
            return json.dumps(
                {
                    "memories": [
                        {
                            "operation": "ADD",
                            "kind": "dislike" if "讨厌" in source else "like",
                            "value": "咖啡",
                            "evidence": source,
                            "confidence": 0.99,
                            "attributed_to": "user",
                        }
                    ]
                }
            )

    async def complete(request, session, request_id):
        return "我记下了。", "test"

    config = MemoryLlmConfig(enabled=True, base_url="https://test.invalid", model="test")
    enrich = partial(enrich_state, config=config, completion=Writer())
    chat = CharacterChatService(repo.database, complete, enrich)
    entry = character(repo)
    session = repo.create("1", entry["id"], "自动记忆")

    async def scenario():
        first = await chat.generate("1", session["id"], revision=1, request_id="like", message="我喜欢咖啡")
        saved = repo.get("1", session["id"])
        assert saved["state"]["memory_processing"] == "completed"
        memory = saved["state"]["memories"][0]
        assert memory["status"] == "confirmed" and memory["source_turn_id"] == first["id"]
        second = await chat.generate("1", session["id"], revision=2, request_id="dislike", message="我讨厌咖啡")
        saved = repo.get("1", session["id"])
        assert [item["status"] for item in saved["state"]["memories"]] == ["confirmed", "pending"]
        candidate = saved["state"]["memories"][-1]
        saved = repo.manage_memory("1", session["id"], 3, memory_id=candidate["id"], content="用户偏好清淡饮料")
        assert len(saved["state"]["memories"]) == 1
        corrected = saved["state"]["memories"][0]
        assert corrected["origin"] == "user" and corrected["evidence"] == candidate["evidence"]
        from services.character_chat import prepare_request

        prompt_context = prepare_request(saved, [], "喝什么？").character_context.conversation_reference_context
        assert "用户偏好清淡饮料" in prompt_context
        assert candidate["evidence"] not in prompt_context
        saved = repo.retract("1", session["id"], second["id"], saved["revision"])
        assert saved["state"]["memories"][0]["id"] == memory["id"]
        repo.retract("1", session["id"], first["id"], saved["revision"])
        assert repo.get("1", session["id"])["state"]["memories"] == []

    asyncio.run(scenario())


def test_memory_extraction_failure_preserves_reply(repo):
    from functools import partial

    from character.memory_llm import MemoryLlmConfig
    from services.character_chat_memory import enrich_state

    class Writer:
        async def complete(self, messages):
            return "invalid json"

    async def complete(request, session, request_id):
        return "正常回复", "test"

    entry = character(repo)
    session = repo.create("1", entry["id"], "提取失败")
    enrich = partial(
        enrich_state,
        config=MemoryLlmConfig(enabled=True, base_url="https://test.invalid", model="test"),
        completion=Writer(),
    )
    chat = CharacterChatService(repo.database, complete, enrich)
    result = asyncio.run(chat.generate("1", session["id"], revision=1, request_id="invalid", message="我喜欢咖啡"))
    assert result["reply"] == "正常回复"
    saved = repo.get("1", session["id"])
    assert saved["state"]["memory_processing"] == "failed" and saved["state"]["memories"] == []


def test_shared_memory_is_explicit_scoped_and_revoked_when_source_changes(repo):
    entry = character(repo)
    other = repo.characters.create("1", CharacterSettings(name="另一个人", persona="热情", greeting="嗨"))
    source = repo.create("1", entry["id"], "来源")
    source, first = turn(repo, source)
    source = repo.manage_memory("1", source["id"], source["revision"], content="喜欢茶")
    memory_id = source["state"]["memories"][0]["id"]
    source = repo.share_memory("1", source["id"], memory_id, source["revision"])
    shared_id = repo.shared_memories("1")[0]["id"]
    assert repo.shared_memories("2") == []
    fresh = repo.create("1", other["id"], "默认初识")
    assert fresh["state"]["memories"] == []
    imported = repo.create("1", other["id"], "明确带入", shared_memory_ids=[shared_id])
    assert imported["state"]["relationship"]["stage"] == "stranger"
    assert imported["state"]["memories"][0]["content"] == "喜欢茶"
    assert imported["state"]["memories"][0]["id"] != memory_id
    source = repo.manage_memory("1", source["id"], source["revision"], memory_id=memory_id, content="喜欢咖啡")
    assert repo.shared_memories("1") == []
    with pytest.raises(CharacterNotFoundError):
        repo.create("1", other["id"], "旧共享失效", shared_memory_ids=[shared_id])
    assert repo.get("1", imported["id"])["state"]["memories"][0]["content"] == "喜欢茶"
    source = repo.share_memory("1", source["id"], memory_id, source["revision"])
    branch = repo.create(
        "1",
        entry["id"],
        "假想",
        source_session_id=source["id"],
        source_revision=source["revision"],
        source_turn_id=first["id"],
    )
    with pytest.raises(ValueError, match="假想"):
        repo.share_memory("1", branch["id"], memory_id, 1)
    repo.retract("1", source["id"], first["id"], source["revision"])
    assert repo.shared_memories("1") == []


def test_shared_memory_http_routes_and_authority(repo):
    entry = character(repo)
    source = repo.create("1", entry["id"], "来源")
    source = repo.manage_memory("1", source["id"], 1, content="喜欢茶")
    memory_id = source["state"]["memories"][0]["id"]
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user] = lambda: {"user_id": "1"}
    app.dependency_overrides[repository] = lambda: repo
    with TestClient(app) as client:
        result = client.post(f"/api/character-chats/{source['id']}/memories/{memory_id}/share", json={"revision": 2})
        assert result.status_code == 200, result.text
        shared = client.get("/api/character-chats/shared-memories").json()["memories"][0]
        app.dependency_overrides[get_current_user] = lambda: {"user_id": "2"}
        assert client.get("/api/character-chats/shared-memories").json() == {"memories": []}
        assert client.delete("/api/character-chats/shared-memories/" + shared["id"]).status_code == 409
        assert len(repo.shared_memories("1")) == 1
        app.dependency_overrides[get_current_user] = lambda: {"user_id": "1"}
        assert client.delete("/api/character-chats/shared-memories/" + shared["id"]).status_code == 200
        assert repo.shared_memories("1") == []


def test_session_pagination_reaches_older_conversations_without_crossing_accounts(repo):
    entry = character(repo)
    created_ids = {repo.create("1", entry["id"], f"会话 {index}")["id"] for index in range(103)}
    other = repo.characters.create("2", CharacterSettings(name="其他账号", persona="谨慎", greeting="你好"))
    repo.create("2", other["id"], "不能读到")
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user] = lambda: {"user_id": "1"}
    app.dependency_overrides[repository] = lambda: repo
    with TestClient(app) as client:
        first = client.get("/api/character-chats", params={"character_id": entry["id"], "limit": 100}).json()[
            "sessions"
        ]
        remaining = client.get(
            "/api/character-chats", params={"character_id": entry["id"], "limit": 100, "offset": 100}
        ).json()["sessions"]
        assert len(first) == 100 and len(remaining) == 3
        assert {item["id"] for item in first + remaining} == created_ids
        assert client.get("/api/character-chats", params={"offset": -1}).status_code == 422


def test_relationship_adjustment_follows_turn_and_branch_boundaries(repo):
    from services.character_chat import prepare_request

    entry = character(repo)
    session = repo.create("1", entry["id"], "关系记录")
    session = repo.update_relationship("1", session["id"], 1, "acquaintance", "已互相介绍")
    session, first = turn(repo, session)
    session, second = turn(repo, session, "一起旅行")
    old_session = session
    session = repo.update_relationship("1", session["id"], session["revision"], "familiar", "一起旅行过")
    request = prepare_request(session, [], "你好")
    assert "一起旅行过" in request.character_context.conversation_reference_context
    with pytest.raises(RevisionConflictError):
        repo.commit_turn(old_session, "stale", "旧请求", "回答", "test")
    before = repo.create(
        "1",
        entry["id"],
        "此前分支",
        source_session_id=session["id"],
        source_revision=session["revision"],
        source_turn_id=first["id"],
    )
    after = repo.create(
        "1",
        entry["id"],
        "此后分支",
        source_session_id=session["id"],
        source_revision=session["revision"],
        source_turn_id=second["id"],
    )
    assert before["state"]["relationship"]["stage"] == "acquaintance"
    assert after["state"]["relationship"]["stage"] == "familiar"
    session = repo.retract("1", session["id"], second["id"], session["revision"])
    assert session["state"]["relationship"]["stage"] == "acquaintance"
    assert repo.get("1", after["id"])["state"]["relationship"]["stage"] == "familiar"
    session = repo.retract("1", session["id"], first["id"], session["revision"])
    assert session["state"]["relationship"]["summary"] == "已互相介绍"


def test_relationship_http_validation_and_ownership(repo):
    entry = character(repo)
    session = repo.create("1", entry["id"], "关系")
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user] = lambda: {"user_id": "1"}
    app.dependency_overrides[repository] = lambda: repo
    url = f"/api/character-chats/{session['id']}/relationship"
    with TestClient(app) as client:
        assert client.put(url, json={"revision": 1, "stage": "invented"}).status_code == 422
        assert client.put(url, json={"revision": 1, "stage": "close", "summary": "x" * 2001}).status_code == 422
        assert client.put(url, json={"revision": 1, "stage": "familiar", "dimensions": {"tone": "invented"}}).status_code == 422
        response = client.put(url, json={"revision": 1, "stage": "familiar", "summary": "测试记录",
                                         "dimensions": {"tone": "restrained", "initiative": "offer"}})
        assert response.status_code == 200 and response.json()["state"]["relationship"]["stage"] == "familiar"
        assert response.json()["state"]["relationship"]["dimensions"] == {"tone": "restrained", "initiative": "offer"}
        assert client.put(url, json={"revision": 1, "stage": "close"}).status_code == 409
        legacy = client.put(url, json={"revision": 2, "stage": "familiar", "summary": "旧客户端"}).json()
        assert legacy["state"]["relationship"]["dimensions"] == {"tone": "restrained", "initiative": "offer"}
        cleared = client.put(url, json={"revision": 3, "stage": "familiar", "dimensions": {"tone": None}}).json()
        assert cleared["state"]["relationship"]["dimensions"] == {"initiative": "offer"}
        app.dependency_overrides[get_current_user] = lambda: {"user_id": "2"}
        assert client.put(url, json={"revision": 4, "stage": "close", "dimensions": {"initiative": "wait"}}).status_code == 404
