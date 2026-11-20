"""Character versions and account isolation without model calls."""

import importlib.util
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.character_library import original_sources, repository, router
from app.dependencies import get_current_user
from character.library import CharacterSettings
from db.database import SQLiteDB
from db.errors import RevisionConflictError
from repositories.character_library import CharacterLibraryRepository, CharacterNotFoundError
from repositories.character_sources import CharacterSourceRepository


@pytest.fixture
def repo(tmp_path):
    db = SQLiteDB(tmp_path / "characters.db")
    yield CharacterLibraryRepository(db)
    db.close_connection()


def settings(name="角色"):
    return CharacterSettings(name=name, persona="谨慎，有自己的立场", greeting="你好")


def test_import_preview_validates_without_persisting_and_rejects_private_fields(repo):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user] = lambda: {'user_id': 'import-owner'}
    app.dependency_overrides[repository] = lambda: repo
    package = {'format': 'multipersonal-character-v1', 'settings': settings().model_dump()}
    with TestClient(app) as client:
        preview = client.post('/api/character-library/import-preview', json=package)
        assert preview.status_code == 200
        assert repo.list('import-owner') == []
        assert client.post('/api/character-library/import-preview', json={**package, 'private_memories': ['secret']}).status_code == 422
        assert client.post('/api/character-library/import-preview', json={**package, 'format': 'unknown-card'}).status_code == 422
        imported = client.post('/api/character-library/import', json=preview.json())
        assert imported.status_code == 201
        assert len(repo.list('import-owner')) == 1
        assert repo.list('other-owner') == []
        exported = client.get(f"/api/character-library/{imported.json()['id']}/export")
        assert exported.json() == preview.json()


def test_confirmed_import_can_retry_without_recreating_or_overwriting_a_role(repo):
    app = FastAPI()
    app.include_router(router)
    owner = "import-owner"
    app.dependency_overrides[get_current_user] = lambda: {"user_id": owner}
    app.dependency_overrides[repository] = lambda: repo
    path = "/api/character-library/import"
    package = {"settings": settings("文件角色").model_dump(), "request_id": "confirm-one"}
    with TestClient(app) as client:
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(lambda _: client.post(path, json=package), range(2)))
        assert [item.status_code for item in results] == [201, 201]
        imported, replay = [item.json() for item in results]
        assert imported["id"] == replay["id"] and len(repo.list(owner)) == 1
        character_id = imported["id"]
        updated = repo.update(owner, character_id, 1, settings("编辑后"))
        assert client.post(path, json=package).json() == updated
        mismatch = {**package, "settings": settings("另一个文件").model_dump()}
        assert client.post(path, json=mismatch).status_code == 409
        assert len(repo.list(owner)) == 1
        intentional_copy = client.post(path, json={**package, "request_id": "confirm-two"})
        assert intentional_copy.status_code == 201 and intentional_copy.json()["id"] != character_id
        owner = "other-owner"
        foreign = client.post(path, json=package)
        assert foreign.status_code == 201 and foreign.json()["id"] != character_id
        assert client.get("/api/character-library/" + character_id).status_code == 404


def test_preset_installation_is_independent_repeatable_and_scope_checked(repo):
    sources = CharacterSourceRepository(repo.database)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user] = lambda: {"user_id": "first"}
    app.dependency_overrides[original_sources] = lambda: sources
    url = "/api/character-library/presets/kamimaho-kisaki"
    with TestClient(app) as client, ThreadPoolExecutor(2) as pool:
        responses = list(pool.map(lambda _: client.post(url), range(2)))
        assert all(response.status_code == 201 for response in responses)
        first, duplicate = [response.json() for response in responses]
        assert first["settings"]["name"] == "月社妃"
        assert first["id"] == duplicate["id"] and len(repo.list("first")) == 1
        edition, = sources.list("first", first["id"], 1)
        assert edition["index_status"] == "pending" and first["share_token"] is None
        corpus = sources.get("first", edition["id"])
        assert "学校用月社" in corpus["sources"][1]["content"]
        assert "https://kagura.rdy.jp/kamimaho/qa_old02.html" in corpus["sources"][1]["content"]
        repo.update("first", first["id"], 1, settings("自己的版本"))
        sources.revoke("first", edition["id"])
        reinstalled = client.post(url).json()
        assert reinstalled["settings"]["name"] == "自己的版本"
        assert sources.list("first", first["id"], 1) == []
        app.dependency_overrides[get_current_user] = lambda: {"user_id": "second"}
        second = client.post(url).json()
        assert second["id"] != first["id"] and second["settings"]["name"] == "月社妃"
        with pytest.raises(CharacterNotFoundError):
            sources.list("second", first["id"], 1)
        app.dependency_overrides[get_current_user] = lambda: {"user_id": "second", "auth_type": "api_key"}
        assert client.post(url).status_code == 403


def test_versions_and_stale_updates(repo):
    created = repo.create("1", settings())
    updated = repo.update("1", created["id"], 1, settings("新名字"))
    assert updated["revision"] == 2
    assert repo.get("1", created["id"], 1)["settings"]["name"] == "角色"
    with pytest.raises(RevisionConflictError):
        repo.update("1", created["id"], 1, settings("覆盖"))
    assert repo.get("1", created["id"])["settings"]["name"] == "新名字"
    with pytest.raises(CharacterNotFoundError):
        repo.update("2", created["id"], 2, settings())
    assert repo.list("2") == []


def test_author_update_requires_explicit_import_and_exact_preview(repo):
    from repositories.character_chat import CharacterChatRepository

    original = repo.create("1", settings())
    token = repo.publish("1", original["id"], 1, True)["share_token"]
    copied = repo.copy_shared("2", token)
    chats = CharacterChatRepository(repo.database)
    session = chats.create("2", copied["id"], "保持原设定")
    repo.update("1", original["id"], 1, settings("私有草稿"))
    preview = repo.source_update("2", copied["id"])
    assert not preview["available"] and preview["changes"] == {}
    repo.publish("1", original["id"], 2, True)
    preview = repo.source_update("2", copied["id"])
    assert preview["available"] and preview["source_revision"] == 2
    assert preview["changes"]["name"] == {"current": "角色", "incoming": "私有草稿"}
    assert repo.get("2", copied["id"])["revision"] == 1
    repo.update("2", copied["id"], 1, settings("本地修改"))
    with pytest.raises(RevisionConflictError):
        repo.import_source_update("2", copied["id"], 1, 2)
    repo.publish("1", original["id"], 2, False)
    assert not repo.source_update("2", copied["id"])["available"]
    with pytest.raises(RevisionConflictError):
        repo.import_source_update("2", copied["id"], 2, 2)
    repo.publish("1", original["id"], 2, True)
    imported = repo.import_source_update("2", copied["id"], 2, 2)
    assert imported["revision"] == 3 and imported["source_revision"] == 2
    assert imported["settings"]["name"] == "私有草稿"
    assert chats.get("2", session["id"])["character"]["settings"]["name"] == "角色"
    assert not repo.source_update("2", copied["id"])["available"]
    with pytest.raises(CharacterNotFoundError):
        repo.source_update("3", copied["id"])


def test_source_update_http_contract(repo):
    original = repo.create("1", settings())
    token = repo.publish("1", original["id"], 1, True)["share_token"]
    copied = repo.copy_shared("2", token)
    repo.update("1", original["id"], 1, settings("发布更新"))
    repo.publish("1", original["id"], 2, True)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user] = lambda: {"user_id": "2"}
    app.dependency_overrides[repository] = lambda: repo
    url = f"/api/character-library/{copied['id']}/source-update"
    with TestClient(app) as client:
        preview = client.get(url).json()
        assert preview["available"]
        confirmation = {"publication_id": preview["publication_id"], "local_corpus_id": preview["local_corpus_id"]}
        assert client.post(url, json={"revision": 1, "source_revision": 3, **confirmation}).status_code == 409
        imported = client.post(url, json={"revision": 1, "source_revision": 2, **confirmation})
        assert imported.status_code == 200 and imported.json()["revision"] == 2
        assert client.post(url, json={"revision": 1, "source_revision": 2, **confirmation}).json()["revision"] == 2
        app.dependency_overrides[get_current_user] = lambda: {"user_id": "3"}
        assert client.get(url).status_code == 404


def test_sharing_is_explicit_snapshot_and_copy_is_independent(repo):
    created = repo.create("1", settings())
    share = repo.publish("1", created["id"], 1, True)
    token = share["share_token"]
    repo.update("1", created["id"], 1, settings("私有草稿"))
    shared = repo.shared(token)
    assert shared["settings"]["name"] == "角色"
    assert "owner_user_id" not in shared
    copied = repo.copy_shared("2", token)
    assert copied["source_revision"] == 1
    repo.publish("1", created["id"], 2, True)
    assert repo.shared(token)["settings"]["name"] == "私有草稿"
    assert repo.get("2", copied["id"])["settings"]["name"] == "角色"
    repo.publish("1", created["id"], 2, False)
    with pytest.raises(CharacterNotFoundError):
        repo.shared(token)
    assert repo.get("2", copied["id"])["revision"] == 1


def test_transaction_failure_rolls_back_character_head(repo):
    created = repo.create("1", settings())
    with pytest.raises(sqlite3.OperationalError):
        repo.database.revision_transaction(
            [
                ("UPDATE character_library SET revision=2 WHERE id=:id", {"id": created["id"]}, 1),
                ("INSERT INTO missing_table VALUES (1)", {}, 1),
            ]
        )
    assert repo.get("1", created["id"])["revision"] == 1


def test_concurrent_edits_only_commit_one_version(repo):
    created = repo.create("1", settings())

    def edit(name):
        try:
            return repo.update("1", created["id"], 1, settings(name))
        except RevisionConflictError:
            return None
        finally:
            repo.database.close_connection()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(edit, ["甲", "乙"]))
    assert sum(result is not None for result in results) == 1
    assert repo.get("1", created["id"])["revision"] == 2


def test_existing_narrative_transaction_keeps_conflict_contract(repo):
    from repositories.narrative import NarrativeConflict

    with pytest.raises(NarrativeConflict):
        repo.database.narrative_transaction(
            [
                ("UPDATE character_library SET revision=2 WHERE id='missing'", {}, 1),
            ]
        )


@pytest.mark.parametrize(
    "migration_name, expected_tables",
    [
        ("017_character_library", {"character_library", "character_library_versions"}),
        ("018_character_chat", {"character_chat_sessions", "character_chat_turns"}),
        ("019_character_shared_memory", {"character_shared_memories"}),
        ("020_character_chat_evidence", {"character_chat_evidence"}),
        ("021_character_chat_generation", {"character_chat_generations"}),
        ("023_character_chat_enrichment", {"character_chat_enrichment_jobs"}),
    ],
)
def test_migration_creates_and_removes_version_tables(tmp_path, migration_name, expected_tables):
    import sqlalchemy as sa
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    path = Path(__file__).parents[1] / f"alembic/versions/{migration_name}.py"
    spec = importlib.util.spec_from_file_location("character_library_migration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    engine = sa.create_engine("sqlite:///" + str(tmp_path / "migration.db"))
    try:
        with engine.begin() as connection:
            module.op = Operations(MigrationContext.configure(connection))
            module.upgrade()
            module.upgrade()
            tables = sa.inspect(connection).get_table_names()
            assert set(tables) == expected_tables
            module.downgrade()
            assert sa.inspect(connection).get_table_names() == []
    finally:
        engine.dispose()


def test_api_ownership_validation_export_and_conflict(repo):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[repository] = lambda: repo
    app.dependency_overrides[get_current_user] = lambda: {"user_id": "1", "role": "user"}
    with TestClient(app) as client:
        assert (
            client.post("/api/character-library", json={"name": " ", "persona": "x", "greeting": "x"}).status_code
            == 422
        )
        response = client.post("/api/character-library", json=settings().model_dump())
        assert response.status_code == 201
        entry = response.json()
        url = "/api/character-library/" + entry["id"]
        package = client.get(url + "/export").json()
        assert set(package) == {"format", "settings"}
        assert client.post("/api/character-library/import", json=package).status_code == 201
        body = {"revision": 1, "settings": settings("二版").model_dump()}
        assert client.put(url, json=body).status_code == 200
        assert client.put(url, json=body).status_code == 409
        app.dependency_overrides[get_current_user] = lambda: {"user_id": "2", "role": "user"}
        assert client.get(url).status_code == 404
        assert client.put(url, json=body).status_code == 404
        app.dependency_overrides[get_current_user] = lambda: {"user_id": None, "auth_type": "api_key"}
        assert client.get("/api/character-library").status_code == 403


@pytest.mark.parametrize("change", ["revoke", "republish", "rotate_token"])
def test_copy_rechecks_publication_in_creation_transaction(repo, monkeypatch, change):
    original = repo.create("author", settings())
    token = repo.publish("author", original["id"], 1, True)["share_token"]
    read_shared = repo.shared

    def change_after_read(value):
        snapshot = read_shared(value)
        if change == "republish":
            repo.update("author", original["id"], 1, settings("new publication"))
            repo.publish("author", original["id"], 2, True)
        else:
            repo.publish("author", original["id"], 1, False)
            if change == "rotate_token":
                repo.publish("author", original["id"], 1, True)
        return snapshot

    monkeypatch.setattr(repo, "shared", change_after_read)
    with pytest.raises(RevisionConflictError):
        repo.copy_shared("recipient", token)
    assert repo.list("recipient") == []
    assert len(repo.database.execute_sql("SELECT * FROM character_library_versions")) == (2 if change == "republish" else 1)


def test_private_draft_does_not_invalidate_published_copy(repo, monkeypatch):
    original = repo.create("author", settings())
    token = repo.publish("author", original["id"], 1, True)["share_token"]
    read_shared = repo.shared

    def edit_after_read(value):
        snapshot = read_shared(value)
        repo.update("author", original["id"], 1, settings("private draft"))
        return snapshot

    monkeypatch.setattr(repo, "shared", edit_after_read)
    copied = repo.copy_shared("recipient", token)
    assert copied["settings"] == settings().model_dump()
    assert copied["source_revision"] == 1
