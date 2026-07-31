from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.messages import router
from app.dependencies import get_current_admin, get_current_user
from app.runtime import RuntimeContainer
from repositories.messages import DatabaseMessageRepository, MessageQuery


class FakeMessageDatabase:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def get_message_count(self) -> int:
        self.calls.append(("count_all", None))
        return 12

    def get_messages_filtered(self, **kwargs):
        self.calls.append(("list", kwargs))
        return [{"id": 7, "message": "hello"}]

    def get_message_count_filtered(self, **kwargs) -> int:
        self.calls.append(("count_filtered", kwargs))
        return 3

    def get_session_summaries(self):
        self.calls.append(("sessions", None))
        return [{"sessionId": "session-1"}]

    def set_session_bot_enabled(self, *args) -> None:
        self.calls.append(("toggle", args))

    def delete_messages_by_filter(self, **kwargs) -> int:
        self.calls.append(("delete_filtered", kwargs))
        return 4

    def delete_message(self, message_id: int) -> bool:
        self.calls.append(("delete", message_id))
        return message_id == 7


async def test_database_message_repository_maps_domain_query_to_database_facade() -> None:
    database = FakeMessageDatabase()
    repository = DatabaseMessageRepository(database)
    query = MessageQuery(
        search="hello",
        session_type="group",
        lora_name="writer",
        session_id="session-1",
        session_name="demo",
        platform="telegram",
    )

    page = await repository.list_page(query, limit=25, offset=50)

    expected_filters = {
        "search": "hello",
        "session_type": "group",
        "lora_name": "writer",
        "session_id": "session-1",
        "session_name": "demo",
        "platform": "telegram",
    }
    assert page.messages == [{"id": 7, "message": "hello"}]
    assert page.total == 3
    assert page.total_all == 12
    assert database.calls == [
        ("count_all", None),
        ("list", {**expected_filters, "limit": 25, "offset": 50}),
        ("count_filtered", expected_filters),
    ]


async def test_database_message_repository_maps_mutations() -> None:
    database = FakeMessageDatabase()
    repository = DatabaseMessageRepository(database)

    await repository.set_session_bot_enabled(
        "session-1",
        False,
        platform="qq",
        conversation_id="group-2",
        conversation_type="group",
    )
    deleted = await repository.delete_filtered(
        MessageQuery(
            search="old",
            session_type="private",
            lora_name="base",
            session_name="archive",
            platform="qq",
        )
    )
    found = await repository.delete(7)

    assert deleted == 4
    assert found is True
    assert database.calls == [
        ("toggle", ("session-1", False, "qq", "group-2", "group")),
        (
            "delete_filtered",
            {
                "search": "old",
                "sessionType": "private",
                "lora": "base",
                "sessionName": "archive",
                "platform": "qq",
            },
        ),
        ("delete", 7),
    ]


def test_message_routes_resolve_repository_from_runtime_container() -> None:
    database = FakeMessageDatabase()
    application = FastAPI()
    application.state.runtime_container = RuntimeContainer(
        db=database,
        is_pg_mode=lambda: False,
        startup_env={},
    )
    application.include_router(router)
    application.dependency_overrides[get_current_user] = lambda: {"user_id": 1, "username": "tester"}
    application.dependency_overrides[get_current_admin] = lambda: {"user_id": 1, "username": "admin"}

    with TestClient(application) as client:
        response = client.get(
            "/api/messages",
            params={"sessionType": "all", "lora": "writer", "platform": "all", "limit": 5, "offset": 10},
        )
        toggle = client.put(
            "/api/sessions/bot-toggle",
            json={
                "sessionId": "session-1",
                "enabled": False,
                "platform": "qq",
                "conversationId": "group-2",
                "conversationType": "group",
            },
        )
        batch = client.request(
            "DELETE",
            "/api/messages/batch",
            json={"sessionType": "all", "lora": "all", "platform": "all"},
        )
        single = client.delete("/api/messages/7")

    assert response.status_code == 200
    assert response.json() == {
        "messages": [{"id": 7, "message": "hello"}],
        "total": 3,
        "total_all": 12,
    }
    assert toggle.status_code == 200
    assert batch.json()["deleted"] == 4
    assert single.status_code == 200

    list_call = next(value for name, value in database.calls if name == "list")
    assert list_call == {
        "search": None,
        "session_type": None,
        "lora_name": "writer",
        "session_id": None,
        "session_name": None,
        "platform": None,
        "limit": 5,
        "offset": 10,
    }
    assert ("toggle", ("session-1", False, "qq", "group-2", "group")) in database.calls
    assert (
        "delete_filtered",
        {"search": None, "sessionType": None, "lora": None, "sessionName": None, "platform": None},
    ) in database.calls
