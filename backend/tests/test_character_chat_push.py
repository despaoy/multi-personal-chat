"""Outreach remains authoritative when browser notification delivery fails."""

import base64
from types import SimpleNamespace

import pytest
import requests
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.character_chat import repository, router
from app.dependencies import get_current_user
from character.library import CharacterSettings
from db.database import SQLiteDB
from repositories import character_chat_push
from repositories.character_chat_proactive import CharacterChatProactiveRepository
from repositories.character_chat_push import CharacterChatPushRepository
from repositories.character_library import CharacterNotFoundError
from services.character_web_push import WebPushConfig, _PushSession, validate_endpoint


def encoded(value):
    return base64.urlsafe_b64encode(value).decode().rstrip("=")


def device_keys():
    public = ec.generate_private_key(ec.SECP256R1()).public_key().public_bytes(Encoding.X962, PublicFormat.UncompressedPoint)
    return encoded(public), encoded(bytes(16))


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(character_chat_push, "time", lambda: 100)
    database = SQLiteDB(tmp_path / "push.db")
    proactive = CharacterChatProactiveRepository(database)
    character = proactive.chats.characters.create("a", CharacterSettings(name="角色", persona="直率", greeting="你好"))
    session = proactive.chats.create("a", character["id"], "会话")
    settings = dict(enabled=True, cadence_hours=24, timezone="UTC", quiet_start=None, quiet_end=None)
    proactive.set_policy("a", session["id"], 0, **settings)
    try:
        yield proactive, CharacterChatPushRepository(database), session, settings
    finally:
        database.close_connection()


def test_retry_never_duplicates_application_message(setup):
    proactive, push, session, _ = setup
    identity = session["id"]
    endpoint = "https://fcm.googleapis.com/fcm/send/fixture"
    push.register("a", identity, endpoint, *device_keys())
    proactive.schedule("a", identity, 1, "message", "event", "PRIVATE TOPIC", 101, now=100)
    assert proactive.deliver_due(now=101) == 1
    received = []

    class Sender:
        def send(self, address, public_key, auth, tag):
            assert address == endpoint
            received.append(tag)
            return 503 if len(received) == 1 else 201

    for now in (101, 110, 161, 250):
        push.deliver(Sender(), now=now)
    assert len(received) == 2 and received[0] == received[1]
    message, = proactive.view("a", identity)["jobs"]
    assert message["status"] == "sent" and "PRIVATE TOPIC" in message["content"]
    assert len(proactive.inbox("a")) == 1


def test_quiet_candidates_do_not_starve_later_devices_or_expand_send_budget(setup):
    proactive, push, original, settings = setup
    quiet_begins = 22 * 3600
    keys = device_keys()
    expected = set()
    for index in range(13):
        quiet = index < 10
        session = proactive.chats.create("a", original["character_id"], f"session-{index}")
        policy = {**settings, "quiet_start": 22 * 60, "quiet_end": 8 * 60} if quiet else settings
        proactive.set_policy("a", session["id"], 0, **policy)
        for device in range(5):
            endpoint = f"https://fcm.googleapis.com/fcm/send/fixture-{index}-{device}"
            push.register("a", session["id"], endpoint, *keys)
            if not quiet:
                expected.add(endpoint)
        due = quiet_begins - 1 if quiet else quiet_begins + 1
        proactive.schedule("a", session["id"], 1, "message", "event", "fixture", due, now=100)

    assert proactive.deliver_due(now=quiet_begins - 1) == 10
    assert proactive.deliver_due(now=quiet_begins + 1) == 3
    received = []

    class Sender:
        def send(self, endpoint, p256dh, auth, tag):
            received.append(endpoint)
            return 201

    for count in (12, 15, 15):
        push.deliver(Sender(), now=quiet_begins + 1)
        assert len(received) == count
    assert set(received) == expected


def test_unregister_and_revocation_prevent_queued_push(setup):
    proactive, push, session, settings = setup
    identity = session["id"]
    endpoint = "https://fcm.googleapis.com/fcm/send/fixture"
    keys = device_keys()
    record = push.register("a", identity, endpoint, *keys)
    with pytest.raises(CharacterNotFoundError):
        push.unregister("other", identity, record["id"])
    proactive.schedule("a", identity, 1, "message", "event", "fixture", 101, now=100)
    proactive.deliver_due(now=101)
    push.unregister("a", identity, record["id"])
    assert not push.has_due(101)
    push.register("a", identity, endpoint, *keys)
    proactive.set_policy("a", identity, 1, **{**settings, "enabled": False})
    assert not push.has_due(101)
    assert len(proactive.inbox("a")) == 1
    proactive.set_policy("a", identity, 2, **settings)
    assert not push.registered("a", identity)


def test_browser_push_http_owner_and_endpoint_validation(setup, monkeypatch):
    proactive, _, session, _ = setup
    owner = "a"
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user] = lambda: {"user_id": owner}
    app.dependency_overrides[repository] = lambda: proactive.chats
    path = f"/api/character-chats/{session['id']}/browser-push"
    public_key, auth = device_keys()
    payload = {"endpoint": "https://fcm.googleapis.com/fcm/send/fixture", "p256dh": public_key, "auth": auth}
    with TestClient(app) as client:
        monkeypatch.delenv("CHARACTER_WEB_PUSH_PRIVATE_KEY", raising=False)
        assert client.get(path).json() == {"public_key": None, "subscription_ids": []}
        assert client.post(path, json=payload).status_code == 503
        monkeypatch.setenv("CHARACTER_WEB_PUSH_PRIVATE_KEY", encoded((1).to_bytes(32, "big")))
        monkeypatch.setenv("CHARACTER_WEB_PUSH_SUBJECT", "mailto:test@example.invalid")
        assert client.post(path, json={**payload, "endpoint": "https://127.0.0.1/fixture"}).status_code == 422
        response = client.post(path, json=payload)
        assert response.status_code == 201
        identity = response.json()["id"]
        assert client.get(path).json()["subscription_ids"] == [identity]
        owner = "other"
        assert client.get(path).status_code == 404
        assert client.post(path, json=payload).status_code == 404
        assert client.delete(f"{path}/{identity}").status_code == 404
        owner = "a"
        assert client.delete(f"{path}/{identity}").status_code == 200
        assert client.get(path).json()["subscription_ids"] == []


@pytest.mark.parametrize("endpoint", [
    "http://fcm.googleapis.com/fcm/send/fixture", "https://127.0.0.1/fixture",
    "https://fcm.googleapis.com@127.0.0.1/fixture", "https://fcm.googleapis.com.example.org/fixture",
    "https://fcm.googleapis.com:8443/fixture",
])
def test_endpoint_validation_rejects_non_provider_authorities(endpoint):
    with pytest.raises(ValueError):
        validate_endpoint(endpoint)


def test_web_push_encrypts_generic_payload_and_disallows_redirects(monkeypatch):
    monkeypatch.setenv("CHARACTER_WEB_PUSH_PRIVATE_KEY", encoded((1).to_bytes(32, "big")))
    monkeypatch.setenv("CHARACTER_WEB_PUSH_SUBJECT", "mailto:test@example.invalid")
    config = WebPushConfig.from_env()
    endpoint = "https://fcm.googleapis.com/fcm/send/fixture"
    tag = "f" * 32
    calls = []

    def capture(client, method, address, **kwargs):
        assert not client.trust_env
        assert method == "POST" and address == endpoint
        assert kwargs["allow_redirects"] is False
        headers = {key.lower(): value for key, value in kwargs["headers"].items()}
        assert headers["topic"] == tag
        assert headers["content-encoding"] == "aes128gcm"
        assert headers["authorization"].startswith("vapid ")
        assert tag.encode() not in kwargs["data"]
        calls.append(address)
        return SimpleNamespace(status_code=201, text="", headers={})

    monkeypatch.setattr(requests.Session, "request", capture)
    assert config.send(endpoint, *device_keys(), tag) == 201
    assert calls == [endpoint]
    with _PushSession() as session, pytest.raises(ValueError):
        session.post("https://127.0.0.1/private")
