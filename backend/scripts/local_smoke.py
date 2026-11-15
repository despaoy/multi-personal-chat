"""Isolated startup and explicit unavailable-provider smoke checks.

This verifies no fabricated successful reply or history is produced in mock mode.
It does not validate real model inference.
"""

from __future__ import annotations

import os
import base64
import secrets
import sys
import time
from tempfile import mkdtemp
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

TEMP_ROOT = BACKEND_ROOT / ".test_tmp"
TEMP_ROOT.mkdir(parents=True, exist_ok=True)
# Retain diagnostic artifacts locally; worker-thread SQLite handles may remain
# open until process exit on Windows, so do not delete from an exit finalizer.
TEMP_DIR = Path(mkdtemp(prefix="api-smoke-", dir=TEMP_ROOT))

os.environ["ENVIRONMENT"] = "development"
os.environ["MODEL_PROVIDER"] = "mock"
os.environ["VLLM_ENABLED"] = "false"
os.environ["VLLM_BASE_URLS"] = ""
os.environ["VLLM_BASE_URL"] = ""
os.environ["SECURITY_MIDDLEWARE_ENABLED"] = "false"
os.environ["ASTRBOT_ENABLED"] = "true"
os.environ["ASTRBOT_INTEGRATION_TOKEN"] = "local-verify-token"
os.environ["INTEGRATION_SIGNATURE_REQUIRED"] = "false"
os.environ["DATABASE_PATH"] = str(TEMP_DIR / "local_verify.db")
os.environ["DATABASE_URL"] = ""
os.environ["USE_POSTGRESQL"] = "false"
os.environ["BACKUP_DIR"] = str(TEMP_DIR / "backups")
# Explicit synthetic fixture key, paired with this disposable database only.
os.environ["ENCRYPTION_KEY"] = base64.b64encode(secrets.token_bytes(32)).decode("ascii")
os.environ["MULTIPERSONAL_BACKEND_URL"] = "http://127.0.0.1:8000"
os.environ["QQCHAT_BACKEND_URL"] = "http://127.0.0.1:8000"  # legacy alias
os.environ["JWT_SECRET"] = "local-smoke-jwt-secret-value-32-chars"
os.environ["ALLOWED_ORIGINS"] = "http://localhost:5000"
os.environ["LOG_LEVEL"] = "INFO"

from fastapi.testclient import TestClient

from app.dependencies import get_current_admin, get_current_user
from app.main import app


async def _local_user_override():
    return {"user_id": 0, "username": "local-smoke"}


def _assert_status(resp, expected: int, label: str) -> None:
    if resp.status_code != expected:
        raise AssertionError(f"{label} returned {resp.status_code}: {resp.text[:500]}")


def main() -> None:
    app.dependency_overrides[get_current_user] = _local_user_override
    app.dependency_overrides[get_current_admin] = _local_user_override
    message_id = f"local-smoke-{int(time.time())}"
    payload = {
        "platform": "telegram",
        "adapter": "telegram",
        "messageId": message_id,
        "conversationId": "local-smoke-chat",
        "conversationType": "private",
        "senderId": "local-user",
        "senderName": "Local User",
        "text": "local smoke hello",
        "raw": {"conversationName": "Local Smoke", "timestamp": int(time.time())},
    }

    with TestClient(app) as client:
        health = client.get("/health")
        _assert_status(health, 200, "health")
        if health.json().get("status") != "healthy":
            raise AssertionError(f"health payload is unexpected: {health.text}")

        event = client.post(
            "/api/integrations/astrbot/messages",
            headers={"X-Integration-Token": "local-verify-token"},
            json=payload,
        )
        _assert_status(event, 503, "unavailable mock provider")
        if "mock" not in str(event.json().get("detail", "")):
            raise AssertionError("Unavailable mock provider must return an explicit explanation")

        history = client.get("/api/messages", params={"platform": "telegram", "limit": 5})
        _assert_status(history, 200, "history query")
        messages = history.json().get("messages", [])
        if any(item.get("sourceMessageId") == message_id for item in messages):
            raise AssertionError("Rejected mock generation must not create a completed conversation")

    app.dependency_overrides.clear()
    print("Local startup and unavailable-provider smoke checks passed; real inference was not exercised.")


if __name__ == "__main__":
    main()
