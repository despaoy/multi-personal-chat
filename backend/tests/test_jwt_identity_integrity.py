"""Malformed identities and internal JWT failures cannot authorize requests."""

import jwt
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app import config
from middleware.security import SecurityMiddleware


@pytest.mark.parametrize("claims", [{}, {"sub": ""}, {"sub": " "}, {"sub": None}, {"sub": 3}])
def test_jwt_requires_nonempty_subject(claims):
    token = jwt.encode(claims, config.JWT_SECRET, algorithm=config.JWT_ALGORITHM)
    with pytest.raises(HTTPException) as caught:
        config.verify_token(token)
    assert caught.value.status_code == 401


def _client():
    app = FastAPI()
    app.add_middleware(SecurityMiddleware, api_keys=["test-key"])

    @app.get("/private")
    async def private():
        return {"ok": True}

    return TestClient(app)


def test_internal_jwt_failure_cannot_switch_to_api_key(monkeypatch):
    failure = OSError("synthetic verification unavailable")

    def fail(token):
        raise failure

    monkeypatch.setattr(config, "verify_token", fail)
    with _client() as client, pytest.raises(OSError) as caught:
        client.get("/private", headers={"Authorization": "Bearer token", "X-API-Key": "test-key"})
    assert caught.value is failure


def test_invalid_jwt_can_still_use_explicit_valid_api_key():
    with _client() as client:
        response = client.get("/private", headers={"Authorization": "Bearer invalid", "X-API-Key": "test-key"})
    assert response.status_code == 200


def test_revocation_failure_cannot_switch_to_api_key(monkeypatch):
    from infra import token_revocation

    def fail(jti):
        raise OSError("synthetic revocation unavailable")

    monkeypatch.setattr(token_revocation, "is_token_revoked", fail)
    token = config.create_access_token("test", 1)
    with _client() as client, pytest.raises(OSError, match="revocation unavailable"):
        client.get("/private", headers={"Authorization": f"Bearer {token}", "X-API-Key": "test-key"})
