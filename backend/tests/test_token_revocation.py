"""Revocations remain effective until expiration; internal failures propagate."""

import time

import jwt
import pytest

from infra import token_revocation as revocation
from app.config import JWT_SECRET, JWT_ALGORITHM


def token(jti, expires):
    return jwt.encode({"jti": jti, "exp": expires, "sub": "synthetic-user"}, JWT_SECRET, algorithm=JWT_ALGORITHM)


def test_cleanup_never_restores_unexpired_revoked_tokens(monkeypatch):
    now = time.time()
    records = {f"old-{i}": now + 3600 for i in range(10000)}
    records["expired"] = now - 1
    monkeypatch.setattr(revocation, "_TOKEN_BLACKLIST", records)
    revocation.revoke_token(token("new", now + 7200))
    assert "expired" not in records
    assert len(records) == 10001
    assert all(revocation.is_token_revoked(identity) for identity in records)


def test_internal_revocation_failure_is_not_reported_as_success(monkeypatch):
    failure = OSError("synthetic revocation storage failure")

    class BrokenStore(dict):
        def __setitem__(self, key, value):
            raise failure

    monkeypatch.setattr(revocation, "_TOKEN_BLACKLIST", BrokenStore())
    with pytest.raises(OSError) as caught:
        revocation.revoke_token(token("new", time.time() + 3600))
    assert caught.value is failure


def test_invalid_or_expired_token_does_not_create_revocation(monkeypatch):
    records = {}
    monkeypatch.setattr(revocation, "_TOKEN_BLACKLIST", records)
    revocation.revoke_token("invalid-token")
    revocation.revoke_token(token("expired", time.time() - 60))
    assert records == {}
