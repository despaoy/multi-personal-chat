"""Revocations remain effective until expiration; internal failures propagate."""

import time

import pytest

from db.database import SQLiteDB
from infra import token_revocation as revocation


@pytest.fixture
def database(tmp_path):
    instance = SQLiteDB(tmp_path / "auth.sqlite")
    try:
        yield instance
    finally:
        instance.close_connection()


def test_cleanup_never_restores_unexpired_revoked_tokens(database):
    now = int(time.time())
    statement = "INSERT INTO auth_token_revocations (jti, expires_at) VALUES (:jti, :expires_at)"
    database.revision_transaction(
        [(statement, {"jti": f"old-{i}", "expires_at": now + 3600}, 1) for i in range(10000)]
        + [(statement, {"jti": "expired", "expires_at": now - 1}, 1)]
    )
    revocation.revoke_token(database, "new", now + 7200)
    revocation.revoke_token(database, "new", now + 7200)
    assert database.execute_sql("SELECT COUNT(*) AS n FROM auth_token_revocations") == [{"n": 10001}]
    assert revocation.is_token_revoked(database, "old-0")
    assert revocation.is_token_revoked(database, "old-9999")
    assert revocation.is_token_revoked(database, "new")
    assert not revocation.is_token_revoked(database, "expired")


def test_internal_revocation_failure_is_not_reported_as_success():
    failure = OSError("synthetic revocation storage failure")

    class BrokenStore:
        def revision_transaction(self, statements):
            raise failure

    with pytest.raises(OSError) as caught:
        revocation.revoke_token(BrokenStore(), "new", int(time.time()) + 3600)
    assert caught.value is failure


def test_empty_or_expired_token_does_not_create_revocation(database):
    now = int(time.time())
    revocation.revoke_token(database, "", now + 60)
    revocation.revoke_token(database, "expired", now - 60)
    assert database.execute_sql("SELECT jti FROM auth_token_revocations") == []
