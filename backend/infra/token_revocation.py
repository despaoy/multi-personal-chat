"""Database-backed revocation identities for verified access tokens."""

import time


def revoke_token(database, jti: str, expires_at: int) -> None:
    now = int(time.time())
    if not jti or expires_at <= now:
        return
    database.revision_transaction(
        [
            ("DELETE FROM auth_token_revocations WHERE expires_at <= :now", {"now": now}, None),
            (
                """
            INSERT INTO auth_token_revocations (jti, expires_at)
            VALUES (:jti, :expires_at)
            ON CONFLICT (jti) DO NOTHING
            """,
                {"jti": jti, "expires_at": expires_at},
                None,
            ),
        ]
    )


def is_token_revoked(database, jti: str) -> bool:
    return bool(
        database.execute_sql(
            "SELECT 1 FROM auth_token_revocations WHERE jti = :jti AND expires_at > :now",
            {"jti": jti, "now": int(time.time())},
        )
    )
