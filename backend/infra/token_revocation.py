"""Database-backed revocation identities for verified access tokens."""

import time

from db.auth_revocation_schema import CHECK_TOKEN_REVOCATION, MERGE_TOKEN_REVOCATION, PURGE_EXPIRED_TOKEN_REVOCATIONS


def revoke_token(database, jti: str, expires_at: int) -> None:
    now = int(time.time())
    if not jti or expires_at <= now:
        return
    database.revision_transaction(
        [
            (PURGE_EXPIRED_TOKEN_REVOCATIONS, {"now": now}, None),
            (MERGE_TOKEN_REVOCATION, {"jti": jti, "expires_at": expires_at}, None),
        ]
    )


def is_token_revoked(database, jti: str) -> bool:
    return bool(
        database.execute_sql(
            CHECK_TOKEN_REVOCATION,
            {"jti": jti, "now": int(time.time())},
        )
    )
