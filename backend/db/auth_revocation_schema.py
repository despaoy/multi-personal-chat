"""Shared authentication revocation statements for runtime and offline recovery."""

TOKEN_REVOCATIONS = "auth_token_revocations"
TOKEN_REVOCATION_EXPIRY_INDEX = "idx_auth_token_revocations_expiry"
CREATE_TOKEN_REVOCATIONS = (
    f"CREATE TABLE IF NOT EXISTS {TOKEN_REVOCATIONS} (jti TEXT PRIMARY KEY NOT NULL, expires_at BIGINT NOT NULL)"
)
CREATE_TOKEN_REVOCATION_EXPIRY_INDEX = (
    f"CREATE INDEX IF NOT EXISTS {TOKEN_REVOCATION_EXPIRY_INDEX} ON {TOKEN_REVOCATIONS}(expires_at)"
)
LIVE_TOKEN_REVOCATIONS = f"SELECT jti, expires_at FROM {TOKEN_REVOCATIONS} WHERE expires_at > :now"
CHECK_TOKEN_REVOCATION = f"SELECT 1 FROM {TOKEN_REVOCATIONS} WHERE jti = :jti AND expires_at > :now"
PURGE_EXPIRED_TOKEN_REVOCATIONS = f"DELETE FROM {TOKEN_REVOCATIONS} WHERE expires_at <= :now"
MERGE_TOKEN_REVOCATION = f"""
    INSERT INTO {TOKEN_REVOCATIONS} (jti, expires_at)
    VALUES (:jti, :expires_at)
    ON CONFLICT (jti) DO UPDATE SET expires_at = excluded.expires_at
    WHERE excluded.expires_at > {TOKEN_REVOCATIONS}.expires_at
"""
