"""Portable source-edition ordering and restore revocation records."""

SOURCE_EDITION_INDEX = (
    "CREATE UNIQUE INDEX IF NOT EXISTS ix_character_source_edition "
    "ON character_source_corpora (owner_user_id,character_id,character_revision,edition_number) "
    "WHERE edition_number>0"
)

SOURCE_REVOCATIONS = "character_source_revocation_receipts"
CREATE_SOURCE_REVOCATIONS = f"CREATE TABLE IF NOT EXISTS {SOURCE_REVOCATIONS} (corpus_id TEXT PRIMARY KEY NOT NULL)"
REVOKED_SOURCE_IDS = "SELECT id FROM character_source_corpora WHERE revoked=1"
BACKFILL_SOURCE_REVOCATIONS = (
    f"INSERT INTO {SOURCE_REVOCATIONS} (corpus_id) {REVOKED_SOURCE_IDS} ON CONFLICT (corpus_id) DO NOTHING"
)
INSERT_SOURCE_REVOCATION = (
    f"INSERT INTO {SOURCE_REVOCATIONS} (corpus_id) VALUES (:id) ON CONFLICT (corpus_id) DO NOTHING"
)
REPLAY_SOURCE_REVOCATIONS = (
    f"UPDATE character_source_corpora SET revoked=1 WHERE revoked=0 AND id IN (SELECT corpus_id FROM {SOURCE_REVOCATIONS})"
)
