"""Portable source-edition ordering constraints."""

SOURCE_EDITION_INDEX = (
    "CREATE UNIQUE INDEX IF NOT EXISTS ix_character_source_edition "
    "ON character_source_corpora (owner_user_id,character_id,character_revision,edition_number) "
    "WHERE edition_number>0"
)
