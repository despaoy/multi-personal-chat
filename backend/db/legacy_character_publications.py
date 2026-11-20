"""Backfill settings-only copy provenance from before publication sharing."""


def backfill_legacy_copy_baselines(execute):
    # Post-publication copies and confirmed updates have receipts. A missing
    # pointer on those records is not evidence of an empty source baseline.
    execute("""
        INSERT INTO character_publications
            (id, character_id, character_revision, settings_json, sources_json, created_at)
        SELECT DISTINCT 'legacy-' || v.character_id || '-' || CAST(v.revision AS TEXT),
               v.character_id, v.revision, v.settings_json, '[]', v.created_at
        FROM character_library_versions v
        JOIN character_library c
          ON c.source_character_id = v.character_id AND c.source_revision = v.revision
        WHERE c.source_publication_id IS NULL
          AND NOT EXISTS (
            SELECT 1 FROM character_publication_copies r
            WHERE r.owner_user_id = c.owner_user_id AND r.character_id = c.id
          )
          AND NOT EXISTS (
            SELECT 1 FROM character_source_update_receipts r
            WHERE r.owner_user_id = c.owner_user_id AND r.character_id = c.id
          )
          AND NOT EXISTS (
            SELECT 1 FROM character_publications p
            WHERE p.id = 'legacy-' || v.character_id || '-' || CAST(v.revision AS TEXT)
          )
    """)
    execute("""
        UPDATE character_library
        SET source_publication_id = 'legacy-' || source_character_id || '-' || CAST(source_revision AS TEXT)
        WHERE source_publication_id IS NULL AND source_character_id IS NOT NULL
          AND NOT EXISTS (
            SELECT 1 FROM character_publication_copies r
            WHERE r.owner_user_id = character_library.owner_user_id AND r.character_id = character_library.id
          )
          AND NOT EXISTS (
            SELECT 1 FROM character_source_update_receipts r
            WHERE r.owner_user_id = character_library.owner_user_id AND r.character_id = character_library.id
          )
          AND EXISTS (
            SELECT 1 FROM character_publications p
            WHERE p.id = 'legacy-' || character_library.source_character_id
                         || '-' || CAST(character_library.source_revision AS TEXT)
              AND p.character_id = character_library.source_character_id
              AND p.character_revision = character_library.source_revision
              AND p.corpus_id IS NULL AND p.sources_json = '[]'
          )
    """)
