"""Account-scoped immutable character versions and explicit sharing snapshots."""

import json
import uuid
from datetime import datetime, timezone

from character.library import CharacterSettings


class CharacterNotFoundError(LookupError):
    pass


class SourceMergeBlockedError(ValueError):
    """The source update requires an explicit resolution that this API cannot express."""


class CharacterLibraryRepository:
    def __init__(self, database):
        self.database = database

    def get(self, owner: str, character_id: str, revision: int | None = None):
        rows = self.database.execute_sql(
            "SELECT c.*, v.settings_json, v.revision AS settings_revision "
            "FROM character_library c JOIN character_library_versions v ON v.character_id=c.id "
            "AND v.revision=COALESCE(:revision,c.revision) "
            "WHERE c.id=:id AND c.owner_user_id=:owner",
            {"id": character_id, "owner": owner, "revision": revision},
        )
        if not rows:
            raise CharacterNotFoundError("角色或版本不存在")
        return self._record(rows[0])

    def list(self, owner: str, limit: int = 50, offset: int = 0):
        rows = self.database.execute_sql(
            "SELECT c.*,v.settings_json,v.revision AS settings_revision FROM character_library c "
            "JOIN character_library_versions v ON v.character_id=c.id AND v.revision=c.revision "
            "WHERE c.owner_user_id=:owner ORDER BY c.updated_at DESC,c.id LIMIT :limit OFFSET :offset",
            {"owner": owner, "limit": limit, "offset": offset},
        )
        return [self._record(row) for row in rows]

    @staticmethod
    def _record(row):
        result = dict(row)
        result["settings"] = CharacterSettings.model_validate(json.loads(result.pop("settings_json"))).model_dump()
        return result

    def create(self, owner: str, settings: CharacterSettings):
        character_id = uuid.uuid4().hex
        self.database.revision_transaction(self.creation_statements(owner, settings, character_id))
        return self.get(owner, character_id)

    def creation_statements(self, owner, settings, character_id):
        timestamp = datetime.now(timezone.utc).isoformat()
        params = {
            "id": character_id,
            "owner": owner,
            "now": timestamp,
            "source": None,
            "source_revision": None,
            "settings": settings.model_dump_json(),
            "revision": 1,
        }
        return [
            (
                "INSERT INTO character_library (id,owner_user_id,revision,source_character_id,source_revision,created_at,updated_at) "
                "VALUES (:id,:owner,1,:source,:source_revision,:now,:now) ON CONFLICT (id) DO NOTHING",
                params,
                1,
            ),
            self._insert_version(params),
        ]

    @staticmethod
    def _insert_version(params):
        return (
            "INSERT INTO character_library_versions (character_id,revision,settings_json,created_at) "
            "VALUES (:id,:revision,:settings,:now)",
            params,
            1,
        )

    def update(self, owner: str, character_id: str, revision: int, settings: CharacterSettings):
        current = self.get(owner, character_id)
        sources = self.publications().carry_local_sources(owner, current, revision + 1)
        params = {
            "id": character_id,
            "owner": owner,
            "previous": revision,
            "revision": revision + 1,
            "now": datetime.now(timezone.utc).isoformat(),
            "settings": settings.model_dump_json(),
        }
        self.database.revision_transaction(
            [
                (
                    "UPDATE character_library SET revision=:revision,updated_at=:now "
                    "WHERE id=:id AND owner_user_id=:owner AND revision=:previous",
                    params,
                    1,
                ),
                self._insert_version(params),
            ] + sources
        )
        return self.get(owner, character_id, revision + 1)

    def publications(self):
        from repositories.character_publications import CharacterPublicationRepository
        return CharacterPublicationRepository(self)

    def publish_preview(self, owner, character_id, revision, corpus_id=None, source_ids=None):
        return self.publications().preview(owner, character_id, revision, corpus_id, source_ids)

    def publish(self, owner, character_id, revision, enabled, corpus_id=None, source_ids=None,
                preview_id=None, previous_publication_id=None):
        return self.publications().publish(owner, character_id, revision, enabled, corpus_id, source_ids,
                                           preview_id, previous_publication_id)

    def shared(self, token):
        return self.publications().shared(token)

    def copy_shared(self, owner, token, publication_id=None, request_id=None):
        return self.publications().copy(owner, token, publication_id, request_id)

    def source_update(self, owner, character_id):
        return self.publications().source_update(owner, character_id)

    def preview_source_update(self, owner, character_id, revision, source_revision, publication_id, local_corpus_id, choices):
        return self.publications().preview_update(owner, character_id, revision, source_revision, publication_id, local_corpus_id, choices)

    def import_source_update(self, owner, character_id, revision, source_revision, publication_id=None, local_corpus_id=None,
                             choices=None, preview_id=None):
        return self.publications().import_update(owner, character_id, revision, source_revision, publication_id, local_corpus_id,
                                                 choices, preview_id)
