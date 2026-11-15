"""Account-scoped immutable character versions and explicit sharing snapshots."""

import json
import secrets
import uuid
from datetime import datetime, timezone

from character.library import CharacterSettings
from db.errors import RevisionConflictError


class CharacterNotFoundError(LookupError):
    pass


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

    def create(self, owner: str, settings: CharacterSettings, *, source=None):
        character_id = uuid.uuid4().hex
        timestamp = datetime.now(timezone.utc).isoformat()
        params = {
            "id": character_id,
            "owner": owner,
            "now": timestamp,
            "source": source["id"] if source else None,
            "source_revision": source["settings_revision"] if source else None,
            "settings": settings.model_dump_json(),
            "revision": 1,
        }
        self.database.revision_transaction(
            [
                (
                    "INSERT INTO character_library (id,owner_user_id,revision,source_character_id,source_revision,created_at,updated_at) "
                    "VALUES (:id,:owner,1,:source,:source_revision,:now,:now)",
                    params,
                    1,
                ),
                self._insert_version(params),
            ]
        )
        return self.get(owner, character_id)

    @staticmethod
    def _insert_version(params):
        return (
            "INSERT INTO character_library_versions (character_id,revision,settings_json,created_at) "
            "VALUES (:id,:revision,:settings,:now)",
            params,
            1,
        )

    def update(self, owner: str, character_id: str, revision: int, settings: CharacterSettings):
        self.get(owner, character_id)
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
            ]
        )
        return self.get(owner, character_id, revision + 1)

    def publish(self, owner: str, character_id: str, revision: int, enabled: bool):
        current = self.get(owner, character_id)
        token = (current["share_token"] or secrets.token_urlsafe(32)) if enabled else None
        self.database.revision_transaction(
            [
                (
                    "UPDATE character_library SET share_token=:token,shared_revision=:published "
                    "WHERE id=:id AND owner_user_id=:owner AND revision=:revision",
                    {
                        "token": token,
                        "published": revision if enabled else None,
                        "id": character_id,
                        "owner": owner,
                        "revision": revision,
                    },
                    1,
                ),
            ]
        )
        return {"share_token": token, "shared_revision": revision if enabled else None}

    def shared(self, token: str):
        rows = self.database.execute_sql(
            "SELECT c.id,v.revision AS settings_revision,v.settings_json FROM character_library c "
            "JOIN character_library_versions v ON v.character_id=c.id AND v.revision=c.shared_revision "
            "WHERE c.share_token=:token",
            {"token": token},
        )
        if not rows:
            raise CharacterNotFoundError("分享不存在或已关闭")
        return self._record(rows[0])

    def copy_shared(self, owner: str, token: str):
        source = self.shared(token)
        return self.create(owner, CharacterSettings.model_validate(source["settings"]), source=source)

    def source_update(self, owner: str, character_id: str):
        current = self.get(owner, character_id)
        rows = self.database.execute_sql(
            "SELECT c.id,v.revision AS settings_revision,v.settings_json FROM character_library c "
            "JOIN character_library_versions v ON v.character_id=c.id AND v.revision=c.shared_revision "
            "WHERE c.id=:source AND c.share_token IS NOT NULL",
            {"source": current["source_character_id"]},
        )
        if not rows:
            return {"available": False, "revision": current["revision"], "source_revision": None, "changes": {}}
        source = self._record(rows[0])
        return {
            "available": source["settings_revision"] > (current["source_revision"] or 0),
            "revision": current["revision"],
            "source_revision": source["settings_revision"],
            "changes": {
                key: {"current": current["settings"][key], "incoming": value}
                for key, value in source["settings"].items()
                if value != current["settings"][key]
            },
        }

    def import_source_update(self, owner: str, character_id: str, revision: int, source_revision: int):
        current = self.get(owner, character_id)
        preview = self.source_update(owner, character_id)
        if (
            current["revision"] != revision
            or preview["revision"] != revision
            or not preview["available"]
            or preview["source_revision"] != source_revision
        ):
            raise RevisionConflictError()
        settings = {**current["settings"], **{key: value["incoming"] for key, value in preview["changes"].items()}}
        params = {
            "id": character_id,
            "owner": owner,
            "previous": revision,
            "revision": revision + 1,
            "source": current["source_character_id"],
            "source_revision": source_revision,
            "settings": CharacterSettings.model_validate(settings).model_dump_json(),
            "now": datetime.now(timezone.utc).isoformat(),
        }
        self.database.revision_transaction(
            [
                # Serialize against publication/revocation; never import an unseen update.
                (
                    "UPDATE character_library SET revision=revision WHERE id=:source "
                    "AND shared_revision=:source_revision AND share_token IS NOT NULL",
                    params,
                    1,
                ),
                (
                    "UPDATE character_library SET revision=:revision,source_revision=:source_revision,updated_at=:now "
                    "WHERE id=:id AND owner_user_id=:owner AND revision=:previous",
                    params,
                    1,
                ),
                self._insert_version(params),
            ]
        )
        return self.get(owner, character_id, revision + 1)
