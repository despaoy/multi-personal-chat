"""Durable generation attempts and fencing, independent of HTTP connections."""

import json
import time
import uuid

from db.errors import RevisionConflictError
from repositories.character_chat import CharacterChatRepository
from repositories.character_library import CharacterNotFoundError


class GenerationInProgressError(RuntimeError):
    pass


class EnrichmentInProgressError(RuntimeError):
    pass


class CharacterChatGenerationRepository:
    def __init__(self, database):
        self.database = database
        self.chats = CharacterChatRepository(database)

    def get(self, owner, session_id, request_id):
        rows = self.database.execute_sql(
            "SELECT g.request_id,g.status,g.lease_until,g.turn_id,g.error_code,g.attempt,g.payload_json,"
            "g.draft_text,g.draft_revision,g.request_revision,s.revision AS session_revision,"
            "EXISTS (SELECT 1 FROM character_chat_turns t WHERE t.id=g.turn_id AND t.session_id=s.id AND t.active=1) AS active_turn "
            "FROM character_chat_sessions s LEFT JOIN character_chat_generations g ON g.session_id=s.id "
            "AND g.request_id=:request WHERE s.id=:session AND s.owner_user_id=:owner",
            {"session": session_id, "owner": owner, "request": request_id},
        )
        if not rows:
            raise CharacterNotFoundError("会话不存在")
        if rows[0]["request_id"] is None:
            return None
        result = dict(rows[0])
        payload = result.pop("payload_json")
        revision = result.pop("request_revision")
        session_revision = result.pop("session_revision")
        active_turn = result.pop("active_turn")
        if result["status"] == "running" and result["lease_until"] <= time.time():
            result["status"] = "interrupted"
        if result["status"] == "completed":
            if not active_turn:
                result["status"] = "superseded"
        elif session_revision != revision:
            result["status"] = "superseded"
        if result["status"] in {"running", "failed", "interrupted"}:
            _, message, replace_turn_id = json.loads(payload)
            result["body"] = {
                "revision": revision, "request_id": request_id,
                "message": message, "replace_turn_id": replace_turn_id,
            }
        else:
            result.pop("draft_text", None)
        return result

    def pending(self, owner, session_id):
        session = self.chats.get(owner, session_id)
        # Paginate rather than loading every historical failed message at once.
        offset = 0
        while True:
            rows = self.database.execute_sql(
                "SELECT request_id FROM character_chat_generations WHERE session_id=:session "
                "AND request_revision=:revision AND status IN ('running','failed') "
                "ORDER BY lease_until DESC,request_id LIMIT 50 OFFSET :offset",
                {"session": session_id, "revision": session["revision"], "offset": offset},
            )
            for row in rows:
                result = self.get(owner, session_id, row["request_id"])
                if result and result["status"] in {"running", "failed", "interrupted"}:
                    return result
            if len(rows) < 50:
                return None
            offset += 50

    def claim(self, session, request_id, message, replace_turn_id=None, *, lease_seconds=660):
        from repositories.character_chat_enrichment import CharacterChatEnrichmentRepository

        if not replace_turn_id and CharacterChatEnrichmentRepository(self.database).is_pending(session):
            raise EnrichmentInProgressError()
        now = time.time()
        payload = json.dumps([session["revision"], message, replace_turn_id], ensure_ascii=False)
        params = {
            "session": session["id"], "owner": session["owner_user_id"], "revision": session["revision"],
            "request": request_id, "payload": payload, "now": now,
            "until": now + lease_seconds, "token": uuid.uuid4().hex,
        }
        rows = self.database.execute_sql(
            "SELECT * FROM character_chat_generations WHERE session_id=:session AND request_id=:request", params
        )
        if rows:
            old = rows[0]
            if old["payload_json"] != payload or old["status"] == "completed":
                raise RevisionConflictError()
            if old["status"] == "running" and old["lease_until"] > now:
                raise GenerationInProgressError()
        statements = [
            ("UPDATE character_chat_sessions SET revision=revision "
             "WHERE id=:session AND owner_user_id=:owner AND revision=:revision", params, 1),
            ("INSERT INTO character_chat_generations "
             "(session_id,request_id,payload_json,request_revision,status,lease_token,lease_until,attempt) "
             "VALUES (:session,:request,:payload,:revision,'failed','',0,0) "
             "ON CONFLICT (session_id,request_id) DO NOTHING", params, None),
            ("UPDATE character_chat_generations SET status='running',lease_token=:token,lease_until=:until,"
             "attempt=attempt+1,error_code=NULL,draft_text='',draft_revision=draft_revision+1 "
             "WHERE session_id=:session AND request_id=:request AND request_revision=:revision "
             "AND payload_json=:payload AND (status='failed' OR (status='running' AND lease_until<=:now)) "
             "AND NOT EXISTS (SELECT 1 FROM character_chat_generations active "
             "WHERE active.session_id=:session AND active.request_revision=:revision "
             "AND active.status='running' AND active.lease_until>:now)", params, 1),
        ]
        self.database.revision_transaction(statements)
        return params["token"]

    def publish(self, session, request_id, token, event):
        kind = event.get("type")
        delta = event.get("text", "")
        if kind not in {"draft_reset", "draft_delta"} or not isinstance(delta, str):
            raise ValueError("Invalid draft event")
        params = {"session": session["id"], "owner": session["owner_user_id"],
                  "revision": session["revision"], "request": request_id, "token": token,
                  "now": time.time(), "delta": delta, "reset": int(kind == "draft_reset")}
        self.database.revision_transaction([
            ("UPDATE character_chat_sessions SET revision=revision "
             "WHERE id=:session AND owner_user_id=:owner AND revision=:revision", params, 1),
            ("UPDATE character_chat_generations SET draft_text=CASE WHEN :reset=1 THEN '' ELSE draft_text || :delta END,"
             "draft_revision=draft_revision+1 WHERE session_id=:session AND request_id=:request "
             "AND lease_token=:token AND status='running' AND lease_until>:now "
             "AND (:reset=1 OR length(draft_text)+length(:delta)<=1000000)", params, 1),
        ])

    def fail(self, session_id, request_id, token, code):
        # A late attempt cannot change a newer attempt or a completed result.
        self.database.revision_transaction([
            ("UPDATE character_chat_generations SET status='failed',error_code=:code "
             "WHERE session_id=:session AND request_id=:request AND lease_token=:token AND status='running'",
             {"session": session_id, "request": request_id, "token": token, "code": code}, None)
        ])

    @staticmethod
    def completion_statement(session_id, request_id, token, turn_id):
        return (
            "UPDATE character_chat_generations SET status='completed',turn_id=:turn,error_code=NULL "
            "WHERE session_id=:session AND request_id=:request AND lease_token=:token "
            "AND status='running' AND lease_until>:now",
            {"session": session_id, "request": request_id, "token": token, "turn": turn_id, "now": time.time()}, 1,
        )
