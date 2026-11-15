"""Database authority for replayable background enrichment activities."""

import copy
import json

from db.errors import RevisionConflictError
from repositories.character_chat import CharacterChatRepository
from repositories.character_library import CharacterNotFoundError


class CharacterChatEnrichmentRepository:
    def __init__(self, database):
        self.database = database
        self.chats = CharacterChatRepository(database)

    def pending(self, limit=100):
        return self.database.execute_sql(
            "SELECT turn_id,expected_revision FROM character_chat_enrichment_jobs WHERE status='pending' "
            "ORDER BY created_at,turn_id LIMIT :limit", {"limit": limit},
        )

    def is_pending(self, session):
        return bool(self.database.execute_sql(
            "SELECT turn_id FROM character_chat_enrichment_jobs WHERE session_id=:session "
            "AND expected_revision=:revision AND status IN ('pending','submitted') LIMIT 1",
            {"session": session["id"], "revision": session["revision"]},
        ))

    def load(self, turn_id, expected_revision=None):
        rows = self.database.execute_sql(
            "SELECT j.*,s.owner_user_id FROM character_chat_enrichment_jobs j "
            "JOIN character_chat_sessions s ON s.id=j.session_id WHERE j.turn_id=:turn", {"turn": turn_id},
        )
        if not rows or rows[0]["status"] not in {"pending", "submitted"}:
            return None
        job = dict(rows[0])
        if expected_revision is not None and job["expected_revision"] != expected_revision:
            return None
        try:
            session, turns = self.chats.snapshot(job["owner_user_id"], job["session_id"])
            turn = next((item for item in turns if item["id"] == turn_id), None)
            if session["revision"] != job["expected_revision"] or turn is None:
                self.supersede(turn_id, job["expected_revision"])
                return None
            return session, turn
        except CharacterNotFoundError:
            return None

    def supersede(self, turn_id, expected_revision):
        self.database.revision_transaction([
            ("UPDATE character_chat_enrichment_jobs SET status='superseded' "
             "WHERE turn_id=:turn AND expected_revision=:revision AND status IN ('pending','submitted')",
             {"turn": turn_id, "revision": expected_revision}, None)
        ])

    def submitted(self, turn_id, expected_revision):
        self.database.revision_transaction([
            ("UPDATE character_chat_enrichment_jobs SET status='submitted' "
             "WHERE turn_id=:turn AND status='pending' AND expected_revision=:revision",
             {"turn": turn_id, "revision": expected_revision}, None)
        ])

    def fail(self, turn_id, expected_revision=None):
        loaded = self.load(turn_id, expected_revision)
        if loaded is None:
            return "inactive"
        session, turn = loaded
        state = copy.deepcopy(session["state"])
        for field in ("memory_processing", "relationship_processing", "summary_processing"):
            if state.get(field) == "pending":
                state[field] = "failed"
        try:
            self.database.revision_transaction([
                self.chats.cas(session, state),
                ("UPDATE character_chat_enrichment_jobs SET status='failed' WHERE turn_id=:turn "
                 "AND expected_revision=:revision AND status IN ('pending','submitted')",
                 {"turn": turn_id, "revision": session["revision"]}, 1),
                ("UPDATE character_chat_turns SET state_after_json=:state WHERE id=:turn AND active=1",
                 {"turn": turn["id"], "state": json.dumps(state, ensure_ascii=False)}, 1),
            ])
        except RevisionConflictError:
            self.supersede(turn_id, session["revision"])
            return "superseded"
        return "failed"

    def retryable(self, session):
        return bool(self.database.execute_sql(
            "SELECT turn_id FROM character_chat_enrichment_jobs WHERE session_id=:session "
            "AND expected_revision=:revision AND status='failed' LIMIT 1",
            {"session": session["id"], "revision": session["revision"] - 1},
        ))

    def retry(self, owner, session_id, revision):
        session, turns = self.chats.snapshot(owner, session_id)
        if session["revision"] != revision or not turns or not self.retryable(session):
            raise RevisionConflictError()
        turn = turns[-1]
        state = copy.deepcopy(session["state"])
        state.update(memory_processing="pending", relationship_processing="pending", summary_processing="pending")
        self.database.revision_transaction([
            self.chats.cas(session, state),
            ("UPDATE character_chat_enrichment_jobs SET status='pending',expected_revision=:next "
             "WHERE turn_id=:turn AND session_id=:session AND expected_revision=:previous AND status='failed'",
             {"turn": turn["id"], "session": session_id, "previous": revision - 1, "next": revision + 1}, 1),
            ("UPDATE character_chat_turns SET state_after_json=:state WHERE id=:turn AND active=1",
             {"turn": turn["id"], "state": json.dumps(state, ensure_ascii=False)}, 1),
        ])
        return {"queued": True}

    def commit(self, session, turn, state, vector_batch=None):
        state = copy.deepcopy(state)
        derived = [state.get("relationship", {}), state.get("relationship_proposal", {}), *state.get("memories", [])]
        for item in derived:
            if item.get("source_request_id") == turn["request_id"] and item.get("source_session_id") == session["id"]:
                item["source_turn_id"] = turn["id"]
        vector_statements = []
        if vector_batch is not None:
            from repositories.character_chat_vectors import CharacterChatVectorRepository

            model_id, vectors = vector_batch
            vector_statements = CharacterChatVectorRepository(self.database).replacement_statements(
                {**session, "state": state, "revision": session["revision"] + 1}, model_id, vectors)
        try:
            self.database.revision_transaction([
                self.chats.cas(session, state),
                ("UPDATE character_chat_enrichment_jobs SET status='completed' WHERE turn_id=:turn "
                 "AND expected_revision=:revision AND status IN ('pending','submitted')",
                 {"turn": turn["id"], "revision": session["revision"]}, 1),
                ("UPDATE character_chat_turns SET state_after_json=:state WHERE id=:turn AND session_id=:session AND active=1",
                 {"turn": turn["id"], "session": session["id"], "state": json.dumps(state, ensure_ascii=False)}, 1),
                *self.chats.revoke_changed_shares(session["id"], state),
                *vector_statements,
            ])
        except RevisionConflictError:
            self.supersede(turn["id"], session["revision"])
            return False
        return True
