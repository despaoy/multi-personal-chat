"""Private pgvector retrieval with authoritative session and content validation."""

import hashlib
import json
import math

from db.errors import RevisionConflictError
from repositories.character_chat import CharacterChatRepository


def digest(memory):
    return hashlib.sha256(json.dumps(memory, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def vector_literal(vector):
    if not vector or len(vector) > 16000 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in vector):
        raise ValueError("Invalid embedding vector")
    if not any(vector):
        raise ValueError("Zero embedding cannot be used for cosine retrieval")
    return json.dumps(vector, allow_nan=False)


class CharacterChatVectorRepository:
    def __init__(self, database):
        self.database = database
        self.chats = CharacterChatRepository(database)

    def replace(self, session, model_id, vectors):
        self.database.revision_transaction(self.replacement_statements(session, model_id, vectors))

    def ready(self, session, model_id, dimensions):
        expected = {m["id"]: digest(m) for m in session["state"]["memories"] if m.get("status") == "confirmed"}
        rows = self.database.execute_sql(
            "SELECT memory_id,content_digest FROM character_chat_memory_vectors WHERE session_id=:session "
            "AND session_revision=:revision AND model_id=:model AND dimensions=:dimensions",
            {"session": session["id"], "revision": session["revision"], "model": model_id, "dimensions": dimensions})
        return {row["memory_id"]: row["content_digest"] for row in rows} == expected

    def replacement_statements(self, session, model_id, vectors):
        if not model_id.strip():
            raise ValueError("Embedding model identity is required")
        memories = {m["id"]: m for m in session["state"]["memories"] if m.get("status") == "confirmed"}
        if set(vectors) != set(memories):
            raise ValueError("Embeddings must cover exactly the current confirmed memories")
        sizes = {len(v) for v in vectors.values()}
        if len(sizes) > 1:
            raise ValueError("Mixed embedding dimensions")
        params = {"session": session["id"], "owner": session["owner_user_id"], "revision": session["revision"]}
        statements = [
            ("UPDATE character_chat_sessions SET revision=revision WHERE id=:session AND owner_user_id=:owner AND revision=:revision", params, 1),
            ("DELETE FROM character_chat_memory_vectors WHERE session_id=:session", params, None),
        ]
        for memory_id, vector in vectors.items():
            statements.append((
                "INSERT INTO character_chat_memory_vectors "
                "(session_id,memory_id,model_id,dimensions,session_revision,content_digest,embedding) "
                "VALUES (:session,:memory,:model,:dimensions,:revision,:digest,CAST(:embedding AS vector))",
                {**params, "memory": memory_id, "model": model_id, "dimensions": len(vector),
                 "digest": digest(memories[memory_id]), "embedding": vector_literal(vector)}, 1,
            ))
        return statements

    def search(self, owner, session_id, model_id, vector, limit=8):
        if not 1 <= limit <= 100:
            raise ValueError("Invalid retrieval limit")
        session = self.chats.get(owner, session_id)
        query = vector_literal(vector)
        rows = self.database.execute_sql(
            "SELECT v.memory_id,v.content_digest FROM character_chat_memory_vectors v "
            "JOIN character_chat_sessions s ON s.id=v.session_id AND s.revision=v.session_revision "
            "WHERE s.owner_user_id=:owner AND s.id=:session AND s.revision=:revision "
            "AND v.model_id=:model AND v.dimensions=:dimensions "
            "ORDER BY v.embedding <=> CAST(:query AS vector),v.memory_id LIMIT :limit",
            {"owner": owner, "session": session_id, "revision": session["revision"], "model": model_id,
             "dimensions": len(vector), "query": query, "limit": limit},
        )
        current = self.chats.get(owner, session_id)
        if current["revision"] != session["revision"]:
            raise RevisionConflictError()
        memories = {m["id"]: m for m in current["state"]["memories"] if m.get("status") == "confirmed"}
        return [memories[row["memory_id"]] for row in rows
                if row["memory_id"] in memories and digest(memories[row["memory_id"]]) == row["content_digest"]]
