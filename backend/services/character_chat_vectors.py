"""Local embedding integration; never substitutes a fabricated vector on failure."""

import json

from repositories.character_chat_vectors import vector_literal


def select_memories(session, message, *, database, provider=None, limit=8):
    from db.errors import RevisionConflictError
    from repositories.character_chat_vectors import CharacterChatVectorRepository

    repo = CharacterChatVectorRepository(database)
    current = repo.chats.get(session["owner_user_id"], session["id"])
    if current["revision"] != session["revision"] or current["state"] != session["state"]:
        raise RevisionConflictError()
    if not any(m.get("status") == "confirmed" for m in current["state"]["memories"]):
        return []
    if provider is None:
        from knowledge.retrieval_core.embedding import get_default_embedding_provider

        provider = get_default_embedding_provider()
    identity = embedding_identity(provider)
    if not repo.ready(current, identity, provider.dimension):
        identity, vectors = build_memory_vectors(current["state"], provider)
        repo.replace(current, identity, vectors)
    query = provider.embed_query(message)
    query = query.tolist() if hasattr(query, "tolist") else list(query)
    if len(query) != provider.dimension:
        raise ValueError("Query embedding dimension mismatch")
    result = repo.search(session["owner_user_id"], session["id"], identity, query, limit)
    if repo.chats.get(session["owner_user_id"], session["id"])["revision"] != session["revision"]:
        raise RevisionConflictError()
    return result


def embedding_identity(provider):
    from knowledge.retrieval_core.embedding import ENCODING_CONTRACT

    fingerprint = provider.model_fingerprint
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("Embedding model fingerprint is required")
    return json.dumps([provider.model_id, fingerprint, provider.dimension, ENCODING_CONTRACT])


def build_memory_vectors(state, provider=None):
    if provider is None:
        from knowledge.retrieval_core.embedding import get_default_embedding_provider

        provider = get_default_embedding_provider()
    identity = embedding_identity(provider)
    memories = [memory for memory in state["memories"] if memory.get("status") == "confirmed"]
    output = provider.embed_texts([memory["content"] for memory in memories])
    if len(output) != len(memories):
        raise ValueError("Embedding output count mismatch")
    vectors = {}
    for memory, row in zip(memories, output, strict=True):
        vector = row.tolist() if hasattr(row, "tolist") else list(row)
        if len(vector) != provider.dimension:
            raise ValueError("Embedding output dimension mismatch")
        vector_literal(vector)
        vectors[memory["id"]] = vector
    return identity, vectors
