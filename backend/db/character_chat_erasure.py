"""Shared SQL scope for erasing selected session records and restoring backups."""

from .character_chat_history import HISTORY_PATH

ERASURE_TABLE = "character_chat_erasure_receipts"

# Children precede parents; lineage pointers in unselected sessions are retained.
SESSION_ROWS = (
    (HISTORY_PATH, "session_id"),
    ("character_chat_push_attempts", "session_id"),
    ("character_chat_push_subscriptions", "session_id"),
    ("character_chat_proactive_jobs", "session_id"),
    ("character_chat_proactive_policies", "session_id"),
    ("character_shared_memories", "source_session_id"),
    ("character_chat_generations", "session_id"),
    ("character_chat_enrichment_jobs", "session_id"),
    ("character_chat_memory_vectors", "session_id"),
    ("character_chat_turns", "session_id"),
    ("character_chat_sessions", "id"),
)
