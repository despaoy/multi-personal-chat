"""Versioned SQLite and PostgreSQL schema shared with migration 034."""

PUSH_SCHEMA_034 = (
    """CREATE TABLE IF NOT EXISTS character_chat_push_subscriptions (
        id TEXT NOT NULL PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES character_chat_sessions(id),
        endpoint TEXT NOT NULL,
        p256dh TEXT NOT NULL,
        auth TEXT NOT NULL,
        created_at DOUBLE PRECISION NOT NULL
    )""",
    """CREATE INDEX IF NOT EXISTS ix_chat_push_subscriptions_session
        ON character_chat_push_subscriptions(session_id)""",
    """CREATE TABLE IF NOT EXISTS character_chat_push_attempts (
        subscription_id TEXT NOT NULL REFERENCES character_chat_push_subscriptions(id),
        session_id TEXT NOT NULL REFERENCES character_chat_sessions(id),
        message_id TEXT NOT NULL,
        attempts INTEGER NOT NULL,
        lease_until DOUBLE PRECISION NOT NULL,
        token TEXT NOT NULL,
        terminal INTEGER NOT NULL,
        PRIMARY KEY (subscription_id, message_id)
    )""",
    """CREATE INDEX IF NOT EXISTS ix_chat_push_attempts_session ON character_chat_push_attempts(session_id)""",
)
