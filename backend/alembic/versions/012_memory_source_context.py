"""Independent, revocable source context; no existing chat text is copied."""
from alembic import op

revision = "012_memory_source_context"
down_revision = "011_conversation_source_indexes"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""CREATE TABLE IF NOT EXISTS memory_source_fences (
        owner_key TEXT PRIMARY KEY, revoked_before TEXT NOT NULL)""")
    op.execute("""CREATE TABLE IF NOT EXISTS memory_sources (
        source_key TEXT PRIMARY KEY, owner_key TEXT NOT NULL, scope_key TEXT NOT NULL,
        source_message_id TEXT NOT NULL, observed_at TEXT, body TEXT,
        state TEXT NOT NULL CHECK (state IN ('pending', 'recorded', 'revoked')),
        CHECK ((state = 'recorded' AND body IS NOT NULL AND observed_at IS NOT NULL)
            OR (state <> 'recorded' AND body IS NULL)))""")
    op.execute("""CREATE INDEX IF NOT EXISTS idx_memory_sources_scope
        ON memory_sources (scope_key, state, observed_at, source_key)""")
    op.execute("""CREATE TABLE IF NOT EXISTS memory_source_links (
        memory_id INTEGER NOT NULL, source_key TEXT NOT NULL,
        PRIMARY KEY (memory_id, source_key))""")


def downgrade():
    op.execute("DROP TABLE IF EXISTS memory_source_links")
    op.execute("DROP TABLE IF EXISTS memory_sources")
    op.execute("DROP TABLE IF EXISTS memory_source_fences")
