"""Inverted source-speech postings. Does not backfill existing source text."""
from alembic import op

revision = "013_memory_source_terms"
down_revision = "012_memory_source_context"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""CREATE TABLE IF NOT EXISTS memory_source_terms (
        scope_key TEXT NOT NULL, term TEXT NOT NULL, source_key TEXT NOT NULL,
        PRIMARY KEY (scope_key, term, source_key))""")
    op.execute("CREATE INDEX IF NOT EXISTS idx_memory_source_terms_source ON memory_source_terms(source_key)")


def downgrade():
    op.execute("DROP TABLE IF EXISTS memory_source_terms")
