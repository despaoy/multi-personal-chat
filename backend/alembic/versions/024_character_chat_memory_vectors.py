"""Revision-bound private memory vectors; requires pgvector on PostgreSQL."""

import sqlalchemy as sa
from sqlalchemy.types import UserDefinedType

from alembic import op

revision = "024_chat_memory_vectors"
down_revision = "023_character_chat_enrichment"
branch_labels = None
depends_on = None


class Vector(UserDefinedType):
    cache_ok = True

    def get_col_spec(self, **kw):
        return "vector"


def upgrade():
    bind = op.get_bind()
    postgres = bind.dialect.name == "postgresql"
    if postgres:
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    if "character_chat_memory_vectors" in sa.inspect(bind).get_table_names():
        return
    op.create_table(
        "character_chat_memory_vectors",
        sa.Column("session_id", sa.Text(), sa.ForeignKey("character_chat_sessions.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("memory_id", sa.Text(), primary_key=True),
        sa.Column("model_id", sa.Text(), nullable=False),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column("session_revision", sa.Integer(), nullable=False),
        sa.Column("content_digest", sa.Text(), nullable=False),
        sa.Column("embedding", Vector() if postgres else sa.Text(), nullable=False),
    )


def downgrade():
    op.drop_table("character_chat_memory_vectors")
    # The database extension may be shared by other tables; never remove it here.
