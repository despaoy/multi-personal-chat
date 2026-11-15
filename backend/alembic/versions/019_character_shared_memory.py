"""Explicit account-private shared memory snapshots."""

import sqlalchemy as sa

from alembic import op

revision = "019_character_shared_memory"
down_revision = "018_character_chat"
branch_labels = None
depends_on = None


def upgrade():
    metadata = sa.MetaData()
    table = sa.Table(
        "character_shared_memories",
        metadata,
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("owner_user_id", sa.Text(), nullable=False),
        sa.Column("source_session_id", sa.Text(), nullable=False),
        sa.Column("source_memory_id", sa.Text(), nullable=False),
        sa.Column("memory_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.UniqueConstraint("source_session_id", "source_memory_id", name="uq_character_shared_source"),
    )
    sa.Index("ix_character_shared_memories_owner_user_id", table.c.owner_user_id)
    metadata.create_all(op.get_bind(), checkfirst=True)


def downgrade():
    op.drop_table("character_shared_memories")
