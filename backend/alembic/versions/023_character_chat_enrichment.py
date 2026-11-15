"""Transactional dispatch records for Temporal enrichment workflows."""

import sqlalchemy as sa

from alembic import op

revision = "023_character_chat_enrichment"
down_revision = "022_character_chat_draft"
branch_labels = None
depends_on = None


def upgrade():
    if "character_chat_enrichment_jobs" in sa.inspect(op.get_bind()).get_table_names():
        return
    op.create_table(
        "character_chat_enrichment_jobs",
        sa.Column("turn_id", sa.Text(), sa.ForeignKey("character_chat_turns.id"), primary_key=True),
        sa.Column("session_id", sa.Text(), nullable=False),
        sa.Column("expected_revision", sa.Integer(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default="pending"),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.CheckConstraint("status IN ('pending','submitted','completed','superseded','failed')", name="ck_chat_enrichment_status"),
    )


def downgrade():
    op.drop_table("character_chat_enrichment_jobs")
