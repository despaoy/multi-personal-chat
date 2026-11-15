"""Persist character chat generation attempts."""

import sqlalchemy as sa

from alembic import op

revision = "021_character_chat_generation"
down_revision = "020_character_chat_evidence"
branch_labels = None
depends_on = None


def upgrade():
    if "character_chat_generations" in sa.inspect(op.get_bind()).get_table_names():
        return
    op.create_table(
        "character_chat_generations",
        sa.Column("session_id", sa.Text(), sa.ForeignKey("character_chat_sessions.id"), primary_key=True),
        sa.Column("request_id", sa.Text(), primary_key=True),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("lease_token", sa.Text(), nullable=False),
        sa.Column("lease_until", sa.Float(), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("turn_id", sa.Text()),
        sa.Column("error_code", sa.Text()),
        sa.CheckConstraint("status IN ('running','completed','failed')", name="ck_chat_generation_status"),
    )


def downgrade():
    op.drop_table("character_chat_generations")
