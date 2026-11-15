"""Version-bound conversations and reversible turns."""

import sqlalchemy as sa

from alembic import op

revision = "018_character_chat"
down_revision = "017_character_library"
branch_labels = None
depends_on = None


def upgrade():
    metadata = sa.MetaData()
    sessions = sa.Table(
        "character_chat_sessions",
        metadata,
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("owner_user_id", sa.Text(), nullable=False),
        sa.Column("character_id", sa.Text(), nullable=False),
        sa.Column("character_revision", sa.Integer(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("mode", sa.Text(), nullable=False),
        sa.Column("source_session_id", sa.Text()),
        sa.Column("source_turn_id", sa.Text()),
        sa.Column("hypothesis", sa.Text(), nullable=False, server_default=""),
        sa.Column("baseline_json", sa.Text(), nullable=False),
        sa.Column("state_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
        sa.CheckConstraint("revision >= 1", name="ck_character_chat_revision"),
        sa.CheckConstraint("mode IN ('initial','continuation','branch')", name="ck_character_chat_mode"),
    )
    sa.Index("ix_character_chat_sessions_owner_user_id", sessions.c.owner_user_id)
    turns = sa.Table(
        "character_chat_turns",
        metadata,
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("session_id", sa.Text(), sa.ForeignKey("character_chat_sessions.id"), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("request_id", sa.Text(), nullable=False),
        sa.Column("request_revision", sa.Integer(), nullable=False),
        sa.Column("replaces_turn_id", sa.Text()),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("reply", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("active", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("state_after_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.UniqueConstraint("session_id", "request_id", name="uq_character_chat_request"),
        sa.UniqueConstraint("session_id", "position", name="uq_character_chat_position"),
    )
    sa.Index("ix_character_chat_turns_history", turns.c.session_id, turns.c.active, turns.c.position)
    metadata.create_all(op.get_bind(), checkfirst=True)


def downgrade():
    op.drop_table("character_chat_turns")
    op.drop_table("character_chat_sessions")
