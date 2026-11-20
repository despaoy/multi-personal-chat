"""Store per-session consent and durable application messages."""

import sqlalchemy as sa

from alembic import op

revision = "032_proactive_chat"
down_revision = "031_chat_erasure_receipts"
branch_labels = None
depends_on = None


def upgrade():
    metadata = sa.MetaData()
    sa.Table("character_chat_sessions", metadata, sa.Column("id", sa.Text(), primary_key=True))
    policy = sa.Table(
        "character_chat_proactive_policies", metadata,
        sa.Column("session_id", sa.Text(), sa.ForeignKey("character_chat_sessions.id"), primary_key=True),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("enabled", sa.Integer(), nullable=False),
        sa.Column("cadence_hours", sa.Integer(), nullable=False),
        sa.Column("timezone", sa.Text(), nullable=False),
        sa.Column("quiet_start", sa.Integer()),
        sa.Column("quiet_end", sa.Integer()),
        sa.Column("last_sent_at", sa.Float()),
    )
    jobs = sa.Table(
        "character_chat_proactive_jobs", metadata,
        sa.Column("session_id", sa.Text(), sa.ForeignKey("character_chat_sessions.id"), primary_key=True),
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("policy_revision", sa.Integer(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("topic", sa.Text(), nullable=False),
        sa.Column("due_at", sa.Float(), nullable=False),
        sa.Column("next_attempt_at", sa.Float(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("sent_at", sa.Float()),
        sa.Column("content", sa.Text()),
    )
    sa.Index("ix_chat_proactive_jobs_ready", jobs.c.status, jobs.c.next_attempt_at, jobs.c.session_id)
    metadata.create_all(op.get_bind(), tables=[policy, jobs], checkfirst=True)


def downgrade():
    op.drop_table("character_chat_proactive_jobs")
    op.drop_table("character_chat_proactive_policies")
