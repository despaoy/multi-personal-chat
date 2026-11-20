"""Store opt-in inactivity timers and index the proactive inbox."""

import sqlalchemy as sa

from alembic import op

revision = "033_proactive_inactivity"
down_revision = "032_proactive_chat"
branch_labels = None
depends_on = None


def upgrade():
    table = "character_chat_proactive_policies"
    bind = op.get_bind()
    existing = {column["name"] for column in sa.inspect(bind).get_columns(table)}
    for name, kind in (
        ("inactivity_hours", sa.Integer()), ("inactivity_due_at", sa.Float()),
        ("inactivity_turn_id", sa.Text()), ("inactivity_last_sent_turn_id", sa.Text()),
    ):
        if name not in existing:
            op.add_column(table, sa.Column(name, kind))
    op.create_index("ix_chat_proactive_inactive_due", table, ["enabled", "inactivity_due_at"], if_not_exists=True)
    op.create_index("ix_chat_proactive_inbox", "character_chat_proactive_jobs", ["session_id", "status", "sent_at"],
                    if_not_exists=True)


def downgrade():
    table = "character_chat_proactive_policies"
    op.drop_index("ix_chat_proactive_inbox", table_name="character_chat_proactive_jobs")
    op.drop_index("ix_chat_proactive_inactive_due", table_name=table)
    for name in ("inactivity_last_sent_turn_id", "inactivity_turn_id", "inactivity_due_at", "inactivity_hours"):
        op.drop_column(table, name)
