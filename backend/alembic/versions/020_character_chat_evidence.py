"""Persist answer-used evidence with character chat turns."""

import sqlalchemy as sa

from alembic import op

revision = "020_character_chat_evidence"
down_revision = "019_character_shared_memory"
branch_labels = None
depends_on = None


def upgrade():
    if "character_chat_evidence" in sa.inspect(op.get_bind()).get_table_names():
        return
    op.create_table(
        "character_chat_evidence",
        sa.Column("turn_id", sa.Text(), sa.ForeignKey("character_chat_turns.id"), primary_key=True),
        sa.Column("payload_json", sa.Text(), nullable=False),
    )


def downgrade():
    op.drop_table("character_chat_evidence")
