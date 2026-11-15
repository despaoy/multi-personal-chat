"""Recoverable provisional generation drafts."""

import sqlalchemy as sa

from alembic import op

revision = "022_character_chat_draft"
down_revision = "021_character_chat_generation"
branch_labels = None
depends_on = None


def upgrade():
    columns = {item["name"] for item in sa.inspect(op.get_bind()).get_columns("character_chat_generations")}
    if "draft_text" not in columns:
        op.add_column("character_chat_generations", sa.Column("draft_text", sa.Text(), nullable=False, server_default=""))
    if "draft_revision" not in columns:
        op.add_column("character_chat_generations", sa.Column("draft_revision", sa.Integer(), nullable=False, server_default="0"))


def downgrade():
    op.drop_column("character_chat_generations", "draft_revision")
    op.drop_column("character_chat_generations", "draft_text")
