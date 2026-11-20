"""Invalidate outstanding character publication previews when sharing closes."""

import sqlalchemy as sa

from alembic import op

revision = "038_character_sharing_epoch"
down_revision = "037_character_chat_history"
branch_labels = None
depends_on = None


def upgrade():
    columns = {column["name"] for column in sa.inspect(op.get_bind()).get_columns("character_library")}
    if "sharing_epoch" not in columns:
        op.add_column("character_library", sa.Column("sharing_epoch", sa.BigInteger(), server_default="0", nullable=False))


def downgrade():
    op.drop_column("character_library", "sharing_epoch")
