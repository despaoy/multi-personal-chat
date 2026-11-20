"""Persist erased chat IDs so restores can suppress older copies."""

import sqlalchemy as sa

from alembic import op

revision = "031_chat_erasure_receipts"
down_revision = "030_legacy_copy_baselines"
branch_labels = None
depends_on = None


def upgrade():
    metadata = sa.MetaData()
    sa.Table("character_chat_erasure_receipts", metadata, sa.Column("session_id", sa.Text(), primary_key=True))
    metadata.create_all(op.get_bind(), checkfirst=True)


def downgrade():
    # Retain the content-free receipts across rollback. Old backups can still
    # contain these sessions, so dropping the receipts is unsafe.
    pass
