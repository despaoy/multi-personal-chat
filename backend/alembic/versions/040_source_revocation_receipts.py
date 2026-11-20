"""Persist revoked original-source edition IDs for offline recovery."""

from alembic import op
from db.character_source_schema import BACKFILL_SOURCE_REVOCATIONS, CREATE_SOURCE_REVOCATIONS

revision = "040_source_revocation_receipts"
down_revision = "039_generation_request_revision"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(CREATE_SOURCE_REVOCATIONS)
    op.execute(BACKFILL_SOURCE_REVOCATIONS)


def downgrade():
    # Older backups may contain active copies of these editions.
    pass
