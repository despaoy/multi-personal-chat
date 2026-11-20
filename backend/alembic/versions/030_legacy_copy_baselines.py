"""Restore the empty source baseline for pre-publication character copies."""

from alembic import op
from db.legacy_character_publications import backfill_legacy_copy_baselines

revision = '030_legacy_copy_baselines'
down_revision = '029_source_update_receipts'
branch_labels = None
depends_on = None


def upgrade():
    backfill_legacy_copy_baselines(op.execute)


def downgrade():
    # The synthesized settings-only publications are valid in revision 029.
    # Retain them so downgrading does not invalidate references from copies.
    pass
