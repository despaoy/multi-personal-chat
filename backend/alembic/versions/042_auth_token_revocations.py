"""Persist signed login revocations across runtime instances."""

from alembic import op
from db.models import metadata

revision = "042_auth_token_revocations"
down_revision = "041_proactive_event_identity"
branch_labels = None
depends_on = None


def upgrade():
    metadata.tables["auth_token_revocations"].create(op.get_bind(), checkfirst=True)


def downgrade():
    # Preserve unexpired revocations for a later upgrade.
    pass
