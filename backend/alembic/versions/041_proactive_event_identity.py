"""Store the identity of explicitly dated outreach events."""

import sqlalchemy as sa

from alembic import op

revision = "041_proactive_event_identity"
down_revision = "040_source_revocation_receipts"
branch_labels = None
depends_on = None

_TABLE = "character_chat_proactive_jobs"


def upgrade():
    columns = {column["name"] for column in sa.inspect(op.get_bind()).get_columns(_TABLE)}
    if "event_key" not in columns:
        op.add_column(_TABLE, sa.Column("event_key", sa.Text()))


def downgrade():
    op.drop_column(_TABLE, "event_key")
