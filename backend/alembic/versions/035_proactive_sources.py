"""Keep confirmed outreach tied to its original chat turn."""

import sqlalchemy as sa

from alembic import op

revision = "035_proactive_sources"
down_revision = "034_chat_web_push"
branch_labels = None
depends_on = None

_TABLE = "character_chat_proactive_jobs"
_INDEX = "ix_chat_proactive_source"


def upgrade():
    columns = {column["name"] for column in sa.inspect(op.get_bind()).get_columns(_TABLE)}
    if "source_turn_id" not in columns:
        op.add_column(_TABLE, sa.Column("source_turn_id", sa.Text()))
    op.create_index(_INDEX, _TABLE, ["session_id", "source_turn_id"], if_not_exists=True)


def downgrade():
    op.drop_index(_INDEX, table_name=_TABLE)
    op.drop_column(_TABLE, "source_turn_id")
