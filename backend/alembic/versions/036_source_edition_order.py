"""Order new source editions by their database-assigned sequence."""

import sqlalchemy as sa

from alembic import op
from db.character_source_schema import SOURCE_EDITION_INDEX

revision = "036_source_edition_order"
down_revision = "035_proactive_sources"
branch_labels = None
depends_on = None

_TABLE = "character_source_corpora"


def upgrade():
    columns = {column["name"] for column in sa.inspect(op.get_bind()).get_columns(_TABLE)}
    if "edition_number" not in columns:
        op.add_column(_TABLE, sa.Column("edition_number", sa.Integer(), server_default="0", nullable=False))
    op.execute(SOURCE_EDITION_INDEX)


def downgrade():
    op.drop_index("ix_character_source_edition", table_name=_TABLE)
    op.drop_column(_TABLE, "edition_number")
