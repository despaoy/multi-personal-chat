"""Backfill the session revision used to claim each generation lease."""

import sqlalchemy as sa

from alembic import op

revision = "039_generation_request_revision"
down_revision = "038_character_sharing_epoch"
branch_labels = None
depends_on = None

_TABLE = "character_chat_generations"


def upgrade():
    connection = op.get_bind()
    columns = {column["name"] for column in sa.inspect(connection).get_columns(_TABLE)}
    if "request_revision" not in columns:
        op.add_column(_TABLE, sa.Column("request_revision", sa.Integer(), server_default="0", nullable=False))
        source = ("CAST(CAST(payload_json AS JSON)->>0 AS INTEGER)" if connection.dialect.name == "postgresql"
                  else "CAST(json_extract(payload_json,'$[0]') AS INTEGER)")
        op.execute(f"UPDATE {_TABLE} SET request_revision=COALESCE({source},0)")


def downgrade():
    op.drop_column(_TABLE, "request_revision")
