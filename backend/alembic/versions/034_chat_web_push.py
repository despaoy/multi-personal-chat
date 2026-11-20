"""Persist independent browser subscriptions and notification attempts."""

from alembic import op
from db.character_push_schema import PUSH_SCHEMA_034

revision = "034_chat_web_push"
down_revision = "033_proactive_inactivity"
branch_labels = None
depends_on = None


def upgrade():
    for statement in PUSH_SCHEMA_034:
        op.execute(statement)


def downgrade():
    op.drop_table("character_chat_push_attempts")
    op.drop_table("character_chat_push_subscriptions")
