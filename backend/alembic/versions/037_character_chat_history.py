"""Split inherited conversation paths from immutable message content."""

import sqlalchemy as sa

from alembic import op
from db.character_chat_history import CREATE_HISTORY_OWNER_INDEX, HISTORY_OWNER_INDEX, migrate_baselines
from db.models import CharacterChatHistoryNode, CharacterChatHistoryStep

revision = "037_character_chat_history"
down_revision = "036_source_edition_order"
branch_labels = None
depends_on = None


def upgrade():
    connection = op.get_bind()
    connection.execute(sa.text(CREATE_HISTORY_OWNER_INDEX))
    CharacterChatHistoryNode.__table__.create(connection, checkfirst=True)
    CharacterChatHistoryStep.__table__.create(connection, checkfirst=True)
    migrate_baselines(lambda sql, params: connection.execute(sa.text(sql), params))


def downgrade():
    connection = op.get_bind()
    migrate_baselines(lambda sql, params: connection.execute(sa.text(sql), params), downgrade=True)
    op.drop_table(CharacterChatHistoryStep.__tablename__)
    op.drop_table(CharacterChatHistoryNode.__tablename__)
    op.drop_index(HISTORY_OWNER_INDEX, table_name="character_chat_sessions")
