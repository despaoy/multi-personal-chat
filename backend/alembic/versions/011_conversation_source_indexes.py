"""Scope-first indexes for original conversation source reads."""
from alembic import op

revision = '011_conversation_source_indexes'
down_revision = '010_message_character_scope'
branch_labels = None
depends_on = None


def upgrade():
    # Frozen migration SQL: future runtime tuning must not alter old migrations.
    op.execute('''CREATE INDEX IF NOT EXISTS idx_messages_private_source
        ON messages (platform, adapter, "senderId", "characterId", "createdAt" DESC, id DESC)
        WHERE "branchId" IS NULL AND "conversationType" IN ('private', '')''')
    op.execute('''CREATE INDEX IF NOT EXISTS idx_messages_local_source
        ON messages (platform, adapter, "senderId", "characterId", "conversationType",
                     "conversationId", "createdAt" DESC, id DESC)
        WHERE "branchId" IS NULL''')


def downgrade():
    op.execute('DROP INDEX IF EXISTS idx_messages_local_source')
    op.execute('DROP INDEX IF EXISTS idx_messages_private_source')
