"""Normalize legacy indexes and remove redundant conversation indexes.

Revision ID: 004_index_cleanup
Revises: 003_indexes
Create Date: 2026-07-31 00:00:00.000000

The conversations unique constraint already owns an index over
(platform, conversationId, conversationType). Older schemas also created a
plain index over the same columns, which only added write and storage cost.

Some SQLite databases also used the legacy idx_messages_createdAt name. Keep
one canonical idx_messages_created_at index across both database backends.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "004_index_cleanup"
down_revision: Union[str, None] = "003_indexes"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _index_names(table_name: str) -> set[str]:
    inspector = sa.inspect(op.get_bind())
    try:
        return {
            str(index["name"])
            for index in inspector.get_indexes(table_name)
            if index.get("name")
        }
    except Exception:
        return set()


def upgrade() -> None:
    conversation_indexes = _index_names("conversations")
    if "idx_conversations_platform_conversation" in conversation_indexes:
        op.drop_index(
            "idx_conversations_platform_conversation",
            table_name="conversations",
        )

    message_indexes = _index_names("messages")
    if "idx_messages_created_at" not in message_indexes:
        op.create_index(
            "idx_messages_created_at",
            "messages",
            ["createdAt"],
        )
    if "idx_messages_createdAt" in message_indexes:
        op.drop_index("idx_messages_createdAt", table_name="messages")


def downgrade() -> None:
    conversation_indexes = _index_names("conversations")
    if "idx_conversations_platform_conversation" not in conversation_indexes:
        op.create_index(
            "idx_conversations_platform_conversation",
            "conversations",
            ["platform", "conversationId", "conversationType"],
        )

    message_indexes = _index_names("messages")
    if "idx_messages_createdAt" not in message_indexes:
        op.create_index(
            "idx_messages_createdAt",
            "messages",
            ["createdAt"],
        )
    if "idx_messages_created_at" in message_indexes:
        op.drop_index("idx_messages_created_at", table_name="messages")
