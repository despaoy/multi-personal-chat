"""Add character relationship and long-term memory tables.

Revision ID: 006_character_memory
Revises: 005_unified_access
Create Date: 2026-08-21 00:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "006_character_memory"
down_revision: Union[str, None] = "005_unified_access"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(table_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return table_name in inspector.get_table_names()


def upgrade() -> None:
    # PgDatabase.init() 会先执行 metadata.create_all()，已有环境可能
    # 已经创建过这两张表，这里保持幂等。
    if not _has_table("character_relationships"):
        op.create_table(
            "character_relationships",
            sa.Column("character_id", sa.Text(), nullable=False),
            sa.Column("platform", sa.Text(), nullable=False),
            sa.Column("adapter", sa.Text(), nullable=False),
            sa.Column("sender_id", sa.Text(), nullable=False),
            sa.Column("conversation_type", sa.Text(), nullable=False),
            sa.Column("conversation_id", sa.Text(), nullable=False),
            sa.Column("relationship_stage", sa.Text(), nullable=False),
            sa.Column("preferred_address", sa.Text(), nullable=False),
            sa.Column("summary", sa.Text(), nullable=False),
            sa.Column("interaction_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("created_at", sa.Text(), nullable=False),
            sa.Column("updated_at", sa.Text(), nullable=False),
            sa.PrimaryKeyConstraint(
                "character_id",
                "platform",
                "adapter",
                "sender_id",
                "conversation_type",
                "conversation_id",
            ),
        )

    if not _has_table("character_memories"):
        op.create_table(
            "character_memories",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("character_id", sa.Text(), nullable=False),
            sa.Column("platform", sa.Text(), nullable=False),
            sa.Column("adapter", sa.Text(), nullable=False),
            sa.Column("sender_id", sa.Text(), nullable=False),
            sa.Column("conversation_type", sa.Text(), nullable=False),
            sa.Column("conversation_id", sa.Text(), nullable=False),
            sa.Column("memory_type", sa.Text(), nullable=False),
            sa.Column("memory_key", sa.Text(), nullable=False),
            sa.Column("content", sa.Text(), nullable=False),
            sa.Column("importance", sa.Float(), nullable=False, server_default="0.0"),
            sa.Column("source_message_id", sa.Text(), nullable=True),
            sa.Column("created_at", sa.Text(), nullable=False),
            sa.Column("updated_at", sa.Text(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "character_id",
                "platform",
                "adapter",
                "sender_id",
                "conversation_type",
                "conversation_id",
                "memory_key",
                name="uq_character_memory_key",
            ),
        )
        op.create_index(
            "idx_character_memories_scope",
            "character_memories",
            [
                "character_id",
                "platform",
                "adapter",
                "sender_id",
                "conversation_type",
                "conversation_id",
            ],
        )


def downgrade() -> None:
    op.drop_index("idx_character_memories_scope", table_name="character_memories")
    op.drop_table("character_memories")
    op.drop_table("character_relationships")
