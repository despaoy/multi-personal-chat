"""Add narrative branch isolation tables and messages.branchId.

Revision ID: 009_narrative_branches
Revises: 008_integration_receipts
Create Date: 2026-09-17 00:00:00.000000

新增三张分支隔离表（narrative_branches / branch_assertions / branch_states），
并给 messages 增加可空 branchId 列：NULL 表示正史，非空指向具体分支。
SQLite 与 PostgreSQL 均通过本迁移获得一致 schema；运行时
SQLiteDB._init_database / PgDatabase.init 的幂等建表只是开发期兜底，
正式环境以本迁移为准。
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "009_narrative_branches"
down_revision: str | None = "008_integration_receipts"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _existing_tables() -> set[str]:
    return set(sa.inspect(op.get_bind()).get_table_names())


def upgrade() -> None:
    tables = _existing_tables()

    if "narrative_branches" not in tables:
        op.create_table(
            "narrative_branches",
            sa.Column("id", sa.Text(), primary_key=True),
            sa.Column("owner_user_id", sa.Text(), nullable=False),
            sa.Column("character_id", sa.Text(), nullable=False),
            sa.Column("title", sa.Text(), nullable=False),
            sa.Column("initial_hypothesis", sa.Text(), nullable=False),
            sa.Column("status", sa.Text(), nullable=False, server_default="active"),
            sa.Column("revision", sa.Integer(), nullable=False, server_default="1"),
            sa.Column("created_at", sa.Text(), nullable=False),
            sa.Column("updated_at", sa.Text(), nullable=False),
        )
        op.create_index(
            "idx_narrative_branches_owner",
            "narrative_branches",
            ["owner_user_id", "status", "updated_at"],
        )
        op.create_index(
            "idx_narrative_branches_character",
            "narrative_branches",
            ["character_id"],
        )

    if "branch_assertions" not in tables:
        op.create_table(
            "branch_assertions",
            sa.Column("id", sa.Text(), primary_key=True),
            sa.Column(
                "branch_id",
                sa.Text(),
                sa.ForeignKey("narrative_branches.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("subject", sa.Text(), nullable=False, server_default=""),
            sa.Column("predicate", sa.Text(), nullable=False, server_default=""),
            sa.Column("object", sa.Text(), nullable=False, server_default=""),
            sa.Column("source_type", sa.Text(), nullable=False),
            sa.Column("assertion_kind", sa.Text(), nullable=False, server_default="event"),
            sa.Column("source_message_id", sa.Text(), nullable=True),
            sa.Column("source_assertion_ids", sa.Text(), nullable=False, server_default="[]"),
            sa.Column("status", sa.Text(), nullable=False, server_default="pending"),
            sa.Column("created_at", sa.Text(), nullable=False),
            sa.Column("updated_at", sa.Text(), nullable=False),
        )
        op.create_index(
            "idx_branch_assertions_branch_status",
            "branch_assertions",
            ["branch_id", "status", "updated_at"],
        )

    if "branch_states" not in tables:
        op.create_table(
            "branch_states",
            sa.Column(
                "branch_id",
                sa.Text(),
                sa.ForeignKey("narrative_branches.id", ondelete="CASCADE"),
                primary_key=True,
            ),
            sa.Column("relationship_stage", sa.Text(), nullable=False, server_default="stranger"),
            sa.Column("preferred_address", sa.Text(), nullable=False, server_default=""),
            sa.Column("summary", sa.Text(), nullable=False, server_default=""),
            sa.Column("interaction_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("created_at", sa.Text(), nullable=False),
            sa.Column("updated_at", sa.Text(), nullable=False),
        )

    message_columns = {c["name"] for c in sa.inspect(op.get_bind()).get_columns("messages")}
    if "branchId" not in message_columns:
        with op.batch_alter_table("messages") as batch_op:
            batch_op.add_column(sa.Column("branchId", sa.Text(), nullable=True))
    indexes = {i["name"] for i in sa.inspect(op.get_bind()).get_indexes("messages")}
    if "idx_messages_branch" not in indexes:
        op.create_index("idx_messages_branch", "messages", ["branchId", "createdAt"])


def downgrade() -> None:
    message_columns = {c["name"] for c in sa.inspect(op.get_bind()).get_columns("messages")}
    if "branchId" in message_columns:
        indexes = {i["name"] for i in sa.inspect(op.get_bind()).get_indexes("messages")}
        if "idx_messages_branch" in indexes:
            op.drop_index("idx_messages_branch", table_name="messages")
        with op.batch_alter_table("messages") as batch_op:
            batch_op.drop_column("branchId")

    tables = _existing_tables()
    if "branch_states" in tables:
        op.drop_table("branch_states")
    if "branch_assertions" in tables:
        op.drop_index("idx_branch_assertions_branch_status", table_name="branch_assertions")
        op.drop_table("branch_assertions")
    if "narrative_branches" in tables:
        op.drop_index("idx_narrative_branches_character", table_name="narrative_branches")
        op.drop_index("idx_narrative_branches_owner", table_name="narrative_branches")
        op.drop_table("narrative_branches")
