"""Add scoped, append-only character memory claims.

Revision ID: 007_memory_claims
Revises: 006_character_memory
Create Date: 2026-08-30 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "007_memory_claims"
down_revision: str | None = "006_character_memory"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_NEW_COLUMNS = (
    sa.Column("scope_level", sa.Text(), nullable=False, server_default="conversation"),
    # Existing rows are the compatibility revision. The default is changed to
    # 1 after backfill so direct future inserts follow the append-only contract.
    sa.Column("revision", sa.Integer(), nullable=False, server_default="0"),
    sa.Column("relation_type", sa.Text(), nullable=False, server_default="ADD"),
    sa.Column("status", sa.Text(), nullable=False, server_default="active"),
    sa.Column("parent_memory_id", sa.Integer(), nullable=True),
    sa.Column("supersedes_memory_id", sa.Integer(), nullable=True),
    sa.Column("confidence", sa.Float(), nullable=False, server_default="1.0"),
    sa.Column("source_message_ids_json", sa.Text(), nullable=False, server_default="[]"),
    sa.Column("evidence_json", sa.Text(), nullable=False, server_default="[]"),
    sa.Column("metadata_json", sa.Text(), nullable=False, server_default="{}"),
    sa.Column("valid_from", sa.Text(), nullable=True),
    sa.Column("valid_to", sa.Text(), nullable=True),
    sa.Column("observed_at", sa.Text(), nullable=True),
)

_REVISION_COLUMNS = [
    "character_id",
    "platform",
    "adapter",
    "sender_id",
    "conversation_type",
    "conversation_id",
    "scope_level",
    "memory_key",
    "revision",
]


def _schema_state():
    inspector = sa.inspect(op.get_bind())
    columns = {column["name"] for column in inspector.get_columns("character_memories")}
    uniques = {
        item.get("name"): item.get("column_names", [])
        for item in inspector.get_unique_constraints("character_memories")
    }
    indexes = {item.get("name"): item for item in inspector.get_indexes("character_memories")}
    return columns, uniques, indexes


def upgrade() -> None:
    columns, uniques, indexes = _schema_state()
    missing = [column for column in _NEW_COLUMNS if column.name not in columns]
    old_unique = next(
        (name for name, names in uniques.items() if names == _REVISION_COLUMNS[:-3] + ["memory_key"]),
        None,
    )
    has_revision_unique = any(names == _REVISION_COLUMNS for names in uniques.values()) or any(
        item.get("unique") and item.get("column_names") == _REVISION_COLUMNS for item in indexes.values()
    )

    if missing or old_unique or not has_revision_unique:
        # batch mode rebuilds SQLite tables when a UNIQUE constraint must be
        # replaced and compiles to ordinary ALTER TABLE statements on PG.
        with op.batch_alter_table("character_memories") as batch_op:
            for column in missing:
                batch_op.add_column(column)
            if old_unique:
                batch_op.drop_constraint(old_unique, type_="unique")
            if not has_revision_unique:
                batch_op.create_unique_constraint("uq_character_memory_revision", _REVISION_COLUMNS)

    # Match the ORM/default for fresh append-only rows after old rows were
    # materialized as revision 0.
    columns, _, indexes = _schema_state()
    revision_info = next(
        column for column in sa.inspect(op.get_bind()).get_columns("character_memories") if column["name"] == "revision"
    )
    if str(revision_info.get("default") or "").replace("'", "") not in {"1", "(1)"}:
        with op.batch_alter_table("character_memories") as batch_op:
            batch_op.alter_column(
                "revision",
                existing_type=sa.Integer(),
                existing_nullable=False,
                server_default="1",
            )

    indexes = {item.get("name") for item in sa.inspect(op.get_bind()).get_indexes("character_memories")}
    if "idx_character_memories_active_lookup" not in indexes:
        op.create_index(
            "idx_character_memories_active_lookup",
            "character_memories",
            ["platform", "adapter", "sender_id", "scope_level", "status", "updated_at"],
        )
    if "idx_character_memories_parent" not in indexes:
        op.create_index("idx_character_memories_parent", "character_memories", ["parent_memory_id"])
    if "idx_character_memories_supersedes" not in indexes:
        op.create_index("idx_character_memories_supersedes", "character_memories", ["supersedes_memory_id"])


def downgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name
    # A v1 table can represent one row per logical key. Prefer an active row,
    # then the latest revision, when collapsing a version chain for downgrade.
    ranking = "CASE status WHEN 'active' THEN 0 WHEN 'pending' THEN 1 ELSE 2 END"
    if dialect == "postgresql":
        op.execute(
            sa.text(f"""
            DELETE FROM character_memories WHERE id IN (
                SELECT id FROM (
                    SELECT id, ROW_NUMBER() OVER (
                        PARTITION BY character_id, platform, adapter, sender_id,
                                     conversation_type, conversation_id, memory_key
                        ORDER BY {ranking}, revision DESC, id DESC
                    ) AS row_number
                    FROM character_memories
                ) ranked WHERE row_number > 1
            )
        """)
        )
    else:
        op.execute(
            sa.text(f"""
            DELETE FROM character_memories WHERE id IN (
                SELECT id FROM (
                    SELECT id, ROW_NUMBER() OVER (
                        PARTITION BY character_id, platform, adapter, sender_id,
                                     conversation_type, conversation_id, memory_key
                        ORDER BY {ranking}, revision DESC, id DESC
                    ) AS row_number
                    FROM character_memories
                ) WHERE row_number > 1
            )
        """)
        )

    _, uniques, indexes = _schema_state()
    for name in (
        "idx_character_memories_supersedes",
        "idx_character_memories_parent",
        "idx_character_memories_active_lookup",
    ):
        if name in indexes:
            op.drop_index(name, table_name="character_memories")

    revision_constraint = next(
        (name for name, names in uniques.items() if names == _REVISION_COLUMNS),
        None,
    )
    revision_index = next(
        (
            name
            for name, item in indexes.items()
            if item.get("unique") and item.get("column_names") == _REVISION_COLUMNS
        ),
        None,
    )
    with op.batch_alter_table("character_memories") as batch_op:
        if revision_constraint:
            batch_op.drop_constraint(revision_constraint, type_="unique")
        elif revision_index:
            batch_op.drop_index(revision_index)
        for column in reversed(_NEW_COLUMNS):
            batch_op.drop_column(column.name)
        batch_op.create_unique_constraint(
            "uq_character_memory_key",
            [
                "character_id",
                "platform",
                "adapter",
                "sender_id",
                "conversation_type",
                "conversation_id",
                "memory_key",
            ],
        )
