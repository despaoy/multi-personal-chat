"""Unify access control storage and remove legacy session_settings.

Revision ID: 005_unified_access
Revises: 004_index_cleanup
Create Date: 2026-08-01 00:00:00.000000
"""
from datetime import datetime
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "005_unified_access"
down_revision: Union[str, None] = "004_index_cleanup"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(table_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return table_name in inspector.get_table_names()


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name
    inspector = sa.inspect(bind)

    # API Key table now lives in the main database. It may already exist when
    # PgDatabase.init() ran metadata.create_all() before alembic upgrade.
    if "api_keys" not in inspector.get_table_names():
        op.create_table(
            "api_keys",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("key_hash", sa.Text(), nullable=False, unique=True),
            sa.Column("key_prefix", sa.Text(), nullable=False),
            sa.Column("role", sa.Text(), nullable=False),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("created_at", sa.Float(), nullable=False),
            sa.Column("revoked_at", sa.Float(), nullable=True),
            sa.Column("last_used_at", sa.Float(), nullable=True),
            sa.Column("is_active", sa.Integer(), nullable=False, server_default="1"),
            sa.Column("rate_limit", sa.Integer(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )

    indexes = {idx["name"] for idx in inspector.get_indexes("api_keys")}
    if "idx_api_keys_key_prefix" not in indexes:
        op.create_index("idx_api_keys_key_prefix", "api_keys", ["key_prefix"])
    if "idx_api_keys_role" not in indexes:
        op.create_index("idx_api_keys_role", "api_keys", ["role"])

    # Backfill NULL values before tightening NOT NULL constraints. Server
    # defaults do not rewrite existing rows.
    bind.execute(sa.text("UPDATE claw_tools SET description = '' WHERE description IS NULL"))
    bind.execute(sa.text("UPDATE claw_tools SET code = '' WHERE code IS NULL"))
    bind.execute(sa.text("UPDATE claw_tools SET enabled = 1 WHERE enabled IS NULL"))
    bind.execute(sa.text('UPDATE intent_active_kbs SET "isActive" = 1 WHERE "isActive" IS NULL'))
    bind.execute(sa.text("UPDATE model_invocations SET platform = 'qq' WHERE platform IS NULL"))
    bind.execute(sa.text('UPDATE model_invocations SET "usedRag" = 0 WHERE "usedRag" IS NULL'))
    bind.execute(sa.text('UPDATE model_invocations SET "usedLora" = 0 WHERE "usedLora" IS NULL'))
    bind.execute(sa.text("UPDATE saved_dialogues SET dialogue_count = 0 WHERE dialogue_count IS NULL"))
    bind.execute(sa.text("UPDATE training_tasks SET lora_name = '' WHERE lora_name IS NULL"))

    # Align nullable constraints with db/models.py. SQLite needs Alembic batch
    # mode because it cannot alter these constraints in place.
    constraint_changes = (
        ("claw_tools", "description", sa.Text(), ""),
        ("claw_tools", "code", sa.Text(), ""),
        ("claw_tools", "enabled", sa.Integer(), "1"),
        ("intent_active_kbs", "isActive", sa.Integer(), "1"),
        ("model_invocations", "platform", sa.Text(), "qq"),
        ("model_invocations", "usedRag", sa.Integer(), "0"),
        ("model_invocations", "usedLora", sa.Integer(), "0"),
        ("saved_dialogues", "dialogue_count", sa.Integer(), "0"),
        ("training_tasks", "lora_name", sa.Text(), ""),
    )
    for table_name, column_name, column_type, default in constraint_changes:
        if dialect == "sqlite":
            with op.batch_alter_table(table_name) as batch_op:
                batch_op.alter_column(
                    column_name,
                    existing_type=column_type,
                    nullable=False,
                    server_default=default,
                )
        else:
            op.alter_column(
                table_name,
                column_name,
                existing_type=column_type,
                nullable=False,
                server_default=default,
            )

    # Fold legacy session_settings into conversations, then remove it.
    if _has_table("session_settings"):
        migrated_at = datetime.now().isoformat()
        session_columns = {
            column["name"] for column in sa.inspect(bind).get_columns("session_settings")
        }
        if "conversationType" in session_columns:
            conversation_type = 'COALESCE("conversationType", "sessionType", \'private\')'
            conversation_type_s = 'COALESCE(s."conversationType", s."sessionType", \'private\')'
        elif "sessionType" in session_columns:
            conversation_type = 'COALESCE("sessionType", \'private\')'
            conversation_type_s = 'COALESCE(s."sessionType", \'private\')'
        else:
            conversation_type = "'private'"
            conversation_type_s = "'private'"

        select_sql = f"""
            SELECT
                COALESCE(platform, 'qq'),
                COALESCE("conversationId", "sessionId"),
                {conversation_type},
                COALESCE(NULLIF("sessionName", ''), "sessionId"),
                COALESCE(bot_enabled, 1),
                'default',
                COALESCE(updated_at, :migrated_at),
                COALESCE(updated_at, :migrated_at)
            FROM session_settings
        """
        if dialect == "sqlite":
            bind.execute(
                sa.text(
                    """
                    INSERT OR IGNORE INTO conversations (
                        platform, "conversationId", "conversationType", "displayName",
                        "botEnabled", "replyPolicy", "createdAt", "updatedAt"
                    )
                    """ + select_sql
                ).bindparams(migrated_at=migrated_at)
            )
            bind.execute(
                sa.text(
                    f"""
                    UPDATE conversations
                    SET
                        "botEnabled" = COALESCE((
                            SELECT s.bot_enabled FROM session_settings AS s
                            WHERE COALESCE(s.platform, 'qq') = conversations.platform
                              AND COALESCE(s."conversationId", s."sessionId") = conversations."conversationId"
                              AND {conversation_type_s} = conversations."conversationType"
                        ), conversations."botEnabled"),
                        "displayName" = COALESCE((
                            SELECT COALESCE(NULLIF(s."sessionName", ''), s."sessionId")
                            FROM session_settings AS s
                            WHERE COALESCE(s.platform, 'qq') = conversations.platform
                              AND COALESCE(s."conversationId", s."sessionId") = conversations."conversationId"
                              AND {conversation_type_s} = conversations."conversationType"
                        ), conversations."displayName"),
                        "updatedAt" = COALESCE((
                            SELECT s.updated_at FROM session_settings AS s
                            WHERE COALESCE(s.platform, 'qq') = conversations.platform
                              AND COALESCE(s."conversationId", s."sessionId") = conversations."conversationId"
                              AND {conversation_type_s} = conversations."conversationType"
                        ), conversations."updatedAt")
                    WHERE EXISTS (
                        SELECT 1 FROM session_settings AS s
                        WHERE COALESCE(s.platform, 'qq') = conversations.platform
                          AND COALESCE(s."conversationId", s."sessionId") = conversations."conversationId"
                          AND {conversation_type_s} = conversations."conversationType"
                          AND (conversations."updatedAt" IS NULL OR s.updated_at IS NULL
                               OR s.updated_at >= conversations."updatedAt")
                    )
                    """
                )
            )
        else:
            bind.execute(
                sa.text(
                    """
                INSERT INTO conversations (
                    platform, "conversationId", "conversationType", "displayName",
                    "botEnabled", "replyPolicy", "createdAt", "updatedAt"
                )
                """ + select_sql + """
                ON CONFLICT (platform, "conversationId", "conversationType")
                DO UPDATE SET
                    "botEnabled" = EXCLUDED."botEnabled",
                    "displayName" = EXCLUDED."displayName",
                    "updatedAt" = EXCLUDED."updatedAt"
                WHERE conversations."updatedAt" IS NULL
                   OR EXCLUDED."updatedAt" >= conversations."updatedAt"
                    """
                ).bindparams(migrated_at=migrated_at)
            )
        op.drop_table("session_settings")


def downgrade() -> None:
    # Restore legacy session_settings table (best-effort reverse migration).
    op.create_table(
        "session_settings",
        sa.Column("sessionId", sa.Text(), nullable=False),
        sa.Column("platform", sa.Text(), nullable=False, server_default="qq"),
        sa.Column("conversationId", sa.Text(), nullable=True),
        sa.Column("sessionType", sa.Text(), nullable=False, server_default="private"),
        sa.Column("sessionName", sa.Text(), nullable=True),
        sa.Column("bot_enabled", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("updated_at", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("sessionId"),
    )
    op.alter_column("training_tasks", "lora_name", existing_type=sa.Text(), nullable=True, server_default="")
    op.alter_column("saved_dialogues", "dialogue_count", existing_type=sa.Integer(), nullable=True, server_default="0")
    op.alter_column("model_invocations", "usedLora", existing_type=sa.Integer(), nullable=True, server_default="0")
    op.alter_column("model_invocations", "usedRag", existing_type=sa.Integer(), nullable=True, server_default="0")
    op.alter_column("model_invocations", "platform", existing_type=sa.Text(), nullable=True, server_default="qq")
    op.alter_column("intent_active_kbs", "isActive", existing_type=sa.Integer(), nullable=True, server_default="1")
    op.alter_column("claw_tools", "enabled", existing_type=sa.Integer(), nullable=True, server_default="1")
    op.alter_column("claw_tools", "code", existing_type=sa.Text(), nullable=True, server_default="")
    op.alter_column("claw_tools", "description", existing_type=sa.Text(), nullable=True, server_default="")
    op.drop_index("idx_api_keys_role", table_name="api_keys")
    op.drop_index("idx_api_keys_key_prefix", table_name="api_keys")
    op.drop_table("api_keys")
