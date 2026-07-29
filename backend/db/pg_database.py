"""
PostgreSQL 异步数据库访问层
使用 async SQLAlchemy + asyncpg，提供与 SQLiteDB 相同的方法签名（异步版本）
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, List, Any

from sqlalchemy import (
    MetaData, Table, Column, Integer, BigInteger, Text, Float, String,
    ForeignKey, UniqueConstraint, text,
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

logger = logging.getLogger(__name__)

# ============================================
# 默认连接字符串
# ============================================
DEFAULT_DATABASE_URL = ""  # 必须通过环境变量 DATABASE_URL 配置


def _normalize_database_url(database_url: str) -> str:
    """Use the asyncpg SQLAlchemy dialect for common PostgreSQL URL forms."""
    if database_url.startswith("postgres://"):
        return "postgresql+asyncpg://" + database_url[len("postgres://"):]
    if database_url.startswith("postgresql://"):
        return "postgresql+asyncpg://" + database_url[len("postgresql://"):]
    return database_url

# ============================================
# SQLAlchemy Core 表定义
# ============================================
metadata = MetaData()

messages_table = Table(
    "messages", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("sessionType", Text, nullable=False),
    Column("sessionId", Text, nullable=False),
    Column("sessionName", Text),
    Column("platform", Text, nullable=False, server_default="qq"),
    Column("adapter", Text, nullable=False, server_default="nonebot"),
    Column("conversationId", Text),
    Column("conversationType", Text),
    Column("senderId", Text),
    Column("senderName", Text),
    Column("sourceMessageId", Text),
    Column("traceId", Text),
    Column("userId", Text),
    Column("userName", Text),
    Column("message", Text, nullable=False),
    Column("reply", Text, nullable=False),
    Column("modelName", Text),
    Column("loraName", Text),
    Column("costTime", Float),
    Column("createdAt", Text, nullable=False),
)

loras_table = Table(
    "loras", metadata,
    Column("id", Text, primary_key=True),
    Column("name", Text, nullable=False),
    Column("description", Text),
    Column("status", Text, nullable=False, server_default="inactive"),
    Column("style", Text),
    Column("size", Text),
    Column("trainedSteps", Integer),
    Column("totalSteps", Integer),
    Column("createdAt", Text),
)

config_table = Table(
    "config", metadata,
    Column("key", Text, primary_key=True),
    Column("value", Text, nullable=False),
)

knowledge_bases_table = Table(
    "knowledge_bases", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", Text, nullable=False, unique=True),
    Column("description", Text, server_default=""),
    Column("created_at", Text, nullable=False),
    Column("updated_at", Text, nullable=False),
)

knowledge_folders_table = Table(
    "knowledge_folders", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("knowledge_base_id", Integer, ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False),
    Column("name", Text, nullable=False),
    Column("description", Text, server_default=""),
    Column("created_at", Text, nullable=False),
    Column("updated_at", Text, nullable=False),
    UniqueConstraint("knowledge_base_id", "name"),
)

knowledge_documents_table = Table(
    "knowledge_documents", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("title", Text, nullable=False),
    Column("content", Text, nullable=False),
    Column("category", Text, nullable=False, server_default="未分类"),
    Column("knowledge_base_id", Integer, ForeignKey("knowledge_bases.id", ondelete="SET NULL")),
    Column("folder_id", Integer, ForeignKey("knowledge_folders.id", ondelete="SET NULL")),
    Column("sourceType", Text, nullable=False, server_default="text"),
    Column("sourceUrl", Text),
    Column("fileType", Text),
    Column("fileSize", Integer),
    Column("chunkCount", Integer, server_default="0"),
    Column("createdAt", Text, nullable=False),
    Column("updatedAt", Text, nullable=False),
)

knowledge_chunks_table = Table(
    "knowledge_chunks", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("documentId", Integer, ForeignKey("knowledge_documents.id", ondelete="CASCADE"), nullable=False),
    Column("chunkIndex", Integer, nullable=False),
    Column("content", Text, nullable=False),
    Column("embedding", BigInteger),  # Faiss 向量 ID（不存 BLOB）
    Column("createdAt", Text, nullable=False),
)

users_table = Table(
    "users", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("username", Text, nullable=False, unique=True),
    Column("password_hash", Text, nullable=False),
    Column("created_at", Text, nullable=False),
    # C4 fix: PG 模式下缺失 role 字段导致 RBAC 完全失效。
    # SQLite 实现已通过 _ensure_column 添加此列，PG 实现遗漏。
    # 默认 'user'，首个用户自动升级为 'admin'（见 add_user 实现）。
    Column("role", Text, nullable=False, server_default="user"),
)

user_data_table = Table(
    "user_data", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
    Column("page_key", Text, nullable=False),
    Column("data_json", Text, nullable=False),
    Column("updated_at", Text, nullable=False),
    UniqueConstraint("user_id", "page_key"),
)

saved_dialogues_table = Table(
    "saved_dialogues", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", Text, nullable=False),
    Column("character_desc", Text, nullable=False),
    Column("style", Text),
    Column("dialogue_count", Integer, nullable=False, server_default="0"),
    Column("dialogues_json", Text, nullable=False),
    Column("turn_stats", Text),
    Column("scene_stats", Text),
    Column("created_at", Text, nullable=False),
    Column("updated_at", Text, nullable=False),
)

session_settings_table = Table(
    "session_settings", metadata,
    Column("sessionId", Text, primary_key=True),
    Column("platform", Text, nullable=False, server_default="qq"),
    Column("conversationId", Text),
    Column("sessionType", Text, nullable=False, server_default="private"),
    Column("sessionName", Text),
    Column("bot_enabled", Integer, nullable=False, server_default="1"),
    Column("updated_at", Text),
)

claw_tools_table = Table(
    "claw_tools", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", Text, nullable=False, unique=True),
    Column("description", Text, nullable=False, server_default=""),
    Column("code", Text, nullable=False, server_default=""),
    Column("enabled", Integer, nullable=False, server_default="1"),
    Column("created_at", Text, nullable=False),
    Column("updated_at", Text, nullable=False),
)

integration_message_dedup_table = Table(
    "integration_message_dedup", metadata,
    Column("dedupKey", Text, primary_key=True),
    Column("platform", Text, nullable=False),
    Column("adapter", Text, nullable=False),
    Column("messageId", Text, nullable=False),
    Column("createdAt", Text, nullable=False),
)

conversations_table = Table(
    "conversations", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("platform", Text, nullable=False),
    Column("conversationId", Text, nullable=False),
    Column("conversationType", Text, nullable=False, server_default="private"),
    Column("displayName", Text),
    Column("botEnabled", Integer, nullable=False, server_default="1"),
    Column("replyPolicy", Text, nullable=False, server_default="default"),
    Column("createdAt", Text, nullable=False),
    Column("updatedAt", Text, nullable=False),
    UniqueConstraint("platform", "conversationId", "conversationType", name="uq_conversations_platform_conversation"),
)

integration_events_table = Table(
    "integration_events", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("platform", Text, nullable=False),
    Column("adapter", Text, nullable=False),
    Column("sourceMessageId", Text),
    Column("conversationId", Text),
    Column("conversationType", Text),
    Column("senderId", Text),
    Column("eventType", Text, nullable=False, server_default="message"),
    Column("eventHash", Text, nullable=False),
    Column("rawSummary", Text),
    Column("traceId", Text),
    Column("status", Text, nullable=False, server_default="received"),
    Column("createdAt", Text, nullable=False),
    UniqueConstraint("platform", "adapter", "eventHash", name="uq_integration_events_platform_hash"),
)

model_invocations_table = Table(
    "model_invocations", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("traceId", Text),
    Column("platform", Text, nullable=False, server_default="qq"),
    Column("conversationId", Text),
    Column("sessionId", Text),
    Column("modelName", Text),
    Column("loraName", Text),
    Column("costTime", Float, server_default="0"),
    Column("promptTokens", Integer, server_default="0"),
    Column("completionTokens", Integer, server_default="0"),
    Column("totalTokens", Integer, server_default="0"),
    Column("usedRag", Integer, nullable=False, server_default="0"),
    Column("usedLora", Integer, nullable=False, server_default="0"),
    Column("errorType", Text, server_default=""),
    Column("createdAt", Text, nullable=False),
)

audit_logs_table = Table(
    "audit_logs", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", Float, nullable=False),
    Column("api_key_hash", Text, nullable=False),
    Column("role", Text, nullable=False),
    Column("action", Text, nullable=False),
    Column("resource", Text),
    Column("detail", Text),
    Column("ip_address", Text),
)

intent_samples_table = Table(
    "intent_samples", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("kbName", Text, nullable=False),
    Column("text", Text, nullable=False),
    Column("label", Text, nullable=False),
    Column("createdAt", Text, nullable=False),
)

intent_active_kbs_table = Table(
    "intent_active_kbs", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("kbName", Text, nullable=False),
    Column("isActive", Integer, nullable=False, server_default="1"),
)

training_tasks_table = Table(
    "training_tasks", metadata,
    Column("id", Text, primary_key=True),
    Column("task_id", Text, unique=True),
    Column("lora_name", Text, server_default=""),
    Column("status", Text, nullable=False, server_default="pending"),
    Column("progress", Float, server_default="0"),
    Column("error_message", Text, server_default=""),
    Column("config_json", Text, server_default="{}"),
    Column("created_at", Text, server_default=""),
    Column("updated_at", Text, server_default=""),
    # Legacy columns retained for existing PostgreSQL deployments.
    # 已废弃：新代码不再写入这些列，仅保留以兼容已有数据库（NOT NULL DEFAULT 约束保证旧数据不报错）。
    # 后续应通过 Alembic 迁移删除这些列。
    Column("taskType", Text, nullable=False, server_default="lora"),
    Column("config", Text),
    Column("result", Text),
    Column("createdAt", Text, nullable=False, server_default=""),
    Column("updatedAt", Text, nullable=False, server_default=""),
)

# ============================================
# 研究与评估相关表（LLM Research Enhancement Roadmap）
# ============================================
gold_eval_runs_table = Table(
    "gold_eval_runs", metadata,
    Column("id", Text, primary_key=True),
    Column("run_at", Text, nullable=False),
    Column("adapter_name", Text),
    Column("model_label", Text),
    Column("total_prompts", Integer, server_default="0"),
    Column("category_breakdown", Text),
    Column("metrics", Text),
    Column("config_snapshot", Text),
    Column("notes", Text),
)

experiment_runs_table = Table(
    "experiment_runs", metadata,
    Column("id", Text, primary_key=True),
    Column("experiment_type", Text, nullable=False),
    Column("hypothesis", Text),
    Column("status", Text, nullable=False, server_default="pending"),
    Column("started_at", Text, nullable=False),
    Column("completed_at", Text),
    Column("results", Text),
    Column("config_path", Text),
    Column("report_path", Text),
)

retrieval_eval_questions_table = Table(
    "retrieval_eval_questions", metadata,
    Column("id", Text, primary_key=True),
    Column("question", Text, nullable=False),
    Column("expected_doc_ids", Text),
    Column("expected_doc_titles", Text),
    Column("gold_answer", Text),
    Column("category", Text),
    Column("created_at", Text, nullable=False),
)

preference_pairs_table = Table(
    "preference_pairs", metadata,
    Column("id", Text, primary_key=True),
    Column("prompt", Text, nullable=False),
    Column("chosen", Text, nullable=False),
    Column("rejected", Text, nullable=False),
    Column("rubric", Text),
    Column("annotator", Text),
    Column("metadata", Text),
    Column("review_status", Text, nullable=False, server_default="pending"),
    Column("created_at", Text, nullable=False),
)

adapter_compatibility_table = Table(
    "adapter_compatibility", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("adapter_name", Text, nullable=False),
    Column("checked_at", Text, nullable=False),
    Column("compatible", Integer, nullable=False),
    Column("checks", Text),
    Column("warnings", Text),
    Column("errors", Text),
)

feedback_table = Table(
    "feedback", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("trace_id", Text),
    Column("message_id", Text),
    Column("rating", Text),
    Column("reason", Text),
    Column("adapter_name", Text),
    Column("kb_revision", Text),
    Column("prompt_version", Text),
    Column("detail", Text),
    Column("created_at", Text, nullable=False),
)


# ============================================
# 辅助：将 Row 映射为 dict
# ============================================
def _row_to_dict(row) -> dict:
    """将 SQLAlchemy Row / RowMapping 转为普通 dict"""
    if row is None:
        return None
    try:
        return dict(row._mapping)
    except Exception:
        return dict(row)


# ============================================
# PgDatabase 类
# ============================================
class PgDatabase:
    """PostgreSQL 异步数据库类 - 与 SQLiteDB 相同接口的异步版本"""

    def __init__(self, database_url: Optional[str] = None):
        self.database_url = _normalize_database_url(database_url or os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL))
        if not self.database_url:
            raise ValueError("DATABASE_URL is required when USE_POSTGRESQL=true")
        self.engine = create_async_engine(
            self.database_url,
            echo=False,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
        )
        self.async_session = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False,
        )
        self._initialized = False

    async def init(self):
        """初始化数据库：创建所有表（如果不存在）"""
        if self._initialized:
            return
        async with self.engine.begin() as conn:
            await conn.run_sync(metadata.create_all)
            await self._ensure_column(conn, "messages", "platform", "TEXT NOT NULL DEFAULT 'qq'")
            await self._ensure_column(conn, "messages", "adapter", "TEXT NOT NULL DEFAULT 'nonebot'")
            await self._ensure_column(conn, "messages", "conversationId", "TEXT")
            await self._ensure_column(conn, "messages", "senderId", "TEXT")
            await self._ensure_column(conn, "messages", "sourceMessageId", "TEXT")
            await self._ensure_column(conn, "messages", "traceId", "TEXT")
            await self._ensure_column(conn, "messages", "conversationType", "TEXT")
            await self._ensure_column(conn, "messages", "senderName", "TEXT")
            await self._ensure_column(conn, "session_settings", "platform", "TEXT NOT NULL DEFAULT 'qq'")
            await self._ensure_column(conn, "session_settings", "conversationId", "TEXT")
            await self._ensure_column(conn, "training_tasks", "task_id", "TEXT")
            await self._ensure_column(conn, "training_tasks", "lora_name", "TEXT DEFAULT ''")
            await self._ensure_column(conn, "training_tasks", "error_message", "TEXT DEFAULT ''")
            await self._ensure_column(conn, "training_tasks", "config_json", "TEXT DEFAULT '{}'")
            await self._ensure_column(conn, "training_tasks", "created_at", "TEXT DEFAULT ''")
            await self._ensure_column(conn, "training_tasks", "updated_at", "TEXT DEFAULT ''")
            # C4 fix: 为已有 PG 数据库添加 users.role 列（新数据库由 create_all 创建）
            await self._ensure_column(conn, "users", "role", "TEXT NOT NULL DEFAULT 'user'")
            await conn.execute(text('CREATE UNIQUE INDEX IF NOT EXISTS idx_training_tasks_task_id ON training_tasks (task_id)'))
            await conn.execute(text('CREATE INDEX IF NOT EXISTS idx_messages_platform_conversation ON messages (platform, "conversationId", "createdAt")'))
            await conn.execute(text('CREATE INDEX IF NOT EXISTS idx_messages_source_dedup ON messages (platform, adapter, "sourceMessageId")'))
            await conn.execute(text('CREATE INDEX IF NOT EXISTS idx_messages_session_created ON messages ("sessionId", "createdAt")'))
            await conn.execute(text('CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages ("createdAt")'))
            # idx_conversations_platform_conversation 已删除：UNIQUE(platform, conversationId, conversationType)
            # 约束会自动创建索引，显式 CREATE INDEX 完全冗余。
            await conn.execute(text('CREATE INDEX IF NOT EXISTS idx_integration_events_trace ON integration_events ("traceId")'))
            await conn.execute(text('CREATE INDEX IF NOT EXISTS idx_integration_events_platform_created ON integration_events (platform, "createdAt")'))
            await conn.execute(text('CREATE INDEX IF NOT EXISTS idx_model_invocations_trace ON model_invocations ("traceId")'))
            await conn.execute(text('CREATE INDEX IF NOT EXISTS idx_model_invocations_created ON model_invocations ("createdAt")'))
        self._initialized = True
        logger.info(f"✅ PostgreSQL 数据库初始化完成: {self.database_url.split('@')[-1]}")

    async def _ensure_column(self, conn, table_name: str, column_name: str, definition: str):
        await conn.execute(text(f'ALTER TABLE "{table_name}" ADD COLUMN IF NOT EXISTS "{column_name}" {definition}'))

    async def close(self):
        """关闭引擎连接池"""
        await self.engine.dispose()

    # ============================================
    # 消息管理
    # ============================================
    async def add_message(self, message: Dict) -> Dict:
        """Add a message record and keep the conversation index in sync."""
        created_at = message.get("createdAt", datetime.now().isoformat())
        conversation_type = message.get("conversationType") or message.get("sessionType", "private")
        sender_name = message.get("senderName") or message.get("userName", "")
        conversation_id = message.get("conversationId", message.get("sessionId", ""))
        platform = message.get("platform", "qq")
        async with self.async_session() as session:
            await self._upsert_conversation_session(
                session,
                platform=platform,
                conversation_id=conversation_id,
                conversation_type=conversation_type,
                display_name=message.get("sessionName") or conversation_id or message.get("sessionId", ""),
            )
            stmt = messages_table.insert().values(
                sessionType=message.get("sessionType", conversation_type),
                sessionId=message.get("sessionId", ""),
                sessionName=message.get("sessionName", ""),
                platform=platform,
                adapter=message.get("adapter", "nonebot"),
                conversationId=conversation_id,
                conversationType=conversation_type,
                senderId=message.get("senderId", message.get("userId", "")),
                senderName=sender_name,
                sourceMessageId=message.get("sourceMessageId", ""),
                traceId=message.get("traceId", ""),
                userId=message.get("userId", ""),
                userName=message.get("userName", sender_name),
                message=message.get("message", ""),
                reply=message.get("reply", ""),
                modelName=message.get("modelName", ""),
                loraName=message.get("loraName", ""),
                costTime=message.get("costTime", 0.0),
                createdAt=created_at,
            )
            result = await session.execute(stmt)
            await session.commit()
            message_id = result.inserted_primary_key[0]
            return {**message, "id": str(message_id), "conversationType": conversation_type, "senderName": sender_name, "createdAt": created_at}

    async def get_messages(self, limit: int = 100, offset: int = 0, session_id: Optional[str] = None) -> List[Dict]:
        """获取消息记录，支持按会话 ID 筛选。

        Args:
            limit: 返回条数上限
            offset: 偏移量
            session_id: 可选，指定会话 ID 时在 SQL 层过滤（与 SQLite 实现对齐）
        """
        async with self.async_session() as session:
            stmt = (
                messages_table.select()
                .order_by(messages_table.c.createdAt.desc())
                .limit(limit).offset(offset)
            )
            if session_id:
                stmt = stmt.where(messages_table.c.sessionId == session_id)
            result = await session.execute(stmt)
            return [_row_to_dict(row) for row in result.fetchall()]

    async def get_message_count(self) -> int:
        """获取消息总数"""
        async with self.async_session() as session:
            stmt = text("SELECT COUNT(*) FROM messages")
            result = await session.execute(stmt)
            return result.scalar()


    def _message_filter_conditions(
        self,
        search: Optional[str] = None,
        session_type: Optional[str] = None,
        lora_name: Optional[str] = None,
        session_id: Optional[str] = None,
        session_name: Optional[str] = None,
        platform: Optional[str] = None,
    ):
        conditions = []
        if search:
            pattern = f"%{search}%"
            conditions.append(
                (messages_table.c.message.like(pattern))
                | (messages_table.c.reply.like(pattern))
                | (messages_table.c.userName.like(pattern))
            )
        if session_type:
            conditions.append(messages_table.c.sessionType == session_type)
        if lora_name:
            conditions.append(messages_table.c.loraName == lora_name)
        if session_id:
            conditions.append(messages_table.c.sessionId == session_id)
        if session_name:
            conditions.append(messages_table.c.sessionName.like(f"%{session_name}%"))
        if platform:
            conditions.append(messages_table.c.platform == platform)
        return conditions

    async def get_message_count_filtered(
        self,
        search: Optional[str] = None,
        session_type: Optional[str] = None,
        lora_name: Optional[str] = None,
        session_id: Optional[str] = None,
        session_name: Optional[str] = None,
        platform: Optional[str] = None,
    ) -> int:
        """Return the exact count for the same filters used by get_messages_filtered."""
        async with self.async_session() as session:
            from sqlalchemy import and_, func, select

            stmt = select(func.count()).select_from(messages_table)
            conditions = self._message_filter_conditions(
                search, session_type, lora_name, session_id, session_name, platform
            )
            if conditions:
                stmt = stmt.where(and_(*conditions))
            result = await session.execute(stmt)
            return int(result.scalar() or 0)

    async def get_messages_filtered(
        self,
        search: Optional[str] = None,
        session_type: Optional[str] = None,
        lora_name: Optional[str] = None,
        session_id: Optional[str] = None,
        session_name: Optional[str] = None,
        platform: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict]:
        """Get messages with SQL-level filtering and pagination."""
        async with self.async_session() as session:
            from sqlalchemy import and_

            conditions = self._message_filter_conditions(
                search=search,
                session_type=session_type,
                lora_name=lora_name,
                session_id=session_id,
                session_name=session_name,
                platform=platform,
            )

            stmt = messages_table.select()
            if conditions:
                stmt = stmt.where(and_(*conditions))
            stmt = (
                stmt.order_by(messages_table.c.createdAt.desc())
                .limit(min(limit, 1000))
                .offset(offset)
            )
            result = await session.execute(stmt)
            return [_row_to_dict(row) for row in result.fetchall()]

    async def delete_message(self, msg_id: int) -> bool:
        """删除单条消息记录"""
        async with self.async_session() as session:
            stmt = messages_table.delete().where(messages_table.c.id == msg_id)
            result = await session.execute(stmt)
            await session.commit()
            return result.rowcount > 0

    async def delete_messages_by_filter(
        self,
        search: Optional[str] = None,
        sessionType: Optional[str] = None,
        lora: Optional[str] = None,
        sessionName: Optional[str] = None,
        platform: Optional[str] = None,
    ) -> int:
        """批量删除消息（基于筛选条件），返回删除数量"""
        async with self.async_session() as session:
            conditions = []
            if search:
                conditions.append(
                    (messages_table.c.message.like(f"%{search}%"))
                    | (messages_table.c.reply.like(f"%{search}%"))
                    | (messages_table.c.userName.like(f"%{search}%"))
                )
            if sessionType and sessionType != "all":
                conditions.append(messages_table.c.sessionType == sessionType)
            if lora and lora != "all":
                conditions.append(messages_table.c.loraName == lora)
            if sessionName:
                conditions.append(messages_table.c.sessionName.like(f"%{sessionName}%"))
            if platform and platform != "all":
                conditions.append(messages_table.c.platform == platform)

            stmt = messages_table.delete()
            if conditions:
                from sqlalchemy import and_
                stmt = stmt.where(and_(*conditions))

            result = await session.execute(stmt)
            await session.commit()
            return result.rowcount

    async def get_recent_messages(self, limit: int = 10) -> List[Dict]:
        """获取最近消息"""
        async with self.async_session() as session:
            stmt = (
                messages_table.select()
                .order_by(messages_table.c.createdAt.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            return [_row_to_dict(row) for row in result.fetchall()]

    # ============================================
    # 配置管理
    # ============================================
    async def get_config(self) -> Dict:
        """获取所有配置"""
        async with self.async_session() as session:
            result = await session.execute(config_table.select())
            config_dict = {}
            for row in result.fetchall():
                d = _row_to_dict(row)
                key, value = d["key"], d["value"]
                if value.lower() == "true":
                    config_dict[key] = True
                elif value.lower() == "false":
                    config_dict[key] = False
                else:
                    try:
                        config_dict[key] = float(value) if "." in value else int(value)
                    except (ValueError, TypeError):
                        config_dict[key] = value
            return config_dict

    async def get_config_value(self, key: str, default=None):
        """获取单个配置项的值"""
        config_dict = await self.get_config()
        return config_dict.get(key, default)

    async def set_config(self, new_config: Dict):
        """更新配置（upsert）"""
        async with self.async_session() as session:
            for key, value in new_config.items():
                if isinstance(value, bool):
                    value_str = str(value).lower()
                else:
                    value_str = str(value)
                # PostgreSQL upsert: INSERT ... ON CONFLICT DO UPDATE
                stmt = text(
                    "INSERT INTO config (key, value) VALUES (:k, :v) "
                    "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value"
                )
                await session.execute(stmt, {"k": key, "v": value_str})
            await session.commit()

    async def set_config_value(self, key: str, value: Any):
        """设置单个配置项"""
        await self.set_config({key: value})

    # ============================================
    # LoRA 管理
    # ============================================
    async def get_loras(self, status: Optional[str] = None) -> List[Dict]:
        """获取 LoRA 模型列表"""
        async with self.async_session() as session:
            if status and status != "all":
                stmt = loras_table.select().where(loras_table.c.status == status)
            else:
                stmt = loras_table.select()
            result = await session.execute(stmt)
            return [_row_to_dict(row) for row in result.fetchall()]

    async def add_lora(self, lora: Dict) -> Dict:
        """添加 LoRA 模型"""
        async with self.async_session() as session:
            stmt = loras_table.insert().values(
                id=lora.get("id"),
                name=lora.get("name", ""),
                description=lora.get("description", ""),
                status=lora.get("status", "inactive"),
                style=lora.get("style", ""),
                size=lora.get("size", ""),
                trainedSteps=lora.get("trainedSteps", 0),
                totalSteps=lora.get("totalSteps", 0),
                createdAt=lora.get("createdAt", datetime.now().strftime("%Y-%m-%d")),
            )
            await session.execute(stmt)
            await session.commit()
            return lora

    async def update_lora_status(self, lora_id: str, status: str) -> Optional[Dict]:
        """更新 LoRA 模型状态"""
        async with self.async_session() as session:
            if status == "active":
                # 先将所有其他 LoRA 设为 inactive
                await session.execute(
                    loras_table.update().where(loras_table.c.id != lora_id).values(status="inactive")
                )
            await session.execute(
                loras_table.update().where(loras_table.c.id == lora_id).values(status=status)
            )
            await session.commit()

            # 获取更新后的记录
            stmt = loras_table.select().where(loras_table.c.id == lora_id)
            result = await session.execute(stmt)
            row = result.fetchone()
            return _row_to_dict(row) if row else None

    async def delete_lora(self, lora_id: str) -> bool:
        """删除 LoRA 模型"""
        async with self.async_session() as session:
            stmt = loras_table.delete().where(loras_table.c.id == lora_id)
            result = await session.execute(stmt)
            await session.commit()
            return result.rowcount > 0

    # ============================================
    # 知识库管理
    # ============================================
    async def create_knowledge_base(self, name: str, description: str = "") -> Optional[Dict]:
        """创建知识库"""
        now = datetime.now().isoformat()
        async with self.async_session() as session:
            try:
                stmt = knowledge_bases_table.insert().values(
                    name=name, description=description, created_at=now, updated_at=now,
                )
                result = await session.execute(stmt)
                await session.commit()
                kb_id = result.inserted_primary_key[0]
                return {"id": kb_id, "name": name, "description": description, "created_at": now, "updated_at": now}
            except Exception as e:
                await session.rollback()
                if "unique" in str(e).lower() or "duplicate" in str(e).lower():
                    return None
                raise

    async def get_knowledge_bases(self) -> List[Dict]:
        """获取所有知识库"""
        async with self.async_session() as session:
            stmt = knowledge_bases_table.select().order_by(knowledge_bases_table.c.updated_at.desc())
            result = await session.execute(stmt)
            rows = result.fetchall()
            kb_list = []
            for row in rows:
                kb = _row_to_dict(row)
                # 统计文档数
                cnt_stmt = text("SELECT COUNT(*) as cnt FROM knowledge_documents WHERE knowledge_base_id = :kb_id")
                cnt_result = await session.execute(cnt_stmt, {"kb_id": kb["id"]})
                kb["documentCount"] = cnt_result.scalar()
                # 统计文件夹数
                folder_cnt_stmt = text("SELECT COUNT(*) as cnt FROM knowledge_folders WHERE knowledge_base_id = :kb_id")
                folder_cnt_result = await session.execute(folder_cnt_stmt, {"kb_id": kb["id"]})
                kb["folderCount"] = folder_cnt_result.scalar()
                kb_list.append(kb)
            return kb_list

    async def get_knowledge_base(self, kb_id: int) -> Optional[Dict]:
        """获取单个知识库"""
        async with self.async_session() as session:
            stmt = knowledge_bases_table.select().where(knowledge_bases_table.c.id == kb_id)
            result = await session.execute(stmt)
            row = result.fetchone()
            return _row_to_dict(row) if row else None

    async def update_knowledge_base(self, kb_id: int, data: Dict) -> Optional[Dict]:
        """更新知识库"""
        now = datetime.now().isoformat()
        values = {"updated_at": now}
        if "name" in data and data["name"] is not None:
            values["name"] = data["name"]
        if "description" in data and data["description"] is not None:
            values["description"] = data["description"]
        async with self.async_session() as session:
            stmt = (
                knowledge_bases_table.update()
                .where(knowledge_bases_table.c.id == kb_id)
                .values(**values)
            )
            await session.execute(stmt)
            await session.commit()
            # 获取更新后的记录
            sel_stmt = knowledge_bases_table.select().where(knowledge_bases_table.c.id == kb_id)
            result = await session.execute(sel_stmt)
            row = result.fetchone()
            return _row_to_dict(row) if row else None

    async def delete_knowledge_base(self, kb_id: int) -> bool:
        """删除知识库（级联删除文件夹和文档）"""
        async with self.async_session() as session:
            # 先删除关联文档的 chunks
            await session.execute(
                text(
                    "DELETE FROM knowledge_chunks WHERE \"documentId\" IN "
                    "(SELECT id FROM knowledge_documents WHERE knowledge_base_id = :kb_id)"
                ),
                {"kb_id": kb_id},
            )
            await session.execute(
                knowledge_documents_table.delete().where(knowledge_documents_table.c.knowledge_base_id == kb_id)
            )
            await session.execute(
                knowledge_folders_table.delete().where(knowledge_folders_table.c.knowledge_base_id == kb_id)
            )
            await session.execute(
                knowledge_bases_table.delete().where(knowledge_bases_table.c.id == kb_id)
            )
            await session.commit()
            return True

    # ============================================
    # 知识库文件夹管理
    # ============================================
    async def create_knowledge_folder(self, kb_id: int, name: str, description: str = "") -> Optional[Dict]:
        """创建知识库文件夹"""
        now = datetime.now().isoformat()
        async with self.async_session() as session:
            try:
                stmt = knowledge_folders_table.insert().values(
                    knowledge_base_id=kb_id, name=name, description=description, created_at=now, updated_at=now,
                )
                result = await session.execute(stmt)
                await session.commit()
                folder_id = result.inserted_primary_key[0]
                return {
                    "id": folder_id, "knowledge_base_id": kb_id,
                    "name": name, "description": description, "created_at": now, "updated_at": now,
                }
            except Exception as e:
                await session.rollback()
                if "unique" in str(e).lower() or "duplicate" in str(e).lower():
                    return None
                raise

    async def get_knowledge_folders(self, kb_id: int) -> List[Dict]:
        """获取知识库下的所有文件夹"""
        async with self.async_session() as session:
            stmt = (
                knowledge_folders_table.select()
                .where(knowledge_folders_table.c.knowledge_base_id == kb_id)
                .order_by(knowledge_folders_table.c.name)
            )
            result = await session.execute(stmt)
            rows = result.fetchall()
            folder_list = []
            for row in rows:
                folder = _row_to_dict(row)
                cnt_stmt = text("SELECT COUNT(*) as cnt FROM knowledge_documents WHERE folder_id = :fid")
                cnt_result = await session.execute(cnt_stmt, {"fid": folder["id"]})
                folder["documentCount"] = cnt_result.scalar()
                folder_list.append(folder)
            return folder_list

    async def get_knowledge_folder(self, folder_id: int) -> Optional[Dict]:
        """获取单个文件夹"""
        async with self.async_session() as session:
            stmt = knowledge_folders_table.select().where(knowledge_folders_table.c.id == folder_id)
            result = await session.execute(stmt)
            row = result.fetchone()
            return _row_to_dict(row) if row else None

    async def delete_knowledge_folder(self, folder_id: int) -> bool:
        """删除文件夹（文档的 folder_id 置空）"""
        async with self.async_session() as session:
            await session.execute(
                knowledge_documents_table.update()
                .where(knowledge_documents_table.c.folder_id == folder_id)
                .values(folder_id=None)
            )
            await session.execute(
                knowledge_folders_table.delete().where(knowledge_folders_table.c.id == folder_id)
            )
            await session.commit()
            return True

    # ============================================
    # 知识库文档管理
    # ============================================
    async def add_knowledge_document(self, document: Dict) -> Dict:
        """添加知识库文档"""
        now = datetime.now().isoformat()
        async with self.async_session() as session:
            stmt = knowledge_documents_table.insert().values(
                title=document.get("title", ""),
                content=document.get("content", ""),
                category=document.get("category", "未分类"),
                knowledge_base_id=document.get("knowledge_base_id"),
                folder_id=document.get("folder_id"),
                sourceType=document.get("sourceType", "text"),
                sourceUrl=document.get("sourceUrl"),
                fileType=document.get("fileType"),
                fileSize=document.get("fileSize"),
                chunkCount=document.get("chunkCount", 0),
                createdAt=now,
                updatedAt=now,
            )
            result = await session.execute(stmt)
            await session.commit()
            doc_id = result.inserted_primary_key[0]
            return {**document, "id": doc_id, "createdAt": now, "updatedAt": now}

    async def get_knowledge_documents(
        self,
        limit: int = 100,
        offset: int = 0,
        category: Optional[str] = None,
        knowledge_base_id: Optional[int] = None,
        folder_id: Optional[int] = None,
    ) -> List[Dict]:
        """获取知识库文档列表"""
        async with self.async_session() as session:
            conditions = []
            if category and category != "全部":
                conditions.append(knowledge_documents_table.c.category == category)
            if knowledge_base_id is not None:
                conditions.append(knowledge_documents_table.c.knowledge_base_id == knowledge_base_id)
            if folder_id is not None:
                conditions.append(knowledge_documents_table.c.folder_id == folder_id)

            stmt = knowledge_documents_table.select()
            if conditions:
                from sqlalchemy import and_
                stmt = stmt.where(and_(*conditions))
            stmt = stmt.order_by(knowledge_documents_table.c.updatedAt.desc()).limit(limit).offset(offset)

            result = await session.execute(stmt)
            return [_row_to_dict(row) for row in result.fetchall()]

    async def get_knowledge_document(self, doc_id: int) -> Optional[Dict]:
        """获取单个知识库文档"""
        async with self.async_session() as session:
            stmt = knowledge_documents_table.select().where(knowledge_documents_table.c.id == doc_id)
            result = await session.execute(stmt)
            row = result.fetchone()
            return _row_to_dict(row) if row else None

    KNOWLEDGE_DOC_UPDATABLE_COLUMNS = {
        "title", "content", "category", "knowledge_base_id", "folder_id",
        "sourceType", "sourceUrl", "fileType", "fileSize",
        "chunkCount", "updatedAt",
    }

    async def update_knowledge_document(self, doc_id: int, document: Dict) -> Optional[Dict]:
        """更新知识库文档 - 只更新提供的字段（白名单校验）"""
        now = datetime.now().isoformat()
        async with self.async_session() as session:
            values = {"updatedAt": now}
            for key, value in document.items():
                if key in ("id", "createdAt"):
                    continue
                if key not in self.KNOWLEDGE_DOC_UPDATABLE_COLUMNS:
                    logger.warning(f"update_knowledge_document: 忽略非法列名 '{key}'")
                    continue
                if value is not None:
                    values[key] = value

            stmt = (
                knowledge_documents_table.update()
                .where(knowledge_documents_table.c.id == doc_id)
                .values(**values)
            )
            await session.execute(stmt)
            await session.commit()

            # 获取更新后的文档
            sel_stmt = knowledge_documents_table.select().where(knowledge_documents_table.c.id == doc_id)
            result = await session.execute(sel_stmt)
            row = result.fetchone()
            return _row_to_dict(row) if row else None

    async def delete_knowledge_document(self, doc_id: int) -> bool:
        """删除知识库文档"""
        async with self.async_session() as session:
            await session.execute(
                knowledge_chunks_table.delete().where(knowledge_chunks_table.c.documentId == doc_id)
            )
            await session.execute(
                knowledge_documents_table.delete().where(knowledge_documents_table.c.id == doc_id)
            )
            await session.commit()
            return True

    # ============================================
    # 知识库分块管理
    # ============================================
    async def add_knowledge_chunk(self, chunk: Dict) -> Dict:
        """添加知识库文档片段"""
        now = datetime.now().isoformat()
        async with self.async_session() as session:
            stmt = knowledge_chunks_table.insert().values(
                documentId=chunk.get("documentId"),
                chunkIndex=chunk.get("chunkIndex"),
                content=chunk.get("content"),
                embedding=chunk.get("embedding"),
                createdAt=now,
            )
            result = await session.execute(stmt)
            await session.commit()
            chunk_id = result.inserted_primary_key[0]
            return {**chunk, "id": chunk_id, "createdAt": now}

    async def get_knowledge_chunks(self, doc_id: int) -> List[Dict]:
        """获取文档的所有片段"""
        async with self.async_session() as session:
            stmt = (
                knowledge_chunks_table.select()
                .where(knowledge_chunks_table.c.documentId == doc_id)
                .order_by(knowledge_chunks_table.c.chunkIndex)
            )
            result = await session.execute(stmt)
            return [_row_to_dict(row) for row in result.fetchall()]

    async def get_all_knowledge_chunks(self) -> List[Dict]:
        """获取所有知识库片段（用于检索）"""
        async with self.async_session() as session:
            stmt = knowledge_chunks_table.select().order_by(knowledge_chunks_table.c.documentId, knowledge_chunks_table.c.chunkIndex)
            result = await session.execute(stmt)
            return [_row_to_dict(row) for row in result.fetchall()]

    async def get_knowledge_stats(self) -> Dict:
        """获取知识库统计数据"""
        async with self.async_session() as session:
            total_docs_result = await session.execute(text("SELECT COUNT(*) FROM knowledge_documents"))
            total_docs = total_docs_result.scalar()

            total_chunks_result = await session.execute(text("SELECT COUNT(*) FROM knowledge_chunks"))
            total_chunks = total_chunks_result.scalar()

            total_chars_result = await session.execute(text("SELECT SUM(LENGTH(content)) FROM knowledge_documents"))
            total_chars = total_chars_result.scalar() or 0

            return {
                "totalDocuments": total_docs,
                "totalChunks": total_chunks,
                "totalCharacters": total_chars,
            }

    # ============================================
    # 用户管理
    # ============================================
    async def add_user(self, username: str, password_hash: str) -> Dict:
        """添加用户"""
        now = datetime.now().isoformat()
        async with self.async_session() as session:
            # C4 fix: 首个用户自动成为 admin，后续用户为普通 user。
            # 与 SQLite 实现 (database.py:1819-1822) 保持一致。
            count_stmt = users_table.select().order_by(users_table.c.id).limit(1)
            existing = await session.execute(count_stmt)
            role = "admin" if existing.fetchone() is None else "user"
            stmt = users_table.insert().values(
                username=username, password_hash=password_hash, created_at=now, role=role,
            )
            result = await session.execute(stmt)
            await session.commit()
            user_id = result.inserted_primary_key[0]
            return {"id": user_id, "username": username, "created_at": now, "role": role}

    async def get_user(self, user_id: int) -> Optional[Dict]:
        """获取用户 by ID"""
        async with self.async_session() as session:
            stmt = users_table.select().where(users_table.c.id == user_id)
            result = await session.execute(stmt)
            row = result.fetchone()
            return _row_to_dict(row) if row else None

    async def get_user_by_username(self, username: str) -> Optional[Dict]:
        """获取用户 by username"""
        async with self.async_session() as session:
            # C4 fix: 显式选择 role 列（此前 select(*) 已包含，但 _row_to_dict 需确认）
            stmt = users_table.select().where(users_table.c.username == username)
            result = await session.execute(stmt)
            row = result.fetchone()
            return _row_to_dict(row) if row else None

    # ============================================
    # 用户数据持久化
    # ============================================
    async def get_user_data(self, user_id: int, page_key: Optional[str] = None) -> Any:
        """获取用户表单数据"""
        async with self.async_session() as session:
            if page_key:
                stmt = user_data_table.select().where(
                    (user_data_table.c.user_id == user_id) & (user_data_table.c.page_key == page_key)
                )
                result = await session.execute(stmt)
                row = result.fetchone()
                if not row:
                    return None
                d = _row_to_dict(row)
                return {"page_key": d["page_key"], "data_json": d["data_json"], "updated_at": d["updated_at"]}
            else:
                stmt = user_data_table.select().where(user_data_table.c.user_id == user_id)
                result = await session.execute(stmt)
                data = {}
                for row in result.fetchall():
                    d = _row_to_dict(row)
                    data[d["page_key"]] = {"data_json": d["data_json"], "updated_at": d["updated_at"]}
                return data

    async def save_user_data(self, user_id: int, page_key: str, data_json: str) -> bool:
        """保存用户表单数据（upsert）"""
        now = datetime.now().isoformat()
        async with self.async_session() as session:
            stmt = text(
                "INSERT INTO user_data (user_id, page_key, data_json, updated_at) "
                "VALUES (:uid, :pk, :dj, :ua) "
                "ON CONFLICT (user_id, page_key) DO UPDATE SET "
                "data_json = EXCLUDED.data_json, updated_at = EXCLUDED.updated_at"
            )
            await session.execute(stmt, {"uid": user_id, "pk": page_key, "dj": data_json, "ua": now})
            await session.commit()
            return True

    # ============================================
    # 会话管理
    # ============================================
    async def _upsert_conversation_session(
        self,
        session: AsyncSession,
        *,
        platform: str,
        conversation_id: str,
        conversation_type: str = "private",
        display_name: str = "",
        bot_enabled: bool | None = None,
        reply_policy: str | None = None,
    ) -> None:
        if not conversation_id:
            return
        now = datetime.now().isoformat()
        await session.execute(text('''
            INSERT INTO conversations (platform, "conversationId", "conversationType", "displayName", "botEnabled", "replyPolicy", "createdAt", "updatedAt")
            VALUES (:platform, :conversation_id, :conversation_type, :display_name, :bot_enabled, :reply_policy, :created_at, :updated_at)
            ON CONFLICT (platform, "conversationId", "conversationType") DO UPDATE SET
                "displayName" = COALESCE(NULLIF(EXCLUDED."displayName", ''), conversations."displayName"),
                "botEnabled" = CASE WHEN :bot_enabled_is_null THEN conversations."botEnabled" ELSE EXCLUDED."botEnabled" END,
                "replyPolicy" = CASE WHEN :reply_policy_is_null THEN conversations."replyPolicy" ELSE EXCLUDED."replyPolicy" END,
                "updatedAt" = EXCLUDED."updatedAt"
        '''), {
            "platform": platform,
            "conversation_id": conversation_id,
            "conversation_type": conversation_type,
            "display_name": display_name,
            "bot_enabled": 1 if bot_enabled is None else int(bot_enabled),
            "reply_policy": reply_policy or "default",
            "created_at": now,
            "updated_at": now,
            "bot_enabled_is_null": bot_enabled is None,
            "reply_policy_is_null": reply_policy is None,
        })

    async def upsert_conversation(self, data: Dict) -> None:
        async with self.async_session() as session:
            await self._upsert_conversation_session(
                session,
                platform=data.get("platform", "qq"),
                conversation_id=data.get("conversationId") or data.get("sessionId", ""),
                conversation_type=data.get("conversationType") or data.get("sessionType", "private"),
                display_name=data.get("displayName") or data.get("sessionName", ""),
                bot_enabled=data.get("botEnabled") if "botEnabled" in data else None,
                reply_policy=data.get("replyPolicy"),
            )
            await session.commit()

    async def get_conversation(self, platform: str, conversation_id: str, conversation_type: Optional[str] = None) -> Optional[Dict]:
        async with self.async_session() as session:
            if conversation_type:
                stmt = text('SELECT * FROM conversations WHERE platform = :platform AND "conversationId" = :conversation_id AND "conversationType" = :conversation_type LIMIT 1')
                result = await session.execute(stmt, {"platform": platform, "conversation_id": conversation_id, "conversation_type": conversation_type})
            else:
                stmt = text('SELECT * FROM conversations WHERE platform = :platform AND "conversationId" = :conversation_id ORDER BY "updatedAt" DESC LIMIT 1')
                result = await session.execute(stmt, {"platform": platform, "conversation_id": conversation_id})
            row = result.fetchone()
            return _row_to_dict(row) if row else None

    async def add_integration_event(self, event: Dict) -> None:
        raw_summary = event.get("rawSummary", "")
        if not isinstance(raw_summary, str):
            raw_summary = json.dumps(raw_summary, ensure_ascii=False, default=str)
        event_hash = event.get("eventHash") or f"{event.get('sourceMessageId', '')}:{event.get('traceId', '')}"
        async with self.async_session() as session:
            await session.execute(text('''
                INSERT INTO integration_events (platform, adapter, "sourceMessageId", "conversationId", "conversationType", "senderId", "eventType", "eventHash", "rawSummary", "traceId", status, "createdAt")
                VALUES (:platform, :adapter, :source_message_id, :conversation_id, :conversation_type, :sender_id, :event_type, :event_hash, :raw_summary, :trace_id, :status, :created_at)
                ON CONFLICT (platform, adapter, "eventHash") DO UPDATE SET
                    "traceId" = EXCLUDED."traceId", status = EXCLUDED.status, "rawSummary" = EXCLUDED."rawSummary"
            '''), {
                "platform": event.get("platform", "qq"),
                "adapter": event.get("adapter", "other"),
                "source_message_id": event.get("sourceMessageId", ""),
                "conversation_id": event.get("conversationId", ""),
                "conversation_type": event.get("conversationType", "private"),
                "sender_id": event.get("senderId", ""),
                "event_type": event.get("eventType", "message"),
                "event_hash": event_hash,
                "raw_summary": raw_summary[:4096],
                "trace_id": event.get("traceId", ""),
                "status": event.get("status", "received"),
                "created_at": event.get("createdAt", datetime.now().isoformat()),
            })
            await session.commit()

    async def add_model_invocation(self, invocation: Dict) -> None:
        prompt_tokens = int(invocation.get("promptTokens", 0) or 0)
        completion_tokens = int(invocation.get("completionTokens", 0) or 0)
        total_tokens = int(invocation.get("totalTokens", prompt_tokens + completion_tokens) or 0)
        async with self.async_session() as session:
            await session.execute(text('''
                INSERT INTO model_invocations ("traceId", platform, "conversationId", "sessionId", "modelName", "loraName", "costTime", "promptTokens", "completionTokens", "totalTokens", "usedRag", "usedLora", "errorType", "createdAt")
                VALUES (:trace_id, :platform, :conversation_id, :session_id, :model_name, :lora_name, :cost_time, :prompt_tokens, :completion_tokens, :total_tokens, :used_rag, :used_lora, :error_type, :created_at)
            '''), {
                "trace_id": invocation.get("traceId", ""),
                "platform": invocation.get("platform", "qq"),
                "conversation_id": invocation.get("conversationId", ""),
                "session_id": invocation.get("sessionId", ""),
                "model_name": invocation.get("modelName", ""),
                "lora_name": invocation.get("loraName", ""),
                "cost_time": float(invocation.get("costTime", 0.0) or 0.0),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "used_rag": int(bool(invocation.get("usedRag", False))),
                "used_lora": int(bool(invocation.get("usedLora", False))),
                "error_type": invocation.get("errorType", ""),
                "created_at": invocation.get("createdAt", datetime.now().isoformat()),
            })
            await session.commit()

    async def get_session_summaries(self) -> List[Dict]:
        """获取所有会话的聚合统计信息"""
        async with self.async_session() as session:
            stmt = text("""
                SELECT
                    "sessionId",
                    "sessionType",
                    "sessionName",
                    COALESCE(platform, 'qq') as platform,
                    COALESCE(adapter, 'nonebot') as adapter,
                    COALESCE("conversationId", "sessionId") as "conversationId",
                    COUNT(*) as message_count,
                    MAX("createdAt") as last_active,
                    STRING_AGG(message, '||' ORDER BY "createdAt") as recent_messages
                FROM messages
                GROUP BY "sessionId", "sessionType", "sessionName", platform, adapter, "conversationId"
                ORDER BY last_active DESC
            """)
            result = await session.execute(stmt)
            sessions = []
            for row in result.fetchall():
                d = _row_to_dict(row)
                session_id = d["sessionId"]
                session_type = d["sessionType"]
                session_name = d["sessionName"] or session_id
                platform = d.get("platform") or "qq"
                adapter = d.get("adapter") or "nonebot"
                conversation_id = d.get("conversationId") or session_id
                message_count = d["message_count"]
                last_active = d["last_active"]

                raw_msgs = (d.get("recent_messages") or "").split("||")
                recent = [m for m in raw_msgs if m.strip()][-3:]
                summary = "；".join(recent[:3])
                if len(summary) > 100:
                    summary = summary[:100] + "..."

                # 查询该会话的机器人开关状态（统一从 conversations 表查询）
                setting_stmt = text(
                    'SELECT "botEnabled" FROM conversations WHERE platform = :platform AND "conversationId" = :cid ORDER BY "updatedAt" DESC LIMIT 1'
                )
                setting_result = await session.execute(setting_stmt, {"platform": platform, "cid": conversation_id})
                setting_row = setting_result.fetchone()
                bot_enabled = _row_to_dict(setting_row)["botEnabled"] if setting_row else 1

                sessions.append({
                    "sessionId": session_id,
                    "sessionType": session_type,
                    "sessionName": session_name,
                    "platform": platform,
                    "adapter": adapter,
                    "conversationId": conversation_id,
                    "messageCount": message_count,
                    "lastActive": last_active,
                    "summary": summary,
                    "botEnabled": bool(bot_enabled),
                })
            return sessions

    async def set_session_bot_enabled(self, session_id: str, enabled: bool, platform: str = "qq", conversation_id: Optional[str] = None) -> None:
        """设置某个会话的机器人开关。

        统一写入 conversations 表（此前同时写 session_settings + conversations 双表，
        现合并为单表，消除冗余）。
        """
        async with self.async_session() as session:
            resolved_conversation_id = conversation_id or session_id
            await self._upsert_conversation_session(
                session,
                platform=platform,
                conversation_id=resolved_conversation_id,
                conversation_type="private",
                display_name=session_id,
                bot_enabled=enabled,
            )
            await session.commit()

    async def is_session_bot_enabled(self, session_id: str, platform: str = "qq", conversation_id: Optional[str] = None) -> bool:
        """检查某个会话的机器人是否启用（默认启用）。

        统一从 conversations 表查询（此前先查 session_settings 失败再查 conversations，
        现直接查 conversations，消除双表冗余查询）。
        """
        async with self.async_session() as session:
            resolved_conversation_id = conversation_id or session_id
            stmt = text(
                'SELECT "botEnabled" FROM conversations WHERE platform = :platform AND "conversationId" = :cid ORDER BY "updatedAt" DESC LIMIT 1'
            )
            result = await session.execute(stmt, {"platform": platform, "cid": resolved_conversation_id})
            row = result.fetchone()
            if row is None:
                return True
            return bool(_row_to_dict(row)["botEnabled"])

    # ============================================
    # Claw 工具 CRUD
    # ============================================
    async def mark_integration_message_processed(self, platform: str, adapter: str, message_id: str) -> bool:
        if not message_id:
            return True
        key = f"{platform}:{adapter}:{message_id}"
        async with self.async_session() as session:
            stmt = text(
                'INSERT INTO integration_message_dedup ("dedupKey", platform, adapter, "messageId", "createdAt") '
                'VALUES (:key, :platform, :adapter, :message_id, :created_at) '
                'ON CONFLICT ("dedupKey") DO NOTHING'
            )
            result = await session.execute(stmt, {
                "key": key,
                "platform": platform,
                "adapter": adapter,
                "message_id": message_id,
                "created_at": datetime.now().isoformat(),
            })
            await session.commit()
            return result.rowcount > 0

    async def get_claw_tools(self) -> List[Dict]:
        """获取所有自定义 Claw 工具"""
        async with self.async_session() as session:
            stmt = claw_tools_table.select().order_by(claw_tools_table.c.created_at.desc())
            result = await session.execute(stmt)
            return [_row_to_dict(row) for row in result.fetchall()]

    async def get_claw_tool_by_name(self, name: str) -> Optional[Dict]:
        """按名称获取单个工具"""
        async with self.async_session() as session:
            stmt = claw_tools_table.select().where(claw_tools_table.c.name == name)
            result = await session.execute(stmt)
            row = result.fetchone()
            return _row_to_dict(row) if row else None

    async def save_claw_tool(self, name: str, description: str, code: str, enabled: bool = True) -> int:
        """创建或更新自定义 Claw 工具，返回工具 id"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        async with self.async_session() as session:
            stmt = text(
                "INSERT INTO claw_tools (name, description, code, enabled, created_at, updated_at) "
                "VALUES (:n, :d, :c, :e, :ca, :ua) "
                "ON CONFLICT (name) DO UPDATE SET "
                "description = EXCLUDED.description, code = EXCLUDED.code, "
                "enabled = EXCLUDED.enabled, updated_at = EXCLUDED.updated_at"
            )
            result = await session.execute(stmt, {
                "n": name, "d": description, "c": code, "e": int(enabled), "ca": now, "ua": now,
            })
            await session.commit()
            # 获取 upsert 后的 id
            sel_stmt = claw_tools_table.select().where(claw_tools_table.c.name == name)
            sel_result = await session.execute(sel_stmt)
            row = sel_result.fetchone()
            return _row_to_dict(row)["id"] if row else 0

    async def delete_claw_tool(self, name: str) -> bool:
        """删除自定义 Claw 工具"""
        async with self.async_session() as session:
            stmt = claw_tools_table.delete().where(claw_tools_table.c.name == name)
            result = await session.execute(stmt)
            await session.commit()
            return result.rowcount > 0

    # ============================================
    # 审计日志
    # ============================================
    async def add_audit_log(
        self,
        api_key_hash: str,
        role: str,
        action: str,
        resource: Optional[str] = None,
        detail: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> None:
        """记录审计日志"""
        import time
        async with self.async_session() as session:
            stmt = audit_logs_table.insert().values(
                timestamp=time.time(),
                api_key_hash=api_key_hash,
                role=role,
                action=action,
                resource=resource,
                detail=detail,
                ip_address=ip_address,
            )
            await session.execute(stmt)
            await session.commit()

    async def get_audit_logs(
        self,
        limit: int = 100,
        offset: int = 0,
        role: Optional[str] = None,
        action: Optional[str] = None,
    ) -> List[Dict]:
        """查询审计日志"""
        async with self.async_session() as session:
            conditions = []
            if role:
                conditions.append(audit_logs_table.c.role == role)
            if action:
                conditions.append(audit_logs_table.c.action == action)

            stmt = audit_logs_table.select()
            if conditions:
                from sqlalchemy import and_
                stmt = stmt.where(and_(*conditions))
            stmt = stmt.order_by(audit_logs_table.c.id.desc()).limit(limit).offset(offset)

            result = await session.execute(stmt)
            return [_row_to_dict(row) for row in result.fetchall()]

    # ============================================
    # 意图样本管理
    # ============================================
    async def add_intent_sample(self, kb_name: str, text: str, label: str) -> Dict:
        """添加意图样本"""
        now = datetime.now().isoformat()
        async with self.async_session() as session:
            stmt = intent_samples_table.insert().values(
                kbName=kb_name, text=text, label=label, createdAt=now,
            )
            result = await session.execute(stmt)
            await session.commit()
            sample_id = result.inserted_primary_key[0]
            return {"id": sample_id, "kbName": kb_name, "text": text, "label": label, "createdAt": now}

    async def get_intent_samples(self, kb_name: Optional[str] = None) -> List[Dict]:
        """获取意图样本"""
        async with self.async_session() as session:
            stmt = intent_samples_table.select()
            if kb_name:
                stmt = stmt.where(intent_samples_table.c.kbName == kb_name)
            result = await session.execute(stmt)
            return [_row_to_dict(row) for row in result.fetchall()]

    async def get_active_kbs(self) -> List[Dict]:
        """获取活跃的知识库列表"""
        async with self.async_session() as session:
            stmt = intent_active_kbs_table.select().where(intent_active_kbs_table.c.isActive == 1)
            result = await session.execute(stmt)
            return [_row_to_dict(row) for row in result.fetchall()]

    async def set_active_kb(self, kb_name: str, is_active: bool) -> None:
        """设置知识库活跃状态"""
        async with self.async_session() as session:
            stmt = text(
                "INSERT INTO intent_active_kbs (\"kbName\", \"isActive\") "
                "VALUES (:kbn, :ia) "
                "ON CONFLICT DO NOTHING"
            )
            await session.execute(stmt, {"kbn": kb_name, "ia": int(is_active)})
            # Update if exists
            upd_stmt = (
                intent_active_kbs_table.update()
                .where(intent_active_kbs_table.c.kbName == kb_name)
                .values(isActive=int(is_active))
            )
            await session.execute(upd_stmt)
            await session.commit()

    # ============================================
    # 训练任务管理
    # ============================================
    @staticmethod
    def _normalize_training_task(row) -> Optional[Dict]:
        if row is None:
            return None
        data = _row_to_dict(row)
        try:
            config = json.loads(data.get("config_json") or data.get("config") or "{}")
        except (TypeError, json.JSONDecodeError):
            config = {}
        return {
            "task_id": data.get("task_id") or data.get("id"),
            "lora_name": data.get("lora_name", ""),
            "status": data.get("status", "pending"),
            "progress": float(data.get("progress", 0) or 0),
            "error_message": data.get("error_message", ""),
            "config": config,
            "created_at": data.get("created_at") or data.get("createdAt", ""),
            "updated_at": data.get("updated_at") or data.get("updatedAt", ""),
        }

    async def save_training_task(self, task_id: str, task_data: Dict) -> None:
        created_at = task_data.get("created_at", "")
        updated_at = task_data.get("updated_at", "")
        config_json = json.dumps(task_data.get("config", {}), ensure_ascii=False)
        async with self.async_session() as session:
            await session.execute(text('''
                INSERT INTO training_tasks (
                    id, task_id, lora_name, status, progress, error_message,
                    config_json, created_at, updated_at
                ) VALUES (
                    :id, :task_id, :lora_name, :status, :progress, :error_message,
                    :config_json, :created_at, :updated_at
                )
                ON CONFLICT (id) DO UPDATE SET
                    task_id = EXCLUDED.task_id,
                    lora_name = EXCLUDED.lora_name,
                    status = EXCLUDED.status,
                    progress = EXCLUDED.progress,
                    error_message = EXCLUDED.error_message,
                    config_json = EXCLUDED.config_json,
                    updated_at = EXCLUDED.updated_at
            '''), {
                "id": task_id,
                "task_id": task_id,
                "lora_name": task_data.get("lora_name", ""),
                "status": task_data.get("status", "pending"),
                "progress": float(task_data.get("progress", 0) or 0),
                "error_message": task_data.get("error_message") or "",
                "config_json": config_json,
                "created_at": created_at,
                "updated_at": updated_at,
            })
            await session.commit()

    async def get_all_training_tasks(self) -> List[Dict]:
        async with self.async_session() as session:
            result = await session.execute(
                training_tasks_table.select().order_by(training_tasks_table.c.created_at.desc())
            )
            return [self._normalize_training_task(row) for row in result.fetchall()]

    async def get_active_training_by_lora_name(self, lora_name: str) -> List[Dict]:
        async with self.async_session() as session:
            result = await session.execute(
                training_tasks_table.select().where(
                    training_tasks_table.c.lora_name == lora_name,
                    training_tasks_table.c.status.in_(("pending", "running", "training")),
                )
            )
            return [self._normalize_training_task(row) for row in result.fetchall()]

    async def add_training_task(self, task: Dict) -> Dict:
        """添加训练任务"""
        now = datetime.now().isoformat()
        async with self.async_session() as session:
            stmt = training_tasks_table.insert().values(
                id=task.get("id"),
                taskType=task.get("taskType", "lora"),
                status=task.get("status", "pending"),
                config=task.get("config"),
                progress=task.get("progress", 0),
                result=task.get("result"),
                createdAt=now,
                updatedAt=now,
            )
            await session.execute(stmt)
            await session.commit()
            return {**task, "createdAt": now, "updatedAt": now}

    async def get_training_tasks(self, status: Optional[str] = None) -> List[Dict]:
        """获取训练任务列表"""
        async with self.async_session() as session:
            stmt = training_tasks_table.select()
            if status:
                stmt = stmt.where(training_tasks_table.c.status == status)
            stmt = stmt.order_by(training_tasks_table.c.createdAt.desc())
            result = await session.execute(stmt)
            return [_row_to_dict(row) for row in result.fetchall()]

    async def get_training_task(self, task_id: str) -> Optional[Dict]:
        """获取单个训练任务"""
        async with self.async_session() as session:
            stmt = training_tasks_table.select().where(training_tasks_table.c.id == task_id)
            result = await session.execute(stmt)
            return self._normalize_training_task(result.fetchone())

    async def update_training_task(self, task_id: str, data: Dict) -> Optional[Dict]:
        """更新训练任务"""
        now = datetime.now().isoformat()
        async with self.async_session() as session:
            values = {"updatedAt": now}
            for key in ("status", "progress", "result", "config"):
                if key in data:
                    values[key] = data[key]
            stmt = (
                training_tasks_table.update()
                .where(training_tasks_table.c.id == task_id)
                .values(**values)
            )
            await session.execute(stmt)
            await session.commit()

            sel_stmt = training_tasks_table.select().where(training_tasks_table.c.id == task_id)
            result = await session.execute(sel_stmt)
            row = result.fetchone()
            return _row_to_dict(row) if row else None

    async def delete_training_task(self, task_id: str) -> bool:
        """删除训练任务"""
        async with self.async_session() as session:
            stmt = training_tasks_table.delete().where(training_tasks_table.c.id == task_id)
            result = await session.execute(stmt)
            await session.commit()
            return result.rowcount > 0

    # ============================================
    # 通用 SQL 执行（兼容 SQLite 直接 SQL 调用）
    # ============================================
    async def execute_sql(self, query: str, params: Optional[dict] = None) -> Any:
        """执行原始 SQL 语句，返回结果。

        对于 SELECT 语句，返回行列表（每行为 dict）。
        对于 INSERT/UPDATE/DELETE，返回受影响行数。
        """
        async with self.async_session() as session:
            # 将 SQLite 风格的 ? 占位符替换为 :param 风格
            if params and "?" in query:
                # 不支持 ? 占位符自动转换，需要调用方使用命名参数
                raise ValueError("PostgreSQL execute_sql 不支持 ? 占位符，请使用命名参数 :name")

            result = await session.execute(text(query), params or {})

            if query.strip().upper().startswith("SELECT"):
                rows = result.fetchall()
                return [_row_to_dict(row) for row in rows]
            else:
                await session.commit()
                return result.rowcount

    async def execute_sql_insert(self, query: str, params: Optional[dict] = None) -> dict:
        """执行 INSERT SQL 并返回插入的行信息（包含自动生成的 ID）"""
        async with self.async_session() as session:
            result = await session.execute(text(query), params or {})
            await session.commit()
            # 尝试获取 lastrowid
            try:
                last_id_result = await session.execute(text("SELECT lastval()"))
                last_id = last_id_result.scalar()
            except Exception:
                last_id = None
            return {"lastrowid": last_id, "rowcount": result.rowcount}

    # ============================================
    # 兼容属性（与 SQLiteDB 保持一致）
    # ============================================
    @property
    def config(self):
        """兼容属性 - 同步获取配置（不推荐，建议使用 await get_config()）"""
        raise RuntimeError("PgDatabase.config 是异步操作，请使用 await get_config()")

    @property
    def messages(self):
        """兼容属性"""
        raise RuntimeError("PgDatabase.messages 是异步操作，请使用 await get_messages()")

    @property
    def loras(self):
        """兼容属性"""
        raise RuntimeError("PgDatabase.loras 是异步操作，请使用 await get_loras()")


# ============================================
# 全局单例
# ============================================
pg_db = PgDatabase()


# ============================================
# 同步适配器 - 让现有同步代码无需修改即可使用 PostgreSQL
# ============================================
import asyncio
import threading


class SyncPgAdapter:
    """同步适配器：将 PgDatabase 的异步方法包装为同步方法。

    使用独立事件循环在后台线程中运行异步方法，
    确保与现有同步代码（如 SQLite Database 类）兼容。
    """

    def __init__(self, pg: PgDatabase):
        self._pg = pg
        self._loop = None
        self._thread = None

    def _ensure_loop(self):
        """确保后台事件循环正在运行"""
        if self._loop is None or not self._loop.is_running():
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
            self._thread.start()
            # 初始化数据库
            asyncio.run_coroutine_threadsafe(self._pg.init(), self._loop).result(timeout=30)

    def _run(self, coro):
        """在后台事件循环中运行协程并等待结果"""
        self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=30)

    # 代理所有 PgDatabase 的方法为同步调用
    def init(self):
        self._run(self._pg.init())

    def get_config(self):
        return self._run(self._pg.get_config())

    def get_config_value(self, key, default=None):
        return self._run(self._pg.get_config_value(key, default))

    def set_config(self, new_config):
        return self._run(self._pg.set_config(new_config))

    def set_config_value(self, key, value):
        return self._run(self._pg.set_config_value(key, value))

    def add_message(self, message):
        return self._run(self._pg.add_message(message))

    def get_messages(self, **kwargs):
        return self._run(self._pg.get_messages(**kwargs))

    def get_messages_filtered(self, **kwargs):
        return self._run(self._pg.get_messages_filtered(**kwargs))

    def get_message_count_filtered(self, **kwargs):
        return self._run(self._pg.get_message_count_filtered(**kwargs))

    def get_message_count(self, **kwargs):
        return self._run(self._pg.get_message_count(**kwargs))

    def delete_message(self, msg_id):
        return self._run(self._pg.delete_message(msg_id))

    def delete_messages_by_filter(self, **kwargs):
        return self._run(self._pg.delete_messages_by_filter(**kwargs))

    def get_recent_messages(self, limit=10):
        return self._run(self._pg.get_recent_messages(limit))

    def get_loras(self, status=None):
        return self._run(self._pg.get_loras(status))

    def add_lora(self, lora_data):
        return self._run(self._pg.add_lora(lora_data))

    def update_lora_status(self, lora_id, status):
        # fix: 参数名 active 暗示布尔值，但 PgDatabase/SQLite 实际需要字符串状态
        # (如 "active"/"inactive")。与 SQLite Database.update_lora_status(lora_id, status) 对齐。
        return self._run(self._pg.update_lora_status(lora_id, status))

    def delete_lora(self, lora_id):
        return self._run(self._pg.delete_lora(lora_id))

    def get_knowledge_bases(self):
        return self._run(self._pg.get_knowledge_bases())

    def get_knowledge_base(self, kb_id):
        return self._run(self._pg.get_knowledge_base(kb_id))

    def get_knowledge_folder(self, folder_id):
        return self._run(self._pg.get_knowledge_folder(folder_id))

    def create_knowledge_base(self, name, description=""):
        return self._run(self._pg.create_knowledge_base(name, description))

    def update_knowledge_base(self, kb_id, data):
        return self._run(self._pg.update_knowledge_base(kb_id, data))

    def delete_knowledge_base(self, kb_id):
        return self._run(self._pg.delete_knowledge_base(kb_id))

    def get_knowledge_folders(self, kb_id):
        return self._run(self._pg.get_knowledge_folders(kb_id))

    def create_knowledge_folder(self, kb_id, name, description=""):
        # fix: 此前第三参数名为 parent_id，被当作 description 传给 PgDatabase，
        # 导致文件夹描述被设为 parent_id 值。与 SQLite Database.create_knowledge_folder
        # (kb_id, name, description="") 对齐。
        return self._run(self._pg.create_knowledge_folder(kb_id, name, description))

    # 注意：SQLite Database 和 PgDatabase 均未实现 update_knowledge_folder。
    # 此前 SyncPgAdapter 委托给不存在的 self._pg.update_knowledge_folder 会抛 AttributeError。
    # 若未来需要更新文件夹，应在 SQLite/PgDatabase 中先实现，再在此添加委托。

    def delete_knowledge_folder(self, folder_id):
        return self._run(self._pg.delete_knowledge_folder(folder_id))

    def get_knowledge_documents(self, **kwargs):
        return self._run(self._pg.get_knowledge_documents(**kwargs))

    def add_knowledge_document(self, doc_data):
        return self._run(self._pg.add_knowledge_document(doc_data))

    def get_knowledge_document(self, doc_id):
        return self._run(self._pg.get_knowledge_document(doc_id))

    def update_knowledge_document(self, doc_id, document):
        return self._run(self._pg.update_knowledge_document(doc_id, document))

    def delete_knowledge_document(self, doc_id):
        return self._run(self._pg.delete_knowledge_document(doc_id))

    def get_knowledge_chunks(self, doc_id):
        return self._run(self._pg.get_knowledge_chunks(doc_id))

    def get_all_knowledge_chunks(self):
        return self._run(self._pg.get_all_knowledge_chunks())

    def add_knowledge_chunk(self, chunk_data):
        return self._run(self._pg.add_knowledge_chunk(chunk_data))

    def get_knowledge_stats(self):
        return self._run(self._pg.get_knowledge_stats())

    def add_user(self, username, password_hash):
        return self._run(self._pg.add_user(username, password_hash))

    def get_user(self, user_id):
        return self._run(self._pg.get_user(user_id))

    def get_user_by_username(self, username):
        return self._run(self._pg.get_user_by_username(username))

    def get_user_data(self, user_id, page_key=None):
        return self._run(self._pg.get_user_data(user_id, page_key))

    def save_user_data(self, user_id, page_key, data_json):
        return self._run(self._pg.save_user_data(user_id, page_key, data_json))

    def get_session_summaries(self):
        return self._run(self._pg.get_session_summaries())

    def set_session_bot_enabled(self, session_id, enabled, platform="qq", conversation_id=None):
        return self._run(self._pg.set_session_bot_enabled(session_id, enabled, platform, conversation_id))

    def is_session_bot_enabled(self, session_id, platform="qq", conversation_id=None):
        return self._run(self._pg.is_session_bot_enabled(session_id, platform, conversation_id))

    def mark_integration_message_processed(self, platform, adapter, message_id):
        return self._run(self._pg.mark_integration_message_processed(platform, adapter, message_id))

    def upsert_conversation(self, data):
        return self._run(self._pg.upsert_conversation(data))

    def get_conversation(self, platform, conversation_id, conversation_type=None):
        return self._run(self._pg.get_conversation(platform, conversation_id, conversation_type))

    def add_integration_event(self, event):
        return self._run(self._pg.add_integration_event(event))

    def add_model_invocation(self, invocation):
        return self._run(self._pg.add_model_invocation(invocation))

    def get_claw_tools(self):
        return self._run(self._pg.get_claw_tools())

    def get_claw_tool_by_name(self, name):
        return self._run(self._pg.get_claw_tool_by_name(name))

    def save_claw_tool(self, name, description, code, enabled=True):
        return self._run(self._pg.save_claw_tool(name, description, code, enabled))

    def delete_claw_tool(self, name):
        # fix: 参数名 tool_id 误导调用方传 ID，但 PgDatabase/SQLite 实际按 name 删除。
        # 与 SQLite Database.delete_claw_tool(name) 和 api/claw.py:110 调用方对齐。
        return self._run(self._pg.delete_claw_tool(name))

    def add_audit_log(self, **kwargs):
        return self._run(self._pg.add_audit_log(**kwargs))

    def get_audit_logs(self, **kwargs):
        return self._run(self._pg.get_audit_logs(**kwargs))

    def add_intent_sample(self, kb_name, text, label):
        return self._run(self._pg.add_intent_sample(kb_name, text, label))

    def get_intent_samples(self, kb_name=None):
        return self._run(self._pg.get_intent_samples(kb_name))

    def get_active_kbs(self):
        return self._run(self._pg.get_active_kbs())

    def set_active_kb(self, kb_name, is_active):
        return self._run(self._pg.set_active_kb(kb_name, is_active))

    def save_training_task(self, task_id, task_data):
        return self._run(self._pg.save_training_task(task_id, task_data))

    def get_all_training_tasks(self):
        return self._run(self._pg.get_all_training_tasks())

    def get_active_training_by_lora_name(self, lora_name):
        return self._run(self._pg.get_active_training_by_lora_name(lora_name))

    def add_training_task(self, task_data):
        return self._run(self._pg.add_training_task(task_data))

    def get_training_tasks(self, status=None):
        return self._run(self._pg.get_training_tasks(status))

    def get_training_task(self, task_id):
        return self._run(self._pg.get_training_task(task_id))

    def update_training_task(self, task_id, data=None, **kwargs):
        updates = dict(data or {})
        updates.update(kwargs)
        return self._run(self._pg.update_training_task(task_id, updates))

    def delete_training_task(self, task_id):
        return self._run(self._pg.delete_training_task(task_id))

    # 兼容属性
    @property
    def db_path(self):
        return "postgresql://localhost:5432/qqassistant"

    @property
    def config(self):
        """兼容 SQLite 的 config 属性"""
        return self.get_config()

    @property
    def messages(self):
        """兼容 SQLite 的 messages 属性"""
        return self.get_messages(limit=10000)

    @property
    def loras(self):
        """兼容 SQLite 的 loras 属性"""
        return self.get_loras()

    def update_config(self, new_config):
        """兼容 SQLite 的 update_config 方法"""
        return self.set_config(new_config)

    def execute_sql(self, query, params=None):
        """兼容 SQLite 的直接 SQL 执行"""
        return self._run(self._pg.execute_sql(query, params))

    def execute_sql_insert(self, query, params=None):
        """兼容 SQLite 的 INSERT SQL 执行"""
        return self._run(self._pg.execute_sql_insert(query, params))

    def get_connection(self):
        """兼容 SQLite 的 get_connection 方法"""
        raise NotImplementedError("PostgreSQL 不支持直接获取连接，请使用 API 方法")

    def _get_connection(self):
        raise NotImplementedError("PostgreSQL 不支持直接获取连接，请使用 API 方法")


# 同步适配器单例
sync_pg_db = SyncPgAdapter(pg_db)
