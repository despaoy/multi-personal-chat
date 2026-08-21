"""
PostgreSQL 异步数据库访问层
使用 async SQLAlchemy + asyncpg，提供与 SQLiteDB 相同的方法签名（异步版本）
"""

import os
import time
import json
import logging
from datetime import datetime
from typing import Optional, Dict, List, Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from cache.ttl_value_cache import BoundedTTLCache
from db.errors import RegistrationClosedError
from db.urls import resolve_runtime_database_url

logger = logging.getLogger(__name__)



# ============================================
# SQLAlchemy Core 表定义
# ============================================
# Phase 2 fix: 表定义从 db.models 的 ORM metadata 派生，消除 26 张表的重复定义。
# models.py 是数据库 schema 的单一真相源，pg_database.py 不再自行声明 Table 对象。
from db.models import metadata

messages_table = metadata.tables["messages"]
loras_table = metadata.tables["loras"]
config_table = metadata.tables["config"]
knowledge_bases_table = metadata.tables["knowledge_bases"]
knowledge_folders_table = metadata.tables["knowledge_folders"]
knowledge_documents_table = metadata.tables["knowledge_documents"]
knowledge_chunks_table = metadata.tables["knowledge_chunks"]
users_table = metadata.tables["users"]
user_data_table = metadata.tables["user_data"]
saved_dialogues_table = metadata.tables["saved_dialogues"]
api_keys_table = metadata.tables["api_keys"]
claw_tools_table = metadata.tables["claw_tools"]
integration_message_dedup_table = metadata.tables["integration_message_dedup"]
conversations_table = metadata.tables["conversations"]
integration_events_table = metadata.tables["integration_events"]
model_invocations_table = metadata.tables["model_invocations"]
audit_logs_table = metadata.tables["audit_logs"]
intent_samples_table = metadata.tables["intent_samples"]
intent_active_kbs_table = metadata.tables["intent_active_kbs"]
training_tasks_table = metadata.tables["training_tasks"]
gold_eval_runs_table = metadata.tables["gold_eval_runs"]
experiment_runs_table = metadata.tables["experiment_runs"]
retrieval_eval_questions_table = metadata.tables["retrieval_eval_questions"]
preference_pairs_table = metadata.tables["preference_pairs"]
adapter_compatibility_table = metadata.tables["adapter_compatibility"]
feedback_table = metadata.tables["feedback"]


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
        self.database_url = resolve_runtime_database_url(database_url or None)
        if not self.database_url:
            raise ValueError("DATABASE_URL or PG_PASSWORD is required when USE_POSTGRESQL=true")
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
        self._bot_enabled_cache: BoundedTTLCache[tuple[str, str, str], bool] = BoundedTTLCache(
            ttl=float(os.getenv("SESSION_SWITCH_CACHE_TTL", "60")),
            max_size=int(os.getenv("SESSION_SWITCH_CACHE_MAX_SIZE", "4096")),
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
            # One-way compatibility migration: legacy session_settings is folded into
            # conversations and then removed. Fresh databases never create this table.
            try:
                result = await conn.execute(text("SELECT to_regclass('public.session_settings')"))
                if result.scalar():
                    migrated_at = datetime.now().isoformat()
                    await conn.execute(text('''
                        INSERT INTO conversations (
                            platform, "conversationId", "conversationType", "displayName",
                            "botEnabled", "replyPolicy", "createdAt", "updatedAt"
                        )
                        SELECT
                            COALESCE(platform, 'qq'),
                            COALESCE("conversationId", "sessionId"),
                            COALESCE("sessionType", 'private'),
                            COALESCE(NULLIF("sessionName", ''), "sessionId"),
                            bot_enabled,
                            'default',
                            COALESCE(updated_at, :migrated_at),
                            COALESCE(updated_at, :migrated_at)
                        FROM session_settings
                        ON CONFLICT (platform, "conversationId", "conversationType")
                        DO UPDATE SET
                            "botEnabled" = EXCLUDED."botEnabled",
                            "displayName" = EXCLUDED."displayName",
                            "updatedAt" = EXCLUDED."updatedAt"
                    '''), {"migrated_at": migrated_at})
                    await conn.execute(text('DROP TABLE IF EXISTS session_settings'))
            except Exception:
                logger.warning("session_settings migration skipped", exc_info=True)
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
            await conn.execute(text('DROP INDEX IF EXISTS idx_conversations_platform_conversation'))
            # UNIQUE(platform, conversationId, conversationType) 已自动创建索引。
            await conn.execute(text('CREATE INDEX IF NOT EXISTS idx_integration_events_trace ON integration_events ("traceId")'))
            await conn.execute(text('CREATE INDEX IF NOT EXISTS idx_integration_events_platform_created ON integration_events (platform, "createdAt")'))
            await conn.execute(text('CREATE INDEX IF NOT EXISTS idx_model_invocations_trace ON model_invocations ("traceId")'))
            await conn.execute(text('CREATE INDEX IF NOT EXISTS idx_model_invocations_created ON model_invocations ("createdAt")'))
            await conn.execute(text('CREATE INDEX IF NOT EXISTS idx_experiment_runs_type ON experiment_runs (experiment_type)'))
            await conn.execute(text('CREATE INDEX IF NOT EXISTS idx_preference_pairs_status ON preference_pairs (review_status)'))
            await conn.execute(text('CREATE INDEX IF NOT EXISTS idx_feedback_created ON feedback (created_at)'))
            await conn.execute(text('CREATE INDEX IF NOT EXISTS idx_adapter_compat_name ON adapter_compatibility (adapter_name)'))
            # 补充此前缺失的高频外键/过滤列索引（与 SQLite 一致）
            await conn.execute(text('CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_documentId ON knowledge_chunks ("documentId")'))
            await conn.execute(text('CREATE INDEX IF NOT EXISTS idx_knowledge_documents_kb_id ON knowledge_documents (knowledge_base_id)'))
            await conn.execute(text('CREATE INDEX IF NOT EXISTS idx_knowledge_documents_folder_id ON knowledge_documents (folder_id)'))
            await conn.execute(text('CREATE INDEX IF NOT EXISTS idx_training_tasks_lora_name ON training_tasks (lora_name)'))
            await conn.execute(text('CREATE INDEX IF NOT EXISTS idx_training_tasks_status ON training_tasks (status)'))
            await conn.execute(text('CREATE INDEX IF NOT EXISTS idx_intent_samples_kbName ON intent_samples ("kbName")'))
            await conn.execute(text('CREATE INDEX IF NOT EXISTS idx_feedback_trace_id ON feedback (trace_id)'))
            await conn.execute(text('CREATE INDEX IF NOT EXISTS idx_feedback_message_id ON feedback (message_id)'))
            await conn.execute(text('CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs (timestamp)'))
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
                from db.config_utils import coerce_config_value
                config_dict[key] = coerce_config_value(value)
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
        """获取所有知识库（单次 LEFT JOIN + 聚合，消除 N+1 查询）"""
        async with self.async_session() as session:
            # 一次查询获取 KB 列表 + 文档数 + 文件夹数，避免每个 KB 单独 COUNT
            stmt = text("""
                SELECT
                    kb.*,
                    COALESCE(d.cnt, 0) as documentCount,
                    COALESCE(f.cnt, 0) as folderCount
                FROM knowledge_bases kb
                LEFT JOIN (
                    SELECT knowledge_base_id, COUNT(*) as cnt
                    FROM knowledge_documents
                    GROUP BY knowledge_base_id
                ) d ON d.knowledge_base_id = kb.id
                LEFT JOIN (
                    SELECT knowledge_base_id, COUNT(*) as cnt
                    FROM knowledge_folders
                    GROUP BY knowledge_base_id
                ) f ON f.knowledge_base_id = kb.id
                ORDER BY kb.updated_at DESC
            """)
            result = await session.execute(stmt)
            kb_list = []
            for row in result.fetchall():
                kb = _row_to_dict(row)
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
        """获取知识库下的所有文件夹（LEFT JOIN 聚合，消除 N+1）"""
        async with self.async_session() as session:
            stmt = text("""
                SELECT
                    f.*,
                    COALESCE(d.cnt, 0) as documentCount
                FROM knowledge_folders f
                LEFT JOIN (
                    SELECT folder_id, COUNT(*) as cnt
                    FROM knowledge_documents
                    GROUP BY folder_id
                ) d ON d.folder_id = f.id
                WHERE f.knowledge_base_id = :kb_id
                ORDER BY f.name
            """)
            result = await session.execute(stmt, {"kb_id": kb_id})
            folder_list = []
            for row in result.fetchall():
                folder = _row_to_dict(row)
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

    async def get_all_knowledge_chunks(self, limit: int | None = None, offset: int = 0) -> List[Dict]:
        """获取所有知识库片段（用于检索）。

        Args:
            limit: 返回数量上限，None 表示全量（谨慎使用，大库可能 OOM）
            offset: 跳过前 N 条
        """
        async with self.async_session() as session:
            stmt = knowledge_chunks_table.select().order_by(knowledge_chunks_table.c.documentId, knowledge_chunks_table.c.chunkIndex)
            if limit is not None:
                stmt = stmt.limit(limit).offset(offset)
            result = await session.execute(stmt)
            return [_row_to_dict(row) for row in result.fetchall()]

    async def iter_all_knowledge_chunks(self, batch_size: int = 500):
        """分页迭代所有知识库片段，避免大库 OOM。

        异步生成器，每次 yield 一条 chunk dict。
        """
        async with self.async_session() as session:
            offset = 0
            while True:
                stmt = (
                    knowledge_chunks_table.select()
                    .order_by(knowledge_chunks_table.c.documentId, knowledge_chunks_table.c.chunkIndex)
                    .limit(batch_size)
                    .offset(offset)
                )
                result = await session.execute(stmt)
                rows = result.fetchall()
                if not rows:
                    break
                for row in rows:
                    yield _row_to_dict(row)
                offset += len(rows)
                if len(rows) < batch_size:
                    break

    async def iter_chunks_with_document(self, batch_size: int = 500):
        """分页迭代 chunk 及其所属文档（LEFT JOIN），避免 N+1 查询。

        与 SQLiteDB.iter_chunks_with_document 行为一致：每次 yield 一条 dict，
        包含 chunk 字段和文档字段（doc_title / doc_category / doc_kb_id）。
        孤儿 chunk（文档已删除）的 doc_title 为 None，调用方可跳过。
        """
        offset = 0
        while True:
            batch = await self.get_chunks_with_document(limit=batch_size, offset=offset)
            if not batch:
                break
            for row in batch:
                yield row
            offset += len(batch)
            if len(batch) < batch_size:
                break

    async def get_chunks_with_document(self, limit: int = 500, offset: int = 0) -> List[Dict]:
        """单页查询 chunk + document（LEFT JOIN），返回 dict 列表。

        供 SyncPgAdapter 同步包装使用（async generator 无法直接 _run）。
        """
        from sqlalchemy import select
        async with self.async_session() as session:
            stmt = (
                select(
                    knowledge_chunks_table,
                    knowledge_documents_table.c.title.label("doc_title"),
                    knowledge_documents_table.c.category.label("doc_category"),
                    knowledge_documents_table.c.knowledge_base_id.label("doc_kb_id"),
                )
                .select_from(
                    knowledge_chunks_table.outerjoin(
                        knowledge_documents_table,
                        knowledge_chunks_table.c.documentId == knowledge_documents_table.c.id,
                    )
                )
                .order_by(knowledge_chunks_table.c.documentId, knowledge_chunks_table.c.chunkIndex)
                .limit(limit)
                .offset(offset)
            )
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
    async def add_user(
        self,
        username: str,
        password_hash: str,
        bootstrap_only: bool = False,
    ) -> Dict:
        """Add a user while serializing the first-admin decision."""
        now = datetime.now().isoformat()
        async with self.async_session() as session:
            # Serialize the first-admin decision across API workers. The lock is
            # transaction-scoped and is released automatically on commit/rollback.
            await session.execute(
                text("SELECT pg_advisory_xact_lock(hashtext('qqchat:first-admin'))")
            )
            count_stmt = users_table.select().order_by(users_table.c.id).limit(1)
            existing = await session.execute(count_stmt)
            has_users = existing.fetchone() is not None
            if bootstrap_only and has_users:
                raise RegistrationClosedError("bootstrap administrator already exists")
            role = "user" if has_users else "admin"
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
    # 角色关系与长期记忆
    # ============================================
    @staticmethod
    def _character_scope_params(
        character_id: str,
        platform: str,
        adapter: str,
        sender_id: str,
        conversation_type: str,
        conversation_id: str,
        **extra: Any,
    ) -> dict:
        """组装角色记忆隔离范围的 SQL 绑定参数（与 SQLite 侧语义一致）。"""
        params = {
            "character_id": character_id,
            "platform": platform,
            "adapter": adapter,
            "sender_id": sender_id,
            "conversation_type": conversation_type,
            "conversation_id": conversation_id,
        }
        params.update(extra)
        return params

    _CHARACTER_SCOPE_SQL = (
        "character_id = :character_id AND platform = :platform AND adapter = :adapter "
        "AND sender_id = :sender_id AND conversation_type = :conversation_type "
        "AND conversation_id = :conversation_id"
    )

    async def get_character_relationship(
        self,
        character_id: str,
        platform: str,
        adapter: str,
        sender_id: str,
        conversation_type: str,
        conversation_id: str,
    ) -> Optional[dict]:
        """读取指定角色+用户范围的关系状态，不存在时返回 None。"""
        async with self.async_session() as session:
            stmt = text(
                "SELECT * FROM character_relationships WHERE "
                + self._CHARACTER_SCOPE_SQL
            )
            result = await session.execute(
                stmt,
                self._character_scope_params(
                    character_id, platform, adapter, sender_id, conversation_type, conversation_id
                ),
            )
            row = result.fetchone()
            return _row_to_dict(row) if row else None

    async def upsert_character_relationship(
        self,
        character_id: str,
        platform: str,
        adapter: str,
        sender_id: str,
        conversation_type: str,
        conversation_id: str,
        relationship_stage: str,
        preferred_address: str = "",
        summary: str = "",
        interaction_count: Optional[int] = None,
    ) -> dict:
        """写入关系状态（单条 UPSERT 原子完成）。

        interaction_count 为 None 时 UPDATE 子句不触碰该列、保留数据库
        当前值：先 SELECT 计数再写回的实现在并发下会用旧计数覆盖
        increment_character_interaction 刚自增的结果（管理端更新关系与
        新消息并发时计数回退）。RETURNING * 返回写入后的真实记录。
        """
        now = datetime.now().isoformat()
        set_clauses = [
            "relationship_stage = EXCLUDED.relationship_stage",
            "preferred_address = EXCLUDED.preferred_address",
            "summary = EXCLUDED.summary",
        ]
        if interaction_count is not None:
            set_clauses.append("interaction_count = EXCLUDED.interaction_count")
        set_clauses.append("updated_at = EXCLUDED.updated_at")
        stmt = text(
            "INSERT INTO character_relationships ("
            "character_id, platform, adapter, sender_id, conversation_type, "
            "conversation_id, relationship_stage, preferred_address, summary, "
            "interaction_count, created_at, updated_at) VALUES ("
            ":character_id, :platform, :adapter, :sender_id, :conversation_type, "
            ":conversation_id, :relationship_stage, :preferred_address, :summary, "
            ":interaction_count, :now, :now) "
            "ON CONFLICT (character_id, platform, adapter, sender_id, conversation_type, conversation_id) "
            f"DO UPDATE SET {', '.join(set_clauses)} "
            "RETURNING *"
        )
        params = self._character_scope_params(
            character_id, platform, adapter, sender_id, conversation_type, conversation_id,
            relationship_stage=relationship_stage,
            preferred_address=preferred_address,
            summary=summary,
            interaction_count=(
                int(interaction_count) if interaction_count is not None else 0
            ),
            now=now,
        )
        async with self.async_session() as session:
            result = await session.execute(stmt, params)
            row = result.fetchone()
            await session.commit()
            if row is None:  # pragma: no cover - RETURNING 必返回一行
                raise RuntimeError("upsert_character_relationship RETURNING 未返回行")
            return _row_to_dict(row)

    async def increment_character_interaction(
        self,
        character_id: str,
        platform: str,
        adapter: str,
        sender_id: str,
        conversation_type: str,
        conversation_id: str,
    ) -> int:
        """交互轮数 +1，返回自增后的值（首次交互时从 1 开始）。

        单条 UPSERT ... RETURNING 原子完成：先 SELECT 再 UPDATE 的
        实现在并发下会丢失更新（两条并发消息都从 10 更新到 11）。
        """
        now = datetime.now().isoformat()
        params = self._character_scope_params(
            character_id, platform, adapter, sender_id, conversation_type, conversation_id, now=now
        )
        async with self.async_session() as session:
            stmt = text(
                "INSERT INTO character_relationships ("
                "character_id, platform, adapter, sender_id, conversation_type, "
                "conversation_id, relationship_stage, preferred_address, summary, "
                "interaction_count, created_at, updated_at) VALUES ("
                ":character_id, :platform, :adapter, :sender_id, :conversation_type, "
                ":conversation_id, 'stranger', '', '', 1, :now, :now) "
                "ON CONFLICT (character_id, platform, adapter, sender_id, conversation_type, conversation_id) "
                "DO UPDATE SET interaction_count = character_relationships.interaction_count + 1, "
                "updated_at = EXCLUDED.updated_at "
                "RETURNING interaction_count"
            )
            result = await session.execute(stmt, params)
            new_count = int(result.scalar_one())
            await session.commit()
            return new_count

    async def list_character_memories(
        self,
        character_id: str,
        platform: str,
        adapter: str,
        sender_id: str,
        conversation_type: str,
        conversation_id: str,
        limit: int = 30,
    ) -> List[dict]:
        """读取指定范围内最近的记忆（按 updated_at 倒序，最多 limit 条）。"""
        async with self.async_session() as session:
            stmt = text(
                "SELECT * FROM character_memories WHERE "
                + self._CHARACTER_SCOPE_SQL
                + " ORDER BY updated_at DESC LIMIT :limit"
            )
            result = await session.execute(
                stmt,
                self._character_scope_params(
                    character_id, platform, adapter, sender_id, conversation_type, conversation_id,
                    limit=max(1, min(int(limit), 200)),
                ),
            )
            return [_row_to_dict(row) for row in result.fetchall()]

    async def add_or_update_character_memory(
        self,
        character_id: str,
        platform: str,
        adapter: str,
        sender_id: str,
        conversation_type: str,
        conversation_id: str,
        memory_type: str,
        memory_key: str,
        content: str,
        importance: float = 0.0,
        source_message_id: Optional[str] = None,
    ) -> dict:
        """写入一条记忆（同 memory_key upsert，更新内容与时间戳）。"""
        now = datetime.now().isoformat()
        async with self.async_session() as session:
            stmt = text(
                "INSERT INTO character_memories ("
                "character_id, platform, adapter, sender_id, conversation_type, "
                "conversation_id, memory_type, memory_key, content, importance, "
                "source_message_id, created_at, updated_at) VALUES ("
                ":character_id, :platform, :adapter, :sender_id, :conversation_type, "
                ":conversation_id, :memory_type, :memory_key, :content, :importance, "
                ":source_message_id, :now, :now) "
                "ON CONFLICT (character_id, platform, adapter, sender_id, conversation_type, "
                "conversation_id, memory_key) DO UPDATE SET "
                "memory_type = EXCLUDED.memory_type, content = EXCLUDED.content, "
                "importance = EXCLUDED.importance, source_message_id = EXCLUDED.source_message_id, "
                "updated_at = EXCLUDED.updated_at RETURNING id, created_at"
            )
            result = await session.execute(
                stmt,
                self._character_scope_params(
                    character_id, platform, adapter, sender_id, conversation_type, conversation_id,
                    memory_type=memory_type,
                    memory_key=memory_key,
                    content=content,
                    importance=float(importance),
                    source_message_id=source_message_id,
                    now=now,
                ),
            )
            row = result.fetchone()
            await session.commit()
            record = _row_to_dict(row) if row else {}
            return {
                "id": record.get("id"),
                "created_at": record.get("created_at", now),
                "character_id": character_id,
                "memory_type": memory_type,
                "memory_key": memory_key,
                "content": content,
                "importance": float(importance),
                "source_message_id": source_message_id,
                "updated_at": now,
            }

    async def delete_character_memory(
        self,
        memory_id: int,
        character_id: str,
        platform: str,
        adapter: str,
        sender_id: str,
        conversation_type: str,
        conversation_id: str,
    ) -> bool:
        """删除一条记忆。必须同时匹配隔离范围，防止越权删除其他用户记忆。"""
        async with self.async_session() as session:
            stmt = text(
                "DELETE FROM character_memories WHERE id = :memory_id AND "
                + self._CHARACTER_SCOPE_SQL
            )
            result = await session.execute(
                stmt,
                self._character_scope_params(
                    character_id, platform, adapter, sender_id, conversation_type, conversation_id,
                    memory_id=int(memory_id),
                ),
            )
            await session.commit()
            return bool(result.rowcount)

    async def clear_character_memories(
        self,
        character_id: str,
        platform: str,
        adapter: str,
        sender_id: str,
        conversation_type: str,
        conversation_id: str,
    ) -> int:
        """清空指定范围内的全部记忆，返回删除条数。"""
        async with self.async_session() as session:
            stmt = text(
                "DELETE FROM character_memories WHERE " + self._CHARACTER_SCOPE_SQL
            )
            result = await session.execute(
                stmt,
                self._character_scope_params(
                    character_id, platform, adapter, sender_id, conversation_type, conversation_id
                ),
            )
            await session.commit()
            return int(result.rowcount)

    async def list_conversation_history(
        self,
        platform: str,
        adapter: str,
        sender_id: str,
        conversation_type: str,
        conversation_id: str,
        limit: int = 8,
        max_chars: int = 6000,
    ) -> List[dict]:
        """按用户范围读取最近对话历史，组装成角色生成用的消息列表。

        语义与 SQLite 侧 list_conversation_history 一致：
        - 私聊：platform+adapter+senderId 下全部私聊记录；
        - 群聊/频道：再加 conversationId（群/频道）过滤；
        - 返回按时间正序的 [{"role", "content"}]，超预算从最旧一侧截断。
        """
        async with self.async_session() as session:
            if conversation_type in ("group", "channel"):
                stmt = text(
                    'SELECT message, reply FROM messages WHERE platform = :platform '
                    'AND adapter = :adapter AND "senderId" = :sender_id '
                    'AND "conversationId" = :conversation_id '
                    'ORDER BY "createdAt" DESC LIMIT :limit'
                )
                params: dict = {
                    "platform": platform,
                    "adapter": adapter,
                    "sender_id": sender_id,
                    "conversation_id": conversation_id,
                    "limit": max(1, min(int(limit), 50)),
                }
            else:
                stmt = text(
                    'SELECT message, reply FROM messages WHERE platform = :platform '
                    'AND adapter = :adapter AND "senderId" = :sender_id '
                    'AND ("conversationType" = :private OR "conversationType" = :empty) '
                    'ORDER BY "createdAt" DESC LIMIT :limit'
                )
                params = {
                    "platform": platform,
                    "adapter": adapter,
                    "sender_id": sender_id,
                    "private": "private",
                    "empty": "",
                    "limit": max(1, min(int(limit), 50)),
                }
            result = await session.execute(stmt, params)
            rows = result.fetchall()
        turns: List[dict] = []
        for row in reversed(rows):
            d = _row_to_dict(row)
            message = (d.get("message") or "").strip()
            reply = (d.get("reply") or "").strip()
            if message:
                turns.append({"role": "user", "content": message})
            if reply:
                turns.append({"role": "assistant", "content": reply})
        if max_chars > 0:
            kept: List[dict] = []
            total = 0
            for item in reversed(turns):
                total += len(item["content"])
                if total > max_chars and kept:
                    break
                kept.append(item)
            kept.reverse()
            turns = kept
        return turns

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
        """获取所有会话的聚合统计信息（相关子查询消除 N+1）"""
        async with self.async_session() as session:
            stmt = text("""
                WITH normalized AS (
                    SELECT
                        id, "sessionId", "sessionName",
                        COALESCE(platform, 'qq') AS platform,
                        COALESCE(adapter, 'nonebot') AS adapter,
                        COALESCE("conversationId", "sessionId") AS "conversationId",
                        COALESCE("conversationType", "sessionType") AS "conversationType",
                        message, "createdAt"
                    FROM messages
                ), ranked AS (
                    SELECT *,
                        ROW_NUMBER() OVER (
                            PARTITION BY platform, "conversationType", "conversationId"
                            ORDER BY "createdAt" DESC, id DESC
                        ) AS rn
                    FROM normalized
                ), aggregated AS (
                    SELECT
                        MAX(CASE WHEN rn = 1 THEN "sessionId" END) AS "sessionId",
                        "conversationType" AS "sessionType",
                        MAX(CASE WHEN rn = 1 THEN "sessionName" END) AS "sessionName",
                        platform,
                        MAX(CASE WHEN rn = 1 THEN adapter END) AS adapter,
                        "conversationId", "conversationType",
                        COUNT(*) AS message_count,
                        MAX("createdAt") AS last_active,
                        MAX(CASE WHEN rn = 1 THEN message END) AS recent_1,
                        MAX(CASE WHEN rn = 2 THEN message END) AS recent_2,
                        MAX(CASE WHEN rn = 3 THEN message END) AS recent_3
                    FROM ranked
                    GROUP BY platform, "conversationId", "conversationType"
                )
                SELECT a.*, COALESCE(c."botEnabled", 1) AS bot_enabled
                FROM aggregated a
                LEFT JOIN conversations c
                  ON c.platform = a.platform
                 AND c."conversationId" = a."conversationId"
                 AND c."conversationType" = a."conversationType"
                ORDER BY a.last_active DESC
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

                recent = [d.get(key) for key in ("recent_3", "recent_2", "recent_1") if d.get(key) and d[key].strip()]
                summary = "；".join(recent)
                if len(summary) > 100:
                    summary = summary[:100] + "..."

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
                    "botEnabled": bool(d.get("bot_enabled", 1)),
                })
            return sessions

    async def set_session_bot_enabled(self, session_id: str, enabled: bool, platform: str = "qq", conversation_id: Optional[str] = None, conversation_type: str = "private") -> None:
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
                conversation_type=conversation_type,
                display_name=session_id,
                bot_enabled=enabled,
            )
            await session.commit()
        self._bot_enabled_cache.invalidate((platform, resolved_conversation_id, conversation_type))

    async def is_session_bot_enabled(self, session_id: str, platform: str = "qq", conversation_id: Optional[str] = None, conversation_type: str = "private") -> bool:
        """检查某个会话的机器人是否启用（默认启用）。

        统一从 conversations 表查询（此前先查 session_settings 失败再查 conversations，
        现直接查 conversations，消除双表冗余查询）。
        """
        resolved_conversation_id = conversation_id or session_id
        cache_key = (platform, resolved_conversation_id, conversation_type)
        cached = self._bot_enabled_cache.get(cache_key)
        if cached is not None:
            return cached

        async with self.async_session() as session:
            stmt = text(
                'SELECT "botEnabled" FROM conversations WHERE platform = :platform AND "conversationId" = :cid AND "conversationType" = :ctype LIMIT 1'
            )
            result = await session.execute(stmt, {"platform": platform, "cid": resolved_conversation_id, "ctype": conversation_type})
            row = result.fetchone()
            value = True if row is None else bool(_row_to_dict(row)["botEnabled"])
            self._bot_enabled_cache.set(cache_key, value)
            return value

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
    # ============================================
    # API Key 管理（统一访问控制）
    # ============================================
    async def create_api_key_record(self, key_hash: str, key_prefix: str, role: str,
                                    description: Optional[str] = None,
                                    rate_limit: Optional[int] = None) -> Dict:
        """Create a managed API key row in the main database."""
        async with self.async_session() as session:
            created_at = time.time()
            result = await session.execute(
                api_keys_table.insert().values(
                    key_hash=key_hash,
                    key_prefix=key_prefix,
                    role=role,
                    description=description,
                    created_at=created_at,
                    rate_limit=rate_limit,
                )
            )
            await session.commit()
            return {
                "id": result.inserted_primary_key[0],
                "key_hash": key_hash,
                "key_prefix": key_prefix,
                "role": role,
                "description": description,
                "created_at": created_at,
                "rate_limit": rate_limit,
            }

    async def get_api_key_by_hash(self, key_hash: str) -> Optional[Dict]:
        """Return one managed API key row by stored hash."""
        async with self.async_session() as session:
            result = await session.execute(
                api_keys_table.select().where(api_keys_table.c.key_hash == key_hash)
            )
            row = result.fetchone()
            return _row_to_dict(row) if row else None

    async def get_api_key_by_id(self, key_id: int) -> Optional[Dict]:
        """Return one managed API key row by database id."""
        async with self.async_session() as session:
            result = await session.execute(
                api_keys_table.select().where(api_keys_table.c.id == key_id)
            )
            row = result.fetchone()
            return _row_to_dict(row) if row else None

    async def list_api_keys(self, include_revoked: bool = False) -> List[Dict]:
        """List managed API key metadata from the main database."""
        async with self.async_session() as session:
            stmt = api_keys_table.select()
            if not include_revoked:
                stmt = stmt.where(api_keys_table.c.is_active == 1)
            stmt = stmt.order_by(api_keys_table.c.created_at.desc())
            result = await session.execute(stmt)
            return [_row_to_dict(row) for row in result.fetchall()]

    async def revoke_api_key_by_hash(self, key_hash: str) -> bool:
        """Revoke a managed API key by its stored hash."""
        async with self.async_session() as session:
            result = await session.execute(
                api_keys_table.update()
                .where(api_keys_table.c.key_hash == key_hash)
                .where(api_keys_table.c.is_active == 1)
                .values(is_active=0, revoked_at=time.time())
            )
            await session.commit()
            return result.rowcount > 0

    async def revoke_api_key_by_id(self, key_id: int) -> bool:
        """Revoke a managed API key by its database id."""
        async with self.async_session() as session:
            result = await session.execute(
                api_keys_table.update()
                .where(api_keys_table.c.id == key_id)
                .where(api_keys_table.c.is_active == 1)
                .values(is_active=0, revoked_at=time.time())
            )
            await session.commit()
            return result.rowcount > 0

    async def get_api_key_rows_by_prefix(self, prefix: str) -> List[Dict]:
        """Return active/inactive key rows matching a key prefix."""
        async with self.async_session() as session:
            stmt = api_keys_table.select().where(api_keys_table.c.key_prefix == prefix)
            result = await session.execute(stmt)
            return [_row_to_dict(row) for row in result.fetchall()]

    async def touch_api_key(self, key_hash: str) -> None:
        """Update last_used_at for a managed API key."""
        async with self.async_session() as session:
            await session.execute(
                api_keys_table.update()
                .where(api_keys_table.c.key_hash == key_hash)
                .values(last_used_at=time.time())
            )
            await session.commit()

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
        # M5 fix: 拒绝空 task_id，与 SQLite 侧保持一致
        if not task_id or not str(task_id).strip():
            raise ValueError("task_id 不能为空")
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
        """添加训练任务并返回规范化记录。

        P1-M2 fix: 与 SQLite 侧对齐，统一走 save_training_task + get_training_task，
        返回 _normalize_training_task 规范化的 DTO，而非 legacy 列。
        """
        task_id = task.get("task_id") or task.get("id") or ""
        now = datetime.now().isoformat()
        task_data = {
            "lora_name": task.get("lora_name", ""),
            "status": task.get("status", "pending"),
            "progress": float(task.get("progress", 0) or 0),
            "error_message": task.get("error_message", ""),
            "config": task.get("config", {}),
            "created_at": task.get("created_at", now),
            "updated_at": task.get("updated_at", now),
        }
        await self.save_training_task(task_id, task_data)
        return await self.get_training_task(task_id) or {**task, "task_id": task_id}

    async def get_training_tasks(self, status: Optional[str] = None) -> List[Dict]:
        """获取训练任务列表。

        P1-M2 fix: 返回 _normalize_training_task 规范化的 DTO，与 SQLite 侧一致；
        排序改用 created_at（与 SQLite 对齐），不再依赖 legacy createdAt 列。
        """
        async with self.async_session() as session:
            stmt = training_tasks_table.select()
            if status:
                stmt = stmt.where(training_tasks_table.c.status == status)
            # 优先用 created_at（新 schema），回退到 createdAt（legacy）
            stmt = stmt.order_by(training_tasks_table.c.created_at.desc())
            result = await session.execute(stmt)
            return [self._normalize_training_task(row) for row in result.fetchall()]

    async def get_training_task(self, task_id: str) -> Optional[Dict]:
        """获取单个训练任务"""
        async with self.async_session() as session:
            stmt = training_tasks_table.select().where(training_tasks_table.c.id == task_id)
            result = await session.execute(stmt)
            return self._normalize_training_task(result.fetchone())

    async def update_training_task(self, task_id: str, data: Dict) -> Optional[Dict]:
        """更新训练任务字段。

        P1-M2 fix: 与 SQLite 侧对齐，更新 config_json/updated_at/error_message 等
        新 schema 列；config 入参为 dict 时序列化为 config_json 存储。
        返回 _normalize_training_task 规范化的 DTO。
        """
        now = datetime.now().isoformat()
        values: Dict = {"updated_at": now}
        for key in ("status", "progress", "error_message", "lora_name"):
            if key in data:
                values[key] = data[key]
        if "config" in data:
            # config 是 dict，存到 config_json 列
            try:
                values["config_json"] = json.dumps(data["config"], ensure_ascii=False)
            except (TypeError, ValueError):
                pass
        if "config_json" in data:
            values["config_json"] = data["config_json"]
        async with self.async_session() as session:
            stmt = (
                training_tasks_table.update()
                .where(training_tasks_table.c.id == task_id)
                .values(**values)
            )
            await session.execute(stmt)
            await session.commit()

            sel_stmt = training_tasks_table.select().where(training_tasks_table.c.id == task_id)
            result = await session.execute(sel_stmt)
            return self._normalize_training_task(result.fetchone())

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
import concurrent.futures
import threading


class SyncPgAdapter:
    """同步适配器：将 PgDatabase 的异步方法包装为同步方法。

    使用独立事件循环在后台线程中运行异步方法，
    确保与现有同步代码（如 SQLite Database 类）兼容。
    """

    def __init__(
        self,
        pg: PgDatabase,
        *,
        init_timeout: float = 30.0,
        operation_timeout: float = 30.0,
        close_timeout: float = 15.0,
        thread_join_timeout: float = 5.0,
    ):
        self._pg = pg
        self._loop = None
        self._thread = None
        self._state_lock = threading.RLock()
        self._pending: set[concurrent.futures.Future] = set()
        self._closed = False
        self._init_timeout = self._validate_timeout("init_timeout", init_timeout)
        self._operation_timeout = self._validate_timeout(
            "operation_timeout", operation_timeout
        )
        self._close_timeout = self._validate_timeout("close_timeout", close_timeout)
        self._thread_join_timeout = self._validate_timeout(
            "thread_join_timeout", thread_join_timeout
        )

    @staticmethod
    def _validate_timeout(name: str, value: float) -> float:
        timeout = float(value)
        if timeout <= 0:
            raise ValueError(f"{name} must be greater than zero")
        return timeout

    @staticmethod
    def _run_loop(loop: asyncio.AbstractEventLoop, started: threading.Event) -> None:
        asyncio.set_event_loop(loop)
        started.set()
        loop.run_forever()

    @staticmethod
    def _close_unsubmitted_coroutine(coro) -> None:
        close = getattr(coro, "close", None)
        if callable(close):
            close()

    async def _shutdown_backend(self) -> None:
        """Cancel adapter-owned tasks before disposing the async engine."""
        current = asyncio.current_task()
        tasks = [
            task
            for task in asyncio.all_tasks()
            if task is not current and not task.done()
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self._pg.close()

    def _stop_loop_locked(self, *, close_backend: bool) -> None:
        loop = self._loop
        thread = self._thread
        if loop is None:
            self._thread = None
            return

        if close_backend and loop.is_running():
            shutdown_future = asyncio.run_coroutine_threadsafe(
                self._shutdown_backend(), loop
            )
            try:
                shutdown_future.result(timeout=self._close_timeout)
            except concurrent.futures.TimeoutError:
                shutdown_future.cancel()
                logger.warning("Timed out while closing SyncPgAdapter backend")
            except Exception as exc:
                logger.warning("Failed to close SyncPgAdapter backend: %s", exc)

        if loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=self._thread_join_timeout)
        if thread is not None and thread.is_alive():
            logger.warning("SyncPgAdapter event-loop thread did not stop in time")
        elif not loop.is_running() and not loop.is_closed():
            loop.close()

        self._loop = None
        self._thread = None

    def _ensure_loop_locked(self) -> None:
        """Start and initialize the private loop while holding the state lock."""
        if self._closed:
            raise RuntimeError("SyncPgAdapter is closed")
        if self._loop is None or not self._loop.is_running():
            if self._loop is not None:
                self._stop_loop_locked(close_backend=False)
            loop = asyncio.new_event_loop()
            started = threading.Event()
            thread = threading.Thread(
                target=self._run_loop,
                args=(loop, started),
                name="sync-pg-adapter",
                daemon=True,
            )
            self._loop = loop
            self._thread = thread
            thread.start()
            if not started.wait(timeout=self._init_timeout):
                self._stop_loop_locked(close_backend=False)
                raise TimeoutError("SyncPgAdapter event loop did not start in time")

            init_future = asyncio.run_coroutine_threadsafe(self._pg.init(), loop)
            try:
                init_future.result(timeout=self._init_timeout)
            except concurrent.futures.TimeoutError:
                init_future.cancel()
                self._stop_loop_locked(close_backend=True)
                raise
            except Exception:
                self._stop_loop_locked(close_backend=True)
                raise

    def _ensure_loop(self) -> None:
        """确保后台事件循环正在运行"""
        with self._state_lock:
            self._ensure_loop_locked()
            # 初始化数据库

    def _run(self, coro):
        """在后台事件循环中运行协程并等待结果"""
        future = None
        try:
            with self._state_lock:
                self._ensure_loop_locked()
                future = asyncio.run_coroutine_threadsafe(coro, self._loop)
                self._pending.add(future)
        except BaseException:
            if future is None:
                self._close_unsubmitted_coroutine(coro)
            raise

        try:
            return future.result(timeout=self._operation_timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise
        finally:
            with self._state_lock:
                self._pending.discard(future)

    def close(self):
        """Cancel pending work and close the private loop exactly once."""
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            for future in tuple(self._pending):
                future.cancel()
            self._pending.clear()
            self._stop_loop_locked(close_backend=True)

    # 代理所有 PgDatabase 的方法为同步调用
    def init(self):
        self._ensure_loop()

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

    def get_all_knowledge_chunks(self, limit: int | None = None, offset: int = 0):
        return self._run(self._pg.get_all_knowledge_chunks(limit=limit, offset=offset))

    def iter_all_knowledge_chunks(self, batch_size: int = 500):
        """SyncPgAdapter 不支持异步生成器委托，回退到分页批量拉取。

        调用方如需流式迭代，应直接使用 PgDatabase.iter_all_knowledge_chunks（async）。
        """
        offset = 0
        while True:
            batch = self._run(self._pg.get_all_knowledge_chunks(limit=batch_size, offset=offset))
            if not batch:
                break
            for chunk in batch:
                yield chunk
            offset += len(batch)
            if len(batch) < batch_size:
                break

    def iter_chunks_with_document(self, batch_size: int = 500):
        """同步生成器版本：分页迭代 chunk + document（LEFT JOIN）。

        与 iter_all_knowledge_chunks 同样回退到分页批量拉取，每条 dict 包含
        chunk 字段及 doc_title / doc_category / doc_kb_id。
        """
        # PgDatabase.iter_chunks_with_document 是 async generator，无法直接 _run。
        # 改为分页调用底层 SQL，与 SQLite 实现保持语义一致。
        offset = 0
        while True:
            batch = self._run(self._pg.get_chunks_with_document(limit=batch_size, offset=offset))
            if not batch:
                break
            for row in batch:
                yield row
            offset += len(batch)
            if len(batch) < batch_size:
                break

    def add_knowledge_chunk(self, chunk_data):
        return self._run(self._pg.add_knowledge_chunk(chunk_data))

    def get_knowledge_stats(self):
        return self._run(self._pg.get_knowledge_stats())

    def add_user(self, username, password_hash, bootstrap_only=False):
        return self._run(self._pg.add_user(username, password_hash, bootstrap_only))

    def get_user(self, user_id):
        return self._run(self._pg.get_user(user_id))

    def get_user_by_username(self, username):
        return self._run(self._pg.get_user_by_username(username))

    def get_user_data(self, user_id, page_key=None):
        return self._run(self._pg.get_user_data(user_id, page_key))

    def save_user_data(self, user_id, page_key, data_json):
        return self._run(self._pg.save_user_data(user_id, page_key, data_json))

    # 角色关系与长期记忆（委托 PgDatabase 异步实现）
    def get_character_relationship(self, character_id, platform, adapter, sender_id, conversation_type, conversation_id):
        return self._run(self._pg.get_character_relationship(
            character_id, platform, adapter, sender_id, conversation_type, conversation_id
        ))

    def upsert_character_relationship(self, character_id, platform, adapter, sender_id, conversation_type, conversation_id, relationship_stage, preferred_address="", summary="", interaction_count=None):
        return self._run(self._pg.upsert_character_relationship(
            character_id, platform, adapter, sender_id, conversation_type, conversation_id,
            relationship_stage, preferred_address, summary, interaction_count
        ))

    def increment_character_interaction(self, character_id, platform, adapter, sender_id, conversation_type, conversation_id):
        return self._run(self._pg.increment_character_interaction(
            character_id, platform, adapter, sender_id, conversation_type, conversation_id
        ))

    def list_character_memories(self, character_id, platform, adapter, sender_id, conversation_type, conversation_id, limit=30):
        return self._run(self._pg.list_character_memories(
            character_id, platform, adapter, sender_id, conversation_type, conversation_id, limit
        ))

    def add_or_update_character_memory(self, character_id, platform, adapter, sender_id, conversation_type, conversation_id, memory_type, memory_key, content, importance=0.0, source_message_id=None):
        return self._run(self._pg.add_or_update_character_memory(
            character_id, platform, adapter, sender_id, conversation_type, conversation_id,
            memory_type, memory_key, content, importance, source_message_id
        ))

    def delete_character_memory(self, memory_id, character_id, platform, adapter, sender_id, conversation_type, conversation_id):
        return self._run(self._pg.delete_character_memory(
            memory_id, character_id, platform, adapter, sender_id, conversation_type, conversation_id
        ))

    def clear_character_memories(self, character_id, platform, adapter, sender_id, conversation_type, conversation_id):
        return self._run(self._pg.clear_character_memories(
            character_id, platform, adapter, sender_id, conversation_type, conversation_id
        ))

    def list_conversation_history(self, platform, adapter, sender_id, conversation_type, conversation_id, limit=8, max_chars=6000):
        return self._run(self._pg.list_conversation_history(
            platform, adapter, sender_id, conversation_type, conversation_id, limit, max_chars
        ))

    def create_api_key_record(self, key_hash, key_prefix, role, description=None, rate_limit=None):
        return self._run(self._pg.create_api_key_record(key_hash, key_prefix, role, description, rate_limit))

    def get_api_key_by_hash(self, key_hash):
        return self._run(self._pg.get_api_key_by_hash(key_hash))

    def get_api_key_by_id(self, key_id):
        return self._run(self._pg.get_api_key_by_id(key_id))

    def list_api_keys(self, include_revoked=False):
        return self._run(self._pg.list_api_keys(include_revoked))

    def revoke_api_key_by_hash(self, key_hash):
        return self._run(self._pg.revoke_api_key_by_hash(key_hash))

    def revoke_api_key_by_id(self, key_id):
        return self._run(self._pg.revoke_api_key_by_id(key_id))

    def get_api_key_rows_by_prefix(self, prefix):
        return self._run(self._pg.get_api_key_rows_by_prefix(prefix))

    def touch_api_key(self, key_hash):
        return self._run(self._pg.touch_api_key(key_hash))

    def get_session_summaries(self):
        return self._run(self._pg.get_session_summaries())

    def set_session_bot_enabled(self, session_id, enabled, platform="qq", conversation_id=None, conversation_type="private"):
        return self._run(self._pg.set_session_bot_enabled(session_id, enabled, platform, conversation_id, conversation_type))

    def is_session_bot_enabled(self, session_id, platform="qq", conversation_id=None, conversation_type="private"):
        return self._run(self._pg.is_session_bot_enabled(session_id, platform, conversation_id, conversation_type))

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
        # 与 SQLiteDB.messages 对齐，硬编码 limit=1000，避免 PG 模式下加载 10000 行造成内存压力
        return self.get_messages(limit=1000)

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
