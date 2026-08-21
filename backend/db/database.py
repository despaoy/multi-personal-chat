import os
import json
import sqlite3
import time
import logging
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

from cache.ttl_value_cache import BoundedTTLCache
from db.errors import RegistrationClosedError

logger = logging.getLogger(__name__)

# LoRA路径映射 - 自动扫描 backend/loras/ 目录
BACKEND_DIR = Path(__file__).resolve().parent.parent


def _runtime_path_from_env(name: str, default: Path) -> Path:
    """Resolve a persistent runtime path without depending on the process cwd."""
    configured = os.getenv(name, "").strip()
    path = Path(configured).expanduser() if configured else default
    return path if path.is_absolute() else BACKEND_DIR / path


LORA_ROOT = _runtime_path_from_env("LORA_PATH", BACKEND_DIR / "loras")


def _scan_lora_dirs(lora_base: Optional[Path] = None):
    """扫描 loras 目录，自动发现 LoRA 适配器"""
    lora_base = lora_base or LORA_ROOT
    path_map = {}
    if lora_base.exists():
        for d in lora_base.iterdir():
            if d.is_dir():
                # 检查是否包含 adapter_config.json（LoRA 适配器标志）
                # 支持直接在目录下或 final/ 子目录下
                has_adapter = (d / "adapter_config.json").exists() or (d / "final" / "adapter_config.json").exists()
                if has_adapter:
                    path_map[d.name] = str(d)
    return path_map


LORA_DIR_MAP = _scan_lora_dirs()
_LORA_DIR_MAP_LOCK = threading.RLock()


def refresh_lora_dir_map(lora_base: Optional[Path] = None) -> dict[str, str]:
    """Refresh the shared adapter directory map without replacing its identity."""

    discovered = _scan_lora_dirs(lora_base)
    with _LORA_DIR_MAP_LOCK:
        LORA_DIR_MAP.clear()
        LORA_DIR_MAP.update(discovered)
        return dict(LORA_DIR_MAP)


# C-R1 fix: 原先 LORA_PATH_MAP = {} 是静态空 dict，全程无代码填充，
# 导致 vLLM 故障回退到 TransformersPeftProvider 时 LoRA 永远不会被加载，
# 静默退化为 base model。改为函数动态查找：从 db.loras 表的 id 列映射
# 到 LORA_DIR_MAP 的目录路径。调用方需改为 get_lora_path_by_id(id)。
def get_lora_path_by_id(lora_id: str, *, loras: list | None = None) -> str | None:
    """根据 LoRA id 查找其文件系统路径。

    查找逻辑：
    1. 遍历 loras（默认 db.loras），找到匹配 id 的 LoRA 记录
    2. 用 LoRA 的 name 字段在 LORA_DIR_MAP 中查找路径
    3. 若 name 未命中，尝试用 id 本身作为目录名（历史兼容）

    loras 由容器注入的调用方传入（从当前应用容器数据库读出的
    列表）：默认查全局 db.loras 时，自定义容器下 LoRA 列表来自
    容器、路径却在全局库查不到，会静默退回基座模型。
    """
    try:
        directory_map = refresh_lora_dir_map()
        for lora in (loras if loras is not None else db.loras):
            if str(lora.get("id")) == str(lora_id):
                name = lora.get("name", "")
                if name and name in directory_map:
                    return directory_map[name]
                # 历史兼容：部分旧 LoRA 的目录名等于 id
                if str(lora_id) in directory_map:
                    return directory_map[str(lora_id)]
                return None
    except Exception:
        pass
    return None


# 保留 LORA_PATH_MAP 作为空 dict 的向后兼容别名，但标记为废弃。
# 新代码应使用 get_lora_path_by_id()。
LORA_PATH_MAP = {}  # deprecated, use get_lora_path_by_id()


def _resolve_path(p: str) -> str:
    if os.path.isabs(p):
        return p
    return str(Path(__file__).parent.parent / p)

# 数据库路径
def _database_path_from_env() -> Path:
    configured = os.getenv("DATABASE_PATH", "").strip()
    if configured:
        candidate = Path(configured).expanduser()
        return candidate if candidate.is_absolute() else BACKEND_DIR / candidate
    return BACKEND_DIR / "qq_assistant.db"

DB_PATH = _database_path_from_env()

# ============================================
# SQLite数据库类
# ============================================
class SQLiteDB:
    """SQLite数据库类 - 实现数据持久化"""
    def __init__(self, db_path: Path | str = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._bot_enabled_cache: BoundedTTLCache[tuple[str, str, str], bool] = BoundedTTLCache(
            ttl=float(os.getenv("SESSION_SWITCH_CACHE_TTL", "60")),
            max_size=int(os.getenv("SESSION_SWITCH_CACHE_MAX_SIZE", "4096")),
        )
        self._init_database()

    def _get_connection(self):
        """获取数据库连接 - 线程本地复用 + WAL模式"""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute('PRAGMA journal_mode=WAL')
            conn.execute('PRAGMA busy_timeout=5000')
            conn.execute('PRAGMA synchronous=NORMAL')
            conn.execute('PRAGMA cache_size=-8000')
            conn.execute('PRAGMA foreign_keys=ON')
            self._local.conn = conn
        return self._local.conn

    def get_connection(self):
        """获取数据库连接（公开接口）"""
        return self._get_connection()

    def close_connection(self):
        """关闭当前线程的数据库连接"""
        if hasattr(self._local, 'conn') and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None

    def _init_database(self):
        """初始化数据库表结构"""
        conn = self._get_connection()
        cursor = conn.cursor()

        # 创建消息表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sessionType TEXT NOT NULL,
                sessionId TEXT NOT NULL,
                sessionName TEXT,
                platform TEXT NOT NULL DEFAULT 'qq',
                adapter TEXT NOT NULL DEFAULT 'nonebot',
                conversationId TEXT,
                senderId TEXT,
                sourceMessageId TEXT,
                traceId TEXT,
                userId TEXT,
                userName TEXT,
                message TEXT NOT NULL,
                reply TEXT NOT NULL,
                modelName TEXT,
                loraName TEXT,
                costTime REAL,
                createdAt TEXT NOT NULL
            )
        ''')

        # 创建LoRA模型表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS loras (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                status TEXT NOT NULL DEFAULT 'inactive',
                style TEXT,
                size TEXT,
                trainedSteps INTEGER,
                totalSteps INTEGER,
                createdAt TEXT
            )
        ''')

        # 创建配置表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        ''')

        # 创建知识库表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_bases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        ''')

        # 创建知识库文件夹表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_folders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                knowledge_base_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (knowledge_base_id) REFERENCES knowledge_bases(id) ON DELETE CASCADE,
                UNIQUE(knowledge_base_id, name)
            )
        ''')

        # 创建知识库文档表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT '未分类',
                knowledge_base_id INTEGER,
                folder_id INTEGER,
                sourceType TEXT NOT NULL DEFAULT 'text',
                sourceUrl TEXT,
                fileType TEXT,
                fileSize INTEGER,
                chunkCount INTEGER DEFAULT 0,
                createdAt TEXT NOT NULL,
                updatedAt TEXT NOT NULL,
                FOREIGN KEY (knowledge_base_id) REFERENCES knowledge_bases(id) ON DELETE SET NULL,
                FOREIGN KEY (folder_id) REFERENCES knowledge_folders(id) ON DELETE SET NULL
            )
        ''')

        # 迁移：为旧表添加 category 字段（如果不存在）
        try:
            cursor.execute("SELECT category FROM knowledge_documents LIMIT 1")
        except sqlite3.OperationalError:
            cursor.execute("ALTER TABLE knowledge_documents ADD COLUMN category TEXT NOT NULL DEFAULT '未分类'")
            logger.info("已为 knowledge_documents 表添加 category 字段")

        # 迁移：为旧表添加 knowledge_base_id 和 folder_id 字段
        try:
            cursor.execute("SELECT knowledge_base_id FROM knowledge_documents LIMIT 1")
        except sqlite3.OperationalError:
            cursor.execute("ALTER TABLE knowledge_documents ADD COLUMN knowledge_base_id INTEGER REFERENCES knowledge_bases(id) ON DELETE SET NULL")
            logger.info("已为 knowledge_documents 表添加 knowledge_base_id 字段")
        try:
            cursor.execute("SELECT folder_id FROM knowledge_documents LIMIT 1")
        except sqlite3.OperationalError:
            cursor.execute("ALTER TABLE knowledge_documents ADD COLUMN folder_id INTEGER REFERENCES knowledge_folders(id) ON DELETE SET NULL")
            logger.info("已为 knowledge_documents 表添加 folder_id 字段")

        # 创建知识库向量表（用于RAG）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                documentId INTEGER NOT NULL,
                chunkIndex INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding INTEGER,
                createdAt TEXT NOT NULL,
                FOREIGN KEY (documentId) REFERENCES knowledge_documents(id) ON DELETE CASCADE
            )
        ''')
        # 老库迁移：将 embedding 列从 BLOB 类型重建为 INTEGER（SQLite 无法直接
        # 修改列类型，需要重建表）。
        #
        # 旧实现用 `SELECT typeof(embedding) FROM knowledge_chunks LIMIT 1` 判断，
        # 但 typeof() 返回的是"某一行实际存储的值的类型"而非"列声明类型"：
        #   - 空表 → fetchone() 返回 None，跳过迁移（但列仍是 BLOB）
        #   - 首行 embedding 为 NULL → typeof 返回 'null'，跳过迁移
        #   - 首行恰好存了整数 → typeof 返回 'integer'，跳过迁移
        #   - 只有首行恰好是 BLOB 字节流时才迁移，概率极低。
        # 正确做法是用 PRAGMA table_info 读取列的声明类型 dtd_type。
        #
        # 数据保留策略：旧 BLOB 列可能保存了向量字节流（pickle/np.tobytes()），
        # 也可能在新版代码运行后保存了有效的 FAISS ID 整数。用 CASE 在行级
        # 判断 typeof()，仅保留整数，BLOB/NULL 置 NULL 让上层重建向量索引。
        #
        # 异常处理：用 SAVEPOINT 包裹表重建，失败时回滚到保存点，避免留下
        # 新空表和 knowledge_chunks_old 残留导致旧数据不可见。
        try:
            cursor.execute("PRAGMA table_info(knowledge_chunks)")
            columns = cursor.fetchall()
            # columns: (cid, name, type, notnull, dflt_value, pk)
            embedding_decl_type = ""
            for col in columns:
                if col[1] == "embedding":
                    embedding_decl_type = (col[2] or "").upper()
                    break

            needs_migration = embedding_decl_type in ("", "BLOB")
            if needs_migration:
                cursor.execute("SAVEPOINT embed_migration")
                try:
                    cursor.execute("ALTER TABLE knowledge_chunks RENAME TO knowledge_chunks_old")
                    cursor.execute('''
                        CREATE TABLE knowledge_chunks (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            documentId INTEGER NOT NULL,
                            chunkIndex INTEGER NOT NULL,
                            content TEXT NOT NULL,
                            embedding INTEGER,
                            createdAt TEXT NOT NULL,
                            FOREIGN KEY (documentId) REFERENCES knowledge_documents(id) ON DELETE CASCADE
                        )
                    ''')
                    # 行级判断：仅保留 typeof()='integer' 的值，BLOB/NULL 置 NULL。
                    # 这避免了"把向量字节流当作 FAISS ID 复制"的数据损坏，
                    # 同时不丢失已经正确存储为整数的有效 FAISS ID。
                    cursor.execute('''
                        INSERT INTO knowledge_chunks (id, documentId, chunkIndex, content, embedding, createdAt)
                        SELECT id, documentId, chunkIndex, content,
                               CASE WHEN typeof(embedding) = 'integer' THEN embedding ELSE NULL END,
                               createdAt
                        FROM knowledge_chunks_old
                    ''')
                    cursor.execute("DROP TABLE knowledge_chunks_old")
                    cursor.execute("RELEASE SAVEPOINT embed_migration")
                    logger.info(
                        "已将 knowledge_chunks.embedding 从 %s 重建为 INTEGER（保留有效整数，BLOB/NULL 置 NULL）",
                        embedding_decl_type or "未声明",
                    )
                except Exception as migrate_exc:
                    cursor.execute("ROLLBACK TO SAVEPOINT embed_migration")
                    cursor.execute("RELEASE SAVEPOINT embed_migration")
                    logger.error(
                        "knowledge_chunks.embedding 迁移失败，已回滚到保存点: %s",
                        migrate_exc,
                        exc_info=True,
                    )
        except sqlite3.OperationalError:
            pass  # 表不存在（新库），_init_database 会创建

        # 创建用户表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        ''')
        # C-S1 fix: 新增 role 列支持 RBAC。默认 'user'，首个用户自动升级为 'admin'。
        # 使用 _ensure_column 做幂等迁移，已有数据库升级时自动添加该列。
        self._ensure_column(cursor, 'users', 'role', 'TEXT NOT NULL DEFAULT \'user\'')

        # 创建 API Key 表（统一访问控制，SQLite 与 PostgreSQL 共用主库）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key_hash TEXT NOT NULL UNIQUE,
                key_prefix TEXT NOT NULL,
                role TEXT NOT NULL,
                description TEXT,
                created_at REAL NOT NULL,
                revoked_at REAL,
                last_used_at REAL,
                is_active INTEGER NOT NULL DEFAULT 1,
                rate_limit INTEGER
            )
        ''')

        # 创建用户数据表（表单数据持久化）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                page_key TEXT NOT NULL,
                data_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                UNIQUE(user_id, page_key)
            )
        ''')

        # 创建已保存对话表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS saved_dialogues (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                character_desc TEXT NOT NULL,
                style TEXT,
                dialogue_count INTEGER NOT NULL DEFAULT 0,
                dialogues_json TEXT NOT NULL,
                turn_stats TEXT,
                scene_stats TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        ''')

        # 创建训练任务表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_tasks (
                id TEXT PRIMARY KEY,
                task_id TEXT UNIQUE,
                lora_name TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'pending',
                progress REAL DEFAULT 0,
                error_message TEXT DEFAULT '',
                config_json TEXT DEFAULT '{}',
                created_at TEXT DEFAULT '',
                updated_at TEXT DEFAULT ''
            )
        ''')
        # 老库迁移：如果 training_tasks 表缺少 id 列（旧 schema 用 task_id 作 PK），补齐
        # 此前仅 ADD COLUMN id TEXT，但旧表的主键仍是 task_id，id 列既非主键也非非空，
        # 导致与 ORM/Alembic 的 schema 不一致。现通过表重建将 id 设为主键。
        try:
            cursor.execute("SELECT id FROM training_tasks LIMIT 1")
            # 检查 id 列是否为主键（通过 PRAGMA table_info）
            cursor.execute("PRAGMA table_info(training_tasks)")
            cols = {row[1]: row for row in cursor.fetchall()}
            id_col = cols.get("id")
            task_id_col = cols.get("task_id")
            # 旧库情况：task_id 为主键 (pk=1)，id 为普通列 (pk=0)
            # 新库情况：id 为主键 (pk=1)，task_id 为 UNIQUE 列 (pk=0)
            if task_id_col and task_id_col[5] == 1 and id_col and id_col[5] == 0:
                # 重建表，将 id 设为主键，task_id 改为 UNIQUE
                cursor.execute("ALTER TABLE training_tasks RENAME TO training_tasks_old")
                cursor.execute('''
                    CREATE TABLE training_tasks (
                        id TEXT PRIMARY KEY,
                        task_id TEXT UNIQUE,
                        lora_name TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'pending',
                        progress REAL DEFAULT 0,
                        error_message TEXT DEFAULT '',
                        config_json TEXT DEFAULT '{}',
                        created_at TEXT DEFAULT '',
                        updated_at TEXT DEFAULT ''
                    )
                ''')
                cursor.execute('''
                    INSERT INTO training_tasks (id, task_id, lora_name, status, progress, error_message, config_json, created_at, updated_at)
                    SELECT id, task_id, lora_name, status, progress, error_message, config_json, created_at, updated_at
                    FROM training_tasks_old
                ''')
                cursor.execute("DROP TABLE training_tasks_old")
                logger.info("已将 training_tasks.id 重建为主键（旧库 task_id 主键 → 新库 id 主键）")
        except sqlite3.OperationalError:
            # 旧表完全无 id 列，先添加 id 列并从 task_id 回填，再走主键重建路径
            try:
                cursor.execute("SELECT task_id FROM training_tasks LIMIT 1")
                # 旧表存在但无 id 列
                cursor.execute("ALTER TABLE training_tasks RENAME TO training_tasks_old")
                cursor.execute('''
                    CREATE TABLE training_tasks (
                        id TEXT PRIMARY KEY,
                        task_id TEXT UNIQUE,
                        lora_name TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'pending',
                        progress REAL DEFAULT 0,
                        error_message TEXT DEFAULT '',
                        config_json TEXT DEFAULT '{}',
                        created_at TEXT DEFAULT '',
                        updated_at TEXT DEFAULT ''
                    )
                ''')
                cursor.execute('''
                    INSERT INTO training_tasks (id, task_id, lora_name, status, progress, error_message, config_json, created_at, updated_at)
                    SELECT task_id, task_id, lora_name, status, progress, error_message, config_json, created_at, updated_at
                    FROM training_tasks_old
                ''')
                cursor.execute("DROP TABLE training_tasks_old")
                logger.info("已重建 training_tasks 表（task_id 主键 → id 主键，并从 task_id 回填 id）")
            except sqlite3.OperationalError:
                pass  # 新库，表已正确创建

        # 创建 Claw 自定义工具表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS claw_tools (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT NOT NULL DEFAULT '',
                code TEXT NOT NULL DEFAULT '',
                enabled INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS integration_message_dedup (
                dedupKey TEXT PRIMARY KEY,
                platform TEXT NOT NULL,
                adapter TEXT NOT NULL,
                messageId TEXT NOT NULL,
                createdAt TEXT NOT NULL
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                platform TEXT NOT NULL,
                conversationId TEXT NOT NULL,
                conversationType TEXT NOT NULL DEFAULT 'private',
                displayName TEXT,
                botEnabled INTEGER NOT NULL DEFAULT 1,
                replyPolicy TEXT NOT NULL DEFAULT 'default',
                createdAt TEXT NOT NULL,
                updatedAt TEXT NOT NULL,
                UNIQUE(platform, conversationId, conversationType)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS integration_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                platform TEXT NOT NULL,
                adapter TEXT NOT NULL,
                sourceMessageId TEXT,
                conversationId TEXT,
                conversationType TEXT,
                senderId TEXT,
                eventType TEXT NOT NULL DEFAULT 'message',
                eventHash TEXT NOT NULL,
                rawSummary TEXT,
                traceId TEXT,
                status TEXT NOT NULL DEFAULT 'received',
                createdAt TEXT NOT NULL,
                UNIQUE(platform, adapter, eventHash)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_invocations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                traceId TEXT,
                platform TEXT NOT NULL DEFAULT 'qq',
                conversationId TEXT,
                sessionId TEXT,
                modelName TEXT,
                loraName TEXT,
                costTime REAL DEFAULT 0,
                promptTokens INTEGER DEFAULT 0,
                completionTokens INTEGER DEFAULT 0,
                totalTokens INTEGER DEFAULT 0,
                usedRag INTEGER NOT NULL DEFAULT 0,
                usedLora INTEGER NOT NULL DEFAULT 0,
                errorType TEXT DEFAULT '',
                createdAt TEXT NOT NULL
            )
        ''')

        # ============================================
        # 研究与评估相关表（LLM Research Enhancement Roadmap）
        # ============================================

        # Gold 评估运行记录
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS gold_eval_runs (
                id TEXT PRIMARY KEY,
                run_at TEXT NOT NULL,
                adapter_name TEXT,
                model_label TEXT,
                total_prompts INTEGER DEFAULT 0,
                category_breakdown TEXT,
                metrics TEXT,
                config_snapshot TEXT,
                notes TEXT
            )
        ''')

        # 实验运行记录（LoRA消融/RAG消融/量化基准）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiment_runs (
                id TEXT PRIMARY KEY,
                experiment_type TEXT NOT NULL,
                hypothesis TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                started_at TEXT NOT NULL,
                completed_at TEXT,
                results TEXT,
                config_path TEXT,
                report_path TEXT
            )
        ''')

        # 检索评估数据集
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS retrieval_eval_questions (
                id TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                expected_doc_ids TEXT,
                expected_doc_titles TEXT,
                gold_answer TEXT,
                category TEXT,
                created_at TEXT NOT NULL
            )
        ''')

        # 偏好数据对（DPO/ORPO 训练用）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS preference_pairs (
                id TEXT PRIMARY KEY,
                prompt TEXT NOT NULL,
                chosen TEXT NOT NULL,
                rejected TEXT NOT NULL,
                rubric TEXT,
                annotator TEXT,
                metadata TEXT,
                review_status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL
            )
        ''')

        # 适配器兼容性检查记录
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS adapter_compatibility (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                adapter_name TEXT NOT NULL,
                checked_at TEXT NOT NULL,
                compatible INTEGER NOT NULL,
                checks TEXT,
                warnings TEXT,
                errors TEXT
            )
        ''')

        # 用户反馈（在线反馈闭环）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT,
                message_id TEXT,
                rating TEXT,
                reason TEXT,
                adapter_name TEXT,
                kb_revision TEXT,
                prompt_version TEXT,
                detail TEXT,
                created_at TEXT NOT NULL
            )
        ''')

        # 审计日志表（与 pg_database.py 对齐，此前 SQLite 缺失）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                api_key_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                action TEXT NOT NULL,
                resource TEXT,
                detail TEXT,
                ip_address TEXT
            )
        ''')

        # 意图样本表（与 pg_database.py 对齐，此前 SQLite 缺失）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS intent_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kbName TEXT NOT NULL,
                text TEXT NOT NULL,
                label TEXT NOT NULL,
                createdAt TEXT NOT NULL
            )
        ''')

        # 意图激活知识库表（与 pg_database.py 对齐，此前 SQLite 缺失）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS intent_active_kbs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kbName TEXT NOT NULL,
                isActive INTEGER NOT NULL DEFAULT 1
            )
        ''')

        # 角色关系表：主键为完整记忆隔离范围（与 UserScope.memory_scope_key 一致）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS character_relationships (
                character_id TEXT NOT NULL,
                platform TEXT NOT NULL,
                adapter TEXT NOT NULL,
                sender_id TEXT NOT NULL,
                conversation_type TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                relationship_stage TEXT NOT NULL,
                preferred_address TEXT NOT NULL,
                summary TEXT NOT NULL,
                interaction_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (character_id, platform, adapter, sender_id, conversation_type, conversation_id)
            )
        ''')

        # 角色长期记忆表：memory_key 在同一范围内唯一（upsert 键）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS character_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                character_id TEXT NOT NULL,
                platform TEXT NOT NULL,
                adapter TEXT NOT NULL,
                sender_id TEXT NOT NULL,
                conversation_type TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                memory_key TEXT NOT NULL,
                content TEXT NOT NULL,
                importance REAL NOT NULL DEFAULT 0.0,
                source_message_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(character_id, platform, adapter, sender_id, conversation_type, conversation_id, memory_key)
            )
        ''')

        self._ensure_column(cursor, "messages", "platform", "TEXT NOT NULL DEFAULT 'qq'")
        self._ensure_column(cursor, "messages", "adapter", "TEXT NOT NULL DEFAULT 'nonebot'")
        self._ensure_column(cursor, "messages", "conversationId", "TEXT")
        self._ensure_column(cursor, "messages", "senderId", "TEXT")
        self._ensure_column(cursor, "messages", "sourceMessageId", "TEXT")
        self._ensure_column(cursor, "messages", "traceId", "TEXT")
        self._ensure_column(cursor, "messages", "conversationType", "TEXT")
        self._ensure_column(cursor, "messages", "senderName", "TEXT")
        # One-way compatibility migration: legacy session_settings is folded into
        # conversations and then removed. Fresh databases never create this table.
        try:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='session_settings'")
            if cursor.fetchone():
                # Older legacy tables may lack the columns introduced later.
                cursor.execute("PRAGMA table_info(session_settings)")
                legacy_cols = {row[1] for row in cursor.fetchall()}
                if "platform" not in legacy_cols:
                    cursor.execute("ALTER TABLE session_settings ADD COLUMN platform TEXT NOT NULL DEFAULT 'qq'")
                if "conversationId" not in legacy_cols:
                    cursor.execute("ALTER TABLE session_settings ADD COLUMN conversationId TEXT")
                if "conversationType" not in legacy_cols:
                    cursor.execute("ALTER TABLE session_settings ADD COLUMN conversationType TEXT NOT NULL DEFAULT 'private'")
                    cursor.execute("UPDATE session_settings SET conversationType = sessionType")
                if "sessionName" not in legacy_cols:
                    cursor.execute("ALTER TABLE session_settings ADD COLUMN sessionName TEXT")
                if "bot_enabled" not in legacy_cols:
                    cursor.execute("ALTER TABLE session_settings ADD COLUMN bot_enabled INTEGER NOT NULL DEFAULT 1")
                if "updated_at" not in legacy_cols:
                    cursor.execute("ALTER TABLE session_settings ADD COLUMN updated_at TEXT")
                migrated_at = datetime.now().isoformat()
                cursor.execute('''
                    INSERT OR IGNORE INTO conversations (
                        platform, conversationId, conversationType, displayName,
                        botEnabled, replyPolicy, createdAt, updatedAt
                    )
                    SELECT
                        COALESCE(platform, 'qq'),
                        COALESCE(conversationId, sessionId),
                        COALESCE(conversationType, sessionType, 'private'),
                        COALESCE(NULLIF(sessionName, ''), sessionId),
                        bot_enabled,
                        'default',
                        COALESCE(updated_at, ?),
                        COALESCE(updated_at, ?)
                    FROM session_settings
                ''', (migrated_at, migrated_at))
                # SQLite does not allow DO UPDATE with INSERT...SELECT...FROM;
                # refresh existing conversations so the latest legacy state wins.
                cursor.execute('''
                    UPDATE conversations
                    SET
                        botEnabled = COALESCE((SELECT s.bot_enabled FROM session_settings s
                                              WHERE COALESCE(s.platform, 'qq') = conversations.platform
                                                AND COALESCE(s.conversationId, s.sessionId) = conversations.conversationId
                                                AND COALESCE(s.conversationType, s.sessionType, 'private') = conversations.conversationType),
                                             conversations.botEnabled),
                        displayName = COALESCE((SELECT COALESCE(NULLIF(s.sessionName, ''), s.sessionId) FROM session_settings s
                                              WHERE COALESCE(s.platform, 'qq') = conversations.platform
                                                AND COALESCE(s.conversationId, s.sessionId) = conversations.conversationId
                                                AND COALESCE(s.conversationType, s.sessionType, 'private') = conversations.conversationType),
                                             conversations.displayName),
                        updatedAt = COALESCE((SELECT s.updated_at FROM session_settings s
                                             WHERE COALESCE(s.platform, 'qq') = conversations.platform
                                               AND COALESCE(s.conversationId, s.sessionId) = conversations.conversationId
                                               AND COALESCE(s.conversationType, s.sessionType, 'private') = conversations.conversationType),
                                             conversations.updatedAt)
                    WHERE EXISTS (
                        SELECT 1 FROM session_settings s
                        WHERE COALESCE(s.platform, 'qq') = conversations.platform
                          AND COALESCE(s.conversationId, s.sessionId) = conversations.conversationId
                          AND COALESCE(s.conversationType, s.sessionType, 'private') = conversations.conversationType
                    )
                ''')
                cursor.execute("DROP TABLE session_settings")
                logger.info("已迁移并删除旧的 session_settings 表")
        except sqlite3.OperationalError:
            pass

        conn.commit()

        # 高并发优化：WAL模式 + busy_timeout
        cursor.execute('PRAGMA journal_mode=WAL')
        cursor.execute('PRAGMA busy_timeout=5000')
        cursor.execute('PRAGMA synchronous=NORMAL')
        cursor.execute('PRAGMA cache_size=-8000')  # 8MB cache
        cursor.execute('PRAGMA temp_store=MEMORY')

        # 为高频查询建索引
        try:
            # 删除旧命名和与 UNIQUE 约束重复的索引，避免重复维护。
            cursor.execute('DROP INDEX IF EXISTS idx_messages_sessionId_createdAt')
            cursor.execute('DROP INDEX IF EXISTS idx_messages_createdAt')
            cursor.execute('DROP INDEX IF EXISTS idx_conversations_platform_conversation')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_session_created ON messages(sessionId, createdAt)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_platform_conversation ON messages(platform, conversationId, createdAt)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_source_dedup ON messages(platform, adapter, sourceMessageId)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(createdAt)')
            # UNIQUE(platform, conversationId, conversationType) 已自动创建索引。
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_integration_events_trace ON integration_events(traceId)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_integration_events_platform_created ON integration_events(platform, createdAt)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_invocations_trace ON model_invocations(traceId)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_invocations_created ON model_invocations(createdAt)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_experiment_runs_type ON experiment_runs(experiment_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_preference_pairs_status ON preference_pairs(review_status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_created ON feedback(created_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_adapter_compat_name ON adapter_compatibility(adapter_name)')
            # 补充此前缺失的高频外键/过滤列索引
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_documentId ON knowledge_chunks(documentId)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_knowledge_documents_kb_id ON knowledge_documents(knowledge_base_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_knowledge_documents_folder_id ON knowledge_documents(folder_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_training_tasks_lora_name ON training_tasks(lora_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_training_tasks_status ON training_tasks(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_intent_samples_kbName ON intent_samples(kbName)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_trace_id ON feedback(trace_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_message_id ON feedback(message_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_character_memories_scope ON character_memories(character_id, platform, adapter, sender_id, conversation_type, conversation_id)')
        except Exception:
            pass  # 索引已存在或 SQLite 版本不支，不影响功能

        # 初始化LoRA数据（如果表为空）
        cursor.execute('SELECT COUNT(*) FROM loras')
        if cursor.fetchone()[0] == 0:
            self._init_default_loras(cursor)
        else:
            # 清理无对应文件的旧记录（如硬编码的 hutao_style）
            self._cleanup_stale_loras(cursor)
            # 同步：扫描 loras/ 目录，自动注册新增的 LoRA
            self._sync_loras_from_disk(cursor)

        # 初始化配置数据（如果表为空）
        cursor.execute('SELECT COUNT(*) FROM config')
        if cursor.fetchone()[0] == 0:
            self._init_default_config(cursor)

        conn.commit()
        logger.info(f"✅ 数据库初始化完成: {self.db_path}")

    def _ensure_column(self, cursor, table: str, column: str, definition: str):
        cursor.execute(f"PRAGMA table_info({table})")
        columns = {row[1] for row in cursor.fetchall()}
        if column not in columns:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _init_default_loras(self, cursor):
        """初始化默认LoRA数据 - 自动扫描 loras/ 目录并注册"""
        lora_base = LORA_ROOT
        default_loras = []

        if lora_base.exists():
            for idx, d in enumerate(sorted(lora_base.iterdir()), start=1):
                if not d.is_dir():
                    continue
                # 检查是否包含 adapter_config.json
                config_path = d / "adapter_config.json"
                final_config_path = d / "final" / "adapter_config.json"
                adapter_path = d
                if not config_path.exists() and final_config_path.exists():
                    config_path = final_config_path
                    adapter_path = d / "final"

                if not config_path.exists():
                    continue

                # 读取元信息
                meta = self._read_lora_metadata(adapter_path)

                # 计算 adapter_model 大小
                adapter_file = adapter_path / "adapter_model.safetensors"
                size_str = "未知"
                if adapter_file.exists():
                    size_mb = adapter_file.stat().st_size / (1024 * 1024)
                    size_str = f"{size_mb:.0f}MB"

                # 确定状态：第一个默认 active
                status = "active" if idx == 1 else "inactive"

                lora_name = d.name
                # 生成描述（包含 rank/alpha 信息）
                desc_map = {
                    "hutao_lora_7b": "往生堂第七十七代堂主胡桃的对话风格",
                    "minamo_lora": "神白水菜萌风格 LoRA",
                }
                base_desc = desc_map.get(lora_name, f"LoRA 适配器 - {lora_name}")
                rank_info = f" (rank={meta['rank']}, alpha={meta['alpha']})" if meta['rank'] > 0 else ""
                description = base_desc + rank_info

                trained_steps = meta["trained_steps"]
                total_steps = meta["total_steps"] if meta["total_steps"] > 0 else trained_steps

                default_loras.append({
                    "id": str(idx),
                    "name": lora_name,
                    "description": description,
                    "status": status,
                    "style": "",
                    "size": size_str,
                    "trainedSteps": trained_steps,
                    "totalSteps": total_steps,
                    "createdAt": datetime.now().strftime("%Y-%m-%d"),
                })

        # 如果没有扫描到任何 LoRA，跳过初始化
        if not default_loras:
            return

        for lora in default_loras:
            cursor.execute('''
                INSERT INTO loras (id, name, description, status, style, size, trainedSteps, totalSteps, createdAt)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                lora["id"], lora["name"], lora["description"], lora["status"],
                lora["style"], lora["size"], lora["trainedSteps"],
                lora["totalSteps"], lora["createdAt"]
            ))

    def _cleanup_stale_loras(self, cursor):
        """清理无对应文件的旧记录（如硬编码的 hutao_style）"""
        lora_base = LORA_ROOT
        # 获取 loras/ 目录下所有有效目录名
        valid_dirs = set()
        if lora_base.exists():
            for d in lora_base.iterdir():
                if d.is_dir():
                    has_adapter = (d / "adapter_config.json").exists() or (d / "final" / "adapter_config.json").exists()
                    if has_adapter:
                        valid_dirs.add(d.name)

        # 删除数据库中无对应文件的记录
        cursor.execute('SELECT id, name FROM loras')
        for row in cursor.fetchall():
            name = row[1]
            if name not in valid_dirs:
                cursor.execute('DELETE FROM loras WHERE id = ?', (row[0],))
                logger.info(f"清理无效 LoRA 记录: {name} (无对应文件)")

    def _read_lora_metadata(self, adapter_path: Path) -> dict:
        """从 adapter_config.json 和 trainer_state.json 读取 LoRA 元信息"""
        import json as _json
        meta = {"rank": 0, "alpha": 0, "trained_steps": 0, "total_steps": 0, "train_completed": False}

        config_path = adapter_path / "adapter_config.json"
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    cfg = _json.load(f)
                meta["rank"] = cfg.get("r", 0)
                meta["alpha"] = cfg.get("lora_alpha", 0)
            except Exception:
                pass

        # 尝试读取训练状态
        state_path = adapter_path / "trainer_state.json"
        if not state_path.exists() and adapter_path.name == "final":
            # 查找 checkpoint 目录中的 trainer_state.json
            try:
                checkpoint_dirs = [d for d in adapter_path.parent.iterdir()
                                   if d.is_dir() and d.name.startswith("checkpoint-")]
                if checkpoint_dirs:
                    max_ckpt = max(checkpoint_dirs, key=lambda d: int(d.name.split("-")[-1]))
                    candidate = max_ckpt / "trainer_state.json"
                    if candidate.exists():
                        state_path = candidate
            except Exception:
                pass

        if state_path and state_path.exists():
            try:
                with open(state_path, 'r', encoding='utf-8') as f:
                    state = _json.load(f)
                meta["trained_steps"] = state.get("global_step", 0)
                meta["total_steps"] = state.get("max_steps", 0)
                meta["train_completed"] = state.get("best_metric") is not None or meta["trained_steps"] > 0
            except Exception:
                pass

        # 如果没有 trainer_state，但有 adapter_model，说明训练已完成
        adapter_file = adapter_path / "adapter_model.safetensors"
        if adapter_file.exists() and meta["trained_steps"] == 0:
            meta["train_completed"] = True
            meta["trained_steps"] = 1
            meta["total_steps"] = 1

        return meta

    def _sync_loras_from_disk(self, cursor):
        """同步：扫描 loras/ 目录，注册新增 LoRA 并更新已有记录的元信息"""
        lora_base = LORA_ROOT
        if not lora_base.exists():
            return

        # 获取数据库中已有的 LoRA
        cursor.execute('SELECT id, name FROM loras')
        existing_loras = {row[1]: row[0] for row in cursor.fetchall()}  # name -> id

        # 获取当前最大 ID
        cursor.execute('SELECT MAX(CAST(id AS INTEGER)) FROM loras')
        max_id_row = cursor.fetchone()
        max_id = max_id_row[0] if max_id_row and max_id_row[0] else 0

        for d in sorted(lora_base.iterdir()):
            if not d.is_dir():
                continue

            # 检查是否包含 adapter_config.json
            config_path = d / "adapter_config.json"
            adapter_path = d
            if not config_path.exists() and (d / "final" / "adapter_config.json").exists():
                config_path = d / "final" / "adapter_config.json"
                adapter_path = d / "final"

            if not config_path.exists():
                continue

            # 读取元信息
            meta = self._read_lora_metadata(adapter_path)

            # 计算 adapter 大小
            adapter_file = adapter_path / "adapter_model.safetensors"
            size_str = "未知"
            if adapter_file.exists():
                size_mb = adapter_file.stat().st_size / (1024 * 1024)
                size_str = f"{size_mb:.0f}MB"

            # 生成描述（包含 rank/alpha 信息）
            desc_map = {
                "hutao_lora_7b": "往生堂第七十七代堂主胡桃的对话风格",
                "minamo_lora": "神白水菜萌风格 LoRA",
            }
            base_desc = desc_map.get(d.name, f"LoRA 适配器 - {d.name}")
            rank_info = f" (rank={meta['rank']}, alpha={meta['alpha']})" if meta['rank'] > 0 else ""
            description = base_desc + rank_info

            trained_steps = meta["trained_steps"]
            total_steps = meta["total_steps"] if meta["total_steps"] > 0 else trained_steps

            if d.name in existing_loras:
                # 更新已有记录的元信息
                cursor.execute('''
                    UPDATE loras SET description = ?, size = ?, trainedSteps = ?, totalSteps = ?
                    WHERE name = ?
                ''', (description, size_str, trained_steps, total_steps, d.name))
            else:
                # 新增记录
                max_id += 1
                cursor.execute('''
                    INSERT INTO loras (id, name, description, status, style, size, trainedSteps, totalSteps, createdAt)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    str(max_id), d.name, description, "inactive",
                    "", size_str, trained_steps, total_steps,
                    datetime.now().strftime("%Y-%m-%d")
                ))
                logger.info(f"自动注册 LoRA: {d.name} (size={size_str})")

    def _init_default_config(self, cursor):
        """初始化默认配置"""
        default_config = {
            "botName": "MultiPersonal Chat System",
            "autoReply": "true",
            "groupReply": "true",
            "privateReply": "true",
            "replyDelay": "1",
            # C11 fix: 不再默认 mock。mock provider 会静默返回罐头回复，
            # 生产部署时若用户未手动配置会误以为服务正常。
            # 默认改为 vllm（需通过环境变量配置 vLLM 服务），
            # 若 vLLM 未启动，调用会显式失败而非返回假数据。
            "modelProvider": "vllm",
            "baseModel": "qwen3-8b",
            "temperature": "0.7",
            "maxTokens": "2048",
            "contextWindow": "8k",
            "useKnowledgeBase": "true",
            "language": "zh-CN",
            "timezone": "Asia/Shanghai",
            "defaultReplyTemplate": "",
            "errorAlert": "true",
            "dailyStats": "true",
            "anomalyDetection": "false",
            "contentFilter": "true",
            "contentReview": "true",
            "adminQqList": "",
            "qqchatBackendUrl": "http://127.0.0.1:8000",
            "astrbotQQEnabled": "true",
            "astrbotTelegramEnabled": "true",
            "astrbotWecomEnabled": "true",
            "astrbotWechatOfficialEnabled": "true",
            "astrbotWechatPersonalEnabled": "false",
            "astrbotWechatPersonalAdapter": "gewechat",
            "astrbotWechatPersonalEndpoint": "",
            "astrbotWechatPersonalNotes": "",
            "openaiCompatBaseUrl": "https://api.deepseek.com",
            "openaiCompatApiKey": "",
            "openaiCompatModel": "deepseek-chat"
        }

        for key, value in default_config.items():
            cursor.execute('INSERT INTO config (key, value) VALUES (?, ?)', (key, value))

    def get_messages(self, limit: int = 100, offset: int = 0, session_id: str | None = None):
        """获取消息记录，支持按会话 ID 筛选。

        Args:
            limit: 返回条数上限
            offset: 偏移量
            session_id: 可选，指定会话 ID 时在 SQL 层过滤（避免全表拉取再 Python 过滤）
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        if session_id:
            cursor.execute('''
                SELECT * FROM messages
                WHERE sessionId = ?
                ORDER BY createdAt DESC
                LIMIT ? OFFSET ?
            ''', (session_id, limit, offset))
        else:
            cursor.execute('''
                SELECT * FROM messages
                ORDER BY createdAt DESC
                LIMIT ? OFFSET ?
            ''', (limit, offset))
        rows = cursor.fetchall()
        messages = []
        for row in rows:
            messages.append(dict(row))
        return messages

    def _build_message_filter(
        self,
        search: str | None = None,
        session_type: str | None = None,
        lora_name: str | None = None,
        session_id: str | None = None,
        session_name: str | None = None,
        platform: str | None = None,
    ) -> tuple[str, list]:
        conditions: list[str] = []
        params: list = []

        if search:
            conditions.append("(message LIKE ? OR reply LIKE ? OR userName LIKE ?)")
            params.extend([f"%{search}%", f"%{search}%", f"%{search}%"])
        if session_type:
            conditions.append("sessionType = ?")
            params.append(session_type)
        if lora_name:
            conditions.append("loraName = ?")
            params.append(lora_name)
        if session_id:
            conditions.append("sessionId = ?")
            params.append(session_id)
        if session_name:
            conditions.append("sessionName LIKE ?")
            params.append(f"%{session_name}%")
        if platform:
            conditions.append("platform = ?")
            params.append(platform)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        return where, params

    def get_messages_filtered(
        self,
        search: str | None = None,
        session_type: str | None = None,
        lora_name: str | None = None,
        session_id: str | None = None,
        session_name: str | None = None,
        platform: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ):
        """Get messages with SQL-level filtering and pagination."""
        limit = min(limit, 1000)
        conn = self._get_connection()
        cursor = conn.cursor()
        where, params = self._build_message_filter(
            search=search,
            session_type=session_type,
            lora_name=lora_name,
            session_id=session_id,
            session_name=session_name,
            platform=platform,
        )
        params.extend([limit, offset])
        cursor.execute(
            f"SELECT * FROM messages {where} ORDER BY createdAt DESC LIMIT ? OFFSET ?",
            params,
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_message_count_filtered(
        self,
        search: str | None = None,
        session_type: str | None = None,
        lora_name: str | None = None,
        session_id: str | None = None,
        session_name: str | None = None,
        platform: str | None = None,
    ) -> int:
        """Return the exact count for the same filters used by get_messages_filtered."""
        conn = self._get_connection()
        cursor = conn.cursor()
        where, params = self._build_message_filter(
            search=search,
            session_type=session_type,
            lora_name=lora_name,
            session_id=session_id,
            session_name=session_name,
            platform=platform,
        )
        cursor.execute(f"SELECT COUNT(*) FROM messages {where}", params)
        return int(cursor.fetchone()[0])

    def get_message_count(self) -> int:
        """获取消息总数"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM messages')
        return cursor.fetchone()[0]

    def get_recent_messages(self, limit: int = 10) -> list:
        """获取最近的 N 条消息（按创建时间倒序）。

        Phase 2 fix: 补齐 PG 侧存在但 SQLite 侧缺失的方法，保持双后端接口一致。
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            'SELECT * FROM messages ORDER BY createdAt DESC LIMIT ?',
            (limit,),
        )
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def add_message(self, message: Dict):
        """Add a message record and keep the conversation index in sync."""
        conn = self._get_connection()
        cursor = conn.cursor()
        created_at = message.get("createdAt", datetime.now().isoformat())
        conversation_type = message.get("conversationType") or message.get("sessionType", "private")
        sender_name = message.get("senderName") or message.get("userName", "")
        conversation_id = message.get("conversationId", message.get("sessionId", ""))
        platform = message.get("platform", "qq")
        display_name = message.get("sessionName") or conversation_id or message.get("sessionId", "")

        self._upsert_conversation_cursor(
            cursor,
            platform=platform,
            conversation_id=conversation_id,
            conversation_type=conversation_type,
            display_name=display_name,
        )

        cursor.execute('''
            INSERT INTO messages (
                sessionType, sessionId, sessionName, platform, adapter, conversationId,
                conversationType, senderId, senderName, sourceMessageId, traceId,
                userId, userName, message, reply, modelName, loraName, costTime, createdAt
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            message.get("sessionType", conversation_type),
            message.get("sessionId", ""),
            message.get("sessionName", ""),
            platform,
            message.get("adapter", "nonebot"),
            conversation_id,
            conversation_type,
            message.get("senderId", message.get("userId", "")),
            sender_name,
            message.get("sourceMessageId", ""),
            message.get("traceId", ""),
            message.get("userId", ""),
            message.get("userName", sender_name),
            message.get("message", ""),
            message.get("reply", ""),
            message.get("modelName", ""),
            message.get("loraName", ""),
            message.get("costTime", 0.0),
            created_at,
        ))

        message_id = cursor.lastrowid
        conn.commit()
        return {
            **message,
            "id": str(message_id),
            "conversationType": conversation_type,
            "senderName": sender_name,
            "createdAt": created_at,
        }

    def delete_message(self, msg_id: int) -> bool:
        """删除单条消息记录"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM messages WHERE id = ?', (msg_id,))
        conn.commit()
        return cursor.rowcount > 0

    def delete_messages_by_filter(self, search: str = None, sessionType: str = None, lora: str = None, sessionName: str = None, platform: str = None) -> int:
        """批量删除消息（基于筛选条件），返回删除数量"""
        conn = self._get_connection()
        cursor = conn.cursor()

        conditions = []
        params = []
        if search:
            conditions.append("(message LIKE ? OR reply LIKE ? OR userName LIKE ?)")
            params.extend([f"%{search}%", f"%{search}%", f"%{search}%"])
        if sessionType and sessionType != "all":
            conditions.append("sessionType = ?")
            params.append(sessionType)
        if lora and lora != "all":
            conditions.append("loraName = ?")
            params.append(lora)
        if sessionName:
            conditions.append("sessionName LIKE ?")
            params.append(f"%{sessionName}%")
        if platform and platform != "all":
            conditions.append("platform = ?")
            params.append(platform)

        if conditions:
            cursor.execute(f"DELETE FROM messages WHERE {' AND '.join(conditions)}", params)
        else:
            cursor.execute("DELETE FROM messages")

        conn.commit()
        return cursor.rowcount

    def get_loras(self, status: Optional[str] = None):
        """获取LoRA模型列表"""
        conn = self._get_connection()
        cursor = conn.cursor()

        if status and status != "all":
            cursor.execute('SELECT * FROM loras WHERE status = ?', (status,))
        else:
            cursor.execute('SELECT * FROM loras')

        rows = cursor.fetchall()

        loras = []
        for row in rows:
            loras.append(dict(row))
        return loras

    def update_lora_status(self, lora_id: str, status: str):
        """更新LoRA模型状态"""
        conn = self._get_connection()
        cursor = conn.cursor()

        if status == "active":
            # 原子操作：用CASE在单条SQL中完成，避免竞态条件
            cursor.execute(
                "UPDATE loras SET status = CASE WHEN id = ? THEN 'active' ELSE 'inactive' END",
                (lora_id,)
            )
        else:
            cursor.execute('UPDATE loras SET status = ? WHERE id = ?', (status, lora_id))

        conn.commit()

        # 获取更新后的LoRA
        cursor.execute('SELECT * FROM loras WHERE id = ?', (lora_id,))
        row = cursor.fetchone()

        if row:
            return dict(row)
        return None

    @property
    def config(self):
        """获取配置"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM config')
        rows = cursor.fetchall()

        config_dict = {}
        from db.config_utils import coerce_config_value
        for row in rows:
            key = row['key']
            value = row['value']
            config_dict[key] = coerce_config_value(value)
        return config_dict

    def get_config_value(self, key: str, default=None):
        """获取单个配置项的值"""
        config_dict = self.config
        return config_dict.get(key, default)

    def update_config(self, new_config: Dict):
        """更新配置"""
        conn = self._get_connection()
        cursor = conn.cursor()

        for key, value in new_config.items():
            # 转换为字符串存储
            if isinstance(value, bool):
                value_str = str(value).lower()
            else:
                value_str = str(value)

            cursor.execute('''
                INSERT OR REPLACE INTO config (key, value)
                VALUES (?, ?)
            ''', (key, value_str))

        conn.commit()

    # SyncPgAdapter 通过 set_config 别名对齐 PG 端方法名
    set_config = update_config

    def set_config_value(self, key: str, value):
        """更新单个配置项。

        Phase 2 fix: 补齐 PG 侧存在但 SQLite 侧缺失的方法，保持双后端接口一致。
        """
        self.update_config({key: value})

    # ============================================
    # 审计日志（与 pg_database.py 对齐，此前 SQLite 缺失）
    # ============================================
    def add_audit_log(self, api_key_hash: str, role: str, action: str,
                      resource: str | None = None, detail: str | None = None,
                      ip_address: str | None = None) -> None:
        """记录审计日志"""
        import time
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO audit_logs (timestamp, api_key_hash, role, action, resource, detail, ip_address)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (time.time(), api_key_hash, role, action, resource, detail, ip_address))
        conn.commit()

    def get_audit_logs(self, limit: int = 100, offset: int = 0,
                       role: str | None = None, action: str | None = None) -> list[dict]:
        """查询审计日志"""
        conn = self._get_connection()
        cursor = conn.cursor()
        if role and action:
            cursor.execute(
                'SELECT * FROM audit_logs WHERE role = ? AND action = ? ORDER BY id DESC LIMIT ? OFFSET ?',
                (role, action, limit, offset))
        elif role:
            cursor.execute(
                'SELECT * FROM audit_logs WHERE role = ? ORDER BY id DESC LIMIT ? OFFSET ?',
                (role, limit, offset))
        elif action:
            cursor.execute(
                'SELECT * FROM audit_logs WHERE action = ? ORDER BY id DESC LIMIT ? OFFSET ?',
                (action, limit, offset))
        else:
            cursor.execute(
                'SELECT * FROM audit_logs ORDER BY id DESC LIMIT ? OFFSET ?',
                (limit, offset))
        return [dict(row) for row in cursor.fetchall()]

    # ============================================
    # 意图样本管理（与 pg_database.py 对齐，此前 SQLite 缺失）
    # ============================================
    def add_intent_sample(self, kb_name: str, text: str, label: str) -> dict:
        """添加意图样本"""
        now = datetime.now().isoformat()
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO intent_samples (kbName, text, label, createdAt) VALUES (?, ?, ?, ?)',
            (kb_name, text, label, now))
        conn.commit()
        sample_id = cursor.lastrowid
        return {"id": sample_id, "kbName": kb_name, "text": text, "label": label, "createdAt": now}

    def get_intent_samples(self, kb_name: str | None = None) -> list[dict]:
        """获取意图样本"""
        conn = self._get_connection()
        cursor = conn.cursor()
        if kb_name:
            cursor.execute('SELECT * FROM intent_samples WHERE kbName = ?', (kb_name,))
        else:
            cursor.execute('SELECT * FROM intent_samples')
        return [dict(row) for row in cursor.fetchall()]

    def get_active_kbs(self) -> list[dict]:
        """获取活跃的知识库列表"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM intent_active_kbs WHERE isActive = 1')
        return [dict(row) for row in cursor.fetchall()]

    def set_active_kb(self, kb_name: str, is_active: bool) -> None:
        """设置知识库活跃状态"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            'INSERT OR IGNORE INTO intent_active_kbs (kbName, isActive) VALUES (?, ?)',
            (kb_name, int(is_active)))
        cursor.execute(
            'UPDATE intent_active_kbs SET isActive = ? WHERE kbName = ?',
            (int(is_active), kb_name))
        conn.commit()

    @property
    def messages(self):
        """获取所有消息（兼容性）"""
        return self.get_messages(limit=1000)

    @property
    def loras(self):
        """获取所有LoRA（兼容性）"""
        return self.get_loras()

    # ============================================
    # 知识库管理
    # ============================================
    def create_knowledge_base(self, name: str, description: str = ""):
        """创建知识库"""
        conn = self._get_connection()
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        try:
            cursor.execute(
                'INSERT INTO knowledge_bases (name, description, created_at, updated_at) VALUES (?, ?, ?, ?)',
                (name, description, now, now)
            )
            conn.commit()
            kb_id = cursor.lastrowid
            return {"id": kb_id, "name": name, "description": description, "created_at": now, "updated_at": now}
        except sqlite3.IntegrityError:
            return None

    def get_knowledge_bases(self):
        """获取所有知识库（单次 JOIN 查询，避免 N+1）"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT
                kb.*,
                COUNT(DISTINCT kd.id) AS documentCount,
                COUNT(DISTINCT kf.id) AS folderCount
            FROM knowledge_bases kb
            LEFT JOIN knowledge_documents kd ON kd.knowledge_base_id = kb.id
            LEFT JOIN knowledge_folders kf ON kf.knowledge_base_id = kb.id
            GROUP BY kb.id
            ORDER BY kb.updated_at DESC
        ''')
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get_knowledge_base(self, kb_id: int):
        """获取单个知识库"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM knowledge_bases WHERE id = ?', (kb_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def update_knowledge_base(self, kb_id: int, data: Dict):
        """更新知识库"""
        conn = self._get_connection()
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        cursor.execute(
            'UPDATE knowledge_bases SET name = ?, description = ?, updated_at = ? WHERE id = ?',
            (data.get("name"), data.get("description", ""), now, kb_id)
        )
        conn.commit()
        cursor.execute('SELECT * FROM knowledge_bases WHERE id = ?', (kb_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def delete_knowledge_base(self, kb_id: int):
        """删除知识库（级联删除文件夹和文档）"""
        conn = self._get_connection()
        cursor = conn.cursor()
        # knowledge_folders 有 ON DELETE CASCADE，会自动级联
        # knowledge_documents 的外键是 ON DELETE SET NULL，需手动删除
        # knowledge_chunks 有 ON DELETE CASCADE（引用 documents），删除文档后自动级联
        # 用 BEGIN IMMEDIATE 包裹确保原子性，防止部分删除
        cursor.execute('BEGIN IMMEDIATE')
        try:
            # 先删除关联文档的chunks（通过子查询）
            cursor.execute(
                'DELETE FROM knowledge_chunks WHERE documentId IN (SELECT id FROM knowledge_documents WHERE knowledge_base_id = ?)',
                (kb_id,)
            )
            cursor.execute('DELETE FROM knowledge_documents WHERE knowledge_base_id = ?', (kb_id,))
            # knowledge_folders 有 ON DELETE CASCADE，但显式删除更安全
            cursor.execute('DELETE FROM knowledge_folders WHERE knowledge_base_id = ?', (kb_id,))
            cursor.execute('DELETE FROM knowledge_bases WHERE id = ?', (kb_id,))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        return True

    # ============================================
    # 知识库文件夹管理
    # ============================================
    def create_knowledge_folder(self, kb_id: int, name: str, description: str = ""):
        """创建知识库文件夹"""
        conn = self._get_connection()
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        try:
            cursor.execute(
                'INSERT INTO knowledge_folders (knowledge_base_id, name, description, created_at, updated_at) VALUES (?, ?, ?, ?, ?)',
                (kb_id, name, description, now, now)
            )
            conn.commit()
            folder_id = cursor.lastrowid
            return {"id": folder_id, "knowledge_base_id": kb_id, "name": name, "description": description, "created_at": now, "updated_at": now}
        except sqlite3.IntegrityError:
            return None

    def get_knowledge_folders(self, kb_id: int):
        """获取知识库下的所有文件夹"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM knowledge_folders WHERE knowledge_base_id = ? ORDER BY name', (kb_id,))
        rows = cursor.fetchall()
        result = []
        for row in rows:
            folder = dict(row)
            cursor.execute('SELECT COUNT(*) as cnt FROM knowledge_documents WHERE folder_id = ?', (folder["id"],))
            folder["documentCount"] = cursor.fetchone()["cnt"]
            result.append(folder)
        return result

    def get_knowledge_folder(self, folder_id: int):
        """获取单个文件夹"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM knowledge_folders WHERE id = ?', (folder_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def delete_knowledge_folder(self, folder_id: int):
        """删除文件夹（文档的folder_id置空）"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('UPDATE knowledge_documents SET folder_id = NULL WHERE folder_id = ?', (folder_id,))
        cursor.execute('DELETE FROM knowledge_folders WHERE id = ?', (folder_id,))
        conn.commit()
        return True

    # ============================================
    # 知识库文档管理
    # ============================================
    def add_knowledge_document(self, document: Dict):
        """添加知识库文档"""
        conn = self._get_connection()
        cursor = conn.cursor()
        now = datetime.now().isoformat()

        cursor.execute('''
            INSERT INTO knowledge_documents (title, content, category, knowledge_base_id, folder_id, sourceType, sourceUrl, fileType, fileSize, chunkCount, createdAt, updatedAt)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            document.get("title", ""),
            document.get("content", ""),
            document.get("category", "未分类"),
            document.get("knowledge_base_id"),
            document.get("folder_id"),
            document.get("sourceType", "text"),
            document.get("sourceUrl"),
            document.get("fileType"),
            document.get("fileSize"),
            document.get("chunkCount", 0),
            now,
            now
        ))

        doc_id = cursor.lastrowid
        conn.commit()

        return {
            **document,
            "id": doc_id,
            "createdAt": now,
            "updatedAt": now
        }

    def get_knowledge_documents(self, limit: int = 100, offset: int = 0, category: Optional[str] = None, knowledge_base_id: Optional[int] = None, folder_id: Optional[int] = None):
        """获取知识库文档列表，支持按分类/知识库/文件夹筛选"""
        conn = self._get_connection()
        cursor = conn.cursor()
        conditions = []
        params = []
        if category and category != "全部":
            conditions.append("category = ?")
            params.append(category)
        if knowledge_base_id is not None:
            conditions.append("knowledge_base_id = ?")
            params.append(knowledge_base_id)
        if folder_id is not None:
            conditions.append("folder_id = ?")
            params.append(folder_id)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        cursor.execute(
            f'SELECT * FROM knowledge_documents {where} ORDER BY updatedAt DESC LIMIT ? OFFSET ?',
            params + [limit, offset]
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get_knowledge_document(self, doc_id: int):
        """获取单个知识库文档"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM knowledge_documents WHERE id = ?', (doc_id,))
        row = cursor.fetchone()

        if row:
            return dict(row)
        return None

    # knowledge_documents 表允许更新的列名白名单
    # 与 pg_database.py 保持一致；此前 SQLite 白名单包含不存在的列
    # (summary/folderId/kbId/charCount/status/tags/source)，已修正。
    KNOWLEDGE_DOC_UPDATABLE_COLUMNS = {
        "title", "content", "category",
        "knowledge_base_id", "folder_id",
        "sourceType", "sourceUrl", "fileType", "fileSize",
        "chunkCount", "updatedAt",
    }

    def update_knowledge_document(self, doc_id: int, document: Dict):
        """更新知识库文档 - 只更新提供的字段（白名单校验）"""
        conn = self._get_connection()
        cursor = conn.cursor()
        now = datetime.now().isoformat()

        # 构建动态SET子句，只更新提供的字段（白名单校验防止SQL注入）
        set_clauses = ["updatedAt = ?"]
        values = [now]

        for key, value in document.items():
            if key in ("id", "createdAt"):
                continue
            if key not in self.KNOWLEDGE_DOC_UPDATABLE_COLUMNS:
                logger.warning(f"update_knowledge_document: 忽略非法列名 '{key}'")
                continue
            if value is not None:
                set_clauses.append(f"{key} = ?")
                values.append(value)

        values.append(doc_id)

        cursor.execute(
            f'UPDATE knowledge_documents SET {", ".join(set_clauses)} WHERE id = ?',
            values
        )

        conn.commit()

        # 获取更新后的文档
        cursor.execute('SELECT * FROM knowledge_documents WHERE id = ?', (doc_id,))
        row = cursor.fetchone()

        if row:
            return dict(row)
        return None

    def delete_knowledge_document(self, doc_id: int):
        """删除知识库文档"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('DELETE FROM knowledge_chunks WHERE documentId = ?', (doc_id,))
        cursor.execute('DELETE FROM knowledge_documents WHERE id = ?', (doc_id,))
        conn.commit()

        return True

    def add_knowledge_chunk(self, chunk: Dict):
        """添加知识库文档片段"""
        conn = self._get_connection()
        cursor = conn.cursor()
        now = datetime.now().isoformat()

        cursor.execute('''
            INSERT INTO knowledge_chunks (documentId, chunkIndex, content, embedding, createdAt)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            chunk.get("documentId"),
            chunk.get("chunkIndex"),
            chunk.get("content"),
            chunk.get("embedding"),
            now
        ))

        chunk_id = cursor.lastrowid
        conn.commit()

        return {
            **chunk,
            "id": chunk_id,
            "createdAt": now
        }

    def get_knowledge_chunks(self, doc_id: int):
        """获取文档的所有片段"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM knowledge_chunks
            WHERE documentId = ?
            ORDER BY chunkIndex
        ''', (doc_id,))
        rows = cursor.fetchall()

        chunks = []
        for row in rows:
            chunks.append(dict(row))
        return chunks

    def get_all_knowledge_chunks(self, limit: int | None = None, offset: int = 0):
        """获取所有知识库片段（用于检索）。

        Args:
            limit: 返回数量上限，None 表示全量（谨慎使用，大库可能 OOM）
            offset: 跳过前 N 条
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        if limit is not None:
            cursor.execute(
                'SELECT * FROM knowledge_chunks ORDER BY documentId, chunkIndex LIMIT ? OFFSET ?',
                (limit, offset)
            )
        else:
            cursor.execute('SELECT * FROM knowledge_chunks ORDER BY documentId, chunkIndex')
        rows = cursor.fetchall()

        chunks = []
        for row in rows:
            chunks.append(dict(row))
        return chunks

    def iter_all_knowledge_chunks(self, batch_size: int = 500):
        """分页迭代所有知识库片段，避免大库 OOM。

        生成器，每次 yield 一批（最多 batch_size 条）chunk dict。
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        offset = 0
        while True:
            cursor.execute(
                'SELECT * FROM knowledge_chunks ORDER BY documentId, chunkIndex LIMIT ? OFFSET ?',
                (batch_size, offset)
            )
            rows = cursor.fetchall()
            if not rows:
                break
            for row in rows:
                yield dict(row)
            offset += len(rows)
            if len(rows) < batch_size:
                break

    def iter_chunks_with_document(self, batch_size: int = 500):
        """分页迭代 chunk 及其所属文档（LEFT JOIN），避免 N+1 查询。

        每次 yield 一条 dict，包含 chunk 字段和 document 字段（以 doc_ 前缀）。
        孤儿 chunk（文档已删除）的 doc_title 为 None，调用方可跳过。
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        offset = 0
        while True:
            cursor.execute(
                '''SELECT c.*, d.title AS doc_title, d.category AS doc_category,
                          d.knowledge_base_id AS doc_kb_id
                   FROM knowledge_chunks c
                   LEFT JOIN knowledge_documents d ON c.documentId = d.id
                   ORDER BY c.documentId, c.chunkIndex
                   LIMIT ? OFFSET ?''',
                (batch_size, offset)
            )
            rows = cursor.fetchall()
            if not rows:
                break
            for row in rows:
                yield dict(row)
            offset += len(rows)
            if len(rows) < batch_size:
                break

    def get_knowledge_stats(self):
        """获取知识库统计数据"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) as total FROM knowledge_documents')
        total_docs = cursor.fetchone()['total']

        cursor.execute('SELECT COUNT(*) as total FROM knowledge_chunks')
        total_chunks = cursor.fetchone()['total']

        cursor.execute('SELECT SUM(LENGTH(content)) as total_chars FROM knowledge_documents')
        total_chars_result = cursor.fetchone()['total_chars']
        total_chars = total_chars_result or 0

        return {
            "totalDocuments": total_docs,
            "totalChunks": total_chunks,
            "totalCharacters": total_chars
        }

    # ============================================
    # 会话管理
    # ============================================

    def _upsert_conversation_cursor(
        self,
        cursor,
        *,
        platform: str,
        conversation_id: str,
        conversation_type: str = "private",
        display_name: str = "",
        bot_enabled: bool | None = None,
        reply_policy: str | None = None,
    ):
        if not conversation_id:
            return
        now = datetime.now().isoformat()
        bot_value = 1 if bot_enabled is None else int(bot_enabled)
        reply_value = reply_policy or "default"
        cursor.execute('''
            INSERT INTO conversations (platform, conversationId, conversationType, displayName, botEnabled, replyPolicy, createdAt, updatedAt)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(platform, conversationId, conversationType) DO UPDATE SET
                displayName = COALESCE(NULLIF(excluded.displayName, ''), conversations.displayName),
                botEnabled = CASE WHEN ? IS NULL THEN conversations.botEnabled ELSE excluded.botEnabled END,
                replyPolicy = CASE WHEN ? IS NULL THEN conversations.replyPolicy ELSE excluded.replyPolicy END,
                updatedAt = excluded.updatedAt
        ''', (platform, conversation_id, conversation_type, display_name, bot_value, reply_value, now, now, bot_enabled, reply_policy))

    def upsert_conversation(self, data: Dict):
        conn = self._get_connection()
        cursor = conn.cursor()
        self._upsert_conversation_cursor(
            cursor,
            platform=data.get("platform", "qq"),
            conversation_id=data.get("conversationId") or data.get("sessionId", ""),
            conversation_type=data.get("conversationType") or data.get("sessionType", "private"),
            display_name=data.get("displayName") or data.get("sessionName", ""),
            bot_enabled=data.get("botEnabled") if "botEnabled" in data else None,
            reply_policy=data.get("replyPolicy"),
        )
        conn.commit()

    def get_conversation(self, platform: str, conversation_id: str, conversation_type: str | None = None):
        conn = self._get_connection()
        cursor = conn.cursor()
        if conversation_type:
            cursor.execute(
                'SELECT * FROM conversations WHERE platform = ? AND conversationId = ? AND conversationType = ? LIMIT 1',
                (platform, conversation_id, conversation_type),
            )
        else:
            cursor.execute(
                'SELECT * FROM conversations WHERE platform = ? AND conversationId = ? ORDER BY updatedAt DESC LIMIT 1',
                (platform, conversation_id),
            )
        row = cursor.fetchone()
        return dict(row) if row else None

    def add_integration_event(self, event: Dict):
        conn = self._get_connection()
        cursor = conn.cursor()
        created_at = event.get("createdAt", datetime.now().isoformat())
        raw_summary = event.get("rawSummary", "")
        if not isinstance(raw_summary, str):
            raw_summary = json.dumps(raw_summary, ensure_ascii=False, default=str)
        event_hash = event.get("eventHash") or f"{event.get('sourceMessageId', '')}:{event.get('traceId', '')}"
        cursor.execute('''
            INSERT INTO integration_events (
                platform, adapter, sourceMessageId, conversationId, conversationType,
                senderId, eventType, eventHash, rawSummary, traceId, status, createdAt
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(platform, adapter, eventHash) DO UPDATE SET
                traceId = excluded.traceId,
                status = excluded.status,
                rawSummary = excluded.rawSummary
        ''', (
            event.get("platform", "qq"),
            event.get("adapter", "other"),
            event.get("sourceMessageId", ""),
            event.get("conversationId", ""),
            event.get("conversationType", "private"),
            event.get("senderId", ""),
            event.get("eventType", "message"),
            event_hash,
            raw_summary[:4096],
            event.get("traceId", ""),
            event.get("status", "received"),
            created_at,
        ))
        conn.commit()

    def add_model_invocation(self, invocation: Dict):
        conn = self._get_connection()
        cursor = conn.cursor()
        created_at = invocation.get("createdAt", datetime.now().isoformat())
        prompt_tokens = int(invocation.get("promptTokens", 0) or 0)
        completion_tokens = int(invocation.get("completionTokens", 0) or 0)
        total_tokens = int(invocation.get("totalTokens", prompt_tokens + completion_tokens) or 0)
        cursor.execute('''
            INSERT INTO model_invocations (
                traceId, platform, conversationId, sessionId, modelName, loraName, costTime,
                promptTokens, completionTokens, totalTokens, usedRag, usedLora, errorType, createdAt
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            invocation.get("traceId", ""),
            invocation.get("platform", "qq"),
            invocation.get("conversationId", ""),
            invocation.get("sessionId", ""),
            invocation.get("modelName", ""),
            invocation.get("loraName", ""),
            float(invocation.get("costTime", 0.0) or 0.0),
            prompt_tokens,
            completion_tokens,
            total_tokens,
            int(bool(invocation.get("usedRag", False))),
            int(bool(invocation.get("usedLora", False))),
            invocation.get("errorType", ""),
            created_at,
        ))
        conn.commit()

    def get_session_summaries(self):
        """获取所有会话的聚合统计信息"""
        conn = self._get_connection()
        cursor = conn.cursor()

        # 按 sessionId 聚合：消息数、最近消息内容、最后活跃时间
        # bot_enabled 统一从 conversations 表查询（与 PG 端对齐），
        # 使用相关子查询消除 N+1，避免读取已废弃的 session_settings 表导致数据陈旧。
        cursor.execute('''
            WITH normalized AS (
                SELECT
                    id, sessionId, sessionName,
                    COALESCE(platform, 'qq') AS platform,
                    COALESCE(adapter, 'nonebot') AS adapter,
                    COALESCE(conversationId, sessionId) AS conversationId,
                    COALESCE(conversationType, sessionType) AS conversationType,
                    message, createdAt
                FROM messages
            ), ranked AS (
                SELECT *,
                    ROW_NUMBER() OVER (
                        PARTITION BY platform, conversationType, conversationId
                        ORDER BY createdAt DESC, id DESC
                    ) AS rn
                FROM normalized
            ), aggregated AS (
                SELECT
                    MAX(CASE WHEN rn = 1 THEN sessionId END) AS sessionId,
                    conversationType AS sessionType,
                    MAX(CASE WHEN rn = 1 THEN sessionName END) AS sessionName,
                    platform,
                    MAX(CASE WHEN rn = 1 THEN adapter END) AS adapter,
                    conversationId, conversationType,
                    COUNT(*) AS message_count,
                    MAX(createdAt) AS last_active,
                    MAX(CASE WHEN rn = 1 THEN message END) AS recent_1,
                    MAX(CASE WHEN rn = 2 THEN message END) AS recent_2,
                    MAX(CASE WHEN rn = 3 THEN message END) AS recent_3
                FROM ranked
                GROUP BY platform, conversationId, conversationType
            )
            SELECT a.*, COALESCE(c.botEnabled, 1) AS bot_enabled
            FROM aggregated a
            LEFT JOIN conversations c
              ON c.platform = a.platform
             AND c.conversationId = a.conversationId
             AND c.conversationType = a.conversationType
            ORDER BY a.last_active DESC
        ''')
        rows = cursor.fetchall()

        sessions = []
        for row in rows:
            session_id = row['sessionId']
            session_type = row['sessionType']
            session_name = row['sessionName'] or session_id
            platform = row['platform'] or 'qq'
            adapter = row['adapter'] or 'nonebot'
            conversation_id = row['conversationId'] or session_id
            message_count = row['message_count']
            last_active = row['last_active']

            # 从最近消息中提取摘要（取最近3条用户消息）
            recent = [row[key] for key in ('recent_3', 'recent_2', 'recent_1') if row[key] and row[key].strip()]
            summary = '；'.join(recent)
            if len(summary) > 100:
                summary = summary[:100] + '...'

            sessions.append({
                'sessionId': session_id,
                'sessionType': session_type,
                'sessionName': session_name,
                'platform': platform,
                'adapter': adapter,
                'conversationId': conversation_id,
                'messageCount': message_count,
                'lastActive': last_active,
                'summary': summary,
                'botEnabled': bool(row['bot_enabled']),
            })

        return sessions

    def set_session_bot_enabled(self, session_id: str, enabled: bool, platform: str = "qq", conversation_id: str | None = None, conversation_type: str = "private"):
        """设置某个会话的机器人开关。

        统一写入 conversations 表。旧 session_settings 数据会在初始化时单向迁移，
        随后删除旧表。
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        resolved_conversation_id = conversation_id or session_id
        self._upsert_conversation_cursor(
            cursor,
            platform=platform,
            conversation_id=resolved_conversation_id,
            conversation_type=conversation_type,
            display_name=session_id,
            bot_enabled=enabled,
        )
        conn.commit()
        # 变更后主动失效缓存（键包含 platform、conversation_id、conversation_type）
        self._bot_enabled_cache.invalidate((platform, resolved_conversation_id, conversation_type))

    # session bot 开关内存缓存（减少高频读库）
    # TTL 60s，变更时主动失效
    def is_session_bot_enabled(self, session_id: str, platform: str = "qq", conversation_id: str | None = None, conversation_type: str = "private") -> bool:
        """检查某个会话的机器人是否启用（默认启用，内存 TTL 缓存）。

        统一从 conversations 表查询（此前先查 session_settings 失败再查 conversations，
        现直接查 conversations，消除双表冗余查询）。

        缓存键包含 platform、conversation_id、conversation_type，避免不同平台或会话类型出现相同
        session_id 时相互污染。
        """
        resolved_conversation_id = conversation_id or session_id
        cache_key = (platform, resolved_conversation_id, conversation_type)
        cached = self._bot_enabled_cache.get(cache_key)
        if cached is not None:
            return cached
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            'SELECT botEnabled FROM conversations WHERE platform = ? AND conversationId = ? AND conversationType = ? LIMIT 1',
            (platform, resolved_conversation_id, conversation_type),
        )
        row = cursor.fetchone()
        result = True if row is None else bool(row['botEnabled'])
        self._bot_enabled_cache.set(cache_key, result)
        return result

    def mark_integration_message_processed(self, platform: str, adapter: str, message_id: str) -> bool:
        if not message_id:
            return True
        conn = self._get_connection()
        cursor = conn.cursor()
        key = f"{platform}:{adapter}:{message_id}"
        try:
            cursor.execute(
                "INSERT INTO integration_message_dedup (dedupKey, platform, adapter, messageId, createdAt) VALUES (?, ?, ?, ?, ?)",
                (key, platform, adapter, message_id, datetime.now().isoformat()),
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    # ── Claw 工具 CRUD ──

    def get_claw_tools(self) -> list:
        """获取所有自定义 Claw 工具"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM claw_tools ORDER BY created_at DESC')
        return [dict(row) for row in cursor.fetchall()]

    def get_claw_tool_by_name(self, name: str) -> dict | None:
        """按名称获取单个工具"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM claw_tools WHERE name = ?', (name,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def save_claw_tool(self, name: str, description: str, code: str, enabled: bool = True) -> int:
        """创建或更新自定义 Claw 工具，返回工具 id"""
        conn = self._get_connection()
        cursor = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('''
            INSERT INTO claw_tools (name, description, code, enabled, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                description = excluded.description,
                code = excluded.code,
                enabled = excluded.enabled,
                updated_at = excluded.updated_at
        ''', (name, description, code, int(enabled), now, now))
        conn.commit()
        return cursor.lastrowid

    def delete_claw_tool(self, name: str) -> bool:
        """删除自定义 Claw 工具"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM claw_tools WHERE name = ?', (name,))
        conn.commit()
        return cursor.rowcount > 0

    # ============================================
    # 通用 SQL 执行（兼容 PostgreSQL 模式）
    # ============================================
    def execute_sql(self, query: str, params: Optional[dict] = None):
        """执行原始 SQL 语句，返回结果。

        对于 SELECT 语句，返回行列表（每行为 dict）。
        对于 INSERT/UPDATE/DELETE，返回受影响行数。
        params 必须是 dict 格式的命名参数，如 {"username": "test"}。
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(query, params or {})
            if query.strip().upper().startswith("SELECT"):
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
            else:
                conn.commit()
                return cursor.rowcount
        except Exception:
            conn.rollback()
            raise

    def execute_sql_insert(self, query: str, params: Optional[dict] = None) -> dict:
        """执行 INSERT SQL 并返回插入的行信息（包含自动生成的 ID）"""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(query, params or {})
            conn.commit()
            lastrowid = cursor.lastrowid
            return {"lastrowid": lastrowid, "rowcount": cursor.rowcount}
        except Exception:
            conn.rollback()
            raise

    # ============================================
    # 用户管理（高层方法）
    # ============================================
    def get_user_by_username(self, username: str):
        """通过用户名查找用户"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id, username, password_hash, created_at, role FROM users WHERE username = ?', (username,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_user(self, user_id: int):
        """通过用户 ID 查找用户。

        Phase 2 fix: 补齐 PG 侧存在但 SQLite 侧缺失的方法，保持双后端接口一致。
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            'SELECT id, username, password_hash, created_at, role FROM users WHERE id = ?',
            (user_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def add_user(
        self,
        username: str,
        password_hash: str,
        bootstrap_only: bool = False,
    ):
        """Add a user while assigning the first admin atomically."""
        now = datetime.now().isoformat()
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            # Serialize the empty-table check across processes, not only the
            # asyncio lock in one API worker.
            cursor.execute("BEGIN IMMEDIATE")
            cursor.execute("SELECT 1 FROM users LIMIT 1")
            has_users = cursor.fetchone() is not None
            if bootstrap_only and has_users:
                raise RegistrationClosedError("bootstrap administrator already exists")
            role = "user" if has_users else "admin"
            cursor.execute(
                "INSERT INTO users (username, password_hash, created_at, role) "
                "VALUES (?, ?, ?, ?)",
                (username, password_hash, now, role),
            )
            user_id = cursor.lastrowid
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        return {"id": user_id, "username": username, "created_at": now, "role": role}    # ============================================
    # API Key 管理（统一访问控制）
    # ============================================
    def create_api_key_record(self, key_hash: str, key_prefix: str, role: str,
                              description: str | None = None,
                              rate_limit: int | None = None) -> dict:
        """Create a managed API key row in the main database."""
        conn = self._get_connection()
        cursor = conn.cursor()
        created_at = time.time()
        cursor.execute(
            "INSERT INTO api_keys (key_hash, key_prefix, role, description, created_at, rate_limit) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (key_hash, key_prefix, role, description, created_at, rate_limit),
        )
        conn.commit()
        return {
            "id": cursor.lastrowid,
            "key_hash": key_hash,
            "key_prefix": key_prefix,
            "role": role,
            "description": description,
            "created_at": created_at,
            "rate_limit": rate_limit,
        }

    def get_api_key_by_hash(self, key_hash: str) -> dict | None:
        """Return one managed API key row by stored hash."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, key_hash, key_prefix, role, description, created_at, revoked_at, "
            "last_used_at, is_active, rate_limit FROM api_keys WHERE key_hash = ?",
            (key_hash,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_api_key_by_id(self, key_id: int) -> dict | None:
        """Return one managed API key row by database id."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, key_hash, key_prefix, role, description, created_at, revoked_at, "
            "last_used_at, is_active, rate_limit FROM api_keys WHERE id = ?",
            (key_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def list_api_keys(self, include_revoked: bool = False) -> list[dict]:
        """List managed API key metadata from the main database."""
        conn = self._get_connection()
        cursor = conn.cursor()
        query = (
            "SELECT id, key_prefix, role, description, created_at, revoked_at, "
            "is_active, rate_limit FROM api_keys"
        )
        if not include_revoked:
            query += " WHERE is_active = 1"
        query += " ORDER BY created_at DESC"
        cursor.execute(query)
        return [dict(row) for row in cursor.fetchall()]

    def revoke_api_key_by_hash(self, key_hash: str) -> bool:
        """Revoke a managed API key by its stored hash."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE api_keys SET is_active = 0, revoked_at = ? "
            "WHERE key_hash = ? AND is_active = 1",
            (time.time(), key_hash),
        )
        conn.commit()
        return cursor.rowcount > 0

    def revoke_api_key_by_id(self, key_id: int) -> bool:
        """Revoke a managed API key by its database id."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE api_keys SET is_active = 0, revoked_at = ? "
            "WHERE id = ? AND is_active = 1",
            (time.time(), key_id),
        )
        conn.commit()
        return cursor.rowcount > 0

    def get_api_key_rows_by_prefix(self, prefix: str) -> list[dict]:
        """Return active/inactive key rows matching a key prefix."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT key_hash, role, is_active, rate_limit FROM api_keys WHERE key_prefix = ?",
            (prefix,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def touch_api_key(self, key_hash: str) -> None:
        """Update last_used_at for a managed API key."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE api_keys SET last_used_at = ? WHERE key_hash = ?",
            (time.time(), key_hash),
        )
        conn.commit()

    # ============================================
    # 用户数据持久化（高层方法）
    # ============================================
    def get_user_data(self, user_id: int, page_key: Optional[str] = None):
        """获取用户表单数据"""
        conn = self._get_connection()
        cursor = conn.cursor()
        if page_key:
            cursor.execute(
                'SELECT page_key, data_json, updated_at FROM user_data WHERE user_id = ? AND page_key = ?',
                (user_id, page_key)
            )
            row = cursor.fetchone()
            if not row:
                return None
            return {"page_key": row['page_key'], "data_json": row['data_json'], "updated_at": row['updated_at']}
        else:
            cursor.execute('SELECT page_key, data_json, updated_at FROM user_data WHERE user_id = ?', (user_id,))
            rows = cursor.fetchall()
            data = {
                row['page_key']: {
                    "data_json": row['data_json'],
                    "updated_at": row['updated_at']
                }
                for row in rows
            }
            return data

    def save_user_data(self, user_id: int, page_key: str, data_json: str) -> bool:
        """保存用户表单数据（upsert）"""
        now = datetime.now().isoformat()
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO user_data (user_id, page_key, data_json, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, page_key) DO UPDATE SET
                data_json = excluded.data_json,
                updated_at = excluded.updated_at
        ''', (user_id, page_key, data_json, now))
        conn.commit()
        return True

    # ============================================
    # 角色关系与长期记忆（高层方法）
    # ============================================
    def _character_scope_where(
        self,
        character_id: str,
        platform: str,
        adapter: str,
        sender_id: str,
        conversation_type: str,
        conversation_id: str,
        *,
        table: str = "",
        include_character: bool = True,
    ) -> tuple[str, list]:
        """组装角色记忆隔离范围的 WHERE 子句（按 UserScope.memory_scope_key 语义）。"""
        prefix = f"{table}." if table else ""
        conditions = [
            f"{prefix}platform = ?",
            f"{prefix}adapter = ?",
            f"{prefix}sender_id = ?",
            f"{prefix}conversation_type = ?",
            f"{prefix}conversation_id = ?",
        ]
        params: list = [platform, adapter, sender_id, conversation_type, conversation_id]
        if include_character:
            conditions.insert(0, f"{prefix}character_id = ?")
            params.insert(0, character_id)
        return " AND ".join(conditions), params

    def get_character_relationship(
        self,
        character_id: str,
        platform: str,
        adapter: str,
        sender_id: str,
        conversation_type: str,
        conversation_id: str,
    ) -> Optional[Dict]:
        """读取指定角色+用户范围的关系状态，不存在时返回 None。"""
        conn = self._get_connection()
        cursor = conn.cursor()
        where, params = self._character_scope_where(
            character_id, platform, adapter, sender_id, conversation_type, conversation_id
        )
        cursor.execute(f"SELECT * FROM character_relationships WHERE {where}", params)
        row = cursor.fetchone()
        return dict(row) if row else None

    def upsert_character_relationship(
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
    ) -> Dict:
        """写入关系状态（单条 UPSERT 原子完成）。

        interaction_count 为 None 时 UPDATE 子句不触碰该列、保留数据库
        当前值：先 SELECT 计数再写回的实现在并发下会用旧计数覆盖
        increment_character_interaction 刚自增的结果（管理端更新关系与
        新消息并发时计数回退）。
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        where, params = self._character_scope_where(
            character_id, platform, adapter, sender_id, conversation_type, conversation_id
        )
        count = int(interaction_count) if interaction_count is not None else 0
        set_clauses = [
            "relationship_stage = excluded.relationship_stage",
            "preferred_address = excluded.preferred_address",
            "summary = excluded.summary",
        ]
        if interaction_count is not None:
            set_clauses.append("interaction_count = excluded.interaction_count")
        set_clauses.append("updated_at = excluded.updated_at")
        cursor.execute(
            f'''
            INSERT INTO character_relationships (
                character_id, platform, adapter, sender_id, conversation_type,
                conversation_id, relationship_stage, preferred_address, summary,
                interaction_count, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(character_id, platform, adapter, sender_id, conversation_type, conversation_id)
            DO UPDATE SET {", ".join(set_clauses)}
        ''',
            (
                character_id, platform, adapter, sender_id, conversation_type,
                conversation_id, relationship_stage, preferred_address, summary,
                count, now, now,
            ),
        )
        conn.commit()
        # 写入后回读真实记录（计数与 created_at 以数据库为准）
        cursor.execute(
            f"SELECT * FROM character_relationships WHERE {where}", params
        )
        row = cursor.fetchone()
        if row is None:  # pragma: no cover - 提交成功后行必存在
            raise RuntimeError("upsert_character_relationship 提交后读取失败")
        return dict(row)

    def increment_character_interaction(
        self,
        character_id: str,
        platform: str,
        adapter: str,
        sender_id: str,
        conversation_type: str,
        conversation_id: str,
    ) -> int:
        """交互轮数 +1，返回自增后的值（首次交互时从 1 开始）。

        自增在单条 UPSERT 语句内由数据库原子完成：
        "读取-加一-写回" 的多语句实现在并发下会丢失更新
        （两条并发消息都从 10 更新到 11）。
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        where, params = self._character_scope_where(
            character_id, platform, adapter, sender_id, conversation_type, conversation_id
        )
        cursor.execute('''
            INSERT INTO character_relationships (
                character_id, platform, adapter, sender_id, conversation_type,
                conversation_id, relationship_stage, preferred_address, summary,
                interaction_count, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, 'stranger', '', '', 1, ?, ?)
            ON CONFLICT(character_id, platform, adapter, sender_id, conversation_type, conversation_id)
            DO UPDATE SET
                interaction_count = character_relationships.interaction_count + 1,
                updated_at = excluded.updated_at
        ''', (
            character_id, platform, adapter, sender_id, conversation_type,
            conversation_id, now, now,
        ))
        conn.commit()
        # 提交后读取当前值；写路径已原子化，此读取仅为返回值
        cursor.execute(
            f"SELECT interaction_count FROM character_relationships WHERE {where}",
            params,
        )
        row = cursor.fetchone()
        return int(row["interaction_count"]) if row else 1

    def list_character_memories(
        self,
        character_id: str,
        platform: str,
        adapter: str,
        sender_id: str,
        conversation_type: str,
        conversation_id: str,
        limit: int = 30,
    ) -> list:
        """读取指定范围内最近的记忆（按 updated_at 倒序，最多 limit 条）。"""
        conn = self._get_connection()
        cursor = conn.cursor()
        where, params = self._character_scope_where(
            character_id, platform, adapter, sender_id, conversation_type, conversation_id
        )
        cursor.execute(
            f"SELECT * FROM character_memories WHERE {where} ORDER BY updated_at DESC LIMIT ?",
            [*params, max(1, min(int(limit), 200))],
        )
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def add_or_update_character_memory(
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
    ) -> Dict:
        """写入一条记忆（同 memory_key upsert，更新内容与时间戳）。

        单条 UPSERT 原子完成：先 SELECT 再 INSERT/UPDATE 的实现在并发下
        相同 memory_key 会撞唯一约束（两个线程都查不到既有行然后各自 INSERT）。
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        where, params = self._character_scope_where(
            character_id, platform, adapter, sender_id, conversation_type, conversation_id
        )
        cursor.execute('''
            INSERT INTO character_memories (
                character_id, platform, adapter, sender_id, conversation_type,
                conversation_id, memory_type, memory_key, content, importance,
                source_message_id, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(character_id, platform, adapter, sender_id, conversation_type, conversation_id, memory_key)
            DO UPDATE SET
                memory_type = excluded.memory_type,
                content = excluded.content,
                importance = excluded.importance,
                source_message_id = excluded.source_message_id,
                updated_at = excluded.updated_at
        ''', (
            character_id, platform, adapter, sender_id, conversation_type,
            conversation_id, memory_type, memory_key, content, float(importance),
            source_message_id, now, now,
        ))
        conn.commit()
        # 写入后回读真实记录（id 与 created_at 以数据库为准）
        cursor.execute(
            f"SELECT * FROM character_memories WHERE {where} AND memory_key = ?",
            [*params, memory_key],
        )
        row = cursor.fetchone()
        if row is None:  # pragma: no cover - 提交成功后行必存在
            raise RuntimeError("add_or_update_character_memory 提交后读取失败")
        return dict(row)

    def delete_character_memory(
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
        conn = self._get_connection()
        cursor = conn.cursor()
        where, params = self._character_scope_where(
            character_id, platform, adapter, sender_id, conversation_type, conversation_id
        )
        cursor.execute(
            f"DELETE FROM character_memories WHERE id = ? AND {where}",
            [int(memory_id), *params],
        )
        deleted = cursor.rowcount > 0
        conn.commit()
        return deleted

    def clear_character_memories(
        self,
        character_id: str,
        platform: str,
        adapter: str,
        sender_id: str,
        conversation_type: str,
        conversation_id: str,
    ) -> int:
        """清空指定范围内的全部记忆，返回删除条数。"""
        conn = self._get_connection()
        cursor = conn.cursor()
        where, params = self._character_scope_where(
            character_id, platform, adapter, sender_id, conversation_type, conversation_id
        )
        cursor.execute(f"DELETE FROM character_memories WHERE {where}", params)
        deleted = cursor.rowcount
        conn.commit()
        return deleted

    def list_conversation_history(
        self,
        platform: str,
        adapter: str,
        sender_id: str,
        conversation_type: str,
        conversation_id: str,
        limit: int = 8,
        max_chars: int = 6000,
    ) -> list:
        """按用户范围读取最近对话历史，组装成角色生成用的消息列表。

        - 私聊：platform+adapter+senderId 下全部私聊记录；
        - 群聊/频道：再加 conversationId（群/频道）过滤，只看该用户在该群的消息；
        - 返回按时间正序的 [{"role": "user"|"assistant", "content": ...}]，
          总字符数超过 max_chars 时从最旧一侧截断。
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        conditions = ["platform = ?", "adapter = ?", 'senderId = ?']
        params: list = [platform, adapter, sender_id]
        if conversation_type in ("group", "channel"):
            conditions.append('"conversationId" = ?')
            params.append(conversation_id)
        else:
            conditions.append('(conversationType = ? OR conversationType = ?)')
            params.extend(["private", ""])
        cursor.execute(
            f'SELECT message, reply, createdAt FROM messages WHERE {" AND ".join(conditions)} '
            "ORDER BY createdAt DESC LIMIT ?",
            [*params, max(1, min(int(limit), 50))],
        )
        rows = cursor.fetchall()
        # 倒序取出后翻转为时间正序
        turns: list = []
        for row in reversed(rows):
            message = (row["message"] or "").strip()
            reply = (row["reply"] or "").strip()
            if message:
                turns.append({"role": "user", "content": message})
            if reply:
                turns.append({"role": "assistant", "content": reply})
        # 总长度预算：超限时从最旧一侧丢弃整条消息
        if max_chars > 0:
            kept: list = []
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
    # LoRA 管理（高层方法）
    # ============================================
    def add_lora(self, lora: Dict):
        """添加 LoRA 模型"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO loras (id, name, description, status, style, size, trainedSteps, totalSteps, createdAt)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            lora.get("id"),
            lora.get("name", ""),
            lora.get("description", ""),
            lora.get("status", "inactive"),
            lora.get("style", ""),
            lora.get("size", ""),
            lora.get("trainedSteps", 0),
            lora.get("totalSteps", 0),
            lora.get("createdAt", datetime.now().strftime("%Y-%m-%d")),
        ))
        conn.commit()
        return lora

    def delete_lora(self, lora_id: str) -> bool:
        """删除 LoRA 模型"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM loras WHERE id = ?', (lora_id,))
        conn.commit()
        return cursor.rowcount > 0

    # ============================================
    # 训练任务持久化
    # ============================================
    def save_training_task(self, task_id: str, task_data: dict):
        """保存训练任务状态到数据库。

        M5 fix: 拒绝空 task_id，与 PG 侧保持一致。
        """
        if not task_id or not str(task_id).strip():
            raise ValueError("task_id 不能为空")
        conn = self._get_connection()
        cursor = conn.cursor()
        import json
        cursor.execute('''
            INSERT OR REPLACE INTO training_tasks (id, task_id, lora_name, status, progress,
            error_message, config_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            task_id,
            task_id,
            task_data.get('lora_name', ''),
            task_data.get('status', 'pending'),
            task_data.get('progress', 0),
            task_data.get('error_message', ''),
            json.dumps(task_data.get('config', {}), ensure_ascii=False),
            task_data.get('created_at', ''),
            task_data.get('updated_at', '')
        ))
        conn.commit()

    def get_training_task(self, task_id: str) -> dict | None:
        """获取单个训练任务。

        P1-M2 fix: 返回与 PG _normalize_training_task 一致的 DTO，
        只暴露统一字段（task_id/lora_name/status/progress/error_message/
        config/created_at/updated_at），不泄露 config_json 等内部列。
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM training_tasks WHERE task_id = ?', (task_id,))
        row = cursor.fetchone()
        if row:
            import json
            columns = [desc[0] for desc in cursor.description]
            raw = dict(zip(columns, row))
            try:
                config = json.loads(raw.get('config_json', '{}') or '{}')
            except (TypeError, json.JSONDecodeError):
                config = {}
            return {
                "task_id": raw.get("task_id"),
                "lora_name": raw.get("lora_name", ""),
                "status": raw.get("status", "pending"),
                "progress": float(raw.get("progress", 0) or 0),
                "error_message": raw.get("error_message", ""),
                "config": config,
                "created_at": raw.get("created_at", ""),
                "updated_at": raw.get("updated_at", ""),
            }
        return None

    def get_all_training_tasks(self) -> list:
        """获取所有训练任务。

        P1-M2 fix: 返回统一 DTO，与 PG _normalize_training_task 一致。
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM training_tasks ORDER BY created_at DESC')
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        import json
        results = []
        for row in rows:
            raw = dict(zip(columns, row))
            try:
                config = json.loads(raw.get('config_json', '{}') or '{}')
            except (TypeError, json.JSONDecodeError):
                config = {}
            results.append({
                "task_id": raw.get("task_id"),
                "lora_name": raw.get("lora_name", ""),
                "status": raw.get("status", "pending"),
                "progress": float(raw.get("progress", 0) or 0),
                "error_message": raw.get("error_message", ""),
                "config": config,
                "created_at": raw.get("created_at", ""),
                "updated_at": raw.get("updated_at", ""),
            })
        return results

    def add_training_task(self, task: dict) -> dict:
        """添加训练任务并返回任务记录。

        Phase 2 fix: 补齐 PG 侧存在但 SQLite 侧缺失的方法，保持双后端接口一致。
        m3 fix: 自动补 created_at/updated_at，与 PG 侧行为一致。
        """
        task_id = task.get("task_id") or task.get("id") or ""
        now = datetime.now().isoformat()
        task_data = dict(task)
        # 自动补时间戳（调用方未提供时），与 PG add_training_task 一致
        if not task_data.get("created_at"):
            task_data["created_at"] = now
        if not task_data.get("updated_at"):
            task_data["updated_at"] = now
        self.save_training_task(task_id, task_data)
        return self.get_training_task(task_id) or task_data

    def get_training_tasks(self, status: str | None = None) -> list:
        """按状态筛选训练任务，status=None 返回全部。

        P1-M2 fix: 返回统一 DTO，与 PG _normalize_training_task 一致。
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        if status:
            cursor.execute(
                'SELECT * FROM training_tasks WHERE status = ? ORDER BY created_at DESC',
                (status,),
            )
        else:
            cursor.execute('SELECT * FROM training_tasks ORDER BY created_at DESC')
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        import json
        results = []
        for row in rows:
            raw = dict(zip(columns, row))
            try:
                config = json.loads(raw.get('config_json', '{}') or '{}')
            except (TypeError, json.JSONDecodeError):
                config = {}
            results.append({
                "task_id": raw.get("task_id"),
                "lora_name": raw.get("lora_name", ""),
                "status": raw.get("status", "pending"),
                "progress": float(raw.get("progress", 0) or 0),
                "error_message": raw.get("error_message", ""),
                "config": config,
                "created_at": raw.get("created_at", ""),
                "updated_at": raw.get("updated_at", ""),
            })
        return results

    def update_training_task(self, task_id: str, data: dict):
        """更新训练任务字段。

        P1-M2 fix: 与 PG 侧对齐，支持 config (dict) 自动序列化为 config_json；
        返回统一 DTO。
        """
        import json
        conn = self._get_connection()
        cursor = conn.cursor()
        allowed = {
            "status", "progress", "error_message", "config_json",
            "lora_name",
        }
        updates = {k: v for k, v in data.items() if k in allowed}
        # The storage layer owns updated_at for consistent SQLite/PostgreSQL behavior.
        updates["updated_at"] = datetime.now().isoformat()
        # config (dict) → config_json (str)，与 PG 侧一致
        if "config" in data and "config_json" not in data:
            try:
                updates["config_json"] = json.dumps(data["config"], ensure_ascii=False)
            except (TypeError, ValueError):
                pass
        if not updates:
            return self.get_training_task(task_id)
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        params = list(updates.values()) + [task_id]
        cursor.execute(
            f'UPDATE training_tasks SET {set_clause} WHERE task_id = ?',
            params,
        )
        conn.commit()
        return self.get_training_task(task_id)

    def delete_training_task(self, task_id: str) -> bool:
        """删除训练任务记录。

        M5 fix: 与 PG 侧对齐，返回 bool 表示是否删除了行。
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM training_tasks WHERE task_id = ?', (task_id,))
        conn.commit()
        return cursor.rowcount > 0

    def get_active_training_by_lora_name(self, lora_name: str) -> list:
        """查找指定lora_name的运行中任务。

        M5 fix: 返回统一 DTO，与 PG _normalize_training_task 一致。
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM training_tasks WHERE lora_name = ? AND status IN ('pending', 'running', 'training')",
            (lora_name,)
        )
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        import json
        results = []
        for row in rows:
            raw = dict(zip(columns, row))
            try:
                config = json.loads(raw.get('config_json', '{}') or '{}')
            except (TypeError, json.JSONDecodeError):
                config = {}
            results.append({
                "task_id": raw.get("task_id"),
                "lora_name": raw.get("lora_name", ""),
                "status": raw.get("status", "pending"),
                "progress": float(raw.get("progress", 0) or 0),
                "error_message": raw.get("error_message", ""),
                "config": config,
                "created_at": raw.get("created_at", ""),
                "updated_at": raw.get("updated_at", ""),
            })
        return results

# 全局数据库实例
db = SQLiteDB()
