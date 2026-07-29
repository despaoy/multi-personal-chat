"""
SQLite → PostgreSQL 一次性迁移脚本

用法:
    cd backend
    python -m db.migration

功能:
    1. 读取 SQLite 中所有表的数据
    2. 插入到 PostgreSQL（ON CONFLICT DO NOTHING 处理重复键）
    3. 报告迁移进度和结果
"""

import asyncio
import os
import sqlite3
import sys
import logging
from pathlib import Path
from datetime import datetime
from sqlalchemy import text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# SQLite 数据库路径
SQLITE_DB_PATH = Path(os.getenv("DATABASE_PATH") or Path(__file__).parent.parent / "qq_assistant.db")


# ============================================
# 迁移辅助
# ============================================

async def _migrate_table(pg_db, table_name: str, pg_table, sqlite_rows: list[dict], conflict_columns: list[str] | None = None) -> int:
    """将 SQLite 行数据迁移到 PostgreSQL 表

    Args:
        pg_db: PgDatabase 实例
        table_name: 表名（用于日志）
        pg_table: SQLAlchemy Table 对象
        sqlite_rows: SQLite 查询结果（list of dict）
        conflict_columns: ON CONFLICT 的列名列表

    Returns:
        成功插入的行数
    """
    if not sqlite_rows:
        logger.info(f"  ⏭️  {table_name}: 无数据，跳过")
        return 0

    # C2 fix: 使用 pg_table.columns 构建白名单，防止 SQLite 源数据中的
    # 恶意列名通过 f-string 拼接造成 SQL 注入。只允许目标表实际存在的列。
    valid_columns = {col.name for col in pg_table.columns}
    # 同时校验 conflict_columns
    if conflict_columns:
        for cc in conflict_columns:
            if cc not in valid_columns:
                logger.error(f"  ❌ {table_name}: conflict_column '{cc}' 不存在于目标表")
                return 0

    total = len(sqlite_rows)
    inserted = 0
    failed = 0

    async with pg_db.async_session() as session:
        for i, row in enumerate(sqlite_rows, 1):
            try:
                # 过滤掉 None 值中可能不兼容的键
                values = {}
                for key, value in row.items():
                    # 跳过 SQLite 内部字段
                    if key.startswith("_"):
                        continue
                    # C2 fix: 只保留目标表实际存在的列，防止 SQL 注入
                    if key not in valid_columns:
                        continue
                    values[key] = value

                if not values:
                    continue

                if conflict_columns:
                    # 构建 INSERT ... ON CONFLICT DO NOTHING
                    cols = ", ".join(f'"{c}"' for c in values.keys())
                    placeholders = ", ".join(f":{c}" for c in values.keys())
                    conflict_cols = ", ".join(f'"{c}"' for c in conflict_columns)
                    sql = f'INSERT INTO "{table_name}" ({cols}) VALUES ({placeholders}) ON CONFLICT ({conflict_cols}) DO NOTHING'
                else:
                    cols = ", ".join(f'"{c}"' for c in values.keys())
                    placeholders = ", ".join(f":{c}" for c in values.keys())
                    sql = f'INSERT INTO "{table_name}" ({cols}) VALUES ({placeholders}) ON CONFLICT DO NOTHING'

                result = await session.execute(text(sql), values)
                if result.rowcount and result.rowcount > 0:
                    inserted += 1
            except Exception as e:
                failed += 1
                if failed <= 3:
                    logger.warning(f"  ⚠️  {table_name} 行 {i} 插入失败: {e}")
                continue

            # 进度报告
            if i % 500 == 0:
                logger.info(f"  📊 {table_name}: {i}/{total} 已处理...")

        await session.commit()

    logger.info(f"  ✅ {table_name}: 总计 {total} 行，插入 {inserted} 行，跳过/失败 {total - inserted} 行")
    return inserted


def _read_sqlite_table(conn: sqlite3.Connection, table_name: str) -> list[dict]:
    """从 SQLite 读取整张表的数据"""
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {table_name}")
        columns = [desc[0] for desc in cursor.description]
        rows = []
        for row in cursor.fetchall():
            rows.append(dict(zip(columns, row)))
        return rows
    except sqlite3.OperationalError as e:
        if "no such table" in str(e).lower():
            logger.info(f"  ⏭️  {table_name}: 表不存在于 SQLite，跳过")
            return []
        raise


# ============================================
# 主迁移流程
# ============================================

async def migrate(sqlite_path: str | Path = SQLITE_DB_PATH):
    """执行完整的 SQLite → PostgreSQL 迁移"""
    from db.pg_database import (
        pg_db,
        messages_table, loras_table, config_table,
        knowledge_bases_table, knowledge_folders_table,
        knowledge_documents_table, knowledge_chunks_table,
        users_table, user_data_table, saved_dialogues_table,
        session_settings_table, claw_tools_table,
        audit_logs_table, intent_samples_table,
        intent_active_kbs_table, training_tasks_table,
    )

    sqlite_path = Path(sqlite_path)
    if not sqlite_path.exists():
        logger.error(f"❌ SQLite 数据库文件不存在: {sqlite_path}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("🚀 开始 SQLite → PostgreSQL 迁移")
    logger.info(f"   SQLite: {sqlite_path}")
    logger.info("=" * 60)

    # 初始化 PostgreSQL
    await pg_db.init()

    # 连接 SQLite
    sqlite_conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    sqlite_conn.row_factory = sqlite3.Row

    total_inserted = 0
    start_time = datetime.now()

    # 按依赖顺序迁移（先迁移被外键引用的表）
    migration_plan = [
        # (表名, SQLAlchemy Table, ON CONFLICT 列)
        ("config", config_table, ["key"]),
        ("users", users_table, None),
        ("loras", loras_table, ["id"]),
        ("messages", messages_table, None),
        ("knowledge_bases", knowledge_bases_table, None),
        ("knowledge_folders", knowledge_folders_table, None),
        ("knowledge_documents", knowledge_documents_table, None),
        ("knowledge_chunks", knowledge_chunks_table, None),
        ("user_data", user_data_table, None),
        ("saved_dialogues", saved_dialogues_table, None),
        ("session_settings", session_settings_table, ["sessionId"]),
        ("claw_tools", claw_tools_table, ["name"]),
        ("audit_logs", audit_logs_table, None),
        ("intent_samples", intent_samples_table, None),
        ("intent_active_kbs", intent_active_kbs_table, None),
        ("training_tasks", training_tasks_table, ["id"]),
    ]

    for table_name, pg_table, conflict_cols in migration_plan:
        logger.info(f"\n📦 迁移表: {table_name}")
        rows = _read_sqlite_table(sqlite_conn, table_name)
        count = await _migrate_table(pg_db, table_name, pg_table, rows, conflict_cols)
        total_inserted += count

    sqlite_conn.close()

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info("\n" + "=" * 60)
    logger.info(f"🎉 迁移完成！共插入 {total_inserted} 行，耗时 {elapsed:.1f}s")
    logger.info("=" * 60)

    # 关闭连接
    await pg_db.close()


# ============================================
# 入口
# ============================================
if __name__ == "__main__":
    asyncio.run(migrate())
