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
import re
import sqlite3
import sys
import logging
from pathlib import Path
from datetime import datetime
from sqlalchemy import text


def _validate_table_name(name: str) -> str:
    """Validate SQL table name to prevent injection (M-1 fix).

    与 infra/backup_manager.py 中的实现保持一致。
    """
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
        raise ValueError(f"Invalid table name: {name}")
    return name

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

async def _migrate_table(pg_db, table_name: str, pg_table, sqlite_rows: list[dict],
                        conflict_columns: list[str] | None = None,
                        column_mapping: dict[str, str] | None = None) -> tuple[int, int]:
    """将 SQLite 行数据迁移到 PostgreSQL 表

    Args:
        pg_db: PgDatabase 实例
        table_name: 表名（用于日志）
        pg_table: SQLAlchemy Table 对象
        sqlite_rows: SQLite 查询结果（list of dict）
        conflict_columns: ON CONFLICT 的列名列表
        column_mapping: SQLite 列名 → PG 列名的映射（用于字段名不一致的表，
            如 training_tasks 的 task_id → id）。
            M4 fix: 映射是"复制"而非"替换"——SQLite 源列的值会同时写入
            映射后的 PG 列和原始 PG 列（若存在），保证 task_id 等字段不为空。

    Returns:
        (inserted, failed) 元组：成功插入行数与失败行数。
        M1 fix: 失败行数返回给外层，使 migrate() 能据此非零退出。
    """
    if not sqlite_rows:
        logger.info(f"  ⏭️  {table_name}: 无数据，跳过")
        return 0, 0

    # C2 fix: 使用 pg_table.columns 构建白名单，防止 SQLite 源数据中的
    # 恶意列名通过 f-string 拼接造成 SQL 注入。只允许目标表实际存在的列。
    valid_columns = {col.name for col in pg_table.columns}
    # 同时校验 conflict_columns
    if conflict_columns:
        for cc in conflict_columns:
            if cc not in valid_columns:
                logger.error(f"  ❌ {table_name}: conflict_column '{cc}' 不存在于目标表")
                return 0, len(sqlite_rows)

    total = len(sqlite_rows)
    inserted = 0
    failed = 0

    async with pg_db.async_session() as session:
        for i, row in enumerate(sqlite_rows, 1):
            # M1 fix: 用 session.begin_nested() 管理 SAVEPOINT，失败时由
            # context manager 自动 ROLLBACK TO SAVEPOINT，保证事务不进入
            # aborted 状态，后续行仍可正常插入。
            # 若 begin_nested 失败（极少见），降级为整事务 rollback + 重建。
            try:
                # 过滤并映射字段
                values = {}
                for key, value in row.items():
                    if key.startswith("_"):
                        continue
                    # M4 fix: 映射是复制而非替换
                    # 1. 原始列名若存在于 PG 表，保留写入（如 task_id）
                    if key in valid_columns:
                        values[key] = value
                    # 2. 若有映射，额外写入映射后的列（如 task_id → id）
                    if column_mapping and key in column_mapping:
                        mapped_key = column_mapping[key]
                        if mapped_key in valid_columns and mapped_key not in values:
                            values[mapped_key] = value

                if not values:
                    failed += 1
                    if failed <= 3:
                        logger.warning(f"  WARNING: {table_name} row {i} has no migratable columns")
                    continue

                if conflict_columns:
                    cols = ", ".join(f'"{c}"' for c in values.keys())
                    placeholders = ", ".join(f":{c}" for c in values.keys())
                    conflict_cols = ", ".join(f'"{c}"' for c in conflict_columns)
                    sql = f'INSERT INTO "{table_name}" ({cols}) VALUES ({placeholders}) ON CONFLICT ({conflict_cols}) DO NOTHING'
                else:
                    cols = ", ".join(f'"{c}"' for c in values.keys())
                    placeholders = ", ".join(f":{c}" for c in values.keys())
                    sql = f'INSERT INTO "{table_name}" ({cols}) VALUES ({placeholders}) ON CONFLICT DO NOTHING'

                # 用 begin_nested 包裹单行插入
                try:
                    row_inserted = False
                    async with session.begin_nested():
                        result = await session.execute(text(sql), values)
                        row_inserted = bool(result.rowcount and result.rowcount > 0)
                    # Count only after the savepoint exits successfully.
                    if row_inserted:
                        inserted += 1
                except Exception as row_err:
                    # savepoint 已自动回滚，事务仍可用
                    failed += 1
                    if failed <= 3:
                        logger.warning(f"  ⚠️  {table_name} 行 {i} 插入失败: {row_err}")
                    # 若事务整体已损坏（极少见），rollback 重建
                    if not session.is_active:
                        await session.rollback()
            except Exception as outer_err:
                # begin_nested 本身失败或事务损坏，降级处理
                failed += 1
                if failed <= 3:
                    logger.warning(f"  ⚠️  {table_name} 行 {i} 事务异常: {outer_err}")
                try:
                    if session.in_transaction():
                        await session.rollback()
                except Exception:
                    pass

            # 进度报告
            if i % 500 == 0:
                logger.info(f"  📊 {table_name}: {i}/{total} 已处理...")

        await session.commit()

    logger.info(f"  ✅ {table_name}: 总计 {total} 行，插入 {inserted} 行，失败 {failed} 行")
    return inserted, failed


async def _sync_pg_sequences(pg_db) -> tuple[int, int]:
    """P0-C2 fix: 迁移完成后同步 PG 序列，防止后续自增主键冲突。

    C1 fix: 改用 pg_get_serial_sequence()（PostgreSQL 标准函数）获取序列名，
    区分空表/非空表的 setval 语义：
    - 非空表: setval(seq, MAX(col), true)  下次 nextval 返回 MAX+1
    - 空表:   setval(seq, 1, false)         下次 nextval 返回 1
    异常提升为 WARNING，不再静默降级为 DEBUG。

    M1 fix: 返回 (synced, failed) 统计，调用方据此决定退出码。
    """
    async with pg_db.async_session() as session:
        # 查询所有 attach 到本库的序列及其归属表/列
        seq_rows = await session.execute(text("""
            SELECT
                ns.nspname AS schema,
                seq.relname AS sequence_name,
                tbl.relname AS table_name,
                att.attname AS column_name
            FROM pg_class seq
            JOIN pg_namespace ns ON seq.relnamespace = ns.oid
            JOIN pg_depend dep ON dep.objid = seq.oid AND dep.deptype IN ('a', 'i')
            JOIN pg_class tbl ON dep.refobjid = tbl.oid
            JOIN pg_attribute att
                ON att.attrelid = tbl.oid AND att.attnum = dep.refobjsubid
            WHERE seq.relkind = 'S' AND ns.nspname = 'public'
        """))
        sequences = seq_rows.fetchall()

    synced = 0
    failed = 0
    for schema, seq_name, tbl_name, col_name in sequences:
        # 安全校验：catalog 返回的标识符仍需校验，避免边界情况
        try:
            _validate_table_name(tbl_name)
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', col_name):
                continue
        except ValueError:
            continue
        try:
            async with pg_db.async_session() as session:
                # C1 fix: 用 pg_get_serial_sequence 获取序列名（返回 text 或 NULL）
                # 签名: pg_get_serial_sequence(regclass, text) → text
                # 非空表 setval(seq, MAX(col), true)，空表 setval(seq, 1, false)
                await session.execute(text(
                    f'SELECT setval('
                    f'  pg_get_serial_sequence(:tbl, :col), '
                    f'  CASE '
                    f'    WHEN (SELECT COUNT(*) FROM "{tbl_name}") > 0 '
                    f'    THEN (SELECT MAX("{col_name}") FROM "{tbl_name}") '
                    f'    ELSE 1 '
                    f'  END, '
                    f'  (SELECT COUNT(*) FROM "{tbl_name}") > 0'
                    f')'
                ), {"tbl": tbl_name, "col": col_name})
                await session.commit()
                synced += 1
        except Exception as e:
            failed += 1
            # C1 fix: 异常提升为 WARNING，避免静默失效掩盖迁移问题
            logger.warning(f"  ⚠️  序列 {schema}.{seq_name} ({tbl_name}.{col_name}) 同步失败: {e}")

    if synced:
        logger.info(f"  🔧 已同步 {synced} 个 PG 序列" + (f"，{failed} 个失败" if failed else ""))
    return synced, failed


def _read_sqlite_table(conn: sqlite3.Connection, table_name: str) -> list[dict]:
    """从 SQLite 读取整张表的数据"""
    try:
        _validate_table_name(table_name)
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
        api_keys_table, claw_tools_table,
        audit_logs_table, intent_samples_table,
        intent_active_kbs_table, training_tasks_table,
        # Phase 2 fix: 补齐此前遗漏的 10 张表
        integration_message_dedup_table,
        conversations_table,
        integration_events_table,
        model_invocations_table,
        gold_eval_runs_table,
        experiment_runs_table,
        retrieval_eval_questions_table,
        preference_pairs_table,
        adapter_compatibility_table,
        feedback_table,
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
    total_failed = 0  # M1 fix: 统计行级失败
    start_time = datetime.now()

    # 按依赖顺序迁移（先迁移被外键引用的表）
    # P0-C2 fix: 元组增加 column_mapping 字段，用于 SQLite→PG 字段名不一致的表
    migration_plan = [
        # (表名, SQLAlchemy Table, ON CONFLICT 列, 字段映射 dict)
        ("config", config_table, ["key"], None),
        ("users", users_table, None, None),
        ("loras", loras_table, ["id"], None),
        ("messages", messages_table, None, None),
        ("knowledge_bases", knowledge_bases_table, None, None),
        ("knowledge_folders", knowledge_folders_table, None, None),
        ("knowledge_documents", knowledge_documents_table, None, None),
        ("knowledge_chunks", knowledge_chunks_table, None, None),
        ("user_data", user_data_table, None, None),
        ("saved_dialogues", saved_dialogues_table, None, None),
        ("api_keys", api_keys_table, ["id"], None),
        ("claw_tools", claw_tools_table, ["name"], None),
        ("audit_logs", audit_logs_table, None, None),
        ("intent_samples", intent_samples_table, None, None),
        ("intent_active_kbs", intent_active_kbs_table, None, None),
        # M4 fix: SQLite training_tasks 主键是 task_id，PG 主键是 id。
        # 映射 task_id → id 是复制语义：task_id 值同时写入 PG 的 id 和 task_id 两列，
        # 保证 task_id 唯一索引不为空，数据契约完整。
        ("training_tasks", training_tasks_table, ["id"], {"task_id": "id"}),
        # Phase 2 fix: 补齐遗漏的 10 张表
        ("integration_message_dedup", integration_message_dedup_table, ["dedupKey"], None),
        ("conversations", conversations_table, None, None),
        ("integration_events", integration_events_table, None, None),
        ("model_invocations", model_invocations_table, None, None),
        ("gold_eval_runs", gold_eval_runs_table, ["id"], None),
        ("experiment_runs", experiment_runs_table, ["id"], None),
        ("retrieval_eval_questions", retrieval_eval_questions_table, ["id"], None),
        ("preference_pairs", preference_pairs_table, ["id"], None),
        ("adapter_compatibility", adapter_compatibility_table, None, None),
        ("feedback", feedback_table, None, None),
    ]

    for table_name, pg_table, conflict_cols, col_mapping in migration_plan:
        logger.info(f"\n📦 迁移表: {table_name}")
        rows = _read_sqlite_table(sqlite_conn, table_name)
        inserted, failed = await _migrate_table(pg_db, table_name, pg_table, rows, conflict_cols, col_mapping)
        total_inserted += inserted
        total_failed += failed

    # P0-C2 fix: 同步 PG 序列，避免后续 INSERT 因序列值小于已迁移数据主键而冲突。
    # 仅对含 SERIAL/identity 列的表生效，无序列的表会被自动跳过。
    # M1 fix: 序列同步失败必须使迁移命令非零退出，避免未同步的数据库被误判为迁移成功。
    seq_synced, seq_failed = await _sync_pg_sequences(pg_db)
    total_failed += seq_failed

    sqlite_conn.close()

    elapsed = (datetime.now() - start_time).total_seconds()
    # m2 fix: 失败判断在成功日志之前，避免日志语义冲突
    if total_failed:
        logger.error("\n" + "=" * 60)
        logger.error(f"❌ 迁移未完全成功：插入 {total_inserted} 行，失败 {total_failed} 行，耗时 {elapsed:.1f}s")
        logger.error("=" * 60)
    else:
        logger.info("\n" + "=" * 60)
        logger.info(f"🎉 迁移完成！共插入 {total_inserted} 行，耗时 {elapsed:.1f}s")
        logger.info("=" * 60)

    # 关闭连接
    await pg_db.close()

    # M1 fix: 存在行级或序列同步失败时以非零状态退出
    if total_failed:
        sys.exit(1)


# ============================================
# 入口
# ============================================
if __name__ == "__main__":
    asyncio.run(migrate())
