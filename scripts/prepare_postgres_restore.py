#!/usr/bin/env python3
"""Replay chat erasures and clear stale grants in a staged PostgreSQL restore."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from asyncpg import PostgresError
from sqlalchemy import make_url, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import create_async_engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from db.character_chat_erasure import ERASURE_TABLE, SESSION_ROWS
from db.character_chat_history import HISTORY_NODES, ORPHANED_HISTORY_NODE
from db.character_sharing_restore import reset_restored_sharing
from db.schema_readiness import require_current_migrations
from db.urls import normalize_async_database_url

_EVIDENCE = "character_chat_evidence"
_POLICIES = "character_chat_proactive_policies"
_JOBS = "character_chat_proactive_jobs"
_ATTEMPTS = "character_chat_push_attempts"
_SUBSCRIPTIONS = "character_chat_push_subscriptions"


def postgres_url(value):
    url = make_url(normalize_async_database_url(value))
    if url.drivername != "postgresql+asyncpg" or not url.database:
        raise ValueError("Use an explicit PostgreSQL URL with a database name")
    return url


async def prepare_restore(staged_url, receipt_url):
    staged, source = postgres_url(staged_url), postgres_url(receipt_url)
    if staged.database == source.database:
        raise ValueError("Use a separately named staging database")
    receipt_engine = create_async_engine(source, hide_parameters=True)
    try:
        async with receipt_engine.begin() as connection:
            await connection.execute(text("SET TRANSACTION READ ONLY"))
            if await connection.scalar(text("SELECT current_database()")) != source.database:
                raise ValueError("Receipt connection resolved to an unexpected database")
            result = await connection.execute(text(f"SELECT session_id FROM {ERASURE_TABLE}"))
            receipt_ids = result.scalars().all()
    finally:
        await receipt_engine.dispose()

    tables = sorted({ERASURE_TABLE, _EVIDENCE, HISTORY_NODES, "character_library", *(table for table, _ in SESSION_ROWS)})
    staged_engine = create_async_engine(staged, hide_parameters=True)
    try:
        async with staged_engine.begin() as connection:
            if await connection.scalar(text("SELECT current_database()")) != staged.database:
                raise ValueError("Staging connection resolved to an unexpected database")
            await connection.run_sync(require_current_migrations)
            await connection.execute(text(f"LOCK TABLE {','.join(tables)} IN ACCESS EXCLUSIVE MODE NOWAIT"))
            if receipt_ids:
                await connection.execute(
                    text(f"INSERT INTO {ERASURE_TABLE} (session_id) VALUES (:id) ON CONFLICT DO NOTHING"),
                    [{"id": identity} for identity in receipt_ids],
                )
            erased = f"SELECT session_id FROM {ERASURE_TABLE}"
            await connection.execute(text(
                f"DELETE FROM {_EVIDENCE} WHERE turn_id IN "
                f"(SELECT id FROM character_chat_turns WHERE session_id IN ({erased}))"
            ))
            for table, column in SESSION_ROWS:
                await connection.execute(text(f"DELETE FROM {table} WHERE {column} IN ({erased})"))
            await connection.execute(text(f"DELETE FROM {HISTORY_NODES} WHERE {ORPHANED_HISTORY_NODE}"))
            await connection.execute(text(
                f"UPDATE {_POLICIES} SET enabled=0,revision=revision+1,inactivity_due_at=NULL,inactivity_turn_id=NULL "
                "WHERE enabled<>0 OR inactivity_due_at IS NOT NULL OR inactivity_turn_id IS NOT NULL"
            ))
            await connection.execute(text(f"UPDATE {_JOBS} SET status='cancelled',topic='' WHERE status='pending'"))
            await connection.execute(text(f"DELETE FROM {_ATTEMPTS}"))
            await connection.execute(text(f"DELETE FROM {_SUBSCRIPTIONS}"))
            await connection.execute(text("DELETE FROM character_shared_memories"))
            sharing_sql, sharing_params = reset_restored_sharing()
            await connection.execute(text(sharing_sql), sharing_params)
    finally:
        await staged_engine.dispose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-writers-stopped", action="store_true")
    args = parser.parse_args()
    if not args.confirm_writers_stopped:
        parser.error("Stop the API and workers before preparing the staged database")
    staged = os.environ.get("RESTORE_STAGED_DATABASE_URL", "")
    receipts = os.environ.get("RESTORE_RECEIPT_DATABASE_URL", "")
    if not staged or not receipts:
        parser.error("Set RESTORE_STAGED_DATABASE_URL and RESTORE_RECEIPT_DATABASE_URL")
    try:
        asyncio.run(prepare_restore(staged, receipts))
    except (OSError, PostgresError, RuntimeError, SQLAlchemyError, ValueError) as exc:
        # Do not include connection configuration or SQL parameters in operational output.
        print(f"Restore preparation failed ({type(exc).__name__}). Keep the staged database offline.", file=sys.stderr)
        return 1
    print("Staged database prepared. Verify it before switching the application connection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
