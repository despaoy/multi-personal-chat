#!/usr/bin/env python3
"""Restore a gzip-compressed SQLite backup while the backend is stopped."""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
import sqlite3
import sys
import uuid
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.db.character_chat_erasure import ERASURE_TABLE, SESSION_ROWS


def _validate_sqlite(path: Path) -> None:
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        row = connection.execute("PRAGMA integrity_check").fetchone()
    finally:
        connection.close()
    if row is None or str(row[0]).lower() != "ok":
        raise RuntimeError(f"SQLite integrity_check failed: {row!r}")


def _table_names(connection):
    return {name for (name,) in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _read_erasure_receipts(path: Path, *, required=False):
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
        if ERASURE_TABLE in _table_names(connection):
            return {session_id for (session_id,) in connection.execute(f"SELECT session_id FROM {ERASURE_TABLE}")}
    if required:
        raise ValueError("The supplied recovery database has no chat erasure table")
    return set()


def _apply_erasure_receipts(candidate: Path, receipts):
    with closing(sqlite3.connect(candidate)) as connection, connection:
        tables = _table_names(connection)
        if not receipts and (
            ERASURE_TABLE not in tables or not connection.execute(f"SELECT 1 FROM {ERASURE_TABLE} LIMIT 1").fetchone()
        ):
            return False
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA secure_delete=ON")
        connection.execute(f"CREATE TABLE IF NOT EXISTS {ERASURE_TABLE} (session_id TEXT PRIMARY KEY NOT NULL)")
        connection.executemany(f"INSERT OR IGNORE INTO {ERASURE_TABLE} (session_id) VALUES (?)",
                               ((session_id,) for session_id in receipts))
        erased = f"SELECT session_id FROM {ERASURE_TABLE}"
        if {"character_chat_turns", "character_chat_evidence"} <= tables:
            connection.execute("DELETE FROM character_chat_evidence WHERE turn_id IN "
                               f"(SELECT id FROM character_chat_turns WHERE session_id IN ({erased}))")
        for table, column in SESSION_ROWS:
            if table in tables:
                connection.execute(f"DELETE FROM {table} WHERE {column} IN ({erased})")
    return True


def restore_sqlite_backup(
    backup_path: Path, database_path: Path, *, deletion_records_from: Path | None = None,
) -> Path | None:
    """Validate, safety-copy, and atomically install one SQLite backup."""
    backup = backup_path.expanduser().resolve()
    database = database_path.expanduser().resolve()
    if not backup.is_file():
        raise FileNotFoundError(f"Backup does not exist: {backup}")
    # Replacing a SQLite main file while its old WAL still has frames can
    # replay unrelated pages, including pages excluded from the candidate.
    wal = database.with_name(database.name + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise RuntimeError("The target database has uncheckpointed WAL data. Checkpoint it with all writers stopped.")

    database.parent.mkdir(parents=True, exist_ok=True)
    temporary = database.with_name(f".{database.name}.restore-{uuid.uuid4().hex}.tmp")
    safety_copy: Path | None = None
    try:
        with gzip.open(backup, "rb") as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target)
            target.flush()
            os.fsync(target.fileno())
        _validate_sqlite(temporary)
        receipts = _read_erasure_receipts(database) if database.exists() else set()
        if deletion_records_from is not None:
            receipts.update(_read_erasure_receipts(deletion_records_from.expanduser().resolve(), required=True))
        if _apply_erasure_receipts(temporary, receipts):
            _validate_sqlite(temporary)

        if database.exists():
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            safety_copy = database.with_name(
                f"{database.stem}.safety-{timestamp}{database.suffix}"
            )
            shutil.copy2(database, safety_copy)

        os.replace(temporary, database)
        return safety_copy
    finally:
        temporary.unlink(missing_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup", required=True, type=Path)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument(
        "--deletion-records-from", type=Path,
        help="Additional SQLite database containing newer erasure receipts, for recovery to a lost database.",
    )
    parser.add_argument(
        "--confirm-backend-stopped",
        action="store_true",
        help="Required acknowledgement that FastAPI and all SQLite writers are stopped.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.confirm_backend_stopped:
        print(
            "Refusing restore: stop the backend and pass --confirm-backend-stopped.",
            file=sys.stderr,
        )
        return 2
    try:
        safety_copy = restore_sqlite_backup(args.backup, args.database, deletion_records_from=args.deletion_records_from)
    except (OSError, EOFError, sqlite3.Error, RuntimeError, ValueError) as exc:
        print(f"Restore failed: {exc}", file=sys.stderr)
        return 1

    print(f"Restore completed: {args.database.expanduser().resolve()}")
    if safety_copy is not None:
        print(f"Previous database safety copy: {safety_copy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
