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
from datetime import datetime, timezone
from pathlib import Path


def _validate_sqlite(path: Path) -> None:
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        row = connection.execute("PRAGMA integrity_check").fetchone()
    finally:
        connection.close()
    if row is None or str(row[0]).lower() != "ok":
        raise RuntimeError(f"SQLite integrity_check failed: {row!r}")


def restore_sqlite_backup(backup_path: Path, database_path: Path) -> Path | None:
    """Validate, safety-copy, and atomically install one SQLite backup."""
    backup = backup_path.expanduser().resolve()
    database = database_path.expanduser().resolve()
    if not backup.is_file():
        raise FileNotFoundError(f"Backup does not exist: {backup}")

    database.parent.mkdir(parents=True, exist_ok=True)
    temporary = database.with_name(f".{database.name}.restore-{uuid.uuid4().hex}.tmp")
    safety_copy: Path | None = None
    try:
        with gzip.open(backup, "rb") as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target)
            target.flush()
            os.fsync(target.fileno())
        _validate_sqlite(temporary)

        if database.exists():
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
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
        safety_copy = restore_sqlite_backup(args.backup, args.database)
    except Exception as exc:
        print(f"Restore failed: {exc}", file=sys.stderr)
        return 1

    print(f"Restore completed: {args.database.expanduser().resolve()}")
    if safety_copy is not None:
        print(f"Previous database safety copy: {safety_copy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())