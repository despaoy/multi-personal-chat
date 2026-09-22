"""Read-only audit of tracked and non-ignored source files (not a security proof)."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def forbidden_artifact(path: str) -> bool:
    parts = Path(path).parts
    name = Path(path).name
    return (
        any(
            p
            in {
                "node_modules",
                "__pycache__",
                ".next",
                ".pnpm-store",
                ".ruff_cache",
                "local-notes",
            }
            for p in parts
        )
        or any(p.startswith((".codex-test-runtime-", ".test_tmp")) for p in parts)
        or name == ".env"
        or (name.startswith(".env.") and name != ".env.example")
        or name.endswith(
            (".db", ".db-wal", ".db-shm", ".db-journal", ".sqlite", ".sqlite3", ".pyc")
        )
    )


def validate_payload(path: str, payload: bytes) -> None:
    suffix = Path(path).suffix.lower()
    if suffix == ".py":
        ast.parse(payload, filename=path)
    elif suffix == ".json":
        json.loads(payload)
    elif suffix == ".jsonl":
        for line in payload.decode("utf-8-sig").splitlines():
            if line.strip():
                json.loads(line)


def main() -> int:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    paths = sorted(set(result.stdout.decode("utf-8").split("\0")) - {""})
    counts: Counter[str] = Counter()
    hashes: dict[str, list[str]] = defaultdict(list)
    errors = []
    total_bytes = 0
    for relative in paths:
        path = ROOT / relative
        if not path.is_file():
            continue  # Working-tree deletions are handled by Git review.
        if forbidden_artifact(relative):
            errors.append({"path": relative, "issue": "runtime/private artifact"})
            continue
        payload = path.read_bytes()
        counts[path.suffix.lower() or "<no suffix>"] += 1
        total_bytes += len(payload)
        if payload.strip():
            hashes[hashlib.sha256(payload).hexdigest()].append(relative)
        try:
            validate_payload(relative, payload)
        except (ValueError, SyntaxError, UnicodeError) as exc:
            # Do not echo data values: invalid files may contain private content.
            errors.append({"path": relative, "issue": type(exc).__name__})
    duplicates = [group for group in hashes.values() if len(group) > 1]
    print(
        json.dumps(
            {
                "files_checked": sum(counts.values()),
                "bytes": total_bytes,
                "extensions": dict(sorted(counts.items())),
                "errors": errors,
                "identical_file_groups": duplicates,
                "note": "Identical files may be intentional fixtures or frozen evidence; never auto-delete. This does not audit licenses, secrets or runtime behavior.",
            },
            ensure_ascii=True,
            indent=2,
        )
    )
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
