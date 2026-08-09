"""V2 archive registry generator.

Generates ``v2_archive_index.json`` alongside the existing V2 files. V2 files
themselves are NEVER moved or modified — only read for hashing. The registry
records file paths, sha256 hashes (LF-normalized text hash to match the
``sha256_utf8_lf_v1`` mode used by ``canonical_dataset_manifest.json``),
``archived_non_comparable=true`` flag, archive timestamp and V2 commit hash.

Run modes:
  default  scan V2 files, write ``v2_archive_index.json``
  --verify re-hash V2 files and compare with existing index, fail on drift
  --check-only  print what would be archived, do not write the index
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND = PROJECT_ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from evaluation.experiment_contracts import sha256_text_file  # noqa: E402

EXPERIMENT_DIR = BACKEND / "data" / "character_dialogues" / "experiments"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "v2_archive_index.json"

# Files that constitute the V2 frozen dataset and its E1-E5 configs.
# Paths are relative to EXPERIMENT_DIR.
V2_FILES: tuple[str, ...] = (
    "canonical_dataset_manifest.json",
    "tsukiyashiro_kisaki_train.json",
    "tsukiyashiro_kisaki_eval.json",
    "canonical_dataset_exclusions.json",
    "configs/kisaki_e1_canonical.json",
    "configs/kisaki_e2_canonical.json",
    "configs/kisaki_e3_canonical.json",
    "configs/kisaki_e4_canonical.json",
    "configs/kisaki_e5_canonical.json",
)

ARCHIVE_REASON = (
    "V2 validation set contains 8 llm_v3_deepseek samples with systematic "
    "style deviation (96.4% meta-narrative hit rate, 75.7% '正因如此' hit, "
    "monotone laughter, missing sharp expressions); superseded by "
    "KISAKI-CANONICAL-V3 for further experiments. V2 results remain valid "
    "for the E1/E2 experiments they were trained on, but cannot be directly "
    "compared with V3 results across different validation sets."
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_commit_hash(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None


def _file_record(rel_path: str) -> dict[str, Any] | None:
    abs_path = EXPERIMENT_DIR / rel_path
    if not abs_path.exists():
        return None
    stat = abs_path.stat()
    return {
        "relative_path": rel_path,
        "absolute_path": str(abs_path),
        "sha256": sha256_text_file(abs_path),
        "size_bytes": stat.st_size,
        "mtime_iso": datetime.fromtimestamp(stat.st_mtime, timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
    }


def _scan_v2_files() -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for rel_path in V2_FILES:
        record = _file_record(rel_path)
        if record is None:
            missing.append(rel_path)
        else:
            records.append(record)
    return records, missing


def _load_v2_manifest_summary() -> dict[str, Any]:
    manifest_path = EXPERIMENT_DIR / "canonical_dataset_manifest.json"
    if not manifest_path.exists():
        return {"found": False}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"found": True, "parse_error": True}
    return {
        "found": True,
        "dataset_id": manifest.get("dataset_id"),
        "status": manifest.get("status"),
        "schema_version": manifest.get("schema_version"),
        "hash_mode": manifest.get("hash_mode"),
        "train_count": manifest.get("train", {}).get("count"),
        "validation_count": manifest.get("validation", {}).get("count"),
        "train_sha256": manifest.get("train", {}).get("sha256"),
        "validation_sha256": manifest.get("validation", {}).get("sha256"),
    }


def _build_index() -> dict[str, Any]:
    file_records, missing = _scan_v2_files()
    if missing:
        raise RuntimeError(
            "V2 files missing (cannot archive without them): " + ", ".join(missing)
        )
    manifest_summary = _load_v2_manifest_summary()
    return {
        "archive_index_version": 1,
        "dataset_id": "KISAKI-CANONICAL-V2",
        "archived_non_comparable": True,
        "archive_reason": ARCHIVE_REASON,
        "archived_at": _utc_now_iso(),
        "v2_commit_hash": _git_commit_hash(PROJECT_ROOT),
        "v2_manifest_summary": manifest_summary,
        "files": file_records,
        "file_count": len(file_records),
    }


def _write_index(output_path: Path, index: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _verify(existing_path: Path) -> int:
    if not existing_path.exists():
        print(json.dumps({"verified": False, "error": "index file not found"}, ensure_ascii=False))
        return 2
    existing = json.loads(existing_path.read_text(encoding="utf-8"))
    current_records, missing = _scan_v2_files()
    if missing:
        print(json.dumps(
            {"verified": False, "missing_files": missing},
            ensure_ascii=False, indent=2,
        ))
        return 2
    current_by_path = {r["relative_path"]: r for r in current_records}
    existing_by_path = {r["relative_path"]: r for r in existing.get("files", [])}

    drift: list[dict[str, Any]] = []
    for rel_path in V2_FILES:
        cur = current_by_path.get(rel_path)
        old = existing_by_path.get(rel_path)
        if cur is None or old is None:
            drift.append({"relative_path": rel_path, "issue": "missing_in_index_or_disk"})
            continue
        if cur["sha256"] != old["sha256"]:
            drift.append({
                "relative_path": rel_path,
                "issue": "sha256_mismatch",
                "archived_sha256": old["sha256"],
                "current_sha256": cur["sha256"],
            })
    if drift:
        print(json.dumps(
            {"verified": False, "drift": drift},
            ensure_ascii=False, indent=2,
        ))
        return 1
    print(json.dumps(
        {"verified": True, "file_count": len(current_records)},
        ensure_ascii=False, indent=2,
    ))
    return 0


def _check_only() -> int:
    records, missing = _scan_v2_files()
    summary = _load_v2_manifest_summary()
    print(json.dumps({
        "would_archive_files": len(records),
        "missing_files": missing,
        "v2_manifest_summary": summary,
        "files": [
            {"relative_path": r["relative_path"], "sha256": r["sha256"], "size_bytes": r["size_bytes"]}
            for r in records
        ],
    }, ensure_ascii=False, indent=2))
    return 0 if not missing else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate V2 archive index without moving V2 files")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify", action="store_true", help="verify existing index against current V2 files")
    parser.add_argument("--check-only", action="store_true", help="print what would be archived, do not write")
    args = parser.parse_args()

    if args.verify:
        return _verify(args.output)
    if args.check_only:
        return _check_only()

    index = _build_index()
    _write_index(args.output, index)
    print(json.dumps({
        "written": True,
        "output": str(args.output),
        "archived_non_comparable": index["archived_non_comparable"],
        "file_count": index["file_count"],
        "v2_commit_hash": index["v2_commit_hash"],
        "archived_at": index["archived_at"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
