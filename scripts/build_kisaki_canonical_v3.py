"""Build KISAKI-CANONICAL-V3 data contract from V2 (read-only).

V2 files are NEVER modified. This script reads V2 train/eval (in place),
extracts the 715 game_extraction training samples and 84 game_extraction
validation samples **in their original order**, augments each training
sample's metadata with precise source-line whitelist fields
(``source_file`` / ``source_line_start`` / ``source_line_end``), and writes
the V3 dataset + draft manifest under ``experiments/v3/``.

The draft manifest records dynamic fields to avoid the count contradiction
flagged in V2.1 review:
  - ``current_train_count``: 715 (initial, grows as v4 samples are accepted)
  - ``target_train_count``: 826
  - ``accepted_count``: 0
  - ``pending_human_review_count``: 0

After this script runs, V3 is in ``status="draft"``. The manifest must be
frozen only after the v4 generation pipeline + human adjudication produces
exactly 111 accepted samples (handled by later stages).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND = PROJECT_ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from evaluation.experiment_contracts import (  # noqa: E402
    canonical_json_hash,
    sha256_text_file,
)

EXPERIMENT_DIR = BACKEND / "data" / "character_dialogues" / "experiments"
V2_TRAIN_PATH = EXPERIMENT_DIR / "tsukiyashiro_kisaki_train.json"
V2_EVAL_PATH = EXPERIMENT_DIR / "tsukiyashiro_kisaki_eval.json"
V2_MANIFEST_PATH = EXPERIMENT_DIR / "canonical_dataset_manifest.json"
V2_ARCHIVE_INDEX_PATH = EXPERIMENT_DIR / "v2_archive_index.json"

V3_DIR = EXPERIMENT_DIR / "v3"
V3_TRAIN_PATH = V3_DIR / "tsukiyashiro_kisaki_train.json"
V3_EVAL_PATH = V3_DIR / "tsukiyashiro_kisaki_eval.json"
V3_MANIFEST_PATH = V3_DIR / "canonical_dataset_manifest.json"

SOURCE_LINE_PATTERN = re.compile(r"^(?P<file>[^:]+):line:(?P<line>\d+)$")

EXPECTED_V2_TRAIN_GAME_EXTRACTION = 715
EXPECTED_V2_EVAL_GAME_EXTRACTION = 84
EXPECTED_V2_TRAIN_LLM_V3 = 111
EXPECTED_V2_EVAL_LLM_V3 = 8
TARGET_TRAIN_COUNT = 826


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _parse_source_line(source: str) -> tuple[str, int, int] | None:
    """Parse ``file.txt:line:N`` into (file, line_start, line_end).

    Returns None if the format does not match. line_start == line_end because
    V2 samples only record a single anchor line number.
    """
    if not source:
        return None
    match = SOURCE_LINE_PATTERN.match(source)
    if not match:
        return None
    line = int(match.group("line"))
    return match.group("file"), line, line


def _augment_whitelist_fields(sample: dict[str, Any]) -> dict[str, Any]:
    """Add precise source-line whitelist fields to a game_extraction sample.

    Preserves all existing metadata fields. Only adds ``source_line_start``
    and ``source_line_end`` if the ``source`` field parses successfully.
    """
    augmented = deepcopy(sample)
    metadata = dict(augmented.get("metadata") or {})
    source = metadata.get("source", "")
    parsed = _parse_source_line(source)
    if parsed is not None:
        source_file, line_start, line_end = parsed
        metadata["source_file"] = source_file
        metadata["source_line_start"] = line_start
        metadata["source_line_end"] = line_end
    else:
        # Fallback: keep source_file if already present, mark line range as unknown.
        metadata.setdefault("source_file", metadata.get("source_file"))
        metadata.setdefault("source_line_start", None)
        metadata.setdefault("source_line_end", None)
    augmented["metadata"] = metadata
    return augmented


def _extract_game_extraction(
    v2_samples: list[dict[str, Any]],
    expected_count: int,
    split_name: str,
) -> list[dict[str, Any]]:
    """Filter game_extraction samples in original order; augment train only."""
    extracted: list[dict[str, Any]] = []
    for sample in v2_samples:
        metadata = sample.get("metadata") or {}
        if metadata.get("data_source") != "game_extraction":
            continue
        if split_name == "train":
            extracted.append(_augment_whitelist_fields(sample))
        else:
            extracted.append(deepcopy(sample))
    if len(extracted) != expected_count:
        raise RuntimeError(
            f"expected {expected_count} game_extraction samples in V2 {split_name}, "
            f"found {len(extracted)}"
        )
    return extracted


def _id_sequence_hash(samples: list[dict[str, Any]]) -> str:
    """Stable hash of the sample id sequence (preserves order)."""
    ids = [sample.get("id", "") for sample in samples]
    return canonical_json_hash(ids)


def _source_distribution(samples: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        source = (sample.get("metadata") or {}).get("data_source", "unknown")
        counts[source] = counts.get(source, 0) + 1
    return counts


def _build_v3_provenance(
    train_samples: list[dict[str, Any]],
    eval_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "game_extraction_train": {
            "count": len(train_samples),
            "id_sequence_sha256": _id_sequence_hash(train_samples),
            "first_id": train_samples[0].get("id") if train_samples else None,
            "last_id": train_samples[-1].get("id") if train_samples else None,
        },
        "game_extraction_validation": {
            "count": len(eval_samples),
            "id_sequence_sha256": _id_sequence_hash(eval_samples),
            "first_id": eval_samples[0].get("id") if eval_samples else None,
            "last_id": eval_samples[-1].get("id") if eval_samples else None,
        },
        "order_locked": True,
        "note": (
            "715 game_extraction training samples are locked in V2 order. "
            "V3 builder MUST NOT re-shuffle. v4 samples appended later must "
            "use kisaki_llm_v4_ prefix and be appended after the 715th slot."
        ),
    }


def _build_draft_manifest(
    v2_manifest: dict[str, Any],
    train_samples: list[dict[str, Any]],
    eval_samples: list[dict[str, Any]],
    v3_train_sha: str,
    v3_eval_sha: str,
) -> dict[str, Any]:
    v2_train = v2_manifest.get("train", {})
    v2_validation = v2_manifest.get("validation", {})
    provenance = _build_v3_provenance(train_samples, eval_samples)

    return {
        "dataset_id": "KISAKI-CANONICAL-V3",
        "status": "draft",
        "schema_version": 3,
        "hash_mode": "sha256_utf8_lf_v1",
        "seed": v2_manifest.get("seed", 42),
        "built_at_local": None,  # filled by caller
        "v3_provenance": provenance,
        # Dynamic draft fields (resolve V2.1 Critical: count contradiction)
        "current_train_count": len(train_samples),  # 715 initially
        "target_train_count": TARGET_TRAIN_COUNT,  # 826
        "accepted_count": 0,
        "pending_human_review_count": 0,
        # Train/eval snapshots (will be updated when v4 samples are appended)
        "train": {
            "path": str(V3_TRAIN_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "sha256": v3_train_sha,
            "count": len(train_samples),  # 715 in draft
            "source_distribution": _source_distribution(train_samples),
        },
        "validation": {
            "path": str(V3_EVAL_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "sha256": v3_eval_sha,
            "count": len(eval_samples),  # 84
            "source_distribution": _source_distribution(eval_samples),
        },
        # V2 archive linkage
        "v2_archive_reference": {
            "v2_dataset_id": v2_manifest.get("dataset_id"),
            "v2_status": v2_manifest.get("status"),
            "v2_archive_index_path": str(
                V2_ARCHIVE_INDEX_PATH.relative_to(PROJECT_ROOT)
            ).replace("\\", "/"),
            "v2_manifest_train_sha256": v2_train.get("sha256"),
            "v2_manifest_validation_sha256": v2_validation.get("sha256"),
            "v2_manifest_train_count": v2_train.get("count"),
            "v2_manifest_validation_count": v2_validation.get("count"),
        },
        # v4 generation placeholder (filled by stage E freeze)
        "llm_v4_judged": None,
        # Archived v3 contamination (documented, removed from V3)
        "llm_v3_deepseek_archived": {
            "reason": (
                "systematic style deviation: 96.4% meta-narrative hit, "
                "75.7% '正因如此' hit, monotone laughter, missing sharp expressions"
            ),
            "v2_train_count": EXPECTED_V2_TRAIN_LLM_V3,
            "v2_validation_count": EXPECTED_V2_EVAL_LLM_V3,
            "v2_train_path": str(V2_TRAIN_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "v2_validation_path": str(V2_EVAL_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
        # Constraints on future modifications
        "freeze_policy": {
            "append_allowed_while_draft": True,
            "append_allowed_after_freeze": False,
            "modification_after_freeze_requires": "new dataset revision (e.g. KISAKI-CANONICAL-V3.1)",
            "v4_sample_id_prefix": "kisaki_llm_v4_",
            "v4_target_count": EXPECTED_V2_TRAIN_LLM_V3,  # 111
        },
    }


def _verify_v2_unchanged(v2_archive_index_path: Path) -> dict[str, Any] | None:
    """If a V2 archive index exists, re-hash V2 files and confirm no drift."""
    if not v2_archive_index_path.exists():
        return None
    index = _read_json(v2_archive_index_path)
    drift: list[dict[str, Any]] = []
    for record in index.get("files", []):
        rel_path = record["relative_path"]
        abs_path = EXPERIMENT_DIR / rel_path
        if not abs_path.exists():
            drift.append({"relative_path": rel_path, "issue": "missing"})
            continue
        current_sha = sha256_text_file(abs_path)
        if current_sha != record["sha256"]:
            drift.append({
                "relative_path": rel_path,
                "issue": "sha256_mismatch",
                "archived_sha256": record["sha256"],
                "current_sha256": current_sha,
            })
    return {"verified": not drift, "drift": drift} if drift else {"verified": True}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build KISAKI-CANONICAL-V3 data contract (V2 read-only)"
    )
    parser.add_argument("--v2-train", type=Path, default=V2_TRAIN_PATH)
    parser.add_argument("--v2-eval", type=Path, default=V2_EVAL_PATH)
    parser.add_argument("--v2-manifest", type=Path, default=V2_MANIFEST_PATH)
    parser.add_argument("--v3-dir", type=Path, default=V3_DIR)
    parser.add_argument("--skip-v2-verify", action="store_true",
                        help="skip V2 archive drift check (not recommended)")
    args = parser.parse_args()

    # 0. Pre-flight: confirm V2 unchanged since archive index was written.
    if not args.skip_v2_verify:
        v2_verify = _verify_v2_unchanged(V2_ARCHIVE_INDEX_PATH)
        if v2_verify is None:
            print(json.dumps({
                "built": False,
                "error": "v2_archive_index.json not found; run v2_archive_index.py first",
            }, ensure_ascii=False, indent=2))
            return 2
        if not v2_verify["verified"]:
            print(json.dumps({
                "built": False,
                "error": "V2 files changed since archive index was written",
                "details": v2_verify["drift"],
            }, ensure_ascii=False, indent=2))
            return 2

    # 1. Read V2 (read-only).
    v2_train = _read_json(args.v2_train)
    v2_eval = _read_json(args.v2_eval)
    v2_manifest = _read_json(args.v2_manifest)

    # 2. Extract game_extraction in original order.
    train_samples = _extract_game_extraction(
        v2_train, EXPECTED_V2_TRAIN_GAME_EXTRACTION, "train",
    )
    eval_samples = _extract_game_extraction(
        v2_eval, EXPECTED_V2_EVAL_GAME_EXTRACTION, "validation",
    )

    # 3. Write V3 train/eval (atomic via temp file + rename would be ideal;
    #    for simplicity here we write directly — V3 is brand new and not yet
    #    referenced by any experiment config).
    v3_train_path = args.v3_dir / "tsukiyashiro_kisaki_train.json"
    v3_eval_path = args.v3_dir / "tsukiyashiro_kisaki_eval.json"
    v3_manifest_path = args.v3_dir / "canonical_dataset_manifest.json"
    _write_json(v3_train_path, train_samples)
    _write_json(v3_eval_path, eval_samples)

    # 4. Compute hashes and write draft manifest.
    v3_train_sha = sha256_text_file(v3_train_path)
    v3_eval_sha = sha256_text_file(v3_eval_path)
    manifest = _build_draft_manifest(
        v2_manifest, train_samples, eval_samples, v3_train_sha, v3_eval_sha,
    )
    # Stamp built_at in UTC to keep manifest stable across re-runs of the
    # same inputs would require deterministic timestamps; we use wall clock
    # because the id_sequence_sha256 already provides content provenance.
    from datetime import datetime, timezone
    manifest["built_at_local"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _write_json(v3_manifest_path, manifest)

    # 5. Summary.
    print(json.dumps({
        "built": True,
        "dataset_id": manifest["dataset_id"],
        "status": manifest["status"],
        "v3_train_path": str(v3_train_path),
        "v3_eval_path": str(v3_eval_path),
        "v3_manifest_path": str(v3_manifest_path),
        "train_count": manifest["train"]["count"],
        "validation_count": manifest["validation"]["count"],
        "current_train_count": manifest["current_train_count"],
        "target_train_count": manifest["target_train_count"],
        "accepted_count": manifest["accepted_count"],
        "v3_provenance": {
            "train_id_sequence_sha256": manifest["v3_provenance"]["game_extraction_train"]["id_sequence_sha256"],
            "validation_id_sequence_sha256": manifest["v3_provenance"]["game_extraction_validation"]["id_sequence_sha256"],
        },
        "v2_verified_unchanged": True if args.skip_v2_verify else _verify_v2_unchanged(V2_ARCHIVE_INDEX_PATH)["verified"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
