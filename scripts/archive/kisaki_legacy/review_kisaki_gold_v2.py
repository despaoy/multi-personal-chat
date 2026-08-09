"""Export and import human review decisions for Kisaki Gold v2 candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = PROJECT_ROOT / "backend" / "evaluation" / "kisaki_gold_set_v2_candidates.json"
ALLOWED_DECISIONS = {"pending", "approved", "needs_revision", "rejected"}
REVIEW_FIELDS = (
    "id",
    "category",
    "prompt",
    "turns",
    "expected_behavior",
    "expected_refs",
    "safety_policy",
    "content_sha256",
    "human_decision",
    "review_notes",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _content_hash(item: dict[str, Any]) -> str:
    reviewable = {
        key: item.get(key)
        for key in ("id", "category", "prompt", "turns", "expected_behavior", "expected_refs", "safety_policy")
    }
    payload = json.dumps(reviewable, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def review_summary(data: dict[str, Any]) -> dict[str, Any]:
    prompts = data.get("prompts", [])
    return {
        "total": len(prompts),
        "decisions": dict(Counter(item.get("review_status", "pending") for item in prompts)),
        "categories": dict(Counter(item.get("category", "missing") for item in prompts)),
    }


def export_review(data: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for item in data.get("prompts", []):
            writer.writerow(
                {
                    "id": item["id"],
                    "category": item.get("category", ""),
                    "prompt": item.get("prompt", ""),
                    "turns": json.dumps(item.get("turns", []), ensure_ascii=False),
                    "expected_behavior": item.get("expected_behavior", ""),
                    "expected_refs": json.dumps(item.get("expected_refs", []), ensure_ascii=False),
                    "safety_policy": json.dumps(item.get("safety_policy", {}), ensure_ascii=False),
                    "content_sha256": _content_hash(item),
                    "human_decision": item.get("review_status", "pending"),
                    "review_notes": item.get("review_notes", ""),
                }
            )


def import_review(
    data: dict[str, Any],
    review_file: Path,
    *,
    reviewer: str,
) -> dict[str, Any]:
    with review_file.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {item["id"]: item for item in data.get("prompts", [])}
    row_ids = [row.get("id", "") for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("Review file contains duplicate IDs")
    if set(row_ids) != set(by_id):
        missing = sorted(set(by_id) - set(row_ids))
        unknown = sorted(set(row_ids) - set(by_id))
        raise ValueError(f"Review IDs do not match candidates; missing={missing[:5]} unknown={unknown[:5]}")

    now = datetime.now(timezone.utc).isoformat()
    for row in rows:
        item = by_id[row["id"]]
        if row.get("content_sha256") != _content_hash(item):
            raise ValueError(f"Reviewable content changed for {item['id']}; export the sheet again")
        decision = row.get("human_decision", "").strip().lower()
        if decision not in ALLOWED_DECISIONS:
            raise ValueError(f"Invalid human_decision={decision!r} for {item['id']}")
        item["review_status"] = decision
        item["review_notes"] = row.get("review_notes", "").strip()
        if decision == "pending":
            item.pop("reviewed_by", None)
            item.pop("reviewed_at", None)
        else:
            item["reviewed_by"] = reviewer
            item["reviewed_at"] = now
    data["review_summary"] = review_summary(data)
    return data


def _atomic_write(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Human review workflow for Kisaki Gold v2")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("summary")
    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--output", type=Path, required=True)
    import_parser = subparsers.add_parser("import")
    import_parser.add_argument("--review-file", type=Path, required=True)
    import_parser.add_argument("--reviewer", required=True)
    args = parser.parse_args()

    data = _load(args.candidates)
    if args.command == "summary":
        print(json.dumps(review_summary(data), ensure_ascii=False, indent=2))
        return 0
    if args.command == "export":
        export_review(data, args.output)
        print(json.dumps({"exported": str(args.output), **review_summary(data)}, ensure_ascii=False))
        return 0

    import_review(data, args.review_file, reviewer=args.reviewer.strip())
    _atomic_write(args.candidates, data)
    print(json.dumps({"updated": str(args.candidates), **review_summary(data)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
