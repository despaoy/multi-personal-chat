"""Build a deterministic category-balanced blind A/B package."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-category", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.per_category <= 0:
        raise SystemExit("per-category must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_dir}")

    a, b = load(args.a), load(args.b)
    if a.get("mock") or b.get("mock"):
        raise SystemExit("formal blind packages cannot use mock reports")
    for field in ("dataset_sha256", "prompt_content_sha256", "generation_sha256"):
        if a.get("provenance", {}).get(field) != b.get("provenance", {}).get(field):
            raise SystemExit(f"paired provenance mismatch: {field}")
    a_rows = {row["id"]: row for row in a["samples"]}
    b_rows = {row["id"]: row for row in b["samples"]}
    common = sorted(set(a_rows) & set(b_rows))
    grouped: dict[str, list[str]] = defaultdict(list)
    for sample_id in common:
        grouped[str(a_rows[sample_id]["category"])].append(sample_id)

    rng = random.Random(args.seed)
    selected: list[str] = []
    selected_by_category: dict[str, list[str]] = {}
    for category in sorted(grouped):
        candidates = grouped[category]
        if len(candidates) < args.per_category:
            raise SystemExit(f"category {category} has only {len(candidates)} samples")
        chosen = sorted(rng.sample(candidates, args.per_category))
        selected_by_category[category] = chosen
        selected.extend(chosen)

    review, key = [], []
    for sample_id in sorted(selected):
        left_first = rng.choice([True, False])
        left, right = (a_rows[sample_id], b_rows[sample_id]) if left_first else (b_rows[sample_id], a_rows[sample_id])
        review.append({
            "id": sample_id,
            "category": left["category"],
            "prompt": left["prompt"],
            "response_A": left["response"],
            "response_B": right["response"],
            "winner": "",
            "reason": "",
        })
        key.append({
            "id": sample_id,
            "A_model": a["model"] if left_first else b["model"],
            "B_model": b["model"] if left_first else a["model"],
        })

    args.output_dir.mkdir(parents=True)
    common_meta = {
        "schema_version": 2,
        "seed": args.seed,
        "per_category": args.per_category,
        "selected_ids": selected_by_category,
        "source_hashes": {"a": sha256(args.a), "b": sha256(args.b)},
    }
    review_payload = {
        **common_meta,
        "status": "pending_independent_human_review",
        "instructions": "Review prompt/A/B only. Do not open blind_key.json or AI opinions before locking decisions.",
        "samples": review,
    }
    key_payload = {**common_meta, "key": key}
    (args.output_dir / "blind_review.json").write_text(json.dumps(review_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "blind_key.json").write_text(json.dumps(key_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"paired": len(review), "categories": {k: len(v) for k, v in selected_by_category.items()}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
