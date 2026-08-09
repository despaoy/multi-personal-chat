"""Freeze a manually approved Gold v2 candidate set after all leakage audits."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND = PROJECT_ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from evaluation.experiment_contracts import canonical_json_hash  # noqa: E402

DEFAULT_INPUT = BACKEND / "evaluation" / "kisaki_gold_set_v2_candidates.json"
DEFAULT_OUTPUT = BACKEND / "evaluation" / "kisaki_gold_set_v2.json"
DEFAULT_AUDIT = (
    BACKEND
    / "data"
    / "character_dialogues"
    / "experiments"
    / "gold_v2_leakage_audit.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze manually approved Kisaki Gold v2")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    prompts = data.get("prompts", [])
    errors: list[str] = []
    if len(prompts) != 150:
        errors.append(f"expected 150 prompts, found {len(prompts)}")
    counts = Counter(item.get("category") for item in prompts)
    if set(counts.values()) != {30} or len(counts) != 5:
        errors.append(f"expected five categories with 30 prompts each, found {dict(counts)}")
    pending = [item.get("id") for item in prompts if item.get("review_status") != "approved"]
    if pending:
        errors.append(f"{len(pending)} prompts are not manually approved")
    if audit.get("status") != "passed":
        errors.append("text leakage audit has not passed")
    if audit.get("semantic_audit_status") != "passed":
        errors.append("semantic leakage audit has not passed")
    if audit.get("unresolved_semantic_matches"):
        errors.append("semantic leakage audit still has unresolved matches")

    if errors:
        print(json.dumps({"frozen": False, "errors": errors}, ensure_ascii=False, indent=2))
        return 2

    frozen = dict(data)
    frozen["status"] = "frozen"
    frozen["content_sha256"] = canonical_json_hash(prompts)
    frozen["semantic_audit"] = {
        "threshold": audit.get("semantic_similarity_threshold", 0.88),
        "status": "passed",
    }
    args.output.write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"frozen": True, "output": str(args.output), "content_sha256": frozen["content_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
