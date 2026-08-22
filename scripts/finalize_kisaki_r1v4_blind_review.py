"""Lock a completed R1V4 Markdown review before producing unblinded results."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DIMENSIONS = (
    "character_consistency",
    "factual_correctness",
    "context_memory",
    "situational_decision",
    "safety",
)
WINNERS = {"A", "B", "tie", "invalid"}
SEVERE_VALUES = {"A", "B", "both", "none"}
SECTION_PATTERN = re.compile(
    r"(?ms)^## (?P<number>\d+)\. (?P<id>\S+) \[(?P<category>[^]]+)]\n"
    r"(?P<body>.*?)(?=^## \d+\.|\Z)"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _field(body: str, pattern: str, label: str) -> str:
    match = re.search(pattern, body, re.MULTILINE)
    if not match or not match.group(1).strip():
        raise ValueError(f"missing {label}")
    return match.group(1).strip()


def _scores(value: str, label: str) -> dict[str, int | None]:
    parts = value.split("/")
    if len(parts) != len(DIMENSIONS):
        raise ValueError(f"{label}: expected five scores")
    result: dict[str, int | None] = {}
    for dimension, part in zip(DIMENSIONS, parts, strict=True):
        if part == "NA":
            result[dimension] = None
        elif part in {"0", "1", "2"}:
            result[dimension] = int(part)
        else:
            raise ValueError(f"{label}: invalid score {part!r}")
    return result


def parse_markdown(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    decisions = []
    for match in SECTION_PATTERN.finditer(text):
        body = match.group("body")
        sample_id = match.group("id")
        winner = _field(body, r"^- 总体选择：(\S+)\s*$", f"{sample_id} winner")
        severe = _field(
            body,
            r"^- 是否存在严重错误（A/B/both/none）：(\S+)\s*$",
            f"{sample_id} severe_error",
        )
        if winner not in WINNERS:
            raise ValueError(f"{sample_id}: invalid winner {winner!r}")
        if severe not in SEVERE_VALUES:
            raise ValueError(f"{sample_id}: invalid severe_error {severe!r}")
        decisions.append(
            {
                "number": int(match.group("number")),
                "id": sample_id,
                "category": match.group("category"),
                "winner": winner,
                "candidate_A_scores": _scores(
                    _field(
                        body,
                        r"^- A 五项评分（人物/事实/多轮/决策/安全）：(\S+)\s*$",
                        f"{sample_id} A scores",
                    ),
                    f"{sample_id} A scores",
                ),
                "candidate_B_scores": _scores(
                    _field(
                        body,
                        r"^- B 五项评分（人物/事实/多轮/决策/安全）：(\S+)\s*$",
                        f"{sample_id} B scores",
                    ),
                    f"{sample_id} B scores",
                ),
                "severe_error": severe,
                "reason": _field(
                    body, r"^- 评价与依据：(.*)$", f"{sample_id} reason"
                ),
            }
        )
    if not decisions:
        raise ValueError("no completed review sections found")
    numbers = [row["number"] for row in decisions]
    if numbers != list(range(1, len(decisions) + 1)):
        raise ValueError("review section numbers must be consecutive")
    ids = [row["id"] for row in decisions]
    if len(ids) != len(set(ids)):
        raise ValueError("review contains duplicate sample IDs")
    return decisions


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _mean_scores(values: dict[str, list[int]]) -> dict[str, float | None]:
    return {
        dimension: round(statistics.mean(values[dimension]), 4)
        if values[dimension]
        else None
        for dimension in DIMENSIONS
    }


def finalize_review(
    markdown_path: Path,
    review_json_path: Path,
    key_path: Path,
    output_dir: Path,
    *,
    confirmed_by: str,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty output: {output_dir}")
    decisions = parse_markdown(markdown_path)
    review = _load(review_json_path)
    review_ids = [row["id"] for row in review.get("samples", [])]
    decision_ids = [row["id"] for row in decisions]
    if decision_ids != review_ids:
        raise ValueError("Markdown decisions do not match the original blind-review order")

    output_dir.mkdir(parents=True, exist_ok=False)
    locked_at = datetime.now(timezone.utc).isoformat()
    locked = {
        "schema_version": 1,
        "package_id": review.get("package_id"),
        "status": "human_confirmed_ai_assisted_review_locked",
        "locked_at": locked_at,
        "confirmed_by": confirmed_by,
        "method": {
            "blinded": True,
            "ai_assisted": True,
            "independent_human_only": False,
            "note": "The user confirmed the AI-assisted decisions before the key was opened.",
        },
        "source_markdown_sha256": _sha256(markdown_path),
        "source_review_sha256": _sha256(review_json_path),
        "decisions": decisions,
    }
    locked_path = output_dir / "decisions_locked.json"
    locked_path.write_text(
        json.dumps(locked, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    key = _load(key_path)
    key_rows = key.get("key", [])
    key_by_id = {row["id"]: row for row in key_rows}
    if set(key_by_id) != set(decision_ids) or len(key_rows) != len(decisions):
        raise ValueError("blind key does not match locked decisions")

    model_names = sorted({row[side] for row in key_rows for side in ("A", "B")})
    if len(model_names) != 2:
        raise ValueError("blind review must compare exactly two models")
    aggregates: dict[str, dict[str, Any]] = {
        model: {
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "invalid": 0,
            "severe_errors": 0,
            "scores": defaultdict(list),
            "categories": defaultdict(Counter),
        }
        for model in model_names
    }
    revealed_rows = []
    for decision in decisions:
        mapping = key_by_id[decision["id"]]
        side_models = {"A": mapping["A"], "B": mapping["B"]}
        for side, model in side_models.items():
            for dimension, score in decision[f"candidate_{side}_scores"].items():
                if score is not None:
                    aggregates[model]["scores"][dimension].append(score)
        winner = decision["winner"]
        if winner in {"A", "B"}:
            loser = "B" if winner == "A" else "A"
            aggregates[side_models[winner]]["wins"] += 1
            aggregates[side_models[loser]]["losses"] += 1
            aggregates[side_models[winner]]["categories"][decision["category"]]["wins"] += 1
            aggregates[side_models[loser]]["categories"][decision["category"]]["losses"] += 1
        else:
            field = "ties" if winner == "tie" else "invalid"
            for model in model_names:
                aggregates[model][field] += 1
                aggregates[model]["categories"][decision["category"]][field] += 1
        severe = decision["severe_error"]
        severe_models = (
            model_names
            if severe == "both"
            else [side_models[severe]] if severe in {"A", "B"} else []
        )
        for model in severe_models:
            aggregates[model]["severe_errors"] += 1
        revealed_rows.append(
            {
                **decision,
                "A_model": side_models["A"],
                "B_model": side_models["B"],
                "winner_model": side_models.get(winner),
                "severe_error_models": severe_models,
            }
        )

    model_summary = {}
    for model, values in aggregates.items():
        model_summary[model] = {
            "wins": values["wins"],
            "losses": values["losses"],
            "ties": values["ties"],
            "invalid": values["invalid"],
            "severe_errors": values["severe_errors"],
            "mean_scores": _mean_scores(values["scores"]),
            "by_category": {
                category: dict(counts)
                for category, counts in sorted(values["categories"].items())
            },
        }
    unblinded = {
        "schema_version": 1,
        "package_id": review.get("package_id"),
        "status": "unblinded_ai_assisted_review_complete",
        "locked_decisions_sha256": _sha256(locked_path),
        "blind_key_sha256": _sha256(key_path),
        "models": model_summary,
        "samples": revealed_rows,
    }
    (output_dir / "unblinded_summary.json").write_text(
        json.dumps(unblinded, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return unblinded


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--review-json", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confirmed-by", default="project_owner")
    args = parser.parse_args()
    try:
        result = finalize_review(
            args.markdown,
            args.review_json,
            args.key,
            args.output_dir,
            confirmed_by=args.confirmed_by,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result["models"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
