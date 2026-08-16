"""Reveal and score a completed blinded A/B benchmark."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

VALID_DECISIONS = {"A", "B", "tie", "invalid"}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="在盲评完成后揭盲并统计模型胜率")
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    review = load(args.review)
    key = load(args.key)
    samples = {item["id"]: item for item in review.get("samples", [])}
    mappings = {item["id"]: item for item in key.get("key", [])}
    if set(samples) != set(mappings):
        raise SystemExit("审核文件和密钥文件的样本 ID 不一致")

    pending = [item_id for item_id, item in samples.items() if item.get("winner") not in VALID_DECISIONS]
    if pending:
        raise SystemExit(f"仍有 {len(pending)} 条未审核；完成后才能揭盲")

    overall = Counter()
    by_category: dict[str, Counter] = defaultdict(Counter)
    rows = []
    for item_id, sample in samples.items():
        decision = sample["winner"]
        mapping = mappings[item_id]
        if decision in {"A", "B"}:
            model = mapping[f"{decision}_model"]
            overall[f"win:{model}"] += 1
            by_category[sample["category"]][f"win:{model}"] += 1
        overall[decision] += 1
        by_category[sample["category"]][decision] += 1
        rows.append(
            {
                "id": item_id,
                "category": sample["category"],
                "decision": decision,
                "winner_model": mapping.get(f"{decision}_model") if decision in {"A", "B"} else None,
                "reason": sample.get("reason", ""),
            }
        )

    decisive = overall["A"] + overall["B"]
    model_wins = {key[4:]: value for key, value in overall.items() if key.startswith("win:")}
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total": len(samples),
        "decisive": decisive,
        "tie": overall["tie"],
        "invalid": overall["invalid"],
        "model_wins": model_wins,
        "model_win_rate_on_decisive": {
            model: round(value / decisive, 4) if decisive else 0.0
            for model, value in model_wins.items()
        },
        "by_category": {category: dict(value) for category, value in sorted(by_category.items())},
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, ensure_ascii=False, indent=2))
    print(f"统计结果: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
