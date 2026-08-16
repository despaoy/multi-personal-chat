#!/usr/bin/env python3
"""Render generated overfit responses into the only human-review artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "backend/data/character_dialogues/experiments/v4/overfit_20/results.json"
DEFAULT_OUTPUT = ROOT / "docs/research/review_packets/kisaki_v4/09_OVERFIT_TEST/review.md"


def render(results_path: Path, output_path: Path) -> None:
    rows = json.loads(results_path.read_text(encoding="utf-8"))["results"]
    if len(rows) != 20:
        raise ValueError(f"expected 20 generated results, found {len(rows)}")
    lines = [
        "# 月社妃 V4：20 条过拟合结果人工审核",
        "",
        "每条只判断：生成是否完整自然、是否保持月社妃特征、人物关系是否正确、是否虚构关键事实。",
        "",
    ]
    for index, row in enumerate(rows, 1):
        prompt = "\n\n".join(message["content"] for message in row["messages"] if message["role"] == "user")
        lines.extend(
            [
                f"## {index:02d}. `{row['id']}`",
                "",
                f"- 对话者：`{row.get('interlocutor', '用户')}`",
                "",
                "**用户输入**",
                "",
                prompt,
                "",
                "**模型生成**",
                "",
                row["response"] or "（空响应）",
                "",
                "**训练参考（仅帮助判断链路是否成功，不要求逐字一致）**",
                "",
                row["reference_answer"],
                "",
                "- [ ] 通过",
                "- [ ] 不通过：说明问题",
                "",
            ]
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    render(args.results, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
