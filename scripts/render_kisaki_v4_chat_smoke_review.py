#!/usr/bin/env python3
"""Render one trained variant's natural, continuity, and contextual smoke results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def render(results_path: Path, output_path: Path) -> None:
    data = json.loads(results_path.read_text(encoding="utf-8"))
    lines = [f"# 月社妃 V4 聊天审核：{data['variant']}", ""]
    lines += ["## A. 自然聊天（20 条）", ""]
    for row in data["natural_chat"]:
        lines += [
            f"### `{row['id']}`",
            "",
            f"用户：{row['messages'][-1]['content']}",
            "",
            f"妃：{row['response'] or '（空响应）'}",
            "",
            "- [ ] 通过",
            "- [ ] 不通过：说明问题",
            "",
        ]
    lines += ["## B. 六轮连续对话", ""]
    for row in data["continuity"]["turns"]:
        lines += [f"**第 {row['turn']} 轮用户**：{row['user']}", "", f"**妃**：{row['response']}", ""]
    lines += ["- [ ] 整体通过", "- [ ] 不通过：说明遗忘、错指代或语气漂移", ""]
    lines += ["## C. 带上下文原作场景（8 条）", ""]
    for row in data["contextual_story"]:
        lines += [
            f"### `{row['source_sample_id']}`",
            "",
            f"来源：`{row['source_path']}:{row['source_line_start']}`",
            "",
            "**模型生成**",
            "",
            row["response"] or "（空响应）",
            "",
            "**原作参考**",
            "",
            row["reference_answer"],
            "",
            "- [ ] 通过",
            "- [ ] 不通过：说明问题",
            "",
        ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.results, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
