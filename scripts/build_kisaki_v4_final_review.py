#!/usr/bin/env python3
"""Render one clean final human-review package from current V4 assets."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
V4 = ROOT / "backend/data/character_dialogues/experiments/v4"
OUTPUT = ROOT / "docs/research/review_packets/kisaki_v4/10_FINAL_REVIEW"
BATCH_SIZE = 50
REVIEWED_AUGMENTATION_SOURCES = {
    "deepseek_user_simulation_v41_reviewed",
    "codex_user_simulation_v41_reviewed",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def render_record(
    row: dict[str, Any],
    *,
    context_check: bool,
    already_approved: bool = False,
) -> list[str]:
    metadata = row.get("metadata", {})
    lines = [f"## `{row['id']}`", ""]
    lines += [f"- 数据来源：`{metadata.get('data_source', 'unknown')}`"]
    if metadata.get("context_speaker_label"):
        lines += [f"- 当前对话者：`{metadata['context_speaker_label']}`"]
    if metadata.get("source"):
        lines += [f"- 原作位置：`{metadata['source']}`"]
    lines.append("")
    for message in row["messages"]:
        label = "用户" if message["role"] == "user" else "妃"
        lines += [f"**{label}**", "", message["content"], ""]
    if already_approved:
        review = metadata.get("human_review", {})
        lines += [
            f"- 审核状态：`{review.get('status', 'unknown')}`",
            f"- 审核记录：`{review.get('decision_source', 'unknown')}`",
            "- 本页仅供追溯，不重复审核。",
        ]
    elif context_check:
        lines += [
            "- [ ] 可作为独立聊天样本",
            "- [ ] 仅补充上下文后可保留",
            "- [ ] 排除：跨场景、残句、缺少动作或关键指代",
        ]
    else:
        lines += ["- [ ] 通过", "- [ ] 修改或排除：说明原因"]
    lines.append("")
    return lines


def write_batches(
    directory: Path,
    title: str,
    rows: list[dict[str, Any]],
    *,
    context_check: bool,
    already_approved: bool = False,
) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start : start + BATCH_SIZE]
        path = directory / f"batch_{start // BATCH_SIZE + 1:02d}.md"
        lines = [f"# {title}：第 {start // BATCH_SIZE + 1} 批", "", f"本批 {len(batch)} 条。", ""]
        for row in batch:
            lines.extend(
                render_record(
                    row,
                    context_check=context_check,
                    already_approved=already_approved,
                )
            )
        path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
        outputs.append(path)
    return outputs


def markdown_links(paths: list[Path], output: Path) -> list[str]:
    return [f"- [{path.stem}]({path.relative_to(output).as_posix()})" for path in paths]


def build(output: Path = OUTPUT) -> dict[str, Any]:
    if output.exists():
        shutil.rmtree(output)
    train = load_jsonl(V4 / "train.jsonl")
    validation = load_jsonl(V4 / "validation.jsonl")
    game = [row for row in train if row.get("metadata", {}).get("data_source") == "game_extraction"]
    augmentation = [
        row
        for row in train
        if row.get("metadata", {}).get("data_source") in REVIEWED_AUGMENTATION_SOURCES
    ]
    constructed = [
        row
        for row in train
        if row.get("metadata", {}).get("data_source")
        not in {"game_extraction", *REVIEWED_AUGMENTATION_SOURCES}
    ]
    if (len(game), len(constructed), len(validation)) != (576, 150, 70):
        raise ValueError("current V4 counts changed; rebuild inputs before rendering final review")
    if len(train) != len(game) + len(constructed) + len(augmentation):
        raise ValueError("current V4 contains an unclassified training source")

    game_files = write_batches(output / "02_GAME_CONTEXT", "Game Train 上下文独立性复核", game, context_check=True)
    constructed_files = write_batches(output / "03_CONSTRUCTED", "构造训练集最终复核", constructed, context_check=False)
    validation_files = write_batches(output / "04_VALIDATION", "Validation 最终复核", validation, context_check=True)
    augmentation_files = write_batches(
        output / "05_APPROVED_AUGMENTATION",
        "已批准多轮增补（仅供追溯）",
        augmentation,
        context_check=False,
        already_approved=True,
    )

    index = [
        "# 月社妃 V4 最终总审核入口",
        "",
        "只审核本目录列出的当前对象；不要再使用旧 `03_GAME_TRAIN` 或 `legacy_v3` 批次。",
        "",
        "## 1. 人物与系统提示词",
        "",
        "- [人物设定](../01_PROFILE_PROMPT/01_character_profile.md)",
        "- [System Prompt v3](../01_PROFILE_PROMPT/02_system_prompt_v3.md)",
        "",
        "## 2. Game Train：576 条",
        "",
        "重点判断脱离原作现场后是否仍能构成可靠问答。",
        "",
        *markdown_links(game_files, output),
        "",
        "## 3. 构造训练集：150 条",
        "",
        *markdown_links(constructed_files, output),
        "",
        "## 4. Validation：70 条",
        "",
        *markdown_links(validation_files, output),
        "",
        f"## 5. 已批准多轮增补：{len(augmentation)} 条完整五轮会话",
        "",
        "这 4 条已完成逐轮修订、人工批准和污染复审，仅供追溯，不重复审核。",
        "",
        *markdown_links(augmentation_files, output),
        "",
        "## 6. Gold",
        "",
        "- [Gold v2.1 第 1 批](../06_GOLD_V21/batch_01.md)",
        "- [Gold v2.1 第 2 批](../06_GOLD_V21/batch_02.md)",
        "- [Gold v2.1 第 3 批](../06_GOLD_V21/batch_03.md)",
        "- [Gold v3 第 1 批](../07_GOLD_V3/batch_01.md)",
        "- [Gold v3 第 2 批](../07_GOLD_V3/batch_02.md)",
        "- [Gold v3 第 3 批](../07_GOLD_V3/batch_03.md)",
        "",
        "## 7. 技术记录（无需重新判断角色效果）",
        "",
        "- [20 条过拟合技术结论](../09_OVERFIT_TEST/technical_review_decision.json)",
        "",
        "完成后请分别给出 Game Train、构造集、Validation、Gold v2.1、Gold v3 的通过、修改、排除数量；已批准多轮增补不重复计入待审核项。",
    ]
    (output / "00_INDEX.md").write_text("\n".join(index) + "\n", encoding="utf-8", newline="\n")
    manifest = {
        "schema_version": 1,
        "status": "pending_final_human_review",
        "game_train": len(game),
        "constructed_train": len(constructed),
        "approved_multiturn_augmentation": len(augmentation),
        "validation": len(validation),
        "game_batches": len(game_files),
        "constructed_batches": len(constructed_files),
        "approved_multiturn_augmentation_batches": len(augmentation_files),
        "validation_batches": len(validation_files),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
