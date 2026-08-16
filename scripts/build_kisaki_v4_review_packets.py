#!/usr/bin/env python3
"""Build human-readable, non-destructive Kisaki V4 review packets."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHARACTER_DIR = PROJECT_ROOT / "backend" / "data" / "character_dialogues"
EXPERIMENT_DIR = CHARACTER_DIR / "experiments"
EVALUATION_DIR = PROJECT_ROOT / "backend" / "evaluation"
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "research" / "review_packets" / "kisaki_v4"
GAMETEXT_DIR = PROJECT_ROOT / "gametext" / "纸上魔法使"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def safe_name(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("_")
    return value or "unknown"


def messages_of(record: dict[str, Any]) -> list[dict[str, str]]:
    if isinstance(record.get("messages"), list):
        return [
            {"role": str(item.get("role", "")), "content": str(item.get("content", ""))}
            for item in record["messages"]
        ]
    result = []
    for item in record.get("conversations", []):
        role = item.get("role") or item.get("from")
        role = {"human": "user", "gpt": "assistant"}.get(role, role)
        result.append({"role": str(role), "content": str(item.get("content", item.get("value", "")))})
    return result


def recommendation(record: dict[str, Any], category: str) -> tuple[str, list[str]]:
    issues: list[str] = []
    messages = messages_of(record)
    assistants = [item["content"] for item in messages if item["role"] == "assistant"]
    if not assistants and not category.startswith("gold_"):
        issues.append("缺少 assistant 回复")
    if any(len(text) > 100 for text in assistants):
        issues.append("assistant 回复超过 100 字，需确认场景是否确实需要长回答")
    if category == "game_train" and record.get("metadata", {}).get("quality_score", 100) < 60:
        issues.append("自动提取质量分低于 60，重点核对上下文是否完整")
    if category == "constructed_train":
        issues.append("非原作逐字样本，必须核对事实、语气和问题设计")
    if any(text.count("正因如此") > 1 for text in assistants):
        issues.append("同一回复多次出现“正因如此”")
    return ("待确认" if issues else "建议通过"), issues


def format_record(record: dict[str, Any], *, category: str, original: bool) -> str:
    metadata = record.get("metadata", {})
    suggested, issues = recommendation(record, category)
    lines = [
        f"## {record.get('id', 'missing-id')}",
        "",
        f"- 分类：`{category}`",
        f"- 数据来源：`{metadata.get('data_source', metadata.get('source', record.get('source', 'unknown')))}`",
        f"- 原作台词：`{'是' if original else '否'}`",
        f"- 场景：`{metadata.get('scene', metadata.get('source_file', '未标注'))}`",
        f"- 原作定位：`{metadata.get('source', record.get('source', '无'))}`",
        f"- AI 初步建议：`{suggested}`",
        f"- 自动问题：{'；'.join(issues) if issues else '未发现硬性问题'}",
    ]
    exclusion_reasons = metadata.get("exclusion_reasons", [])
    if exclusion_reasons:
        lines.append(f"- 排除原因：`{', '.join(exclusion_reasons)}`")
    lines.append("")
    if metadata.get("context"):
        lines.extend(("**原作上下文**", "", "```text", metadata["context"], "```", ""))
    for message in messages_of(record):
        lines.extend((f"**{message['role']}**", "", message["content"], ""))
    if metadata.get("expected_behavior"):
        lines.extend(("**评分标准 expected_behavior**", "", metadata["expected_behavior"], ""))
    lines.extend(
        (
            "- user_decision：`待填写（通过 / 修改 / 排除 / 需要上下文）`",
            "- user_notes：",
            "- revised_candidate：",
            "",
        )
    )
    return "\n".join(lines)


def write_batches(
    directory: Path,
    records: Iterable[dict[str, Any]],
    *,
    category: str,
    original: bool,
    batch_size: int,
    prefix: str = "batch",
) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    records = list(records)
    outputs = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        number = start // batch_size + 1
        path = directory / f"{prefix}_{number:02d}.md"
        header = (
            f"# {category} 审核批次 {number}\n\n"
            f"> 本批 {len(batch)} 条。请只填写 `user_decision`、`user_notes` 和必要的 "
            "`revised_candidate`，不要直接覆盖原始内容。\n\n"
        )
        body = "\n---\n\n".join(
            format_record(item, category=category, original=original) for item in batch
        )
        path.write_text(header + body + "\n", encoding="utf-8")
        outputs.append(path)
    return outputs


def write_gold_review_batches(
    output_dir: Path,
    records: list[dict[str, Any]],
    batch_size: int = 50,
) -> list[Path]:
    """Render Gold review packets from the current JSON, never old Markdown."""

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for offset in range(0, len(records), batch_size):
        batch = records[offset:offset + batch_size]
        lines = [
            f"# Gold v2.1 审核批次 {offset // batch_size + 1}",
            "",
            f"> 本批 {len(batch)} 条。当前均为 development 候选，不是最终盲测集。",
            "",
        ]
        for record in batch:
            lines.extend(
                [
                    f"## {record['id']}",
                    "",
                    f"- category：`{record['category']}`",
                    f"- cluster_id：`{record.get('cluster_id', '-')}`",
                    f"- interlocutor：`{record.get('interlocutor', '-')}`",
                    "",
                ]
            )
            if record.get("conversation"):
                for turn, message in enumerate(record["conversation"], 1):
                    lines.extend((f"**user {turn}**", "", message["content"], ""))
            else:
                lines.extend(("**prompt**", "", str(record.get("prompt", "")), ""))
            for label in (
                "required_facts",
                "required_behaviors",
                "optional_style_traits",
                "forbidden_claims",
                "evidence_refs",
                "required_answer_facts",
                "gold_answer",
                "distractor_refs",
                "turn_rubrics",
                "rubric",
            ):
                lines.extend(
                    (
                        f"**{label}**",
                        "",
                        "```json",
                        json.dumps(record.get(label), ensure_ascii=False, indent=2),
                        "```",
                        "",
                    )
                )
            lines.extend(
                (
                    "- user_decision：`待填写（通过 / 修改 / 排除 / 需要证据）`",
                    "- user_notes：",
                    "",
                    "---",
                    "",
                )
            )
        target = output_dir / f"batch_{offset // batch_size + 1:02d}.md"
        target.write_text("\n".join(lines), encoding="utf-8", newline="\n")
        outputs.append(target)
    return outputs


def write_source_index(output: Path, raw: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(row["source"].split(":line:", 1)[0] for row in raw)
    source_files = sorted(GAMETEXT_DIR.glob("*.txt"), key=lambda path: path.name)
    lines = [
        "# 原作文件覆盖索引",
        "",
        "> 17 个原作文本文件全部列出；其中没有月社妃直接台词的文件也保留为零计数，避免把‘未出现’误判为‘漏提取’。",
        "",
        "| 原作文件 | 月社妃直接台词 | 状态 |",
        "|---|---:|---|",
    ]
    for path in source_files:
        count = counts.get(path.name, 0)
        lines.append(f"| {path.name} | {count} | {'有审核批次' if count else '该文件无直接台词'} |")
    (output / "02_SOURCE_COVERAGE" / "00_SOURCE_FILE_INDEX.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return {
        "original_files_total": len(source_files),
        "files_with_kisaki_lines": sum(count > 0 for count in counts.values()),
    }


def build(
    output: Path,
    batch_size: int,
    candidate_dir: Path | None = None,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"review output is not empty: {output}; use a new version directory "
            "to preserve human decisions"
        )
    output.mkdir(parents=True, exist_ok=True)

    profile = PROJECT_ROOT / "docs" / "research" / "KISAKI_CHARACTER_PROFILE.md"
    prompt_v3 = CHARACTER_DIR / "kisaki_system_prompt_v3.txt"
    profile_prompt_dir = output / "01_PROFILE_PROMPT"
    profile_prompt_dir.mkdir()
    (profile_prompt_dir / "01_character_profile.md").write_text(
        "# 人物画像审核\n\n> 状态：待项目负责人确认。\n\n" + profile.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (profile_prompt_dir / "02_system_prompt_v3.md").write_text(
        "# System Prompt v3 审核\n\n> 状态：待项目负责人确认。此文件只包含人物身份、关系、性格与表达；安全和 RAG 规则由后端独立注入。\n\n```text\n"
        + prompt_v3.read_text(encoding="utf-8")
        + "\n```\n",
        encoding="utf-8",
    )

    raw = load_jsonl(CHARACTER_DIR / "tsukiyashiro_kisaki_raw.jsonl")
    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in raw:
        source_file = item["source"].split(":line:", 1)[0]
        by_file[source_file].append(item)
    source_dir = output / "02_SOURCE_COVERAGE"
    source_outputs: list[Path] = []
    for index, (source_file, rows) in enumerate(sorted(by_file.items()), start=1):
        directory = source_dir / f"{index:02d}_{safe_name(Path(source_file).stem)}"
        source_path = GAMETEXT_DIR / source_file
        source_lines = source_path.read_text(encoding="utf-8").splitlines()
        converted = []
        for row in rows:
            line_number = int(row["source"].rsplit(":line:", 1)[1])
            context_start = max(0, line_number - 4)
            context_end = min(len(source_lines), line_number + 3)
            context = "\n".join(
                f"{line_index + 1}: {source_lines[line_index]}"
                for line_index in range(context_start, context_end)
            )
            converted.append(
                {
                    "id": row["id"],
                    "source": row["source"],
                    "metadata": {
                        "source": row["source"],
                        "source_file": source_file,
                        "context": context,
                    },
                    "messages": [{"role": "assistant", "content": row["text"]}],
                }
            )
        source_outputs.extend(
            write_batches(
                directory,
                converted,
                category="source_coverage",
                original=True,
                batch_size=batch_size,
                prefix="source",
            )
        )
    source_coverage = write_source_index(output, raw)

    candidate_dir = candidate_dir or EXPERIMENT_DIR / "v4"
    canonical_train = load_jsonl(candidate_dir / "train_candidate.jsonl")
    game_train = [
        record
        for record in canonical_train
        if record.get("metadata", {}).get("data_source") == "game_extraction"
    ]
    game_outputs = write_batches(
        output / "03_GAME_TRAIN",
        game_train,
        category="game_train",
        original=True,
        batch_size=batch_size,
    )
    constructed = [
        record
        for record in canonical_train
        if record.get("metadata", {}).get("data_source") != "game_extraction"
    ]
    constructed_outputs = write_batches(
        output / "04_CONSTRUCTED_TRAIN",
        constructed,
        category="constructed_train",
        original=False,
        batch_size=batch_size,
    )

    validation_dir = output / "05_VALIDATION"
    draft_validation = load_jsonl(candidate_dir / "validation_candidate.jsonl")
    validation_outputs = write_batches(
        validation_dir / "v4_independent",
        draft_validation,
        category="v4_independent_validation",
        original=False,
        batch_size=batch_size,
    )

    gold_v21 = load_json(EVALUATION_DIR / "kisaki_gold_set_v21_candidates.json")
    gold_target = output / "06_GOLD_V21"
    gold_outputs = write_gold_review_batches(gold_target, gold_v21["prompts"])
    gold_v3_dir = output / "07_GOLD_V3"
    gold_v3 = load_json(EVALUATION_DIR / "kisaki_gold_set_v3_candidates.json")
    gold_v3_outputs = write_gold_review_batches(gold_v3_dir, gold_v3["prompts"])

    exclusions = load_jsonl(CHARACTER_DIR / "tsukiyashiro_kisaki_excluded.jsonl")
    exclusion_outputs = write_batches(
        output / "08_EXCLUSIONS",
        exclusions,
        category="excluded_game_candidate",
        original=True,
        batch_size=batch_size,
    )

    config_dir = output / "09_EXPERIMENT_CONFIGS"
    config_dir.mkdir()
    config_rows = [
        {"variant": "E1", "neftune_noise_alpha": 0.0, "use_dora": False, "use_rslora": False, "packing": False},
        {"variant": "E2", "neftune_noise_alpha": 5.0, "use_dora": False, "use_rslora": False, "packing": False},
        {"variant": "E3", "neftune_noise_alpha": 0.0, "use_dora": True, "use_rslora": False, "packing": False},
        {"variant": "E4", "neftune_noise_alpha": 0.0, "use_dora": False, "use_rslora": True, "packing": False},
        {"variant": "E5", "neftune_noise_alpha": 0.0, "use_dora": False, "use_rslora": False, "packing": True},
    ]
    (config_dir / "R1V4_CONFIG_REVIEW.md").write_text(
        "# R1V4 配置审核\n\n"
        "> 正式配置要等数据冻结后生成。此处只审核预注册的单变量设计。\n\n"
        "| 实验 | NEFTune | DoRA | RSLoRA | Packing | V4 checkpoint 保留数 |\n"
        "|---|---:|---|---|---|---:|\n"
        + "\n".join(
            f"| {row['variant']} | {row['neftune_noise_alpha']} | {row['use_dora']} | "
            f"{row['use_rslora']} | {row['packing']} | 1 |"
            for row in config_rows
        )
        + "\n",
        encoding="utf-8",
    )

    counts = {
        "source_lines": len(raw),
        "game_train_candidates": len(game_train),
        "constructed_train_candidates": len(constructed),
        "v4_independent_validation": len(draft_validation),
        "gold_v21": len(gold_v21["prompts"]),
        "gold_v3": len(gold_v3["prompts"]),
        "exclusions": len(exclusions),
    }
    packet_files = (source_outputs + game_outputs + constructed_outputs + validation_outputs
                    + gold_outputs + gold_v3_outputs + exclusion_outputs)
    manifest = {
        "schema_version": 1,
        "review_id": "KISAKI-V4-HUMAN-REVIEW",
        "status": "pending_human_review",
        "batch_size": batch_size,
        "counts": counts,
        "source_coverage": source_coverage,
        "packet_file_count": len(packet_files),
        "approval": {
            "approved_categories": [],
            "all_required_approved": False,
            "category_status": {"profile_prompt": "pending"},
            "items": {
                "character_profile": {"status": "pending"},
                "system_prompt_v3": {"status": "pending"},
            },
        },
    }
    (output / "review_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "00_GUIDE.md").write_text(
        "# 月社妃 V4 人工审核指南\n\n"
        "正式训练保持关闭，直到所有必需分类得到项目负责人明确批准。\n\n"
        "## 建议顺序\n\n"
        "1. `01_PROFILE_PROMPT`\n2. `02_SOURCE_COVERAGE`\n3. `03_GAME_TRAIN`\n"
        "4. `04_CONSTRUCTED_TRAIN`\n5. `05_VALIDATION`\n6. `06_GOLD_V21`\n"
        "7. 数据冻结后再生成 `07_GOLD_V3`\n8. `08_EXCLUSIONS`\n9. `09_EXPERIMENT_CONFIGS`\n\n"
        "## 回复格式\n\n"
        "- `全部通过`\n- `样本 ID：修改建议`\n- `样本 ID：排除，原因`\n"
        "- `样本 ID：需要更多原作上下文`\n\n"
        "Gold 修改不得回流训练集。原作 assistant 台词不可改写，只能修正构造的问题或排除错误提取。\n\n"
        f"当前计数：`{json.dumps(counts, ensure_ascii=False)}`\n",
        encoding="utf-8",
    )
    return manifest


def augment_source_context(output: Path) -> int:
    """Add missing source context without touching any human review fields."""

    raw = load_jsonl(CHARACTER_DIR / "tsukiyashiro_kisaki_raw.jsonl")
    contexts: dict[str, str] = {}
    source_cache: dict[str, list[str]] = {}
    for row in raw:
        source_file, raw_line = row["source"].rsplit(":line:", 1)
        source_lines = source_cache.setdefault(
            source_file,
            (GAMETEXT_DIR / source_file).read_text(encoding="utf-8").splitlines(),
        )
        line_number = int(raw_line)
        start = max(0, line_number - 4)
        end = min(len(source_lines), line_number + 3)
        contexts[row["id"]] = "\n".join(
            f"{index + 1}: {source_lines[index]}" for index in range(start, end)
        )

    changed = 0
    for path in (output / "02_SOURCE_COVERAGE").rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        sections = re.split(r"(?=^## )", text, flags=re.MULTILINE)
        updated_sections = []
        file_changed = False
        for section in sections:
            match = re.match(r"## ([^\n]+)", section)
            if match and "**原作上下文**" not in section:
                context = contexts.get(match.group(1).strip())
                marker = "**assistant**"
                if context and marker in section:
                    block = f"**原作上下文**\n\n```text\n{context}\n```\n\n"
                    section = section.replace(marker, block + marker, 1)
                    file_changed = True
            updated_sections.append(section)
        if file_changed:
            path.write_text("".join(updated_sections), encoding="utf-8")
            changed += 1
    source_coverage = write_source_index(output, raw)
    manifest_path = output / "review_manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        manifest["source_coverage"] = source_coverage
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return changed


def refresh_v4_validation(output: Path, batch_size: int) -> int:
    """Write review packets for the independent V4 held-out validation set."""

    target = output / "05_VALIDATION" / "v4_independent"
    if target.exists() and any(target.glob("*.md")):
        raise RuntimeError(f"refusing to overwrite existing V4 validation review: {target}")
    records = load_jsonl(EXPERIMENT_DIR / "v4" / "validation_candidate.jsonl")
    return len(
        write_batches(
            target,
            records,
            category="v4_independent_validation",
            original=True,
            batch_size=batch_size,
        )
    )


def write_consolidated_constructed_review(output: Path) -> dict[str, int]:
    """Write one final-check document from the currently reviewed V4 records."""

    records = load_jsonl(EXPERIMENT_DIR / "train_v5_clean.jsonl")
    pending = [
        record["id"]
        for record in records
        if not record.get("metadata", {}).get("human_review")
    ]
    if pending:
        raise RuntimeError(
            "refusing to build a final-check document with pending records: "
            + ", ".join(pending)
        )

    lines = [
        "# 月社妃 V4 构造训练数据统一复核稿",
        "",
        f"> 共 {len(records)} 条，全部已完成初次人工审核。本文件展示实际将进入候选数据集的最终问答，供冻结前统一复核。",
        "",
        "统一 system prompt 已单独审核，本文件不在每条样本中重复展示。请重点检查事实、人物关系、角色辨识度、问答对应和批量句式重复。",
        "",
        "复核时可直接回复：`全部通过`，或填写 `样本 ID：修改建议/排除原因`。",
        "",
    ]
    status_counts: Counter[str] = Counter()
    for record in records:
        metadata = record.get("metadata", {})
        review = metadata["human_review"]
        status_counts[review["status"]] += 1
        status_label = {
            "approved_unchanged": "原样通过",
            "approved_after_revision": "修改后通过",
        }.get(review["status"], review["status"])
        lines.extend(
            [
                f"## {record['id']}",
                "",
                f"- 场景：`{metadata.get('scene', '未标注')}`",
                f"- 数据来源：`{metadata.get('data_source', metadata.get('source', 'unknown'))}`",
                f"- 初审状态：`{status_label}`",
                f"- 修改理由：{review.get('reason', '无')}",
                "",
            ]
        )
        turn = 0
        for message in messages_of(record):
            if message["role"] == "system":
                continue
            if message["role"] == "user":
                turn += 1
                label = f"user {turn}"
            else:
                label = f"assistant {turn}"
            lines.extend((f"**{label}**", "", message["content"], ""))
        lines.extend(("- final_decision：`待统一复核`", "- final_notes：", "", "---", ""))

    target = output / "04_CONSTRUCTED_TRAIN" / "ALL_REVIEWED_FINAL_CHECK.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8", newline="\n")
    return {
        "records": len(records),
        "approved_unchanged": status_counts["approved_unchanged"],
        "approved_after_revision": status_counts["approved_after_revision"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--candidate-dir", type=Path)
    parser.add_argument("--augment-source-context", action="store_true")
    parser.add_argument("--consolidated-constructed-review", action="store_true")
    parser.add_argument("--refresh-v4-validation", action="store_true")
    args = parser.parse_args()
    if not 40 <= args.batch_size <= 60:
        raise SystemExit("--batch-size must be between 40 and 60")
    if args.augment_source_context:
        print(json.dumps({"updated_files": augment_source_context(args.output.resolve())}))
        return
    if args.consolidated_constructed_review:
        print(
            json.dumps(
                write_consolidated_constructed_review(args.output.resolve()),
                ensure_ascii=False,
            )
        )
        return
    if args.refresh_v4_validation:
        print(json.dumps({"updated_files": refresh_v4_validation(args.output.resolve(), args.batch_size)}))
        return
    manifest = build(args.output.resolve(), args.batch_size, args.candidate_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
