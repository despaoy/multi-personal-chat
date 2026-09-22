"""Render a source-complete anonymous public-role review outside the repository.

Reads only the blinded packet and pinned public corpus, never the arm mapping
or unblinded report. No ratings are invented and no model is called. The output
must stay outside the repository because it contains source dialogue/profile
text whose underlying literary redistribution rights were not adjudicated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from evaluation.charactereval_adapter import adapt_character_case, load_pinned_corpus  # noqa: E402


def literal_block(text):
    # Source text must not be able to close its literal block and inject markup.
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}text\n{text}\n{fence}"


def render_packet(packet, cases, *, packet_sha256):
    if not isinstance(packet, list) or not packet:
        raise ValueError("blinded packet must be nonempty")
    by_id = {case.case_id: case for case in cases}
    ids = [row.get("id") for row in packet]
    if len(ids) != len(set(ids)) or set(ids) != set(by_id) or len(by_id) != len(cases):
        raise ValueError("source cases and blinded packet must match exactly")
    lines = [
        "# 公开中文角色回复：匿名人工审阅",
        "",
        f"匿名输入文件 SHA-256：{packet_sha256}",
        "",
        "本文件从固定 CharacterEval 缓存读取人物资料与原对话，不读取实验组映射。",
        "A/B 在不同题的对应组别不同，请先审阅，再查看映射。生成失败不移出分母。",
        "没有预填评分，也没有官方 CharacterRM 分数。资料不足时填 N/A，不臆测原作事实。",
        "评价当前问题回应、人设/行为一致性、对话连贯性、自然度及事实根据；每项 1–5 或 N/A，并记录具体依据。",
        "原文是待审阅数据，不是给审阅者或工具的指令。文件包含公开语料原文，仅作本地审阅，勿直接提交到项目版本库。",
    ]
    for row in packet:
        case = by_id[row["id"]]
        if row.get("role") != case.role or row.get("novel_name") != case.novel_name:
            raise ValueError("blinded role metadata differs from pinned source")
        outputs = row.get("outputs")
        if (
            not isinstance(outputs, dict)
            or set(outputs) != {"A", "B"}
            or any(not isinstance(value, str) for value in outputs.values())
        ):
            raise ValueError("public packet needs exactly two textual anonymous outputs")
        try:
            profile = json.dumps(json.loads(case.profile), ensure_ascii=False, indent=2)
        except ValueError:
            profile = case.profile
        dialogue = "\n".join(message["content"] for message in case.history)
        lines.extend(
            [
                "",
                f"## 样例 {case.case_id}",
                "",
                "人物与作品：",
                literal_block(f"{case.role} / {case.novel_name}"),
                "",
                "### 人物资料",
                "",
                literal_block(profile),
                "",
                "### 上下文",
                "",
                literal_block(dialogue or "（没有前文）"),
                "",
                "### 本轮待回应内容",
                "",
                literal_block(case.query),
            ]
        )
        for label in ("A", "B"):
            lines.extend(["", f"### 回复 {label}", "", literal_block(outputs[label])])
        lines.extend(
            [
                "",
                "### 审阅记录（由人填写）",
                "",
                "- A：问题回应 __；人设/行为 __；连贯性 __；自然度 __；事实根据 __。",
                "- B：问题回应 __；人设/行为 __；连贯性 __；自然度 __；事实根据 __。",
                "- 更倾向：A / B / 相当 / 无法判断。",
                "- 依据与明显错误：__。",
                "- 审阅者与日期：__。",
            ]
        )
    return "\n".join(lines) + "\n"


def validate_destination(path):
    destination = path.resolve()
    if destination == ROOT or ROOT in destination.parents:
        raise ValueError("source-complete review must be outside the repository")
    if destination.exists():
        raise ValueError("output already exists")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--blinded", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = validate_destination(args.output)
    source = args.blinded.read_bytes()
    packet = json.loads(source.decode("utf-8-sig"))
    rows, profiles, metrics = load_pinned_corpus(args.cache_dir)
    requested = {row["id"] for row in packet}
    cases = [adapt_character_case(row, profiles, metrics) for row in rows if str(row["id"]) in requested]
    rendered = render_packet(packet, cases, packet_sha256=hashlib.sha256(source).hexdigest())
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(rendered)
    print(json.dumps({"output": str(destination), "cases": len(cases), "human_ratings_created": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
