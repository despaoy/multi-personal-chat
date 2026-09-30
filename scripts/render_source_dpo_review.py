"""Render transparent AI-review candidates and source citations; no human approvals."""

import argparse
import json
from collections import Counter
from pathlib import Path


def render(root):
    records = [
        json.loads(line)
        for line in (root / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    records = list({r["id"]: r for r in records}.values())
    selected = sorted(
        [r for r in records if r["status"] == "ai_selected"], key=lambda r: r["id"]
    )
    counts = Counter(r["status"] for r in records)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    lines = [
        "# 原文 DPO 数据检查与生成结果",
        "",
        "**本报告为 AI 审核，不是人工盲评。生成和初筛使用同一 DeepSeek 模型，可能有自评偏差。**",
        "",
        f"原文通过检查：{manifest['source_checks']}；已处理：{len(records)}；初筛保留：{len(selected)}。",
        "",
        "状态统计：" + json.dumps(counts, ensure_ascii=False),
        "",
        "优选原文逐字保留；输入只有回复前的片段。置信度为模型自报，未经校准。训练文件仍为 pending，不能当作人工批准数据。",
        "",
        "## 初筛保留的偏好对",
        "",
    ]
    for r in selected:
        source = r["source"]
        lines += [
            f"### {r['id']}",
            "",
            f"来源：{source['source_file']}；输入行号 {r['context_start']}–{r['context_end']}；目标事件：{', '.join(source['event_ids'])}。",
            "",
            "**输入**",
            "",
            r["prompt"][-1]["content"],
            "",
            "**优选候选：原文**",
            "",
            r["original"],
            "",
            "**劣选候选：DeepSeek**",
            "",
            r["generated"],
            "",
            "**初筛依据**",
            "",
            r["judgment"]["reason"],
            "",
            "证据行号：" + ", ".join(map(str, r["judgment"]["evidence_lines"])),
            "",
        ]
    lines += ["## 未保留与失败样本", "", "| ID | 状态 | 原因 |", "|---|---|---|"]
    for r in records:
        if r["status"] != "ai_selected":
            reason = r.get("judgment", {}).get("reason", r.get("error_type", ""))
            lines.append(
                f"| {r['id']} | {r['status']} | {reason.replace('|', '/').replace(chr(10), ' ')} |"
            )
    (root / "REVIEW.md").write_text("\n".join(lines), encoding="utf-8")
    sft = [
        {
            "id": r["id"],
            "messages": [*r["prompt"], {"role": "assistant", "content": r["original"]}],
            "metadata": {
                "source_group": r["source_group"],
                "source": r["source"],
                "feedback_source": "ai",
                "review_status": "pending",
                "human_final_approved": False,
            },
        }
        for r in selected
    ]
    (root / "sft.ai-reviewed.pending.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sft), encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    render(parser.parse_args().root)
