"""Export evidence-bound second AI review without claiming human approval."""

import argparse
import json
from pathlib import Path


def export(root, decisions_path):
    records = {
        r["id"]: r
        for r in (
            json.loads(line)
            for line in (root / "results.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    }
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    pairs, sft, report = (
        [],
        [],
        [
            "# 原文对照数据：二次 AI 复核",
            "",
            "已检查游戏原文行号、回复前上下文与 DeepSeek 候选。这里的复核者是 AI，不是人工。",
            "",
            "DeepSeek 初筛分数未经校准；以下选择由第二次逐条证据复核决定，不是把低分样本自动改为高分。原文保持原样。",
            "",
        ],
    )
    for d in decisions["decisions"]:
        r = records[d["id"]]
        if d["audit_record_sha256"] != r["record_sha256"]:
            raise ValueError("decision refers to a different audit record")
        if any(
            not r["context_start"] <= line <= r["context_end"]
            for line in d["evidence_lines"]
        ):
            raise ValueError("decision cites a missing context line")
        report += [
            f"## {d['id']} — {d['decision']}",
            "",
            d["reason"],
            "",
            f"来源：{r['source']['source_file']}，前文行 {r['context_start']}–{r['context_end']}。",
            "",
            "**原文**",
            "",
            r["original"],
            "",
            "**DeepSeek 候选**",
            "",
            r["generated"],
            "",
        ]
        if d["decision"] != "keep":
            continue
        metadata = {
            "schema_version": "source-ai-preference-candidate-v1",
            "feedback_source": "ai",
            "review_method": "two_stage_ai_review",
            "reviewer": "assistant_source_adjudication",
            "human_final_approved": False,
            "split": "train",
            "source_group": r["source_group"],
            "source_ids": [r["source_group"]],
            "source": r["source"],
            "input_sha256": r["input_sha256"],
            "audit_record_sha256": r["record_sha256"],
            "reason": d["reason"],
            "evidence_lines": d["evidence_lines"],
        }
        pairs.append(
            {
                "id": r["id"],
                "prompt": r["prompt"],
                "chosen": [{"role": "assistant", "content": r["original"]}],
                "rejected": [{"role": "assistant", "content": r["generated"]}],
                "review_status": "pending",
                "metadata": metadata,
            }
        )
        sft.append(
            {
                "id": r["id"],
                "messages": [
                    *r["prompt"],
                    {"role": "assistant", "content": r["original"]},
                ],
                "metadata": {**metadata, "review_status": "pending"},
            }
        )
    for name, rows in [
        ("dpo.adjudicated.ai.jsonl", pairs),
        ("sft.adjudicated.ai.jsonl", sft),
    ]:
        with (root / name).open("x", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report[3:3] = [
        f"本次明确复核 {len(decisions['decisions'])} 对，保留 {len(pairs)} 对；其余原始候选留在 results.jsonl。",
        "",
        "这是用于小规模实验的数据候选，不足以证明人物改善；未写入冻结训练集，未启动训练。",
        "",
    ]
    (root / "ADJUDICATED_REVIEW.md").write_text("\n".join(report), encoding="utf-8")
    return len(pairs)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("decisions", type=Path)
    a = p.parse_args()
    print(export(a.root, a.decisions))
