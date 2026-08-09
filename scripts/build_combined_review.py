# -*- coding: utf-8 -*-
"""生成训练数据整体检查清单: combined_merged 全部样本按模块分组展示。"""
import json
from collections import OrderedDict
from pathlib import Path

BASE = Path(r"c:\Users\13474\Desktop\qqchat-enhanced\backend\data\character_dialogues\experiments")
OUT = BASE / "combined_review.md"

recs = [json.loads(l) for l in (BASE / "combined_merged.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

# 按 (module, state) 分组
groups = OrderedDict()
for r in recs:
    md = r["metadata"]
    key = f"{md.get('module','?')} / {md.get('state','?')}"
    groups.setdefault(key, []).append(r)

def first_line(r):
    msgs = r["messages"]
    sys_text = msgs[0]["content"][:40].replace("\n", " ")
    return sys_text

lines = ["# 月社妃训练数据整体检查清单", "",
         f"合计 {len(recs)} 条（merged 口径，含 train+eval）", "",
         "## 统计",
         f"- 总样本: {len(recs)}",
         f"- 总 assistant 回合: {sum(len(r['messages'])//2 for r in recs)}",
         f"- 总 token / assistant loss token: 见 token_stats_qwen3.json", ""]

for key, rs in groups.items():
    lines.append(f"## {key}（{len(rs)} 条）")
    lines.append("")
    lines.append("| id | 场景 | data_branch | split_group | 回合 | system | assistant 内容预览 |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in sorted(rs, key=lambda x: x["id"]):
        md = r["metadata"]
        asst = " ⏎ ".join(m["content"][:40] for m in r["messages"] if m["role"] == "assistant")
        asst = asst.replace("|", "\\|")
        lines.append(f"| {r['id']} | {md.get('scene','')} | {md.get('data_branch','')} | "
                     f"{md.get('split_group','-')} | {len(r['messages'])//2} | {first_line(r)} | {asst} |")
    lines.append("")

OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"写出 {OUT.name}: {len(recs)} 条")
for k, v in groups.items():
    print(f"  {k}: {len(v)}")
