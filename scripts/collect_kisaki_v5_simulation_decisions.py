"""阶段 2 审核：从分批 Markdown 勾选结果回收人工决定（产出 draft）。

职责单一：解析 review_batches/batch_*.md 中每条记录的
"人工选择" 勾选框，校验后生成 draft 决定文件。

规则（任何一条不满足即整体报错退出，不产出部分结果）：
- 每条记录恰好勾选一项：keep / exclude / revise；
  未选择、重复选择、非法值一律报错；
- 勾选 ID 集合必须与 simulation_review_packet.json 的 254 个 ID
  完全相等（未知 ID / 缺失 ID 均报错）；
- revise（待改写）→ 决定值 exclude，同时记入 needs_revision 列表
  （改写属后续工作，本阶段不给 keep）；
- 产出 review_status="draft"——**draft 不算人工批准**，owner 亲自
  核对后把 review_status 改为 "approved" 并填写 reviewed_by，
  build_kisaki_v5_candidate.py --decisions 只认 approved。

用法：
  python scripts/collect_kisaki_v5_simulation_decisions.py \
      [--packet PATH] [--batches-dir DIR] [--output PATH]
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
V5_DIR = REPO_ROOT / "backend/data/character_dialogues/experiments/v5_candidate"
DEFAULT_PACKET = V5_DIR / "simulation_review_packet.json"
DEFAULT_BATCH_DIR = V5_DIR / "review_batches"
DEFAULT_OUTPUT = V5_DIR / "simulation_review_decisions.json"

_ID_LINE = re.compile(r"^- ID: `([^`]+)`")
# 只匹配"人工选择"行，避免误读批次说明文字
_CHOICE_LINE = re.compile(r"^- \*\*人工选择\*\*:\s*(.*)$")
_CHECKED = re.compile(r"\[([xX✓])\]\s*(keep|exclude|revise)")
_ALLOWED = ("keep", "exclude", "revise")


def parse_batches(batch_dir: Path) -> dict[str, str]:
    """解析全部批次 Markdown，返回 {record_id: choice}。

    校验失败（未选择/重复选择/非法值/记录缺选择行）立即 SystemExit。
    """
    batch_files = sorted(batch_dir.glob("batch_*.md"))
    if not batch_files:
        raise SystemExit(f"[ABORT] {batch_dir} 下没有 batch_*.md")

    choices: dict[str, str] = {}
    current_id: str | None = None
    for path in batch_files:
        for line_no, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            id_match = _ID_LINE.match(line)
            if id_match:
                current_id = id_match.group(1)
                continue
            choice_match = _CHOICE_LINE.match(line)
            if not choice_match:
                continue
            if current_id is None:
                raise SystemExit(
                    f"[ABORT] {path.name}:{line_no} 出现选择行但没有前置 ID 行"
                )
            marked = _CHECKED.findall(choice_match.group(1))
            if not marked:
                raise SystemExit(
                    f"[ABORT] {path.name}:{line_no} 记录 {current_id} 未勾选任何选项"
                )
            if len(marked) > 1:
                raise SystemExit(
                    f"[ABORT] {path.name}:{line_no} 记录 {current_id} 勾选了 "
                    f"{len(marked)} 项（{[m[1] for m in marked]}），只能选一项"
                )
            choices[current_id] = marked[0][1]
            current_id = None  # 下一条记录必须重新出现 ID 行

    if current_id is not None:
        raise SystemExit(f"[ABORT] {path.name} 末尾记录 {current_id} 没有选择行")
    return choices


def validate_coverage(choices: dict[str, str], packet_ids: set[str]) -> None:
    unknown = sorted(set(choices) - packet_ids)
    if unknown:
        raise SystemExit(f"[ABORT] 勾选包含未知 ID（{len(unknown)} 个）: {unknown[:5]}")
    missing = sorted(packet_ids - set(choices))
    if missing:
        raise SystemExit(f"[ABORT] {len(missing)} 条记录未出现在勾选中: {missing[:5]}")


def build_draft_document(
    choices: dict[str, str], batch_dir: Path, packet: dict
) -> dict:
    """revise → exclude + needs_revision；产出 draft 决定文件。"""
    decisions = {
        rid: ("exclude" if choice == "revise" else choice)
        for rid, choice in choices.items()
    }
    needs_revision = sorted(rid for rid, c in choices.items() if c == "revise")
    return {
        "review_status": "draft",
        "reviewed_by": None,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "packet_id": packet.get("packet_id"),
            "packet_generated_at": packet.get("generated_at"),
            "batches": sorted(p.name for p in batch_dir.glob("batch_*.md")),
        },
        "decisions": decisions,
        "needs_revision": needs_revision,
        "stats": {
            "total": len(decisions),
            "keep": sum(1 for v in decisions.values() if v == "keep"),
            "exclude": sum(1 for v in decisions.values() if v == "exclude"),
            "needs_revision": len(needs_revision),
        },
        "note": (
            "draft 由勾选回收脚本生成，不算人工批准；owner 核对后将 "
            'review_status 改为 "approved" 并填写 reviewed_by'
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--batches-dir", type=Path, default=DEFAULT_BATCH_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    packet = json.loads(args.packet.read_text(encoding="utf-8"))
    packet_ids = {e["id"] for e in packet["entries"]}

    choices = parse_batches(args.batches_dir)
    validate_coverage(choices, packet_ids)

    doc = build_draft_document(choices, args.batches_dir, packet)
    args.output.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    stats = doc["stats"]
    print(f"勾选回收完成：{stats['total']} 条 → {args.output}")
    print(
        f"  keep={stats['keep']} exclude={stats['exclude']} "
        f"（其中 revise→exclude 待改写 {stats['needs_revision']} 条）"
    )
    print(
        "[提醒] 产出为 draft；owner 核对后将 review_status 改为 "
        '"approved" 并填写 reviewed_by，--decisions 只认 approved'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
