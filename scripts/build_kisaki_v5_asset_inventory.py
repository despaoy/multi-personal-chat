"""阶段 1：建立月社妃 V5 数据资产清单（只读分析，不修改 V4）。

职责单一：读取 KISAKI-CANONICAL-V4 冻结数据，按来源统计暴露量，
输出 V5 候选资产清单与异常报告。不做筛选、不删除数据、不训练。

监督口径（与 backend/training/chat_dataset.py 的训练契约一致）：
- metadata.assistant_supervision ∈ {"all", "last"}，缺省为 "all"；
- "last" 只监督最后一条 assistant 消息（107 条原作多轮记录即此契约）；
- 因此同时统计 raw_assistant_messages（原始 assistant 消息数）与
  supervised_assistant_targets（实际监督目标数），只有被监督的回复
  才计入 supervised_assistant_chars。

产物（全部写入 experiments/v5_candidate/）：
- asset_inventory.json   逐条清单：ID、来源、场景、轮数、长度、状态
- asset_stats.json       来源聚合统计
- inventory_issues.json  异常（重复 ID、空消息、角色顺序、审核状态缺失）
- README.md              说明文档（含统计摘要与代表样本）

V4 保护：
- train.jsonl / validation.jsonl 以只读模式打开；
- 运行时先校验 sha256 与 canonical_dataset_manifest.json 的冻结值一致，
  结束后再次校验，确保本脚本未触碰原文件。

token 估算沿用项目既有口径：Qwen 系 tokenizer 中文约 0.65 token/字符，
占比一律以字符口径为准（token 仅为工程估算）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
V4_DIR = REPO_ROOT / "backend/data/character_dialogues/experiments/v4"
V5_DIR = REPO_ROOT / "backend/data/character_dialogues/experiments/v5_candidate"

TOKEN_PER_CHAR = 0.65  # Qwen 系 tokenizer 中文近似，项目既有口径

# data_source → 四类来源（与 canonical_dataset_manifest 的 source_distribution 对齐）
SOURCE_GROUPS = {
    "game_extraction": "game_extraction_current_sft",
    "codex_user_simulation_v41_reviewed": "codex_user_simulation_v41_reviewed",
    "deepseek_user_simulation_v41_reviewed": "deepseek_user_simulation_v41_reviewed",
}
LLM_V4_PREFIX = "llm_v4_"

# 初始状态：原作数据为冻结核心；两类模拟数据进入后续阶段复审
GROUP_INITIAL_STATUS = {
    "game_extraction_current_sft": "keep_core",
    "llm_v4_reviewed_constructed": "pending_review",
    "codex_user_simulation_v41_reviewed": "pending_review",
    "deepseek_user_simulation_v41_reviewed": "pending_review",
}
ALLOWED_STATUS = {"pending_review", "keep_core", "excluded_candidate"}

# 允许的轮数统计口径：assistant 消息数 > 1 视为多轮
MULTITURN_ASSISTANT_THRESHOLD = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_source(data_source: str) -> str:
    """data_source 原值 → 四类聚合来源。未知值抛错，不静默归入其他。"""
    if data_source in SOURCE_GROUPS:
        return SOURCE_GROUPS[data_source]
    if data_source.startswith(LLM_V4_PREFIX):
        return "llm_v4_reviewed_constructed"
    raise ValueError(f"未知 data_source: {data_source!r}")


def supervised_assistant_messages(messages: list[dict], supervision: str) -> list[dict]:
    """按训练契约返回被监督的 assistant 消息。

    与 backend/training/chat_dataset.py tokenize_assistant_turns 一致：
    "all" 监督全部 assistant 消息；"last" 只监督最后一条。
    """
    assistant_msgs = [m for m in messages if m["role"] == "assistant"]
    if supervision == "last":
        return assistant_msgs[-1:]
    return assistant_msgs


def check_role_order(messages: list[dict]) -> list[str]:
    """role 顺序检查，规则与 normalize_chat_record() 对齐。

    - 非空 content、已知 role；
    - system 仅允许首位；
    - 相邻同角色消息（user-user 与 assistant-assistant 一律拒绝，
      正式训练入口 normalize_chat_record 会 raise）；
    - 末条必须是 assistant（保证存在监督目标）。
    """
    problems = []
    for idx, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content")
        if content is None or not str(content).strip():
            problems.append(f"msg[{idx}] role={role} 内容为空")
        if role not in ("system", "user", "assistant"):
            problems.append(f"msg[{idx}] 未知 role={role!r}")
    if any(m.get("role") == "system" for m in messages[1:]):
        problems.append("system 消息不在首位")
    prev_role = None
    for idx, msg in enumerate(messages):
        role = msg.get("role")
        if role != "system" and prev_role == role:
            problems.append(f"msg[{idx}] 相邻两条 {role} 消息")
        prev_role = role
    if messages and messages[-1]["role"] != "assistant":
        problems.append("最后一条消息不是 assistant（无监督目标）")
    return problems


def check_review_status(metadata: dict, group: str) -> list[str]:
    """审核状态检查：四类来源在 V4 manifest 中均有记录，缺失即异常。"""
    missing = []
    if group == "game_extraction_current_sft" and "final_review" not in metadata:
        missing.append("game 记录缺少 final_review")
    if group == "llm_v4_reviewed_constructed" and "human_review" not in metadata:
        missing.append("构造记录缺少 human_review")
    if (
        group
        in (
            "codex_user_simulation_v41_reviewed",
            "deepseek_user_simulation_v41_reviewed",
        )
        and "human_review" not in metadata
    ):
        missing.append("用户模拟记录缺少 human_review")
    return missing


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path.name} 第 {line_no} 行 JSON 解析失败: {e}")
    return records


def build_inventory(train_path: Path, val_path: Path) -> dict:
    train_records = load_jsonl(train_path)
    val_records = load_jsonl(val_path)
    val_ids = {r["id"] for r in val_records}

    issues: list[dict] = []
    inventory: list[dict] = []
    group_counter: Counter[str] = Counter()

    seen_ids: set[str] = set()
    for rec in train_records:
        rid = rec.get("id")
        meta = rec.get("metadata", {})
        messages = rec.get("messages", [])
        rec_issues: list[str] = []

        if not rid:
            rec_issues.append("记录缺少 id")
        elif rid in seen_ids:
            rec_issues.append(f"重复 id: {rid}")
        else:
            seen_ids.add(rid)

        data_source = str(meta.get("data_source", ""))
        try:
            group = classify_source(data_source)
        except ValueError as e:
            group = "unknown"
            rec_issues.append(str(e))
        group_counter[group] += 1

        rec_issues.extend(check_role_order(messages))
        rec_issues.extend(check_review_status(meta, group))

        # 监督契约：缺省 "all"，107 条原作多轮记录为 "last"
        supervision = str(meta.get("assistant_supervision") or "all")
        if supervision not in ("all", "last"):
            rec_issues.append(f"非法 assistant_supervision: {supervision!r}")

        n_assistant = sum(1 for m in messages if m["role"] == "assistant")
        n_user = sum(1 for m in messages if m["role"] == "user")
        supervised = supervised_assistant_messages(messages, supervision)
        sup_chars = sum(len(m.get("content") or "") for m in supervised)
        t_chars = sum(len(m.get("content") or "") for m in messages)

        if rid in val_ids:
            rec_issues.append("id 与 validation 重叠")

        inventory.append(
            {
                "id": rid,
                "data_source": data_source,
                "source_group": group,
                "status": GROUP_INITIAL_STATUS.get(group, "pending_review"),
                # 场景：game 用 source_file；构造/模拟用 metadata.scene
                "scene": (meta.get("source_file") or meta.get("scene") or "")[:120],
                "interlocutor_kind": meta.get("interlocutor_kind", ""),
                "assistant_supervision": supervision,
                "turns": {"user": n_user, "assistant": n_assistant},
                "multiturn": n_assistant > MULTITURN_ASSISTANT_THRESHOLD,
                "chars": {
                    "raw_assistant": sum(
                        len(m.get("content") or "")
                        for m in messages
                        if m["role"] == "assistant"
                    ),
                    "supervised_assistant": sup_chars,
                    "total": t_chars,
                },
                "supervised_assistant_targets": len(supervised),
                "est_tokens": {
                    "supervised_assistant": int(sup_chars * TOKEN_PER_CHAR),
                    "total": int(t_chars * TOKEN_PER_CHAR),
                },
                "avg_supervised_chars_per_target": (
                    round(sup_chars / len(supervised), 1) if supervised else 0
                ),
                "issues": rec_issues,
            }
        )
        if rec_issues:
            issues.append({"id": rid, "source_group": group, "problems": rec_issues})

    return {
        "inventory": inventory,
        "group_counter": dict(group_counter),
        "issues": issues,
        "train_count": len(train_records),
        "val_count": len(val_records),
    }


def build_stats(result: dict) -> dict:
    inventory = result["inventory"]
    groups: dict[str, dict] = {}
    for item in inventory:
        g = item["source_group"]
        bucket = groups.setdefault(
            g,
            {
                "source_group": g,
                "records": 0,
                "raw_assistant_messages": 0,
                "supervised_assistant_targets": 0,
                "raw_assistant_chars": 0,
                "supervised_assistant_chars": 0,
                "total_chars": 0,
                "est_supervised_assistant_tokens": 0,
                "est_total_tokens": 0,
                "multiturn_records": 0,
                "supervision_counts": {"all": 0, "last": 0},
                "avg_supervised_chars_per_target": 0,
                "avg_supervised_targets_per_record": 0,
                "initial_status": GROUP_INITIAL_STATUS.get(g, "pending_review"),
            },
        )
        bucket["records"] += 1
        bucket["raw_assistant_messages"] += item["turns"]["assistant"]
        bucket["supervised_assistant_targets"] += item["supervised_assistant_targets"]
        bucket["raw_assistant_chars"] += item["chars"]["raw_assistant"]
        bucket["supervised_assistant_chars"] += item["chars"]["supervised_assistant"]
        bucket["total_chars"] += item["chars"]["total"]
        bucket["est_supervised_assistant_tokens"] += item["est_tokens"][
            "supervised_assistant"
        ]
        bucket["est_total_tokens"] += item["est_tokens"]["total"]
        if item["multiturn"]:
            bucket["multiturn_records"] += 1
        bucket["supervision_counts"][item["assistant_supervision"]] += 1

    total_sup_chars = sum(b["supervised_assistant_chars"] for b in groups.values())
    total_sup_targets = sum(b["supervised_assistant_targets"] for b in groups.values())
    total_raw_msgs = sum(b["raw_assistant_messages"] for b in groups.values())
    total_records = sum(b["records"] for b in groups.values())
    for bucket in groups.values():
        bucket["supervised_char_share"] = (
            round(bucket["supervised_assistant_chars"] / total_sup_chars * 100, 2)
            if total_sup_chars
            else 0
        )
        bucket["record_share"] = (
            round(bucket["records"] / total_records * 100, 2) if total_records else 0
        )
        bucket["avg_supervised_chars_per_target"] = (
            round(
                bucket["supervised_assistant_chars"]
                / bucket["supervised_assistant_targets"],
                1,
            )
            if bucket["supervised_assistant_targets"]
            else 0
        )
        bucket["avg_supervised_targets_per_record"] = (
            round(bucket["supervised_assistant_targets"] / bucket["records"], 2)
            if bucket["records"]
            else 0
        )

    return {
        "supervision_contract": (
            "assistant_supervision 缺省 all；last 只监督最后一条 assistant 消息"
            "（与 backend/training/chat_dataset.py 训练契约一致）"
        ),
        "token_estimation": f"chars * {TOKEN_PER_CHAR} (Qwen 系近似；占比以字符口径为准)",
        "multiturn_definition": f"assistant 目标数 > {MULTITURN_ASSISTANT_THRESHOLD}",
        "grand_total": {
            "records": total_records,
            "raw_assistant_messages": total_raw_msgs,
            "supervised_assistant_targets": total_sup_targets,
            "supervised_assistant_chars": total_sup_chars,
            "est_supervised_assistant_tokens": int(total_sup_chars * TOKEN_PER_CHAR),
            "multiturn_records": sum(b["multiturn_records"] for b in groups.values()),
        },
        "by_source": sorted(groups.values(), key=lambda b: -b["records"]),
    }


def pick_samples(inventory: list[dict], per_group: int = 3) -> dict[str, list[str]]:
    """每来源按监督字符排序，取最短/中位/最长三条作为代表样本。"""
    samples: dict[str, list[str]] = {}
    by_group: dict[str, list[dict]] = {}
    for item in inventory:
        by_group.setdefault(item["source_group"], []).append(item)
    for group, items in by_group.items():
        ordered = sorted(items, key=lambda i: i["chars"]["supervised_assistant"])
        if not ordered:
            continue
        picks = [ordered[0]["id"]]
        if len(ordered) > 2:
            picks.append(ordered[len(ordered) // 2]["id"])
        picks.append(ordered[-1]["id"])
        samples[group] = picks[:per_group]
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-group-samples",
        type=int,
        default=3,
        help="README 中每类来源展示的代表样本数（默认 3）",
    )
    args = parser.parse_args()

    train_path = V4_DIR / "train.jsonl"
    val_path = V4_DIR / "validation.jsonl"
    manifest_path = V4_DIR / "canonical_dataset_manifest.json"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # V4 保护：先校验冻结哈希
    frozen = {
        "train": manifest["train"]["sha256"],
        "validation": manifest["validation"]["sha256"],
    }
    before = {"train": sha256_file(train_path), "validation": sha256_file(val_path)}
    for name in ("train", "validation"):
        if before[name] != frozen[name]:
            print(
                f"[ABORT] {name}.jsonl sha256 与冻结 manifest 不一致：\n"
                f"  实际 {before[name]}\n  冻结 {frozen[name]}",
                file=sys.stderr,
            )
            return 2

    result = build_inventory(train_path, val_path)
    stats = build_stats(result)
    samples = pick_samples(result["inventory"], args.per_group_samples)

    # 校验清单完整性：所有 train 记录都在清单中且状态合法
    assert result["train_count"] == len(result["inventory"])
    bad_status = [
        i["id"] for i in result["inventory"] if i["status"] not in ALLOWED_STATUS
    ]
    if bad_status:
        print(f"[ABORT] 存在非法状态: {bad_status[:5]}", file=sys.stderr)
        return 2

    V5_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()

    inventory_doc = {
        "schema_version": 1,
        "dataset_id": "KISAKI-V5-CANDIDATE-ASSET-INVENTORY",
        "generated_at": generated_at,
        "generated_from": {
            "dataset_id": manifest["dataset_id"],
            "git_commit_hint": "see README.md of v5_candidate",
            "train_sha256": before["train"],
            "validation_sha256": before["validation"],
            "train_count": result["train_count"],
            "validation_count": result["val_count"],
        },
        "allowed_status": sorted(ALLOWED_STATUS),
        "status_semantics": {
            "keep_core": "V4 冻结核心（原作提取，approved_after_context_reaudit），默认保留",
            "pending_review": "待后续阶段人工复审（阶段 2 长模拟 / 阶段 3 短构造）",
            "excluded_candidate": "已被排除（本阶段不使用，后续审核阶段写入）",
        },
        "token_estimation": stats["token_estimation"],
        "records": result["inventory"],
    }
    (V5_DIR / "asset_inventory.json").write_text(
        json.dumps(inventory_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    stats_doc = {
        "schema_version": 1,
        "generated_at": generated_at,
        **stats,
        "issues_summary": {
            "records_with_issues": len(result["issues"]),
            "issue_types": dict(
                Counter(p for entry in result["issues"] for p in entry["problems"])
            ),
        },
    }
    (V5_DIR / "asset_stats.json").write_text(
        json.dumps(stats_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    (V5_DIR / "inventory_issues.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at": generated_at,
                "train_sha256": before["train"],
                "records_with_issues": len(result["issues"]),
                "issues": result["issues"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # V4 保护：结束后复验哈希未变
    after = {"train": sha256_file(train_path), "validation": sha256_file(val_path)}
    if after != before:
        print("[ABORT] 运行后 V4 哈希发生变化", file=sys.stderr)
        return 2

    print(json.dumps(stats_doc["grand_total"], ensure_ascii=False))
    for bucket in stats["by_source"]:
        print(
            f"  {bucket['source_group']}: records={bucket['records']} "
            f"raw_a_msgs={bucket['raw_assistant_messages']} "
            f"sup_targets={bucket['supervised_assistant_targets']} "
            f"sup_chars={bucket['supervised_assistant_chars']} "
            f"({bucket['supervised_char_share']}%) "
            f"multiturn={bucket['multiturn_records']}"
        )
    print(f"issues: {len(result['issues'])} 条记录带异常")
    print(f"sample ids: {json.dumps(samples, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
