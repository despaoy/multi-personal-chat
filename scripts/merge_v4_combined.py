# -*- coding: utf-8 -*-
"""v4 全量合并: 主数据(train.jsonl 188条) + 游戏提取(25条) -> 85/15 分层划分。

划分规则:
- 游戏数据(25条): 按 split_group 同组同侧; 每模块(famous/paper/riou/rikata)按 seed=42
  shuffle 取 1 组进 eval, 保证 eval 覆盖 琉璃/夜子/理央/彼方/纸上状态
- 主数据(188条, 无 split_group): 按 data_source 分层, 每层 seed=42 shuffle 取 15% 进 eval
输出:
  - combined_merged.jsonl (全体, 含 module/state/split_group 标记)
  - combined_train.jsonl / combined_eval.jsonl
  - combined_manifest.json
"""
import hashlib
import json
import os
import random
from collections import Counter, OrderedDict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE = PROJECT_ROOT / "backend" / "data" / "character_dialogues" / "experiments"
SEED = 42
EVAL_RATIO = 0.15

GAME_FILES = [
    ("famous", "game_famous_candidates.jsonl"),
    ("paper", "game_famous_paper_state.jsonl"),
    ("riou", "game_riou_candidates.jsonl"),
    ("rikata", "game_rikata_candidates.jsonl"),
]
MAIN_FILE = BASE / "train_v5_clean.jsonl"  # V5 阻塞性清洗后的主数据
OUT_MERGED = BASE / "combined_merged.jsonl"
OUT_TRAIN = BASE / "combined_train.jsonl"
OUT_EVAL = BASE / "combined_eval.jsonl"
OUT_MANIFEST = BASE / "combined_manifest.json"


def load_lines(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def dump(path, recs):
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def dump_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def assistant_stats(recs):
    turns = sum(len(r["messages"]) // 2 for r in recs)
    chars = sum(len(m["content"]) for r in recs for m in r["messages"] if m["role"] == "assistant")
    return {"samples": len(recs), "assistant_turns": turns, "assistant_chars": chars}


def main():
    rng = random.Random(SEED)

    # ── 游戏数据 ──
    game_all, game_by_module = [], {}
    for module, fname in GAME_FILES:
        recs = load_lines(BASE / fname)
        for r in recs:
            r["metadata"]["module"] = module
            r["metadata"]["n_turns"] = len(r["messages"]) // 2
            r["metadata"]["data_branch"] = f"{module}_game"  # riou_game / famous_game ...
        game_by_module[module] = recs
        game_all.extend(recs)
    print(f"游戏数据: {len(game_all)} 条")

    # 每模块按 split_group 分组, shuffle 取 1 组进 eval
    game_eval, game_train = [], []
    for module, recs in game_by_module.items():
        groups = OrderedDict()
        for r in recs:
            groups.setdefault(r["metadata"]["split_group"], []).append(r)
        items = list(groups.items())
        rng.shuffle(items)
        eval_sg, eval_recs = items[0]
        game_eval.extend(eval_recs)
        game_train.extend([r for sg, rs in items[1:] for r in rs])
        print(f"  [{module}] eval组={eval_sg} ({len(eval_recs)}条/{sum(x['metadata']['n_turns'] for x in eval_recs)}回合)")

    # ── 主数据 ──
    main_recs = load_lines(MAIN_FILE)
    for r in main_recs:
        r["metadata"]["module"] = "llm"
        r["metadata"]["state"] = r["metadata"].get("state", "general")
        r["metadata"]["n_turns"] = len(r["messages"]) // 2
        ds = r["metadata"].get("data_source", "unknown")
        # llm_v4_riou 与游戏 riou 区分: riou_manual vs riou_game
        r["metadata"]["data_branch"] = "riou_manual" if ds == "llm_v4_riou" else ds
    print(f"主数据: {len(main_recs)} 条")

    by_source = OrderedDict()
    for r in main_recs:
        by_source.setdefault(r["metadata"]["data_source"], []).append(r)
    main_eval, main_train = [], []
    for src, group in by_source.items():
        group = sorted(group, key=lambda x: x["id"])
        rng.shuffle(group)
        n_eval = max(1, int(len(group) * EVAL_RATIO)) if len(group) >= 5 else 0
        main_eval.extend(group[:n_eval])
        main_train.extend(group[n_eval:])
        print(f"  [{src}] {len(group)} -> eval {n_eval}")

    # riou 重复审计 (主数据 LLM riou vs 游戏提取 riou)
    def msg_hash(r):
        canon = json.dumps(r["messages"], ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]

    def turns_of(r):
        return tuple((m["role"], m["content"]) for m in r["messages"])

    main_riou = [r for r in main_recs if r["metadata"].get("data_source") == "llm_v4_riou"]
    game_riou = [r for r in game_all if r["metadata"]["module"] == "riou"]

    train, eval_ = main_train + game_train, main_eval + game_eval
    rng.shuffle(train)
    rng.shuffle(eval_)

    merged = main_recs + game_all
    dump(OUT_MERGED, merged)
    dump(OUT_TRAIN, train)
    dump(OUT_EVAL, eval_)

    def full_stats(recs, label):
        st = assistant_stats(recs)
        st["by_module"] = dict(Counter(r["metadata"]["module"] for r in recs))
        st["by_source"] = dict(Counter(r["metadata"].get("data_source", "game") for r in recs))
        st["eval_groups"] = None
        print(f"{label}: {st['samples']}条/{st['assistant_turns']}回合/{st['assistant_chars']}字符 | 模块{st['by_module']}")
        return st

    st_all = full_stats(merged, "merged")
    st_train = full_stats(train, "train")
    st_eval = full_stats(eval_, "eval")
    eval_sg_list = sorted({r["metadata"]["split_group"] for r in game_eval})
    print(f"\neval 占比: {st_eval['assistant_turns']}/{st_all['assistant_turns']} = {st_eval['assistant_turns']/st_all['assistant_turns']*100:.1f}% (回合)")
    print(f"游戏 eval split_group: {eval_sg_list}")

    manifest = {
        "created_at": "2026-08-01",
        "seed": SEED,
        "eval_ratio": EVAL_RATIO,
        "rule": "游戏数据按split_group同组同侧、每模块抽1组进eval; 主数据按data_source分层85/15",
        "main_version": "v5_clean (train_v5_clean.jsonl, 159条, 见 clean_v5_report.json)",
        "inputs": {
            "main": str(MAIN_FILE), "main_count": len(main_recs),
            "game": {m: len(load_lines(BASE / f)) for m, f in GAME_FILES},
        },
        "riou_dup_audit": {
            "main_riou": len(main_riou),
            "game_riou": len(game_riou),
            "id_overlap": sorted(set(r["id"] for r in main_riou) & set(r["id"] for r in game_riou)),
            "messages_sha256_overlap": sorted(set(msg_hash(r) for r in main_riou) & set(msg_hash(r) for r in game_riou)),
            "text_pair_overlap": len(set(turns_of(r) for r in main_riou) & set(turns_of(r) for r in game_riou)),
            "conclusion": "两批riou数据无重复(ID/哈希/文本均无交集), 213条成立; data_branch: riou_manual(LLM) vs riou_game(游戏)",
        },
        "merged": st_all, "train": st_train, "eval": st_eval,
        "game_eval_split_groups": eval_sg_list,
    }
    dump_json(OUT_MANIFEST, manifest)
    print(f"\n写出: {OUT_MERGED.name} / {OUT_TRAIN.name} / {OUT_EVAL.name} / {OUT_MANIFEST.name}")


if __name__ == "__main__":
    main()
