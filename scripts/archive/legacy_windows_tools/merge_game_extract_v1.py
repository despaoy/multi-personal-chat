# -*- coding: utf-8 -*-
"""游戏提取数据全量合并 + 按 split_group 划分 train/eval + 模块统计。

输入(28条):
  - game_famous_candidates.jsonl     (通用名场面 8条)
  - game_famous_paper_state.jsonl   (纸上存在专用 2条)
  - game_riou_candidates.jsonl      (理央 12条)
  - game_rikata_candidates.jsonl    (彼方 6条)

规则:
  - 同 split_group 的全部样本同进 train 或同进 eval (防剧情泄漏)
  - seed=42 固定, 贪心抽取 split_group 使 eval assistant 回合占比 ~12%
  - 原子写入: 先写 tmp 再 rename
输出:
  - game_extract_merged.jsonl (28条, 含 module 标记)
  - game_extract_train.jsonl / game_extract_eval.jsonl
  - game_extract_manifest.json (种子/规则/统计/文件hash)

token 估算: Qwen 系 tokenizer 中文约 0.65 token/字符, 占比以字符数口径为准。
"""
import hashlib
import json
import random
from collections import Counter, OrderedDict
from pathlib import Path

BASE = Path(r"c:\Users\13474\Desktop\qqchat-enhanced\backend\data\character_dialogues\experiments")
SEED = 42
EVAL_RATIO = 0.12
TOKEN_PER_CHAR = 0.65  # 估算系数(标注用)

FILES = [
    ("famous", "game_famous_candidates.jsonl"),
    ("paper", "game_famous_paper_state.jsonl"),
    ("riou", "game_riou_candidates.jsonl"),
    ("rikata", "game_rikata_candidates.jsonl"),
]
OUT_MERGED = BASE / "game_extract_merged.jsonl"
OUT_TRAIN = BASE / "game_extract_train.jsonl"
OUT_EVAL = BASE / "game_extract_eval.jsonl"
OUT_MANIFEST = BASE / "game_extract_manifest.json"


def load():
    samples = []
    for module, fname in FILES:
        p = BASE / fname
        recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        for r in recs:
            r["metadata"] = dict(r["metadata"])
            r["metadata"]["module"] = module
            r["metadata"]["n_assistant_turns"] = len(r["messages"]) // 2
            samples.append(r)
        print(f"{module}: {len(recs)} 条")
    return samples


def assistant_chars(sample):
    return sum(len(m["content"]) for m in sample["messages"] if m["role"] == "assistant")


def split_by_group(samples):
    groups = OrderedDict()
    for s in samples:
        sg = s["metadata"]["split_group"]
        groups.setdefault(sg, []).append(s)
    return groups


def greedy_split(groups, total_turns, seed=SEED, ratio=EVAL_RATIO):
    rng = random.Random(seed)
    items = list(groups.items())
    rng.shuffle(items)
    target = total_turns * ratio
    eval_groups, eval_turns = [], 0
    for sg, recs in items:
        if eval_groups and eval_turns >= target:
            break
        eval_groups.append((sg, recs))
        eval_turns += sum(r["metadata"]["n_assistant_turns"] for r in recs)
    train_items = [(sg, recs) for sg, recs in items if (sg, recs) not in eval_groups]
    return train_items, eval_groups


def sha256(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def write_atomic(path, lines):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    tmp.rename(path)


def stats(recs, label):
    n = len(recs)
    turns = sum(r["metadata"]["n_assistant_turns"] for r in recs)
    chars = sum(assistant_chars(r) for r in recs)
    tokens = int(chars * TOKEN_PER_CHAR)
    by_module = Counter(r["metadata"]["module"] for r in recs)
    print(f"{label}: {n}条 | assistant {turns}回合 | {chars}字符 | ~{tokens}token | 模块 {dict(by_module)}")
    return {"samples": n, "assistant_turns": turns, "assistant_chars": chars,
            "assistant_tokens_est": tokens, "by_module": dict(by_module)}


def main():
    samples = load()
    assert len(samples) == 28, f"期望28条, 实际{len(samples)}"
    assert len({s["id"] for s in samples}) == 28, "id 有重复"

    groups = split_by_group(samples)
    total_turns = sum(r["metadata"]["n_assistant_turns"] for r in samples)
    print(f"\nsplit_group 共 {len(groups)} 组, 总 assistant 回合 {total_turns}")
    for sg, recs in groups.items():
        turns = sum(r["metadata"]["n_assistant_turns"] for r in recs)
        print(f"  {sg}: {len(recs)}条/{turns}回合")

    train_items, eval_groups = greedy_split(groups, total_turns)
    train_recs = [r for _, recs in train_items for r in recs]
    eval_recs = [r for _, recs in eval_groups for r in recs]

    print("\n=== 划分结果 ===")
    st_all = stats(samples, "merged")
    st_train = stats(train_recs, "train")
    st_eval = stats(eval_recs, "eval")
    print(f"\neval 回合占比: {st_eval['assistant_turns']}/{total_turns} = "
          f"{st_eval['assistant_turns']/total_turns*100:.1f}%")

    # 写文件(保持原有顺序: 按模块+原文件顺序)
    def dump(recs):
        return [json.dumps(r, ensure_ascii=False) for r in recs]
    write_atomic(OUT_MERGED, dump(samples))
    write_atomic(OUT_TRAIN, dump(train_recs))
    write_atomic(OUT_EVAL, dump(eval_recs))

    manifest = {
        "created_at": "2026-08-01",
        "seed": SEED,
        "eval_ratio_target": EVAL_RATIO,
        "token_estimation": f"chars * {TOKEN_PER_CHAR} (Qwen系近似, 占比以字符口径为准)",
        "split_rule": "同 split_group 全部同进 train 或同进 eval; seed 固定贪心抽取至 eval 回合占比~12%",
        "input_files": {m: str(BASE / f) for m, f in FILES},
        "inputs_sha256": {m: sha256(BASE / f) for m, f in FILES},
        "merged": st_all,
        "train": st_train,
        "eval": st_eval,
        "eval_groups": [sg for sg, _ in eval_groups],
        "outputs_sha256": {
            "merged": sha256(OUT_MERGED), "train": sha256(OUT_TRAIN), "eval": sha256(OUT_EVAL),
        },
    }
    write_atomic(OUT_MANIFEST, [json.dumps(manifest, ensure_ascii=False, indent=2)])
    print(f"\n写出: {OUT_MERGED.name} / {OUT_TRAIN.name} / {OUT_EVAL.name} / {OUT_MANIFEST.name}")


if __name__ == "__main__":
    main()
