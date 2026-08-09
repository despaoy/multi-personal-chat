"""生成月社妃 LoRA 五方对比报告：E1 / E2 / E2' / E2''(旧) / E2''(新重训)。

用法:
  python scripts/generate_e2pp_comparison_report.py
  python scripts/generate_e2pp_comparison_report.py --output docs/research/KISAKI_E2PP_V2_COMPARISON.md

输入文件（均在本地）:
  - backend/data/character_dialogues/experiments/results/kisaki_e1_baseline_eval_v3.json
  - backend/data/character_dialogues/experiments/results/kisaki_e2_neftune_eval.json
  - backend/data/character_dialogues/experiments/results/kisaki_e2p_neftune_eval.json
  - backend/data/character_dialogues/experiments/results/kisaki_e2pp_rag_eval_rp115.json        (旧 E2'' + rp=1.15)
  - backend/data/character_dialogues/experiments/results/kisaki_e2pp_rag_eval_rp115_v2.json     (新 E2'' + rp=1.15)

输出: Markdown 格式的对比报告
"""
# NOTE: 此脚本输出/读取的资产已归档至 archive/legacy_v3_superseded/ 或 docs/research/archive/，保留脚本作历史可追溯性证据。
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

# 默认输入文件
RESULTS_DIR = Path("backend/data/character_dialogues/experiments/results")
DEFAULT_OUTPUT = Path("docs/research/KISAKI_E2PP_V2_COMPARISON.md")

EVAL_FILES = {
    "E1": RESULTS_DIR / "kisaki_e1_baseline_eval_v3.json",
    "E2": RESULTS_DIR / "kisaki_e2_neftune_eval.json",
    "E2'": RESULTS_DIR / "kisaki_e2p_neftune_eval.json",
    "E2''(旧)": RESULTS_DIR / "kisaki_e2pp_rag_eval_rp115.json",
    "E2''(新)": RESULTS_DIR / "kisaki_e2pp_rag_eval_rp115_v2.json",
}

# 实验描述
EXPERIMENT_INFO = {
    "E1": {
        "data": "game_extraction 721 + llm_v3 108 + e2_manual 25 = 854 条",
        "key": "r=32, alpha=64, 无 NEFTune",
    },
    "E2": {
        "data": "E1 + e2_safety_manual 10 = 864 条",
        "key": "+ NEFTune α=2.5, safety 数据增强",
    },
    "E2'": {
        "data": "E2 + e2p_safety_manual 20 = 874 条",
        "key": "+ 扩充 safety 数据（角色化软拒绝）",
    },
    "E2''(旧)": {
        "data": "E2' + e2pp_rag_manual 25 = 899 条（未经审查）",
        "key": "+ RAG 引用数据 + rp=1.15 + fp=0.3",
    },
    "E2''(新)": {
        "data": "Current audited E2PP corpus = 899 samples",
        "key": "+ RAG 引用数据 + rp=1.15 + fp=0.3",
    },
}


def load_eval(path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_metric(data, key, default="—"):
    if data is None:
        return default
    return data["metrics"].get(key, default)


def get_category_metric(data, category, key, default="—"):
    if data is None:
        return default
    cats = data["metrics"].get("by_category", {})
    if category not in cats:
        return default
    return cats[category].get(key, default)


def fmt(v, suffix="", precision=2):
    if v == "—" or v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{precision}f}{suffix}"
    if isinstance(v, int):
        return f"{v}{suffix}"
    return str(v)


def generate_report(eval_data, output_path):
    lines = []
    lines.append("# 月社妃 LoRA E1/E2/E2'/E2''(旧)/E2''(新) 五方对比报告")
    lines.append("")
    lines.append(f"**生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**评估集**：`kisaki_gold_set_v1.json`（100 条，5 类别）")
    lines.append(f"**评估脚本**：`backend/evaluation/character_benchmark.py`（修复版）")
    lines.append(f"**推理参数**：rp=1.15, fp=0.3, temperature=0.0, max_tokens=256")
    lines.append("")
    lines.append(
        "**Known limitation**: one normalized Gold prompt overlaps the historical training "
        "artifacts. Treat reported quality metrics as exploratory; future dataset rebuilds "
        "exclude all Gold prompts automatically."
    )
    lines.append("")
    lines.append("**Input artifact provenance**")
    lines.append("")
    lines.append("| Variant | Result file | SHA-256 |")
    lines.append("|---|---|---|")
    for variant, path in EVAL_FILES.items():
        digest = file_sha256(path) if path.exists() else "missing"
        lines.append(f"| {variant} | `{path.as_posix()}` | `{digest}` |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 一、实验设计
    lines.append("## 一、实验设计")
    lines.append("")
    lines.append("| 实验 | 训练数据 | 关键变量 | 训练量 |")
    lines.append("|------|----------|----------|--------|")
    for name, info in EXPERIMENT_INFO.items():
        train_count = info["data"].split("=")[-1].strip() if "=" in info["data"] else info["data"]
        lines.append(f"| **{name}** | {info['data']} | {info['key']} | {train_count} |")
    lines.append("")
    lines.append("**统一训练参数**：r=32, alpha=64, dropout=0.1, lr=2e-4, 3 epochs, batch=1, "
                 "grad_accum=8, max_seq_len=1024, bf16, gradient_checkpointing, "
                 "target_modules=7 linear（q/k/v/o/gate/up/down_proj）, NEFTune α=2.5（E2+）")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 二、核心指标对比
    lines.append("## 二、核心指标对比")
    lines.append("")
    headers = list(EVAL_FILES.keys())
    lines.append("| 指标 | " + " | ".join(headers) + " |")
    lines.append("|------|" + "|".join(["-----"] * len(headers)) + "|")

    metric_rows = [
        ("format_correct_rate", "format_correct_rate", 4),
        ("average_output_chars", "average_output_chars", 2),
        ("average_output_tokens", "average_output_tokens", 2),
        ("distinct_1", "distinct_1", 4),
        ("distinct_2", "distinct_2", 4),
        ("avg_repetition_rate", "avg_repetition_rate", 4),
        ("average_latency_ms", "average_latency_ms", 0),
        ("p95_latency_ms", "p95_latency_ms", 0),
    ]
    for label, key, prec in metric_rows:
        row = [label]
        for h in headers:
            v = get_metric(eval_data[h], key)
            row.append(fmt(v, precision=prec))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 三、按类别对比
    lines.append("## 三、按类别细项对比")
    lines.append("")
    categories = ["persona", "factual", "rag_grounded", "safety", "multiturn"]
    cat_labels = {
        "persona": "Persona（角色一致性）",
        "factual": "Factual（事实准确性）",
        "rag_grounded": "RAG Grounded（引用接地性）",
        "safety": "Safety（安全拒答）",
        "multiturn": "Multiturn（多轮对话）",
    }

    for cat in categories:
        lines.append(f"### 3.{categories.index(cat)+1} {cat_labels[cat]}")
        lines.append("")
        lines.append("| 指标 | " + " | ".join(headers) + " |")
        lines.append("|------|" + "|".join(["-----"] * len(headers)) + "|")

        cat_metrics = [
            ("format_correct_rate", 4),
            ("average_output_chars", 2),
            ("average_output_tokens", 2),
            ("avg_latency_ms", 0),
            ("safety_pass_rate", 4),
            ("citation_accuracy", 4),
        ]
        if cat == "rag_grounded":
            cat_metrics.append(("citation_ratio_avg", 4))

        for label, prec in cat_metrics:
            row = [label]
            for h in headers:
                v = get_category_metric(eval_data[h], cat, label)
                row.append(fmt(v, precision=prec))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    lines.append("---")
    lines.append("")

    # 四、E2'' 新旧对比
    lines.append("## 四、E2'' 新旧对比（重训效果验证）")
    lines.append("")
    if eval_data["E2''(新)"] is None:
        lines.append("⚠️ E2''(新) 评估结果尚未生成。请先运行 `scripts/run_e2pp_eval_v2.py`。")
    else:
        old = eval_data["E2''(旧)"]
        new = eval_data["E2''(新)"]
        lines.append("| 指标 | E2''(旧, 899条) | E2''(新, 899条) | 变化 | 结论 |")
        lines.append("|------|----------------|----------------|------|------|")

        comparison_metrics = [
            ("format_correct_rate", 4, "越高越好"),
            ("average_output_chars", 2, "接近 E2' 水平(44.63)最佳"),
            ("distinct_1", 4, "越高越好"),
            ("distinct_2", 4, "越高越好"),
            ("avg_repetition_rate", 4, "越低越好"),
            ("average_latency_ms", 0, "越低越好"),
            ("p95_latency_ms", 0, "越低越好"),
        ]
        for key, prec, direction in comparison_metrics:
            old_v = get_metric(old, key)
            new_v = get_metric(new, key)
            if old_v == "—" or new_v == "—":
                delta = "—"
                conclusion = "—"
            else:
                delta_val = new_v - old_v
                delta = f"{delta_val:+.{prec}f}"
                if "越低越好" in direction:
                    conclusion = "✅ 改善" if delta_val < 0 else ("持平" if delta_val == 0 else "⚠️ 退化")
                else:
                    conclusion = "✅ 改善" if delta_val > 0 else ("持平" if delta_val == 0 else "⚠️ 退化")
            lines.append(f"| {key} | {fmt(old_v, precision=prec)} | {fmt(new_v, precision=prec)} | {delta} | {conclusion} |")

        # 按类别对比
        lines.append("")
        lines.append("### 4.1 按类别对比")
        lines.append("")
        for cat in categories:
            lines.append(f"**{cat_labels[cat]}**:")
            lines.append("")
            lines.append("| 指标 | E2''(旧) | E2''(新) | 变化 |")
            lines.append("|------|---------|---------|------|")
            cat_metrics = [
                ("average_output_chars", 2),
                ("safety_pass_rate", 4),
                ("citation_accuracy", 4),
            ]
            if cat == "rag_grounded":
                cat_metrics.append(("citation_ratio_avg", 4))
            for label, prec in cat_metrics:
                old_v = get_category_metric(old, cat, label)
                new_v = get_category_metric(new, cat, label)
                if old_v == "—" or new_v == "—":
                    delta = "—"
                else:
                    delta = f"{new_v - old_v:+.{prec}f}"
                lines.append(f"| {label} | {fmt(old_v, precision=prec)} | {fmt(new_v, precision=prec)} | {delta} |")
            lines.append("")

    lines.append("---")
    lines.append("")

    # 五、关键发现
    lines.append("## 五、关键发现")
    lines.append("")
    if eval_data["E2''(新)"] is None:
        lines.append("（待 E2''(新) 评估完成后补充）")
    else:
        new = eval_data["E2''(新)"]
        old = eval_data["E2''(旧)"]

        # 计算关键变化
        new_rep = get_metric(new, "avg_repetition_rate")
        old_rep = get_metric(old, "avg_repetition_rate")
        new_d1 = get_metric(new, "distinct_1")
        old_d1 = get_metric(old, "distinct_1")
        new_d2 = get_metric(new, "distinct_2")
        old_d2 = get_metric(old, "distinct_2")
        new_cit = get_category_metric(new, "rag_grounded", "citation_accuracy")
        old_cit = get_category_metric(old, "rag_grounded", "citation_accuracy")
        new_safety = get_category_metric(new, "safety", "safety_pass_rate")
        old_safety = get_category_metric(old, "safety", "safety_pass_rate")

        lines.append("### 5.1 重训带来的变化")
        lines.append("")
        lines.append("1. **Data provenance**: current E2PP training artifact contains 899 samples; use artifact hashes below for reproducibility")
        if isinstance(new_rep, (int, float)) and isinstance(old_rep, (int, float)):
            if new_rep < old_rep:
                lines.append(f"2. **重复率**：{old_rep} → {new_rep}（{(new_rep-old_rep)/old_rep*100:+.1f}%）✅")
            else:
                lines.append(f"2. **重复率**：{old_rep} → {new_rep}（{(new_rep-old_rep)/old_rep*100:+.1f}%）⚠️")
        if isinstance(new_d2, (int, float)) and isinstance(old_d2, (int, float)):
            if new_d2 > old_d2:
                lines.append(f"3. **多样性 (distinct_2)**：{old_d2} → {new_d2}（+{(new_d2-old_d2)/old_d2*100:.1f}%）✅")
            else:
                lines.append(f"3. **多样性 (distinct_2)**：{old_d2} → {new_d2}（{(new_d2-old_d2)/old_d2*100:+.1f}%）")
        if isinstance(new_cit, (int, float)) and isinstance(old_cit, (int, float)):
            lines.append(f"4. **RAG citation**：{old_cit} → {new_cit}（{new_cit-old_cit:+.2f}）")
        if isinstance(new_safety, (int, float)) and isinstance(old_safety, (int, float)):
            lines.append(f"5. **Safety 通过率**：{old_safety} → {new_safety}（{new_safety-old_safety:+.2f}）")
        lines.append("")

    lines.append("---")
    lines.append("")

    # 六、结论
    lines.append("## 六、结论")
    lines.append("")
    if eval_data["E2''(新)"] is None:
        lines.append("（待评估完成后补充）")
    else:
        lines.append("E2'' 重训实验验证了数据治理对模型质量的影响：")
        lines.append("")
        lines.append("- 四轮深度审查删除 11 条问题样本（LLM 答非所问 + 输出错误）")
        lines.append("- 重新设计 62 条样本（超长精简 + 短 Prompt 修正 + 跨场景合并错误修复）")
        lines.append("- game_extraction 错误率 0.4%，数据质量达到可重训水平")
        lines.append("")
        lines.append("详细对比见上方表格。")
    lines.append("")

    # 写入文件
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ 报告已生成: {output_path}")
    print(f"   行数: {len(lines)}")


def main():
    parser = argparse.ArgumentParser(description="生成五方对比报告")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"输出路径（默认: {DEFAULT_OUTPUT}）")
    parser.add_argument("--check-only", action="store_true",
                        help="只检查输入文件是否存在，不生成报告")
    args = parser.parse_args()

    print("=" * 60)
    print("五方对比报告生成器")
    print("=" * 60)

    # 检查输入文件
    print("\n输入文件状态:")
    eval_data = {}
    all_ok = True
    for name, path in EVAL_FILES.items():
        exists = path.exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {name}: {path}")
        if exists:
            eval_data[name] = load_eval(path)
        else:
            eval_data[name] = None
            if name in ("E1", "E2'", "E2''(旧)", "E2''(新)"):
                all_ok = False  # 这些是必需的

    if args.check_only:
        print(f"\n必需文件全部存在: {'是' if all_ok else '否'}")
        return

    if not all_ok:
        print("\n⚠️  部分必需文件缺失，仍生成报告（缺失部分用 — 表示）")

    print(f"\n生成报告: {args.output}")
    generate_report(eval_data, args.output)


if __name__ == "__main__":
    main()
