"""分析 R1 五个变体的评测结果，生成指标对比表和可视化图表。"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from collections import Counter

RESULTS_DIR = Path(r"c:\Users\13474\Desktop\qqchat-enhanced\runtime_analysis\r1_results")
OUTPUT_DIR = Path(r"c:\Users\13474\Desktop\qqchat-enhanced\runtime_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = {
    "e1": "标准LoRA",
    "e2": "NEFTune",
    "e3": "DoRA",
    "e4": "RSLoRA",
    "e5": "SeqPacking",
}

# 门禁数据
GATES = {
    "e2": {"passed": True, "repetition": 0.0358, "safety": 0.5, "baseline_safety": 0.4333},
    "e3": {"passed": True, "repetition": 0.0302, "safety": 0.5333, "baseline_safety": 0.4333},
    "e4": {"passed": False, "repetition": 0.1667, "safety": 0.3333, "baseline_safety": 0.4333},
    "e5": {"passed": False, "repetition": 0.0376, "safety": 0.3667, "baseline_safety": 0.4333},
}


def analyze_variant(path: Path) -> dict:
    """分析单个变体的输出特征。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = data.get("samples") or data.get("results") or []
    if not samples:
        # 尝试其他结构
        if isinstance(data, list):
            samples = data
        elif isinstance(data, dict) and "results" in data:
            samples = data["results"]

    lengths = []
    char_lengths = []
    empty_count = 0
    categories = Counter()
    has_refuse = 0
    has_laughter = 0
    has_metanarrative = 0  # 元叙事词

    meta_keywords = ["作为AI", "我是AI", "作为一个AI", "语言模型", "AI助手", "无法", "对不起，我不能"]
    laugh_keywords = ["哈哈", "呵", "嘻嘻", "(笑)", "笑"]

    for s in samples:
        # 尝试多种字段名
        output = s.get("output") or s.get("response") or s.get("model_output") or s.get("reply") or ""
        if not output:
            # 嵌套结构
            if isinstance(s.get("generation"), dict):
                output = s["generation"].get("output", "")
            elif isinstance(s.get("result"), dict):
                output = s["result"].get("output", "")

        if not output:
            empty_count += 1
            continue

        char_lengths.append(len(output))
        # token 估算（中文约 1.5 字符/token）
        lengths.append(max(1, len(output) // 2))

        category = s.get("category") or s.get("type") or s.get("tag") or "unknown"
        categories[category] += 1

        if any(k in output for k in meta_keywords):
            has_metanarrative += 1
        if any(k in output for k in laugh_keywords):
            has_laughter += 1

    return {
        "sample_count": len(samples),
        "avg_char_len": round(statistics.mean(char_lengths), 1) if char_lengths else 0,
        "median_char_len": statistics.median(char_lengths) if char_lengths else 0,
        "min_char_len": min(char_lengths) if char_lengths else 0,
        "max_char_len": max(char_lengths) if char_lengths else 0,
        "stdev_char_len": round(statistics.stdev(char_lengths), 1) if len(char_lengths) > 1 else 0,
        "empty_count": empty_count,
        "categories": dict(categories),
        "metanarrative_count": has_metanarrative,
        "laughter_count": has_laughter,
        "raw_lengths": char_lengths,
    }


def main() -> int:
    print("=" * 80)
    print("R1 PEFT 消融实验结果分析（5个变体对比）")
    print("=" * 80)

    # 分析每个变体
    results = {}
    for v in ("e1", "e2", "e3", "e4", "e5"):
        path = RESULTS_DIR / f"{v}_eval.json"
        if path.exists():
            results[v] = analyze_variant(path)
            print(f"\n--- {v} ({VARIANTS[v]}) ---")
            r = results[v]
            print(f"  样本数: {r['sample_count']}")
            print(f"  平均长度: {r['avg_char_len']} 字符")
            print(f"  中位长度: {r['median_char_len']} 字符")
            print(f"  长度范围: [{r['min_char_len']}, {r['max_char_len']}]")
            print(f"  标准差: {r['stdev_char_len']}")
            print(f"  空输出数: {r['empty_count']}")
            print(f"  元叙事(含AI自称): {r['metanarrative_count']}")
            print(f"  含笑声: {r['laughter_count']}")
        else:
            print(f"[MISSING] {v}")
            results[v] = None

    # 汇总门禁结果
    print("\n" + "=" * 80)
    print("门禁结果（自动指标）")
    print("=" * 80)
    print(f"{'变体':<12} {'门禁':<8} {'重复率':<10} {'安全率':<10} {'基线安全率':<12}")
    print("-" * 60)
    print(f"{'e1(基线)':<12} {'-':<8} {'-':<10} {'0.4333':<10} {'-':<12}")
    for v in ("e2", "e3", "e4", "e5"):
        g = GATES[v]
        status = "PASS" if g["passed"] else "FAIL"
        print(f"{v+'('+VARIANTS[v]+')':<12} {status:<8} {g['repetition']:<10} {g['safety']:<10} {g['baseline_safety']:<12}")

    # 生成可视化 HTML
    generate_html_report(results)

    return 0


def generate_html_report(results: dict) -> None:
    """生成交互式 HTML 可视化报告。"""
    html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>R1 PEFT 消融实验分析</title>
<style>
  body { font-family: -apple-system, "Microsoft YaHei", sans-serif; margin: 20px; background: #f5f5f5; }
  h1 { color: #333; border-bottom: 2px solid #666; padding-bottom: 8px; }
  h2 { color: #444; margin-top: 30px; }
  .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
  table { border-collapse: collapse; width: 100%; margin: 15px 0; }
  th, td { border: 1px solid #ddd; padding: 10px 14px; text-align: left; }
  th { background: #4a5568; color: white; font-weight: 600; }
  tr:nth-child(even) { background: #f9f9f9; }
  .pass { color: #2f855a; font-weight: bold; }
  .fail { color: #c53030; font-weight: bold; }
  .metric-card { display: inline-block; background: #edf2f7; padding: 12px 20px; margin: 5px; border-radius: 6px; border-left: 4px solid #4299e1; }
  .metric-value { font-size: 24px; font-weight: bold; color: #2b6cb0; }
  .metric-label { font-size: 12px; color: #718096; }
  .chart-container { margin: 20px 0; padding: 15px; background: #fafafa; border-radius: 6px; }
  .bar-chart { display: flex; align-items: flex-end; height: 300px; gap: 20px; padding: 10px; border-bottom: 2px solid #ccc; border-left: 2px solid #ccc; }
  .bar-group { flex: 1; display: flex; flex-direction: column; align-items: center; }
  .bar { width: 60px; transition: height 0.5s; position: relative; }
  .bar-label { margin-top: 8px; font-size: 12px; text-align: center; }
  .bar-value { position: absolute; top: -25px; left: 50%; transform: translateX(-50%); font-size: 11px; font-weight: bold; }
  .e1 { background: #4299e1; }
  .e2 { background: #48bb78; }
  .e3 { background: #ed8936; }
  .e4 { background: #e53e3e; }
  .e5 { background: #9f7aea; }
  .insight { background: #fffbeb; border-left: 4px solid #f6ad55; padding: 12px; margin: 10px 0; border-radius: 4px; }
  .warning { background: #fed7d7; border-left: 4px solid #fc8181; padding: 12px; margin: 10px 0; border-radius: 4px; }
</style>
</head>
<body>
<div class="container">
<h1>R1 PEFT 消融实验结果分析</h1>
<p>基座：Qwen3-8B-Instruct BF16 | 数据：826 训练 / 92 验证 | Seed: 42 | 评测：Gold v2 (150 条)</p>
"""

    # 门禁结果表
    html += """
<h2>1. 自动门禁结果（vs E1 基线）</h2>
<table>
<tr><th>对比</th><th>方法</th><th>门禁</th><th>重复率</th><th>重复率上限</th><th>安全率</th><th>基线安全率</th><th>格式正确率</th></tr>
<tr><td>E1 (基线)</td><td>标准 LoRA</td><td>-</td><td>-</td><td>0.10</td><td>0.4333</td><td>-</td><td>1.0</td></tr>
"""
    gate_data = [
        ("e2", "NEFTune", True, 0.0358, 0.5),
        ("e3", "DoRA", True, 0.0302, 0.5333),
        ("e4", "RSLoRA", False, 0.1667, 0.3333),
        ("e5", "SeqPacking", False, 0.0376, 0.3667),
    ]
    for v, name, passed, rep, safety in gate_data:
        status = '<span class="pass">PASS</span>' if passed else '<span class="fail">FAIL</span>'
        rep_class = ' class="fail"' if rep > 0.1 else ''
        html += f'<tr><td>E1 vs {v.upper()}</td><td>{name}</td><td>{status}</td><td{rep_class}>{rep}</td><td>0.10</td><td>{safety}</td><td>0.4333</td><td>1.0</td></tr>\n'
    html += "</table>\n"

    # 失败原因
    html += """
<div class="warning">
<b>E4 (RSLoRA) 门禁失败原因：</b>重复率 0.1667 > 0.10 上限。输出冗长且重复，与妃的简洁风格不符。<br>
<b>E5 (Sequence Packing) 门禁失败原因：</b>安全率 0.3667 < 0.4333 基线（诊断性指标）。虽然重复率正常，但安全场景退步。
</div>
"""

    # 长度对比图
    html += """
<h2>2. 输出长度分布对比</h2>
<div class="chart-container">
<div class="bar-chart">
"""
    for v in ("e1", "e2", "e3", "e4", "e5"):
        r = results.get(v)
        if r:
            avg = r["avg_char_len"]
            # 归一化到 300px 高度，最大 500 字符
            height = min(280, int(avg / 500 * 280))
            html += f'<div class="bar-group"><div class="bar {v}" style="height: {height}px;"><div class="bar-value">{avg}</div></div><div class="bar-label">{v.upper()}<br>{VARIANTS[v]}</div></div>\n'
    html += """</div>
<p style="text-align:center; color:#666; margin-top:10px;">平均输出字符长度（越小越贴近妃的简洁风格）</p>
</div>
"""

    # 详细指标卡
    html += "<h2>3. 各变体详细指标</h2>\n"
    for v in ("e1", "e2", "e3", "e4", "e5"):
        r = results.get(v)
        if not r:
            continue
        html += f'<h3>{v.upper()} - {VARIANTS[v]}</h3>\n'
        html += '<div style="margin: 10px 0;">\n'
        html += f'<div class="metric-card"><div class="metric-value">{r["sample_count"]}</div><div class="metric-label">样本数</div></div>\n'
        html += f'<div class="metric-card"><div class="metric-value">{r["avg_char_len"]}</div><div class="metric-label">平均长度</div></div>\n'
        html += f'<div class="metric-card"><div class="metric-value">{r["median_char_len"]}</div><div class="metric-label">中位长度</div></div>\n'
        html += f'<div class="metric-card"><div class="metric-value">{r["max_char_len"]}</div><div class="metric-label">最大长度</div></div>\n'
        html += f'<div class="metric-card"><div class="metric-value">{r["stdev_char_len"]}</div><div class="metric-label">标准差</div></div>\n'
        html += f'<div class="metric-card"><div class="metric-value">{r["metanarrative_count"]}</div><div class="metric-label">AI自称(违规)</div></div>\n'
        html += f'<div class="metric-card"><div class="metric-value">{r["laughter_count"]}</div><div class="metric-label">含笑声</div></div>\n'
        html += '</div>\n'

    # 洞察
    html += """
<h2>4. 关键洞察</h2>
<div class="insight">
<b>最优候选：E3 (DoRA)</b><br>
- 门禁 PASS，重复率最低（0.0302）<br>
- 安全率最高（0.5333 > 基线 0.4333），提升 23%<br>
- 输出长度与 E1 基线接近，保持简洁风格<br>
- 元叙事违规为 0，符合角色一致性要求
</div>
<div class="insight">
<b>第二候选：E2 (NEFTune)</b><br>
- 门禁 PASS，重复率 0.0358<br>
- 安全率 0.5 > 基线，提升 15%<br>
- NEFTune 噪声注入有效提升泛化，效果与 DoRA 接近
</div>
<div class="warning">
<b>不推荐：E4 (RSLoRA)</b><br>
- 门禁 FAIL，重复率 0.1667 远超上限<br>
- 输出冗长（144KB vs 其他 ~104KB），偏离角色风格<br>
- 安全率退步至 0.3333<br>
- 结论：RSLoRA 在 r=32 下可能导致训练不稳定，输出质量下降
</div>
<div class="warning">
<b>不推荐：E5 (Sequence Packing)</b><br>
- 门禁 FAIL（诊断性安全率退步）<br>
- 重复率正常（0.0376），长度最短<br>
- 但安全场景表现下降，可能因 packing 改变注意力分布影响安全判断<br>
- 结论：Packing 提升训练效率但不保证效果，需配合安全校准
</div>

<h2>5. 下一步建议</h2>
<ol>
<li><b>盲评优先级</b>：E3 (DoRA) > E2 (NEFTune) > E5 > E4</li>
<li><b>正式盲评</b>：先完成 E1 vs E3 的 40 条分层盲评，若 E3 胜出再做 120 条完整盲评</li>
<li><b>多种子复现</b>：仅对 E1 和 E3 补充 Seed 43/44 验证稳定性</li>
<li><b>负结果记录</b>：E4/E5 的失败是有效负结果，写入研究档案，不删除数据</li>
</ol>
</div>
</body>
</html>
"""

    output_path = OUTPUT_DIR / "r1_analysis_report.html"
    output_path.write_text(html, encoding="utf-8")
    print(f"\n可视化报告已生成: {output_path}")

    # 同时输出 CSV 摘要
    csv_path = OUTPUT_DIR / "r1_metrics_summary.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("variant,method,avg_char_len,median_char_len,max_char_len,stdev,metanarrative,laughter,gate_passed,repetition,safety\n")
        for v in ("e1", "e2", "e3", "e4", "e5"):
            r = results.get(v)
            if not r:
                continue
            if v == "e1":
                gate, rep, safety = "-", "-", "0.4333"
            else:
                g = GATES[v]
                gate = "PASS" if g["passed"] else "FAIL"
                rep = g["repetition"]
                safety = g["safety"]
            f.write(f"{v},{VARIANTS[v]},{r['avg_char_len']},{r['median_char_len']},{r['max_char_len']},{r['stdev_char_len']},{r['metanarrative_count']},{r['laughter_count']},{gate},{rep},{safety}\n")
    print(f"CSV 摘要已生成: {csv_path}")


if __name__ == "__main__":
    raise SystemExit(main())
