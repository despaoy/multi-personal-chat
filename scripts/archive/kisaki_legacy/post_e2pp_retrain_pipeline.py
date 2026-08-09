"""E2'' 重训后自动管道：等待训练→重启vLLM→运行评估→分析问题样本。

用法:
  python scripts/post_e2pp_retrain_pipeline.py
  python scripts/post_e2pp_retrain_pipeline.py --skip-wait     # 训练已完成，直接执行后续步骤
  python scripts/post_e2pp_retrain_pipeline.py --no-vllm       # 跳过 vLLM 重启（如果已在运行）
  python scripts/post_e2pp_retrain_pipeline.py --status-only   # 仅检查状态
"""
# NOTE: 此脚本输出/读取的资产已归档至 archive/legacy_v3_superseded/ 或 docs/research/archive/，保留脚本作历史可追溯性证据。
import argparse
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime

from remote_config import connect_ssh

ROOT = "/home/szw/lhm2"
PROJECT = f"{ROOT}/qqchat-enhanced"
ENV_DIR = f"{ROOT}/envs/qqchat-gpu-qwen3"
PYTHON = f"{ENV_DIR}/bin/python"

# 训练相关
LOGFILE = f"{ROOT}/runtime/logs/kisaki_e2pp_rag.log"
OUTPUT_DIR = f"{ROOT}/runtime/loras/kisaki/e2pp_rag_r32"
FINAL_ADAPTER = f"{OUTPUT_DIR}/final/adapter_config.json"

# vLLM 相关
VLLM_PORT = 8002
VLLM_MODEL = "kisaki-e2pp-rag"
VLLM_LOG = f"{ROOT}/runtime/logs/vllm_e2pp_rag.log"
VLLM_PIDFILE = f"{ROOT}/runtime/vllm_e2pp_rag.pid"

# 评估相关
DATASET = "backend/evaluation/kisaki_gold_set_v1.json"
RAG_DOCS = "backend/data/character_dialogues/kisaki_knowledge_base.json"
RESULT_FILE = "backend/data/character_dialogues/experiments/results/kisaki_e2pp_rag_eval_rp115_v2.json"
RP = 1.15
FP = 0.3

# 本地输出
LOCAL_RESULT = "backend/data/character_dialogues/experiments/results/kisaki_e2pp_rag_eval_rp115_v2.json"
LOCAL_SAMPLES = "docs/research/KISAKI_E2PP_V2_PROBLEM_SAMPLES.md"


def run(cli, cmd, timeout=30):
    _, stdout, stderr = cli.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace").strip()
    err = stderr.read().decode("utf-8", errors="replace").strip()
    return out, err


def run_stream(cli, cmd, timeout=600):
    """执行命令并实时流式输出。"""
    _, stdout, stderr = cli.exec_command(cmd, timeout=timeout, get_pty=True)
    output_lines = []
    while True:
        line = stdout.readline()
        if not line:
            break
        line = line.rstrip()
        if line:
            print(f"  {line}")
            output_lines.append(line)
    return "\n".join(output_lines)


def check_training(cli):
    """检查训练状态。返回 (is_complete, progress_info)。"""
    info = {
        "running": False,
        "progress": None,
        "adapter_ready": False,
        "log_tail": [],
    }

    # 检查 adapter
    out, _ = run(cli, f"test -f {FINAL_ADAPTER} && echo yes || echo no")
    info["adapter_ready"] = out.strip() == "yes"
    if info["adapter_ready"]:
        # 获取文件时间
        out, _ = run(cli, f"stat -c '%Y' {FINAL_ADAPTER} 2>/dev/null")
        if out:
            info["adapter_time"] = int(out.strip())

    # 检查进程
    out, _ = run(cli, "pgrep -f 'backend.training.trainer' | head -5")
    pids = [p for p in out.splitlines() if p.strip().isdigit()]
    info["running"] = len(pids) > 0
    info["pids"] = pids

    # 读取日志
    out, _ = run(cli, f"tail -20 {LOGFILE} 2>/dev/null")
    if out:
        info["log_tail"] = out.splitlines()

    # 解析进度
    for line in reversed(info["log_tail"]):
        m = re.search(r"(\d+)/(\d+)\s*\[", line)
        if m:
            info["progress"] = {"step": int(m.group(1)), "total": int(m.group(2))}
            info["progress"]["pct"] = round(int(m.group(1)) / int(m.group(2)) * 100, 1)
            break

    # 判断完成：进程已退出 + adapter 就绪
    info["complete"] = (not info["running"]) and info["adapter_ready"]

    return info


def print_training_status(info):
    """打印训练状态。"""
    print("=" * 60)
    print(f"训练状态 - {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)
    print(f"  运行中: {'是' if info['running'] else '否'}")
    print(f"  Adapter: {'就绪' if info['adapter_ready'] else '未就绪'}")
    if info.get("adapter_time"):
        print(f"  Adapter时间: {datetime.fromtimestamp(info['adapter_time']).strftime('%Y-%m-%d %H:%M:%S')}")
    if info.get("progress"):
        p = info["progress"]
        eta = ""
        if p["step"] > 0 and p["total"] > p["step"]:
            remaining = (p["total"] - p["step"]) * 9.5 / 60  # ~9.5s/step
            eta = f"  预计剩余: {remaining:.0f}分钟"
        print(f"  进度: {p['step']}/{p['total']} ({p['pct']}%){eta}")
    if info.get("pids"):
        print(f"  PID: {', '.join(info['pids'])}")

    # 显示最后几行日志
    print("  日志末尾:")
    for line in info["log_tail"][-3:]:
        print(f"    {line}")
    print("=" * 60)


# ── vLLM ──────────────────────────────────────────────

def find_vllm_pids(cli):
    out, _ = run(cli, "pgrep -f 'vllm serve' || true")
    return [p.strip() for p in out.splitlines() if p.strip().isdigit()]


def kill_vllm(cli):
    """终止旧 vLLM 进程并等待 GPU 内存释放。"""
    pids = find_vllm_pids(cli)
    if not pids:
        print("无运行中的 vLLM 进程")
        return True

    print(f"终止 vLLM 进程: {', '.join(pids)}")
    for pid in pids:
        run(cli, f"kill {pid} 2>/dev/null")

    # 等待退出
    for wait in range(30):
        time.sleep(1)
        alive = []
        for pid in pids:
            out, _ = run(cli, f"kill -0 {pid} 2>/dev/null && echo alive || echo dead")
            if out.strip() == "alive":
                alive.append(pid)
        if not alive:
            print(f"  ✅ 已退出 ({wait+1}s)")
            break
        if wait == 14:
            for pid in alive:
                run(cli, f"kill -9 {pid} 2>/dev/null")
    else:
        print("  ⚠️  进程未完全退出")

    # 等待 GPU 释放
    print("等待 GPU 0 内存释放...")
    for i in range(15):
        time.sleep(2)
        out, _ = run(cli, "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0")
        try:
            used = int(out.strip())
            if used < 2000:
                print(f"  ✅ GPU 0 内存已释放 ({used}MB)")
                return True
        except ValueError:
            pass
    return True


def start_vllm(cli):
    """启动 vLLM 加载新 LoRA adapter。"""
    start_cmd = (
        f"cd {PROJECT} && "
        f"export C_INCLUDE_PATH={ENV_DIR}/include/python3.11 && "
        f"export CPLUS_INCLUDE_PATH={ENV_DIR}/include/python3.11 && "
        f"export TRITON_CACHE_DIR={ROOT}/runtime/cache/triton && "
        f"mkdir -p $TRITON_CACHE_DIR && "
        f"export CUDA_HOME=/usr/local/cuda && "
        f"nohup env CUDA_VISIBLE_DEVICES=0 {ENV_DIR}/bin/vllm serve "
        f"{ROOT}/runtime/models/Qwen3-8B-Instruct "
        f"--served-model-name {VLLM_MODEL} "
        f"--host 127.0.0.1 --port {VLLM_PORT} "
        f"--gpu-memory-utilization 0.90 --max-model-len 4096 "
        f"--enable-lora --max-loras 1 --max-lora-rank 32 "
        f"--lora-modules {VLLM_MODEL}={FINAL_ADAPTER} "
        f">{VLLM_LOG} 2>&1 </dev/null &"
    )

    print(f"启动 vLLM...")
    out, err = run(cli, start_cmd, timeout=15)

    new_pids = find_vllm_pids(cli)
    if new_pids:
        run(cli, f"echo '{new_pids[0]}' > {VLLM_PIDFILE}")
        print(f"  PID: {new_pids[0]}")

    # 等待健康检查
    print("等待 vLLM 就绪（最长 180s）...")
    url = f"http://127.0.0.1:{VLLM_PORT}/v1/models"
    for elapsed in range(0, 180, 5):
        time.sleep(5)
        out, _ = run(cli, f"curl -fsS --max-time 5 '{url}' 2>&1 || echo CURL_FAIL")
        if VLLM_MODEL in out and "CURL_FAIL" not in out:
            print(f"  ✅ vLLM 已就绪 ({elapsed+5}s)")
            return True
        if elapsed % 15 == 0:
            print(f"  [{elapsed+5}s] 等待中...")
    print(f"  ❌ 健康检查超时")
    return False


# ── 评估 ──────────────────────────────────────────────

def run_eval(cli):
    """运行 v2 评估。"""
    eval_cmd = (
        f"cd {PROJECT} && "
        f"{PYTHON} backend/evaluation/character_benchmark.py "
        f"--dataset {DATASET} "
        f"--rag-documents {RAG_DOCS} "
        f"--model {VLLM_MODEL} "
        f"--output {RESULT_FILE} "
        f"--base-url http://127.0.0.1:{VLLM_PORT} "
        f"--max-tokens 256 "
        f"--timeout 120 "
        f"--gpu 0 "
        f"--repetition-penalty {RP} "
        f"--frequency-penalty {FP}"
    )

    print(f"\n运行评估（rp={RP}, fp={FP}）...")
    started = time.time()
    out = run_stream(cli, eval_cmd, timeout=600)
    elapsed = time.time() - started
    print(f"  耗时: {elapsed:.1f}s")
    return out


# ── 分析问题样本 ──────────────────────────────────────

def analyze_results(cli):
    """分析评估结果，提取问题样本。"""
    print("\n" + "=" * 60)
    print("分析评估结果")

    # 读取结果
    _, stdout, _ = cli.exec_command(
        f"python3 -c \"import json; print(json.dumps(json.load(open('{RESULT_FILE}')), ensure_ascii=False, indent=2))\"",
        timeout=15
    )
    raw = stdout.read().decode("utf-8", errors="replace").strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print("  ❌ 无法解析结果 JSON")
        return None

    metrics = data.get("metrics", {})
    results = data.get("results", [])

    # 打印指标
    print(f"\n指标摘要:")
    print(f"  总样本: {metrics.get('total', '?')}")
    print(f"  成功: {metrics.get('success', '?')}")
    print(f"  格式正确: {metrics.get('format_correct_rate', '?')}")
    print(f"  平均输出字符: {metrics.get('average_output_chars', '?')}")
    print(f"  重复率: {metrics.get('avg_repetition_rate', '?')}")
    print(f"  平均延迟: {metrics.get('average_latency_ms', '?')}ms")

    cats = metrics.get("by_category", {})
    if cats:
        print(f"\n按类别:")
        for c in sorted(cats):
            cat = cats[c]
            print(f"  {c}: chars={cat.get('average_output_chars')}, "
                  f"safety={cat.get('safety_pass_rate')}, "
                  f"citation={cat.get('citation_accuracy', '-')}")

    # ── 提取问题样本 ──
    problems = []
    for r in results:
        output = (r.get("response", "") or "").strip()
        if not output:
            continue

        issues = []
        category = r.get("category", "")

        # 1. AI自指
        ai_refs = ["作为AI", "我是AI", "作为人工智能", "作为语言模型", "我不能", "我无法回答"]
        for ref in ai_refs:
            if ref in output:
                issues.append(f"AI自指: \"{ref}\"")
                break

        # 2. 过长（大于100字）
        if len(output) > 100:
            issues.append(f"过长: {len(output)}字")

        # 3. 统计类回答
        if re.search(r"共.*条|占比.*%|平均.*字|总计|统计|约.*条", output):
            issues.append("统计类回答")

        # 4. 第三人称客观（角色漂移）
        if re.search(r"月社妃.*是|她是.*角色|月社妃.*性格", output):
            issues.append("第三人称角色描述")

        # 5. 哈哈/嘿嘿（禁用口癖）
        if "哈哈" in output or "嘿嘿" in output:
            issues.append("禁用口癖（哈哈/嘿嘿）")

        # 6. 过短（少于5字且非拒答）
        if len(output) < 5 and "拒绝" not in output and "不方便" not in output:
            issues.append(f"过短: {len(output)}字")

        # 7. RAG类：仍带引用标签
        if category == "rag" and re.search(r"\[文档\d+\]", output):
            issues.append("RAG仍带引用标签")

        if issues:
            problems.append({
                "id": r.get("id", "?"),
                "category": category,
                "question": r.get("question", ""),
                "response": output,
                "issues": issues,
                "gold_answer_2": r.get("gold_answer_2", ""),
            })

    # 按问题严重性排序
    priorities = {"AI自指": 0, "第三人称角色描述": 1, "RAG仍带引用标签": 2,
                  "禁用口癖（哈哈/嘿嘿）": 2, "统计类回答": 3, "过长": 4, "过短": 5}

    def sort_key(p):
        scores = [priorities.get(i.split(":")[0], 5) for i in p["issues"]]
        return min(scores) if scores else 5

    problems.sort(key=sort_key)

    # 也提取一些无问题但有特点的样本供审查
    normal_samples = []
    for r in results:
        output = (r.get("response", "") or "").strip()
        if not output:
            continue
        has_issue = any(
            (ref in output) for refs in [
                ["作为AI", "我是AI", "作为人工智能", "作为语言模型", "我不能", "我无法回答"],
            ] for ref in refs
        )
        if has_issue:
            continue
        if len(output) > 100:
            continue
        if "哈哈" in output or "嘿嘿" in output:
            continue
        if re.search(r"共.*条|占比.*%|平均.*字", output):
            continue
        normal_samples.append({
            "id": r.get("id", "?"),
            "category": r.get("category", ""),
            "question": r.get("question", ""),
            "response": output,
        })

    return {
        "metrics": metrics,
        "problems": problems,
        "normal_samples": normal_samples,
        "total_results": len(results),
    }


def generate_report(analysis):
    """生成问题样本报告。"""
    problems = analysis["problems"]
    normal = analysis["normal_samples"]

    lines = [
        "# E2'' 重训评估 - 问题样本审查",
        "",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 指标摘要",
        "",
    ]

    m = analysis["metrics"]
    lines.append(f"- 总样本: {m.get('total', '?')}")
    lines.append(f"- 成功率: {m.get('success', '?')}/{m.get('total', '?')}")
    lines.append(f"- 平均输出字符: {m.get('average_output_chars', '?')}")
    lines.append(f"- 重复率: {m.get('avg_repetition_rate', '?')}")
    lines.append(f"- 平均延迟: {m.get('average_latency_ms', '?')}ms")
    lines.append(f"- 问题样本数: {len(problems)}")

    cats = m.get("by_category", {})
    if cats:
        lines.append("")
        lines.append("| 类别 | 字符数 | 安全通过 | 引用准确 |")
        lines.append("|------|--------|----------|----------|")
        for c in sorted(cats):
            cat = cats[c]
            lines.append(f"| {c} | {cat.get('average_output_chars')} | "
                        f"{cat.get('safety_pass_rate')} | {cat.get('citation_accuracy', '-')} |")

    if problems:
        lines.append("")
        lines.append(f"## 问题样本（{len(problems)}条）")
        lines.append("")

        for i, p in enumerate(problems, 1):
            lines.append(f"### #{i} [{p['category']}] {p['id']}")
            lines.append(f"**问题**: {', '.join(p['issues'])}")
            lines.append(f"**提问**: {p['question']}")
            lines.append(f"**回复**: {p['response']}")
            if p.get("gold_answer_2"):
                gold = p["gold_answer_2"]
                if len(gold) > 200:
                    gold = gold[:200] + "..."
                lines.append(f"**Gold参考**: {gold}")
            lines.append("")

    if normal:
        lines.append(f"## 正常样本抽查（前10条）")
        lines.append("")
        for i, s in enumerate(normal[:10], 1):
            lines.append(f"### #{i} [{s['category']}] {s['id']}")
            lines.append(f"**提问**: {s['question']}")
            lines.append(f"**回复**: {s['response']}")
            lines.append("")

    report = "\n".join(lines)

    # 保存到本地
    with open(LOCAL_SAMPLES, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n✅ 问题样本报告已保存: {LOCAL_SAMPLES}")
    print(f"   问题样本: {len(problems)} 条")
    print(f"   正常抽查: {len(normal[:10])} 条")

    return report


# ── 主流程 ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="E2'' 后训练自动管道")
    parser.add_argument("--skip-wait", action="store_true", help="跳过等待训练（训练已完成）")
    parser.add_argument("--no-vllm", action="store_true", help="跳过 vLLM 重启")
    parser.add_argument("--no-eval", action="store_true", help="跳过评估（仅分析已有结果）")
    parser.add_argument("--status-only", action="store_true", help="仅检查训练状态")
    parser.add_argument("--poll-interval", type=int, default=30, help="轮询间隔（秒）")
    parser.add_argument("--max-wait", type=int, default=7200, help="最长等待（秒）")
    args = parser.parse_args()

    cli = connect_ssh()

    # ── 阶段1: 等待训练完成 ──
    if not args.skip_wait:
        print("=" * 60)
        print("阶段1: 等待训练完成")
        print("=" * 60)

        elapsed = 0
        last_step = 0
        while elapsed < args.max_wait:
            info = check_training(cli)
            print_training_status(info)

            if info["complete"] and info.get("progress"):
                # 额外验证：确保不是假完成（训练过少）
                if info["progress"]["step"] >= 290:  # 至少95%
                    print(f"\n✅ 训练已完成！{info['progress']['step']}/{info['progress']['total']}步")
                    break
                elif info["progress"]["step"] < 50:
                    print(f"\n⚠️  训练步数过少（{info['progress']['step']}步），可能异常退出")
                    # 继续等待（可能是日志读取延迟）
                else:
                    print(f"\n⚠️  训练在 {info['progress']['step']}/{info['progress']['total']} 步退出，检查日志...")
                    out, _ = run(cli, f"tail -40 {LOGFILE}")
                    print(out)
                    if "CUDA out of memory" in out:
                        print("OOM! 需要手动处理")
                        cli.close()
                        sys.exit(3)
                    break

            # 卡住检测（训练进程消失但步数没变）
            if info["progress"]:
                if info["progress"]["step"] == last_step and not info["running"]:
                    # 卡住了
                    if info["progress"]["step"] < 290:
                        print(f"\n⚠️  训练在 {info['progress']['step']}步步停止，无运行进程。")
                        out, _ = run(cli, f"tail -40 {LOGFILE}")
                        print(f"\n完整日志尾部:")
                        print(out)
                        # 检查是否有错误
                        if "error" in out.lower() or "exception" in out.lower():
                            print("\n❌ 训练出错，check log above")
                            cli.close()
                            sys.exit(3)
                        break
                last_step = info["progress"]["step"]

            if args.status_only:
                cli.close()
                return

            time.sleep(args.poll_interval)
            elapsed += args.poll_interval
        else:
            print(f"\n⚠️  等待超时 ({args.max_wait}s)")
            cli.close()
            sys.exit(2)
    else:
        print("跳过训练等待（--skip-wait）")
        info = check_training(cli)
        print_training_status(info)

    if args.status_only:
        cli.close()
        return

    # ── 阶段2: 重启 vLLM ──
    if not args.no_vllm:
        print("\n" + "=" * 60)
        print("阶段2: 重启 vLLM")
        print("=" * 60)
        kill_vllm(cli)
        if not start_vllm(cli):
            print("vLLM 启动失败!")
            out, _ = run(cli, f"tail -30 {VLLM_LOG}")
            print(out)
            cli.close()
            sys.exit(4)
    else:
        print("跳过 vLLM 重启（--no-vllm）")

    # ── 阶段3: 运行评估 ──
    if not args.no_eval:
        print("\n" + "=" * 60)
        print("阶段3: 运行评估")
        print("=" * 60)
        run_eval(cli)
    else:
        print("跳过评估（--no-eval）")

    # ── 阶段4: 分析结果 ──
    print("\n" + "=" * 60)
    print("阶段4: 分析结果")
    print("=" * 60)
    analysis = analyze_results(cli)
    if analysis:
        report = generate_report(analysis)

        # 下载结果到本地
        print("\n下载评估结果到本地...")
        sftp = cli.open_sftp()
        try:
            sftp.get(f"{PROJECT}/{RESULT_FILE}", LOCAL_RESULT)
            print(f"  ✅ {LOCAL_RESULT}")
        except Exception as e:
            print(f"  ⚠️  下载失败: {e}")
        sftp.close()

        # 打印问题样本摘要
        if analysis["problems"]:
            print(f"\n{'='*60}")
            print(f"⚠️  发现 {len(analysis['problems'])} 条问题样本:")
            for p in analysis["problems"]:
                print(f"  [{p['category']}] {p['id']}: {', '.join(p['issues'])}")
                print(f"    Q: {p['question'][:60]}...")
                print(f"    A: {p['response'][:80]}...")
                print()
        else:
            print("\n🎉 未发现问题样本！")
    else:
        print("  ❌ 分析失败")

    print("\n" + "=" * 60)
    print("✅ 管道完成")
    print(f"   评估结果: {LOCAL_RESULT}")
    print(f"   问题报告: {LOCAL_SAMPLES}")
    print("=" * 60)

    cli.close()


if __name__ == "__main__":
    main()
