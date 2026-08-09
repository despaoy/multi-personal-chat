"""在服务器上运行 E2'' v2 评估（含推理惩罚 rp=1.15, fp=0.3）。

用法:
  python scripts/run_e2pp_eval_v2.py
  python scripts/run_e2pp_eval_v2.py --no-penalty   # 不加惩罚（用于对比）
  python scripts/run_e2pp_eval_v2.py --mock          # mock 模式（不调用 vLLM）

说明:
  在服务器上调用 backend/evaluation/character_benchmark.py，
  使用 kisaki_gold_set_v1.json 作为评估集，
  使用 kisaki_knowledge_base.json 作为 RAG 文档。
  结果保存为 kisaki_e2pp_rag_eval_rp115_v2.json（v2 表示重训后版本）。
"""
import argparse
import sys
import time

from remote_config import connect_ssh

ROOT = "/home/szw/lhm2"
PROJECT = f"{ROOT}/qqchat-enhanced"
PYTHON = f"{ROOT}/envs/qqchat-gpu-qwen3/bin/python"

DATASET = "backend/evaluation/kisaki_gold_set_v1.json"
RAG_DOCS = "backend/data/character_dialogues/kisaki_knowledge_base.json"
MODEL = "kisaki-e2pp-rag"
VLLM_PORT = 8002

# 结果输出路径（v2 = 重训后）
RESULT_NO_PENALTY = "backend/data/character_dialogues/experiments/results/kisaki_e2pp_rag_eval_v2.json"
RESULT_WITH_PENALTY = "backend/data/character_dialogues/experiments/results/kisaki_e2pp_rag_eval_rp115_v2.json"

# 默认推理参数
DEFAULT_RP = 1.15
DEFAULT_FP = 0.3
DEFAULT_MAX_TOKENS = 256
DEFAULT_TIMEOUT = 120


def run(cli, cmd, timeout=300):
    """执行命令并实时获取输出。"""
    print(f"$ {cmd[:200]}{'...' if len(cmd) > 200 else ''}")
    _, stdout = cli.exec_command(cmd, timeout=timeout, get_pty=True)
    output = []
    while True:
        line = stdout.readline()
        if not line:
            break
        line = line.rstrip()
        if line:
            print(f"  {line}")
            output.append(line)
    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(description="运行 E2'' v2 评估")
    parser.add_argument("--no-penalty", action="store_true",
                        help="不加推理惩罚（默认加 rp=1.15, fp=0.3）")
    parser.add_argument("--mock", action="store_true", help="mock 模式（不调用 vLLM）")
    parser.add_argument("--rp", type=float, default=DEFAULT_RP)
    parser.add_argument("--fp", type=float, default=DEFAULT_FP)
    parser.add_argument("--limit", type=int, default=0, help="只评估前 N 条（0=全部）")
    parser.add_argument("--output", type=str, default="", help="自定义输出路径")
    args = parser.parse_args()

    print("=" * 60)
    print("E2'' v2 评估（重训后）")
    print("=" * 60)
    print(f"惩罚: {'无' if args.no_penalty else f'rp={args.rp}, fp={args.fp}'}")
    print(f"mock: {args.mock}")

    cli = connect_ssh()

    # 1. 检查 vLLM 是否在运行
    print("\n[1] 检查 vLLM 状态...")
    cmd = f"curl -fsS --max-time 5 http://127.0.0.1:{VLLM_PORT}/v1/models 2>&1 || echo CURL_FAIL"
    _, stdout, _ = cli.exec_command(cmd, timeout=10)
    out = stdout.read().decode()
    if MODEL in out and "CURL_FAIL" not in out:
        print(f"  ✅ vLLM 正常，模型 {MODEL} 已加载")
    else:
        print(f"  ❌ vLLM 未就绪或模型未加载")
        print(f"  响应: {out[:200]}")
        print("  请先运行: python scripts/restart_vllm_e2pp.py")
        cli.close()
        sys.exit(1)

    # 2. 确定输出路径
    if args.output:
        remote_output = args.output
    elif args.no_penalty:
        remote_output = RESULT_NO_PENALTY
    else:
        remote_output = RESULT_WITH_PENALTY

    print(f"\n[2] 输出路径: {remote_output}")

    # 3. 构造评估命令
    eval_cmd = (
        f"cd {PROJECT} && "
        f"{PYTHON} backend/evaluation/character_benchmark.py "
        f"--dataset {DATASET} "
        f"--rag-documents {RAG_DOCS} "
        f"--model {MODEL} "
        f"--output {remote_output} "
        f"--base-url http://127.0.0.1:{VLLM_PORT} "
        f"--max-tokens {DEFAULT_MAX_TOKENS} "
        f"--timeout {DEFAULT_TIMEOUT} "
        f"--gpu 0"
    )
    if not args.no_penalty:
        eval_cmd += f" --repetition-penalty {args.rp} --frequency-penalty {args.fp}"
    if args.mock:
        eval_cmd += " --mock"
    if args.limit > 0:
        eval_cmd += f" --limit {args.limit}"

    # 4. 运行评估
    print(f"\n[3] 运行评估（约 2-5 分钟）...")
    started = time.time()
    _, stdout, stderr = cli.exec_command(eval_cmd, timeout=600, get_pty=True)
    while True:
        line = stdout.readline()
        if not line:
            break
        line = line.rstrip()
        if line:
            print(f"  {line}")
    elapsed = time.time() - started
    print(f"\n  评估耗时: {elapsed:.1f}s")

    # 5. 验证结果
    print(f"\n[4] 验证结果...")
    verify_cmd = (
        f"python3 -c \""
        f"import json; "
        f"d=json.load(open('{remote_output}')); "
        f"m=d['metrics']; "
        f"print(f'total={{m[\\\"total\\\"]}}, success={{m[\\\"success\\\"]}}'); "
        f"print(f'format_correct_rate={{m[\\\"format_correct_rate\\\"]}}'); "
        f"print(f'avg_output_chars={{m[\\\"average_output_chars\\\"]}}'); "
        f"print(f'distinct_1={{m[\\\"distinct_1\\\"]}}, distinct_2={{m[\\\"distinct_2\\\"]}}'); "
        f"print(f'avg_repetition_rate={{m[\\\"avg_repetition_rate\\\"]}}'); "
        f"print(f'avg_latency_ms={{m[\\\"average_latency_ms\\\"]}}'); "
        f"cats=m['by_category']; "
        f"print('by_category:'); "
        f"[print(f'  {{c}}: chars={{cats[c][\\\"average_output_chars\\\"]}}, safety={{cats[c][\\\"safety_pass_rate\\\"]}}, citation={{cats[c].get(\\\"citation_accuracy\\\",\\\"-\\\")}}') for c in sorted(cats)]"
        f"\""
    )
    _, stdout, _ = cli.exec_command(verify_cmd, timeout=15)
    out = stdout.read().decode()
    print(out)

    # 6. 下载结果到本地
    print(f"\n[5] 下载结果到本地...")
    local_output = remote_output  # 保持相同相对路径
    # 本地对应路径
    local_path = local_output.replace("backend/", "backend/")
    sftp = cli.open_sftp()
    remote_full = f"{PROJECT}/{remote_output}"
    try:
        sftp.get(remote_full, local_path)
        print(f"  ✅ 已下载: {local_path}")
    except Exception as e:
        print(f"  ⚠️  下载失败: {e}")
    sftp.close()

    print("\n" + "=" * 60)
    print("✅ E2'' v2 评估完成")
    print(f"   结果: {remote_output}")
    print("=" * 60)

    cli.close()


if __name__ == "__main__":
    main()
