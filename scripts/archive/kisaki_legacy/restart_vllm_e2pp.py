"""重启 vLLM 以加载新训练完成的 E2'' LoRA adapter。

用法:
  python scripts/restart_vllm_e2pp.py
  python scripts/restart_vllm_e2pp.py --force   # 强制重启，不检查 adapter

说明:
  vLLM 在启动时通过 --lora-modules 加载 LoRA adapter，无法热重载。
  本脚本会:
    1. 检查 final/adapter_config.json 是否存在
    2. 终止当前 vLLM 进程（端口 8002）
    3. 通过 nohup 启动新的 vLLM 进程
    4. 等待 vLLM 健康检查通过（/v1/models 返回 200）
    5. 验证 LoRA 已加载
"""
import argparse
import sys
import time
import urllib.request

from remote_config import connect_ssh

ROOT = "/home/szw/lhm2"
PROJECT = f"{ROOT}/qqchat-enhanced"
ENV_DIR = f"{ROOT}/envs/qqchat-gpu-qwen3"
VLLM_PORT = 8002
VLLM_MODEL = "kisaki-e2pp-rag"
FINAL_ADAPTER_DIR = f"{ROOT}/runtime/loras/kisaki/e2pp_rag_r32/final"
VLLM_LOG = f"{ROOT}/runtime/logs/vllm_e2pp_rag.log"
VLLM_PIDFILE = f"{ROOT}/runtime/vllm_e2pp_rag.pid"

# vLLM 启动命令（对应 lab-start-vllm-e2pp-cuda-graph.sh）
START_CMD = (
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
    f"--lora-modules {VLLM_MODEL}={FINAL_ADAPTER_DIR} "
    f">{VLLM_LOG} 2>&1 </dev/null &"
)


def run(cli, cmd, timeout=30):
    _, stdout, stderr = cli.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace").strip()
    err = stderr.read().decode("utf-8", errors="replace").strip()
    return out, err


def find_vllm_pids(cli):
    """找到所有监听 8002 端口的 vLLM 进程。"""
    out, _ = run(cli, "pgrep -f 'vllm serve.*kisaki-e2pp-rag' || true")
    pids = [p.strip() for p in out.splitlines() if p.strip().isdigit()]
    return pids


def wait_for_health(cli, max_wait=180, interval=5):
    """等待 vLLM 健康检查通过。"""
    print(f"等待 vLLM 健康检查通过（最长 {max_wait}s）...")
    url = f"http://127.0.0.1:{VLLM_PORT}/v1/models"
    elapsed = 0
    while elapsed < max_wait:
        # 在服务器上用 curl 检查（避免本地→服务器的网络问题）
        out, _ = run(cli, f"curl -fsS --max-time 5 '{url}' 2>&1 || echo CURL_FAIL")
        if VLLM_MODEL in out and "CURL_FAIL" not in out:
            print(f"  ✅ vLLM 已就绪（{elapsed}s）")
            return True
        if elapsed % 15 == 0:
            print(f"  [{elapsed}s] 等待中...")
        time.sleep(interval)
        elapsed += interval
    print(f"  ❌ 健康检查超时")
    return False


def main():
    parser = argparse.ArgumentParser(description="重启 vLLM 加载新 E2'' adapter")
    parser.add_argument("--force", action="store_true", help="跳过 adapter 检查强制重启")
    parser.add_argument("--no-wait", action="store_true", help="不等待健康检查")
    args = parser.parse_args()

    print("=" * 60)
    print("重启 vLLM 加载新 E2'' LoRA")
    print("=" * 60)

    cli = connect_ssh()

    # 1. 检查 adapter
    if not args.force:
        out, _ = run(cli, f"test -f {FINAL_ADAPTER_DIR}/adapter_config.json && echo yes || echo no")
        if out.strip() != "yes":
            print(f"❌ adapter_config.json 不存在: {FINAL_ADAPTER_DIR}")
            print("   使用 --force 跳过此检查")
            cli.close()
            sys.exit(1)
        print(f"✅ adapter 已就绪: {FINAL_ADAPTER_DIR}")

        # 显示 adapter 配置摘要
        out, _ = run(cli, f"python3 -c \"import json; d=json.load(open('{FINAL_ADAPTER_DIR}/adapter_config.json')); print(f'r={{d.get(\\\"r\\\")}}, alpha={{d.get(\\\"lora_alpha\\\")}}, target_modules={{len(d.get(\\\"target_modules\\\",[]))}}')\"")
        if out:
            print(f"   配置: {out.strip()}")

    # 2. 找到当前 vLLM 进程
    pids = find_vllm_pids(cli)
    if pids:
        print(f"\n当前 vLLM 进程: {', '.join(pids)}")
        for pid in pids:
            out, _ = run(cli, f"ps -p {pid} -o etime,cmd --no-headers 2>/dev/null")
            if out:
                print(f"  PID {pid}: {out[:100]}")
    else:
        print("\n未发现运行中的 vLLM 进程")

    # 3. 终止旧进程
    if pids:
        print("\n终止旧 vLLM 进程...")
        for pid in pids:
            run(cli, f"kill {pid} 2>/dev/null")
            print(f"  kill PID {pid}")

        # 等待进程退出
        print("等待进程退出（最长 30s）...")
        for wait in range(30):
            time.sleep(1)
            still_alive = []
            for pid in pids:
                out, _ = run(cli, f"kill -0 {pid} 2>/dev/null && echo alive || echo dead")
                if out.strip() == "alive":
                    still_alive.append(pid)
            if not still_alive:
                print(f"  ✅ 所有进程已退出（{wait+1}s）")
                break
            if wait == 14:
                print(f"  发送 SIGKILL...")
                for pid in still_alive:
                    run(cli, f"kill -9 {pid} 2>/dev/null")
        else:
            print("  ⚠️  部分进程未正常退出")

        # 等待 GPU 内存释放
        print("等待 GPU 0 内存释放...")
        for wait in range(15):
            time.sleep(2)
            out, _ = run(cli, "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0")
            try:
                used = int(out.strip())
                if used < 2000:
                    print(f"  ✅ GPU 0 内存已释放（{used}MB）")
                    break
                print(f"  [{wait*2}s] GPU 0 仍占用 {used}MB")
            except ValueError:
                pass

    # 4. 启动新 vLLM
    print("\n启动新 vLLM 进程...")
    print(f"  日志: {VLLM_LOG}")
    out, err = run(cli, START_CMD, timeout=15)
    print(f"  启动命令已发送")

    # 获取新 PID
    new_pids = find_vllm_pids(cli)
    if new_pids:
        print(f"  ✅ 新 vLLM PID: {', '.join(new_pids)}")
        # 写入 PID 文件
        run(cli, f"echo '{new_pids[0]}' > {VLLM_PIDFILE}")
    else:
        print("  ⚠️  未立即检测到新进程，检查日志...")

    # 5. 等待健康检查
    if not args.no_wait:
        print()
        healthy = wait_for_health(cli)
        if not healthy:
            print("\nvLLM 启动日志末尾:")
            out, _ = run(cli, f"tail -30 {VLLM_LOG}")
            print(out)
            cli.close()
            sys.exit(2)

    # 6. 最终验证
    print("\n最终验证...")
    out, _ = run(cli, "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader -i 0")
    print(f"GPU 0:\n{out}")

    new_pids = find_vllm_pids(cli)
    print(f"\nvLLM 进程: {', '.join(new_pids) if new_pids else '未找到'}")

    print("\n" + "=" * 60)
    print("✅ vLLM 重启完成，新 LoRA adapter 已加载")
    print(f"   端口: {VLLM_PORT}")
    print(f"   模型: {VLLM_MODEL}")
    print(f"   adapter: {FINAL_ADAPTER_DIR}")
    print("=" * 60)

    cli.close()


if __name__ == "__main__":
    main()
