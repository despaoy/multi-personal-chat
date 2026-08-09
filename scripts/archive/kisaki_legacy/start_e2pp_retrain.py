"""在服务器上启动 E2'' 重训。

用法:
  python scripts/start_e2pp_retrain.py
  python scripts/start_e2pp_retrain.py --check  # 仅检查训练数据状态

说明:
  使用 tsukiyashiro_kisaki_e2pp_rag.json 配置，
  训练数据已更新为 899 条的清洗+补充版本。
  输出到 /home/szw/lhm2/runtime/loras/kisaki/e2pp_rag_r32/final/
"""
import argparse
import sys

from remote_config import connect_ssh

ROOT = "/home/szw/lhm2"
PROJECT = f"{ROOT}/qqchat-enhanced"
ENV_DIR = f"{ROOT}/envs/qqchat-gpu-qwen3"
PYTHON = f"{ENV_DIR}/bin/python"

CONFIG = f"{PROJECT}/backend/data/character_dialogues/experiments/configs/tsukiyashiro_kisaki_e2pp_rag.json"
TRAIN_DATA = f"{PROJECT}/backend/data/character_dialogues/tsukiyashiro_kisaki_train_e2pp.json"
OUTPUT_DIR = f"{ROOT}/runtime/loras/kisaki/e2pp_rag_r32"
LOGFILE = f"{ROOT}/runtime/logs/kisaki_e2pp_rag.log"
PIDFILE = f"{ROOT}/runtime/kisaki_e2pp_rag.pid"


def run(cli, cmd, timeout=30):
    _, stdout, stderr = cli.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace").strip()
    err = stderr.read().decode("utf-8", errors="replace").strip()
    return out, err


def main():
    parser = argparse.ArgumentParser(description="启动 E2'' 重训")
    parser.add_argument("--check", action="store_true", help="仅检查状态，不启动训练")
    args = parser.parse_args()

    print("=" * 60)
    print("E2'' 重训启动器")
    print("=" * 60)

    cli = connect_ssh()

    # 1. 检查训练数据
    print("\n[1] 检查训练数据...")
    out, _ = run(cli, f"python3 -c \"import json; d=json.load(open('{TRAIN_DATA}')); print(f'total={{len(d)}}')\"")
    if out:
        print(f"  ✅ 训练数据: {out}")
    else:
        print(f"  ❌ 无法读取训练数据")
        cli.close()
        sys.exit(1)

    out, _ = run(cli, f"ls -lh {TRAIN_DATA}")
    print(f"  {out.strip()}")

    # 2. 检查是否有已运行的训练
    print("\n[2] 检查运行中的训练...")
    out, _ = run(cli, f"cat {PIDFILE} 2>/dev/null && ps -p $(cat {PIDFILE}) -o pid,etime,cmd --no-headers 2>/dev/null || echo 'no running training'")
    if "no running" not in out:
        print(f"  ⚠️  已有训练运行中:\n  {out}")
        print("  请先等待完成或手动终止")
        cli.close()
        sys.exit(1)
    print("  ✅ 无训练运行中")

    if args.check:
        print("\n所有检查通过，可以启动训练。")
        cli.close()
        return

    # 3. 备份旧日志
    print("\n[3] 备份旧日志...")
    run(cli, f"mv {LOGFILE} {LOGFILE}.bak 2>/dev/null; echo done")

    # 4. 启动训练
    print("\n[4] 启动训练...")
    train_cmd = (
        f"cd {PROJECT} && "
        f"export PYTHONPATH={PROJECT}/backend:$PYTHONPATH && "
        f"nohup env CUDA_VISIBLE_DEVICES=0 {PYTHON} -m backend.training.trainer "
        f"--config {CONFIG} "
        f">{LOGFILE} 2>&1 </dev/null &"
    )
    print(f"  命令: {train_cmd[:120]}...")
    out, err = run(cli, train_cmd, timeout=10)

    # 5. 获取 PID
    import time
    time.sleep(3)
    out, _ = run(cli, f"pgrep -f 'backend.training.trainer' | head -1 || echo NONE")
    pid = out.strip()
    if pid != "NONE":
        run(cli, f"echo '{pid}' > {PIDFILE}")
        print(f"  ✅ 训练已启动, PID={pid}")
    else:
        print(f"  ⚠️ 未检测到进程，检查日志...")
        out, _ = run(cli, f"tail -20 {LOGFILE}")
        print(f"  {out}")

    # 6. 显示日志末尾
    print("\n[5] 训练日志末尾（等待5秒初始化）...")
    time.sleep(5)
    out, _ = run(cli, f"tail -15 {LOGFILE}")
    print(out)

    print("\n" + "=" * 60)
    print("✅ E2'' 重训已启动")
    print(f"   PID: {pid}")
    print(f"   配置: {CONFIG}")
    print(f"   数据: 899 条（RAG改写 + 11条知识补充）")
    print(f"   输出: {OUTPUT_DIR}/final/")
    print(f"   日志: {LOGFILE}")
    print(f"   状态: python scripts/check_e2pp_training_status.py")
    print("=" * 60)

    cli.close()


if __name__ == "__main__":
    main()
