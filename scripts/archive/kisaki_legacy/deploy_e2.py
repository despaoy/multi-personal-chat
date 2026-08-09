"""Deploy E2 research artifacts to a configured lab host.

Required authentication: LAB_SSH_KEY or LAB_PASS. Optional connection settings:
LAB_HOST, LAB_USER, and LAB_SSH_AUTO_ADD_HOST_KEY. Never store credentials here.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import paramiko

from remote_config import HOST, USER, connect_ssh

REMOTE_ROOT = "/home/szw/lhm2/qqchat-enhanced"

# (本地路径, 远程相对路径)
UPLOADS = [
    ("backend/data/character_dialogues/experiments/configs/tsukiyashiro_kisaki_e2_neftune.json",
     "backend/data/character_dialogues/experiments/configs/tsukiyashiro_kisaki_e2_neftune.json"),
    ("backend/data/character_dialogues/kisaki_e2_supplement.json",
     "backend/data/character_dialogues/kisaki_e2_supplement.json"),
    ("backend/data/character_dialogues/tsukiyashiro_kisaki_train_e2.json",
     "backend/data/character_dialogues/tsukiyashiro_kisaki_train_e2.json"),
    ("scripts/lab-start-kisaki-e2-training.sh",
     "scripts/lab-start-kisaki-e2-training.sh"),
    ("scripts/lab-start-vllm-e2-lora.sh",
     "scripts/lab-start-vllm-e2-lora.sh"),
    ("backend/evaluation/character_benchmark.py",
     "backend/evaluation/character_benchmark.py"),
]


def run_remote(ssh: paramiko.SSHClient, cmd: str, timeout: int = 30) -> tuple[int, str]:
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace").strip()
    err = stderr.read().decode("utf-8", errors="replace").strip()
    code = stdout.channel.recv_exit_status()
    return code, out if out else err


def main():
    project = Path(__file__).resolve().parent.parent

    print(f"[1/5] 连接服务器 {USER}@{HOST} ...")
    ssh = connect_ssh()
    print("      连接成功")

    # 2. 上传文件
    print(f"[2/5] 上传 {len(UPLOADS)} 个文件 ...")
    sftp = ssh.open_sftp()
    for local_rel, remote_rel in UPLOADS:
        local = project / local_rel
        remote = f"{REMOTE_ROOT}/{remote_rel}"
        remote_dir = os.path.dirname(remote)
        # 确保远程目录存在
        run_remote(ssh, f"mkdir -p '{remote_dir}'")
        sftp.put(str(local), remote)
        print(f"      ✓ {local_rel}")
    sftp.close()
    print("      上传完成")

    # 3. 停止 E1 vLLM（端口 8002）
    print("[3/5] 停止 E1 vLLM 服务（端口 8002）...")
    code, out = run_remote(ssh, "fuser 8002/tcp 2>/dev/null")
    if out:
        pids = out.split()
        for pid in pids:
            run_remote(ssh, f"kill -9 {pid} 2>/dev/null")
            print(f"      killed pid={pid}")
        time.sleep(2)
        print("      E1 vLLM 已停止")
    else:
        print("      端口 8002 无进程（可能已停止）")

    # 4. 检查 GPU 状态
    print("[4/5] 检查 GPU 状态 ...")
    code, out = run_remote(ssh, "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits")
    print(f"      GPU 状态:\n      {out.replace(chr(10), chr(10)+'      ')}")

    # 检查 GPU 0 是否可用（训练用）
    code, gpu0_used = run_remote(ssh, "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0")
    gpu0_used = int(gpu0_used.strip())
    if gpu0_used > 5000:
        print(f"      ⚠️ GPU 0 已使用 {gpu0_used}MB (>5000MB)，训练可能无法启动")
        print("      等待 GPU 0 释放或手动处理后再启动训练")
        ssh.close()
        sys.exit(2)
    else:
        print(f"      GPU 0 可用 ({gpu0_used}MB used)")

    # 5. 启动 E2 训练
    print("[5/5] 启动 E2 训练 ...")
    run_remote(ssh, f"chmod +x {REMOTE_ROOT}/scripts/lab-start-kisaki-e2-training.sh")

    # 用 setsid 启动后台训练，避免 SSH 断开影响
    code, out = run_remote(ssh, f"bash {REMOTE_ROOT}/scripts/lab-start-kisaki-e2-training.sh", timeout=10)
    print(f"      {out}")

    # 等待几秒检查训练是否真的启动
    time.sleep(3)
    code, pid_check = run_remote(ssh, "cat /home/szw/lhm2/runtime/kisaki_e2_neftune.pid 2>/dev/null")
    if pid_check:
        code, alive = run_remote(ssh, f"kill -0 {pid_check} 2>/dev/null && echo ALIVE || echo DEAD")
        print(f"      训练进程 PID={pid_check} 状态={alive}")
    else:
        print("      ⚠️ 未找到 PID 文件，检查日志：")
        code, log_tail = run_remote(ssh, "tail -20 /home/szw/lhm2/runtime/logs/kisaki_e2_neftune.log 2>/dev/null")
        print(f"      {log_tail}")

    ssh.close()
    print("\n✅ E2 部署完成")
    print(f"   训练日志: tail -f /home/szw/lhm2/runtime/logs/kisaki_e2_neftune.log")
    print(f"   训练配置: {REMOTE_ROOT}/backend/data/character_dialogues/experiments/configs/tsukiyashiro_kisaki_e2_neftune.json")


if __name__ == "__main__":
    main()
