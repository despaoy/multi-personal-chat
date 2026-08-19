#!/usr/bin/env python3
"""Upload, run, inspect, and fetch the Kisaki V4 overfit smoke test."""

from __future__ import annotations

import argparse
import os
import posixpath
from pathlib import Path

from remote_config import connect_ssh


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = os.getenv("MULTIPERSONAL_REMOTE_ROOT") or os.getenv("QQCHAT_REMOTE_ROOT", "/workspace/multi-personal-chat")
REMOTE_LAB_ROOT = os.getenv("MULTIPERSONAL_LAB_ROOT") or os.getenv("QQCHAT_LAB_ROOT") or str(Path(REMOTE_ROOT).parent)
REMOTE_PYTHON = os.getenv("MULTIPERSONAL_REMOTE_PYTHON") or os.getenv("QQCHAT_REMOTE_PYTHON", "python")
REMOTE_MODEL = os.getenv("MULTIPERSONAL_REMOTE_MODEL") or os.getenv(
    "QQCHAT_REMOTE_MODEL",
    f"{REMOTE_ROOT}/runtime/models/Qwen3-8B-Instruct",
)
REMOTE_OUTPUT = os.getenv("MULTIPERSONAL_REMOTE_OUTPUT") or os.getenv(
    "QQCHAT_REMOTE_OUTPUT",
    f"{REMOTE_LAB_ROOT}/runtime/loras/kisaki/r1v4/overfit20",
)
LOG = "/tmp/kisaki_v4_overfit20.log"
FILES = (
    "backend/data/character_dialogues/experiments/v4/overfit_20/train.jsonl",
    "backend/data/character_dialogues/experiments/v4/overfit_20/cases.json",
    "backend/data/character_dialogues/experiments/v4/overfit_20/config.json",
    "backend/data/character_dialogues/experiments/v4/overfit_20/manifest.json",
    "backend/data/character_dialogues/kisaki_system_prompt_v3.txt",
    "backend/training/chat_dataset.py",
    "backend/training/trainer.py",
    "scripts/generate_kisaki_v4_overfit_results.py",
    "scripts/render_kisaki_v4_overfit_review.py",
    "scripts/run_kisaki_v4_overfit_test.py",
)


def ensure_remote_dir(sftp, directory: str) -> None:
    current = "/"
    for part in directory.strip("/").split("/"):
        current = posixpath.join(current, part)
        try:
            sftp.stat(current)
        except OSError:
            sftp.mkdir(current)


def exec_text(client, command: str, timeout: int = 30) -> tuple[int, str, str]:
    _, stdout, stderr = client.exec_command(command, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return stdout.channel.recv_exit_status(), out, err


def upload(client) -> None:
    sftp = client.open_sftp()
    try:
        for relative in FILES:
            local = ROOT / relative
            remote = posixpath.join(REMOTE_ROOT, relative.replace("\\", "/"))
            ensure_remote_dir(sftp, posixpath.dirname(remote))
            sftp.put(str(local), remote)
            print(f"uploaded={relative}")
    finally:
        sftp.close()


def start(client) -> None:
    command = (
        f"cd {REMOTE_ROOT} && export MULTIPERSONAL_LAB_ROOT={REMOTE_LAB_ROOT} && "
        f"nohup {REMOTE_PYTHON} scripts/run_kisaki_v4_overfit_test.py "
        f"--base-model {REMOTE_MODEL} --output-dir {REMOTE_OUTPUT} "
        f"> {LOG} 2>&1 < /dev/null & echo $!"
    )
    code, out, err = exec_text(client, command)
    if code:
        raise RuntimeError(err or out)
    print(f"remote_pid={out.strip()}")


def status(client) -> None:
    command = (
        f"tail -40 {LOG} 2>/dev/null || true; "
        "echo STATUS_SEPARATOR; "
        f"pgrep -af 'run_kisaki_v4_overfit_test|training.trainer' || true; "
        "echo GPU_SEPARATOR; "
        "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu "
        "--format=csv,noheader,nounits"
    )
    _, out, err = exec_text(client, command)
    print(out)
    if err.strip():
        print(err)


def fetch(client) -> None:
    pairs = (
        (
            "backend/data/character_dialogues/experiments/v4/overfit_20/results.json",
            "backend/data/character_dialogues/experiments/v4/overfit_20/results.json",
        ),
        (
            "docs/research/review_packets/kisaki_v4/09_OVERFIT_TEST/review.md",
            "docs/research/review_packets/kisaki_v4/09_OVERFIT_TEST/review.md",
        ),
    )
    sftp = client.open_sftp()
    try:
        for remote_relative, local_relative in pairs:
            remote = posixpath.join(REMOTE_ROOT, remote_relative)
            local = ROOT / local_relative
            local.parent.mkdir(parents=True, exist_ok=True)
            sftp.get(remote, str(local))
            print(f"fetched={local_relative}")
    finally:
        sftp.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("upload", "start", "status", "fetch"))
    args = parser.parse_args()
    client = connect_ssh(timeout=30)
    try:
        {"upload": upload, "start": start, "status": status, "fetch": fetch}[args.action](client)
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
