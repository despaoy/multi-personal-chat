#!/usr/bin/env python3
"""Synchronize and operate the formal R1V4 training queue over SSH."""

from __future__ import annotations

import argparse
import os
import posixpath
from pathlib import Path

from remote_config import connect_ssh


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = os.getenv("MULTIPERSONAL_REMOTE_ROOT") or os.getenv("QQCHAT_REMOTE_ROOT", "/workspace/multi-personal-chat")
REMOTE_LAB_ROOT = os.getenv("MULTIPERSONAL_LAB_ROOT") or os.getenv("QQCHAT_LAB_ROOT") or str(Path(REMOTE_ROOT).parent)
PYTHON = os.getenv("MULTIPERSONAL_REMOTE_PYTHON") or os.getenv("MULTIPERSONAL_REMOTE_PYTHON", "python")
QUEUE_LOG = "/tmp/kisaki_r1v4_queue.log"
FILES = (
    "backend/data/character_dialogues/experiments/v4/train.jsonl",
    "backend/data/character_dialogues/experiments/v4/validation.jsonl",
    "backend/data/character_dialogues/experiments/v4/canonical_dataset_manifest.json",
    "backend/data/character_dialogues/experiments/v4/r1v4_base_config.json",
    "backend/data/character_dialogues/experiments/v4/configs/config_manifest.json",
    "backend/data/character_dialogues/experiments/v4/configs/kisaki_r1v4_e1.json",
    "backend/data/character_dialogues/experiments/v4/configs/kisaki_r1v4_e2.json",
    "backend/data/character_dialogues/experiments/v4/configs/kisaki_r1v4_e3.json",
    "backend/data/character_dialogues/experiments/v4/configs/kisaki_r1v4_e4.json",
    "backend/data/character_dialogues/experiments/v4/configs/kisaki_r1v4_e5.json",
    "backend/data/character_dialogues/kisaki_system_prompt_v3.txt",
    "backend/evaluation/kisaki_gold_set_v3.json",
    "backend/evaluation/experiment_contracts.py",
    "backend/inference/prompt_policy.py",
    "backend/training/chat_dataset.py",
    "backend/training/evaluator.py",
    "backend/training/trainer.py",
    "docs/research/review_packets/kisaki_v4/review_manifest.json",
    "scripts/run_kisaki_experiment.py",
    "scripts/validate_kisaki_v4_training_gate.py",
)


def ensure_dir(sftp, directory: str) -> None:
    current = "/"
    for part in directory.strip("/").split("/"):
        current = posixpath.join(current, part)
        try:
            sftp.stat(current)
        except OSError:
            sftp.mkdir(current)


def command(client, value: str, timeout: int = 30) -> tuple[int, str, str]:
    _, stdout, stderr = client.exec_command(value, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return stdout.channel.recv_exit_status(), out, err


def upload(client) -> None:
    sftp = client.open_sftp()
    try:
        for relative in FILES:
            remote = posixpath.join(REMOTE_ROOT, relative)
            ensure_dir(sftp, posixpath.dirname(remote))
            sftp.put(str(ROOT / relative), remote)
            print(f"uploaded={relative}")
    finally:
        sftp.close()


def start(client) -> None:
    queue = " ".join(
        f"echo START_{name}; {PYTHON} scripts/run_kisaki_experiment.py --experiment {name} --seed 42 "
        f"> /tmp/kisaki_r1v4_{name}.log 2>&1 || exit $?; echo DONE_{name};"
        for name in ("e1", "e2", "e3", "e4", "e5")
    )
    shell = (
        f"cd {REMOTE_ROOT}; export MULTIPERSONAL_LAB_ROOT={REMOTE_LAB_ROOT}; "
        f"{queue} touch /tmp/kisaki_r1v4_complete"
    )
    code, out, err = command(
        client,
        f"nohup bash -lc {shell!r} > {QUEUE_LOG} 2>&1 < /dev/null & echo $!",
    )
    if code:
        raise RuntimeError(err or out)
    print(f"queue_pid={out.strip()}")


def status(client) -> None:
    check = (
        f"tail -30 {QUEUE_LOG} 2>/dev/null || true; echo TRAIN_LOG_SEPARATOR; "
        f"for e in e1 e2 e3 e4 e5; do if pgrep -f \"^{PYTHON} scripts/run_kisaki_experiment.py --experiment $e\" >/dev/null; "
        "then echo CURRENT=$e; tail -25 /tmp/kisaki_r1v4_$e.log 2>/dev/null; fi; done; echo STATUS_SEPARATOR; "
        "pgrep -af 'run_kisaki_experiment|training.trainer' || true; echo ADAPTER_SEPARATOR; "
        f"for e in e1 e2 e3 e4 e5; do test -f {REMOTE_LAB_ROOT}/runtime/loras/kisaki/r1v4/$e/seed42/final/adapter_config.json "
        "&& echo $e=complete || echo $e=pending; done; echo GPU_SEPARATOR; "
        "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits"
    )
    _, out, err = command(client, check)
    print(out)
    if err.strip():
        print(err)


def stop(client) -> None:
    stop_command = (
        f"pkill -TERM -f '^{PYTHON} -m training.trainer --config .*/r1v4/' || true; "
        f"pkill -TERM -f '^{PYTHON} scripts/run_kisaki_experiment.py --experiment' || true; "
        f"pkill -TERM -f '^bash -lc cd {REMOTE_ROOT}; echo START_e1' || true; "
        "sleep 3; pgrep -af 'run_kisaki_experiment|training.trainer' || true"
    )
    _, out, err = command(client, stop_command)
    print(out)
    if err.strip():
        print(err)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("upload", "start", "status", "stop"))
    args = parser.parse_args()
    client = connect_ssh(timeout=30)
    try:
        {"upload": upload, "start": start, "status": status, "stop": stop}[args.action](client)
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
