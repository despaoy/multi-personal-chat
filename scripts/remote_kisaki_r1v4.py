#!/usr/bin/env python3
"""Synchronize and operate the formal R1V4 training queue over SSH."""

from __future__ import annotations

import argparse
import hashlib
import os
import posixpath
import re
import shlex
from pathlib import Path

from remote_config import connect_ssh


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = os.getenv("MULTIPERSONAL_REMOTE_ROOT") or os.getenv("QQCHAT_REMOTE_ROOT", "/workspace/multi-personal-chat")
REMOTE_LAB_ROOT = (
    os.getenv("MULTIPERSONAL_LAB_ROOT")
    or os.getenv("QQCHAT_LAB_ROOT")
    or posixpath.dirname(REMOTE_ROOT.rstrip("/"))
)
PYTHON = os.getenv("MULTIPERSONAL_REMOTE_PYTHON") or os.getenv("QQCHAT_REMOTE_PYTHON", "python")
QUEUE_LOG = "/tmp/kisaki_r1v4_queue.log"
REPOSITORY_URL = os.getenv(
    "MULTIPERSONAL_REPOSITORY_URL",
    "https://github.com/despaoy/multi-personal-chat.git",
)
REPOSITORY_BRANCH = os.getenv("MULTIPERSONAL_REPOSITORY_BRANCH", "main")
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


def _validate_remote_settings() -> None:
    """Reject unsafe shell/path settings before opening an SSH connection."""

    safe_path = re.compile(r"^/[A-Za-z0-9._/-]+$")
    for label, value in (("REMOTE_ROOT", REMOTE_ROOT), ("REMOTE_LAB_ROOT", REMOTE_LAB_ROOT)):
        if not safe_path.fullmatch(value) or posixpath.normpath(value) != value.rstrip("/"):
            raise ValueError(f"{label} must be a normalized absolute POSIX path")
    if REMOTE_ROOT.rstrip("/") in {"", "/", "/root", "/workspace"}:
        raise ValueError("REMOTE_ROOT is too broad for remote operations")
    if REMOTE_LAB_ROOT.rstrip("/") in {"", "/"}:
        raise ValueError("REMOTE_LAB_ROOT is too broad for remote operations")
    lab_prefix = REMOTE_LAB_ROOT.rstrip("/") + "/"
    if not REMOTE_ROOT.startswith(lab_prefix):
        raise ValueError("REMOTE_ROOT must be inside REMOTE_LAB_ROOT")
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", PYTHON):
        raise ValueError("REMOTE_PYTHON must be a plain executable path")
    if any(char in REPOSITORY_URL for char in "\r\n\0"):
        raise ValueError("REPOSITORY_URL contains control characters")
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", REPOSITORY_BRANCH):
        raise ValueError("REPOSITORY_BRANCH contains unsafe characters")


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
            local_hash = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            with sftp.open(remote, "rb") as remote_file:
                remote_hash = hashlib.sha256(remote_file.read()).hexdigest()
            if local_hash != remote_hash:
                raise RuntimeError(f"remote hash mismatch after upload: {relative}")
            print(f"uploaded={relative}")
    finally:
        sftp.close()


def clean_stale_training(client) -> None:
    experiment_names = ("e1", "e2", "e3", "e4", "e5")
    targets = [
        *(f"{REMOTE_LAB_ROOT}/runtime/loras/kisaki/r1v4/{name}" for name in experiment_names),
        f"{REMOTE_LAB_ROOT}/runtime/loras/kisaki/r1v4/overfit20",
        *(f"{REMOTE_LAB_ROOT}/runtime/experiments/kisaki/r1v4/{name}" for name in experiment_names),
    ]
    quoted = " ".join(shlex.quote(target) for target in targets)
    runtime_prefix = shlex.quote(f"{REMOTE_LAB_ROOT}/runtime/")
    cleanup = (
        "if pgrep -f '^[^ ]*python[^ ]* scripts/run_kisaki_experiment.py' >/dev/null "
        "|| pgrep -f '^[^ ]*python[^ ]* -m training.trainer' >/dev/null; then "
        "echo active_training_refuses_cleanup; exit 3; fi; "
        f"for p in {quoted}; do case \"$p\" in {runtime_prefix}*) ;; *) exit 4;; esac; "
        "test -e \"$p\" && du -sh \"$p\" || true; done; "
        f"rm -rf -- {quoted}; "
        "rm -f /tmp/kisaki_r1v4_queue.log /tmp/kisaki_r1v4_e?.log "
        "/tmp/kisaki_r1v4_complete; "
        f"for p in {quoted}; do test ! -e \"$p\" || exit 5; done; echo cleanup_complete"
    )
    code, out, err = command(client, cleanup, timeout=120)
    print(out)
    if code:
        raise RuntimeError(err or f"cleanup failed with exit code {code}")


def reclone(client) -> None:
    parent = posixpath.dirname(REMOTE_ROOT.rstrip("/"))
    preserved = f"{REMOTE_LAB_ROOT}/runtime/project-preserved-20260819"
    command_text = (
        "set -e; if pgrep -f '^[^ ]*python[^ ]* scripts/run_kisaki_experiment.py' >/dev/null "
        "|| pgrep -f '^[^ ]*python[^ ]* -m training.trainer' >/dev/null; then "
        "echo active_training_refuses_reclone; exit 3; fi; "
        f"test '{REMOTE_ROOT}' = '{parent}/qqchat-enhanced' || exit 4; "
        f"test ! -e '{preserved}' || exit 5; "
        f"mkdir -p '{preserved}/backend/knowledge' '{preserved}/backend' "
        f"'{preserved}/legacy_adapters' && "
        f"if test -e '{REMOTE_ROOT}/.env'; then mv '{REMOTE_ROOT}/.env' '{preserved}/.env'; fi; "
        f"if test -e '{REMOTE_ROOT}/backend/qq_assistant.db'; then mv "
        f"'{REMOTE_ROOT}/backend/qq_assistant.db' '{preserved}/backend/qq_assistant.db'; fi; "
        f"if test -e '{REMOTE_ROOT}/backend/knowledge/data'; then mv "
        f"'{REMOTE_ROOT}/backend/knowledge/data' '{preserved}/backend/knowledge/data'; fi; "
        f"if test -e '{REMOTE_ROOT}/backend/logs'; then mv "
        f"'{REMOTE_ROOT}/backend/logs' '{preserved}/backend/logs'; fi; "
        f"if test -e '{REMOTE_ROOT}/backend/loras'; then mv "
        f"'{REMOTE_ROOT}/backend/loras' '{preserved}/legacy_adapters/backend_loras'; fi; "
        f"if test -e '{REMOTE_ROOT}/loras'; then mv "
        f"'{REMOTE_ROOT}/loras' '{preserved}/legacy_adapters/root_loras'; fi; "
        f"rm -rf -- '{REMOTE_ROOT}' && "
        f"git clone --branch {shlex.quote(REPOSITORY_BRANCH)} --single-branch "
        f"{shlex.quote(REPOSITORY_URL)} '{REMOTE_ROOT}' && "
        f"if test -e '{preserved}/.env'; then cp -a '{preserved}/.env' '{REMOTE_ROOT}/.env'; fi; "
        f"if test -e '{preserved}/backend/qq_assistant.db'; then cp -a "
        f"'{preserved}/backend/qq_assistant.db' '{REMOTE_ROOT}/backend/qq_assistant.db'; fi; "
        f"if test -e '{preserved}/backend/knowledge/data'; then "
        f"mkdir -p '{REMOTE_ROOT}/backend/knowledge'; cp -a "
        f"'{preserved}/backend/knowledge/data' '{REMOTE_ROOT}/backend/knowledge/data'; fi; "
        f"if test -e '{preserved}/backend/logs'; then cp -a "
        f"'{preserved}/backend/logs' '{REMOTE_ROOT}/backend/logs'; fi; "
        f"cd '{REMOTE_ROOT}' && git rev-parse HEAD && git status --short; "
        f"echo PRESERVED && du -sh '{preserved}'/* '{preserved}'/.[!.]* 2>/dev/null || true"
    )
    code, out, err = command(client, command_text, timeout=300)
    print(out)
    if code:
        raise RuntimeError(err or f"reclone failed with exit code {code}")


def update_checkout(client) -> None:
    """Fast-forward a clean remote checkout to the configured branch."""

    command_text = (
        f"cd '{REMOTE_ROOT}' && "
        "test -z \"$(git status --porcelain --untracked-files=no)\" || "
        "{ echo tracked_changes_refuse_update; exit 3; }; "
        f"git fetch origin {shlex.quote(REPOSITORY_BRANCH)} && "
        "git merge --ff-only FETCH_HEAD && git rev-parse HEAD"
    )
    code, out, err = command(client, command_text, timeout=120)
    print(out)
    if code:
        raise RuntimeError(err or f"remote update failed with exit code {code}")


def start(client, experiments: list[str]) -> None:
    queue = " ".join(
        f"echo START_{name}; {PYTHON} scripts/run_kisaki_experiment.py --experiment {name} --seed 42 "
        f"> /tmp/kisaki_r1v4_{name}.log 2>&1 || exit $?; echo DONE_{name};"
        for name in experiments
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


def inspect_remote(client) -> None:
    check = (
        "echo ROOTS; pwd; "
        f"for p in {REMOTE_ROOT} {REMOTE_LAB_ROOT} '\\workspace' /workspace /root/autodl-tmp; do "
        "if test -e \"$p\"; then echo PATH=$p; du -sh \"$p\" 2>/dev/null || true; fi; done; "
        "echo PROCESSES; pgrep -af 'run_kisaki_experiment|training.trainer' || true; "
        "echo PROJECTS; find /root/autodl-tmp /root -maxdepth 3 -type f "
        "-path '*/scripts/run_kisaki_experiment.py' -print 2>/dev/null || true; "
        "echo RUNTIME_SIZES; for p in /root/autodl-tmp/runtime/loras/kisaki/r1v4/* "
        "/root/autodl-tmp/runtime/experiments/kisaki/r1v4/*; do "
        "test -e \"$p\" && du -sh \"$p\"; done; "
        "echo LOGS; for f in /tmp/kisaki_r1v4_queue.log /tmp/kisaki_r1v4_e*.log; do "
        "test -f \"$f\" && echo FILE=$f && tail -20 \"$f\"; done; "
        "echo OUTPUTS; find /workspace /root/autodl-tmp -path '*/kisaki/r1v4/*' "
        "-maxdepth 10 -type f -name adapter_config.json -print 2>/dev/null || true; "
        "echo DISK; df -h /workspace /root/autodl-tmp 2>/dev/null || df -h /"
    )
    _, out, err = command(client, check, timeout=120)
    print(out)
    if err.strip():
        print(err)


def inspect_project(client) -> None:
    check = (
        f"test -d '{REMOTE_ROOT}' || exit 2; "
        f"echo TOP_LEVEL; du -sh '{REMOTE_ROOT}'/* '{REMOTE_ROOT}'/.[!.]* 2>/dev/null | sort -h; "
        f"echo LARGE_FILES; find '{REMOTE_ROOT}' -xdev -type f -size +20M "
        "-printf '%s %p\\n' 2>/dev/null | sort -n; "
        f"echo RUNTIME_ASSETS; find '{REMOTE_ROOT}' -xdev -maxdepth 5 "
        "\\( -type f -o -type l \\) "
        "\\( -iname '*.safetensors' -o -iname '*.bin' -o -iname '*.pt' -o "
        "-iname '*.pth' -o -iname '*.db' -o -iname '*.sqlite*' -o -iname '*.log' \\) "
        "-print 2>/dev/null; "
        f"echo SYMLINKS; find '{REMOTE_ROOT}' -xdev -type l -printf '%p -> %l\\n' 2>/dev/null; "
        f"echo GIT; cd '{REMOTE_ROOT}' && git status --short 2>/dev/null || true"
    )
    code, out, err = command(client, check, timeout=180)
    print(out)
    if code:
        raise RuntimeError(err or f"project inspection failed with exit code {code}")


def inspect_assets(client) -> None:
    candidates = (
        ".env backend/qq_assistant.db backend/knowledge/data backend/uploads "
        "backend/logs backend/loras loras runtime storage uploads"
    )
    check = (
        f"cd '{REMOTE_ROOT}' || exit 2; echo ASSETS; "
        f"for p in {candidates}; do test -e \"$p\" && du -sh \"$p\"; done; "
        "echo KNOWLEDGE; find backend/knowledge/data -maxdepth 3 -type f "
        "-printf '%s %p\\n' 2>/dev/null | sort -n; "
        "echo DATABASES; find . -xdev -type f "
        "\\( -iname '*.db' -o -iname '*.sqlite' -o -iname '*.sqlite3' \\) "
        "-not -path './node_modules/*' -printf '%s %p\\n' 2>/dev/null | sort -n; "
        "echo ADAPTERS; find backend/loras loras -type f -name adapter_config.json "
        "-print 2>/dev/null || true"
    )
    code, out, err = command(client, check, timeout=120)
    print(out)
    if code:
        raise RuntimeError(err or f"asset inspection failed with exit code {code}")


def preflight(client) -> None:
    model = f"{REMOTE_LAB_ROOT}/runtime/models/Qwen3-8B-Instruct"
    check = (
        f"cd '{REMOTE_ROOT}' && export MULTIPERSONAL_LAB_ROOT='{REMOTE_LAB_ROOT}' && "
        f"test -f '{model}/config.json' && "
        f"{PYTHON} -c \"import json; c=json.load(open('{model}/config.json')); "
        "assert not c.get('quantization_config'), 'prequantized model is forbidden'; "
        "print('model_type='+str(c.get('model_type')));\" && "
        f"{PYTHON} scripts/validate_kisaki_v4_training_gate.py "
        f"--disk-path '{REMOTE_LAB_ROOT}' --minimum-free-gb 15 && "
        f"cd backend && {PYTHON} -c \"import torch,transformers,peft,trl; "
        "print('torch='+torch.__version__); print('transformers='+transformers.__version__); "
        "print('peft='+peft.__version__); print('trl='+trl.__version__); "
        "print('cuda='+str(torch.cuda.is_available())); "
        "print('bf16='+str(torch.cuda.is_bf16_supported()));\" && "
        "cd .. && echo GPU && nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu "
        "--format=csv,noheader,nounits"
    )
    code, out, err = command(client, check, timeout=120)
    print(out)
    if code:
        raise RuntimeError(err or f"preflight failed with exit code {code}")


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
    parser.add_argument(
        "action",
        choices=(
            "upload", "clean-stale", "reclone", "update", "preflight",
            "start", "status", "inspect", "inspect-project", "inspect-assets", "stop",
        ),
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=("e1", "e2", "e3", "e4", "e5"),
        default=["e1"],
        help="Experiments to queue; defaults to E1 only.",
    )
    args = parser.parse_args()
    _validate_remote_settings()
    client = connect_ssh(timeout=30)
    try:
        if args.action == "start":
            start(client, args.experiments)
        else:
            {
                "upload": upload,
                "clean-stale": clean_stale_training,
                "reclone": reclone,
                "update": update_checkout,
                "preflight": preflight,
                "status": status,
                "inspect": inspect_remote,
                "inspect-project": inspect_project,
                "inspect-assets": inspect_assets,
                "stop": stop,
            }[args.action](client)
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
