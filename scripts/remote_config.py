"""Shared, environment-driven SSH configuration for lab automation scripts."""

from __future__ import annotations

import os
from pathlib import Path

import paramiko


HOST = os.getenv("LAB_HOST", "").strip()
USER = os.getenv("LAB_USER", "").strip()
PORT = int(os.getenv("LAB_PORT", "22"))


def connect_ssh(*, timeout: int = 15) -> paramiko.SSHClient:
    """Create an SSH client without embedding credentials in source code."""
    password = os.getenv("LAB_PASS")
    key_file = os.getenv("LAB_SSH_KEY")
    if not HOST or not USER:
        raise RuntimeError("Set LAB_HOST and LAB_USER before running a remote automation script")
    if not password and not key_file:
        raise RuntimeError("Set LAB_PASS or LAB_SSH_KEY before running a remote automation script")
    if key_file and not Path(key_file).expanduser().is_file():
        raise RuntimeError(f"LAB_SSH_KEY does not exist: {key_file}")

    client = paramiko.SSHClient()
    client.load_system_host_keys()
    auto_add = os.getenv("LAB_SSH_AUTO_ADD_HOST_KEY", "false").strip().lower() in {
        "1", "true", "yes", "on"
    }
    if auto_add:
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    else:
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(
        HOST,
        port=PORT,
        username=USER,
        password=password,
        key_filename=str(Path(key_file).expanduser()) if key_file else None,
        timeout=timeout,
    )
    return client
