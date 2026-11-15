"""Read-only inventory of the explicitly configured lhm server, without credentials in output."""

import importlib.util
import json
from pathlib import Path

import paramiko


def connect():
    directory = Path.home() / "Desktop" / "solve" / "automation"
    config = json.loads((directory / "server_config_x99_server2.json").read_text(encoding="utf-8"))
    spec = importlib.util.spec_from_file_location("windows_secret", directory / "windows_secret.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    password = module.load_secret(directory / config["password_dpapi"])
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(config["host"], port=config["port"], username=config["user"], password=password,
                   look_for_keys=False, allow_agent=False, timeout=15)
    return client


if __name__ == "__main__":
    with connect() as client:
        command = "find /home/boot/lhm/multipersonal-runtime -maxdepth 3 -type f \\( -name psql -o -name pg_ctl -o -name postmaster.pid \\) -print; ss -ltn | head -25"
        _, stdout, stderr = client.exec_command(command, timeout=25)
        print(stdout.read().decode())
        print(stderr.read().decode())
