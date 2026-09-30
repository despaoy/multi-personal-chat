"""Test bootstrap must never rotate the workspace deployment credential."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "secret", [None, "", "short-test-value", "multipersonal-jwt-secret-change-in-production", "x" * 48]
)
@pytest.mark.parametrize("environment", ["development", "production"])
def test_bootstrap_normalizes_only_invalid_credentials_without_config_write(tmp_path, secret, environment):
    backend = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update(
        ENVIRONMENT=environment,
        USE_POSTGRESQL="false",
        DATABASE_PATH=str(tmp_path / "bootstrap.sqlite"),
        TMPDIR=str(tmp_path),
        TMP=str(tmp_path),
        TEMP=str(tmp_path),
        PYTHONDONTWRITEBYTECODE="1",
    )
    if secret is None:
        env.pop("JWT_SECRET", None)
    else:
        env["JWT_SECRET"] = secret
    probe = r"""
import json, os, runpy
from pathlib import Path
path = Path(".env")
before = path.read_bytes() if path.exists() else None
runpy.run_path("tests/conftest.py")
credential = os.environ["JWT_SECRET"]
import app.config as config
after = path.read_bytes() if path.exists() else None
print("BOOTSTRAP_RESULT=" + json.dumps(dict(length=len(credential), unchanged=before == after,
    used_by_config=config.JWT_SECRET == credential,
    supplied_valid_preserved=credential == os.environ.get("EXPECTED_VALID", ""))))
"""
    env["EXPECTED_VALID"] = (
        secret if secret and len(secret) >= 32 and secret != "multipersonal-jwt-secret-change-in-production" else ""
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], cwd=backend, env=env, check=True, capture_output=True, text=True, timeout=30
    )
    line = next(line for line in result.stdout.splitlines() if line.startswith("BOOTSTRAP_RESULT="))
    proof = json.loads(line.split("=", 1)[1])
    assert proof["length"] >= 32
    assert proof["unchanged"] and proof["used_by_config"]
    assert proof["supplied_valid_preserved"] is bool(env["EXPECTED_VALID"])
