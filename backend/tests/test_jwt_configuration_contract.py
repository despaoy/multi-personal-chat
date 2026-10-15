"""JWT configuration imports must never generate or persist a signing key.

Every full config import runs from a temporary source copy in a fresh process.
Only synthetic credentials and temporary dotenv/database paths enter that process.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

VALID_SECRET = "synthetic-jwt-contract-key-with-at-least-48-characters"
LEGACY_DEFAULT = "multipersonal-jwt-secret-change-in-production"
START_TEMPLATE = "CHANGE_ME_TO_A_SECURE_JWT_SECRET_AT_LEAST_32_CHARS"
EXAMPLE_TEMPLATE = "your-48-char-secure-secret-here-change-in-production"
KNOWN_PLACEHOLDERS = (
    "", "changeme", "change-me", "change-me-to-a-random-secret",
    "PLEASE_SET_A_SECURE_SECRET_HERE", LEGACY_DEFAULT, START_TEMPLATE,
    EXAMPLE_TEMPLATE, "replace-with-at-least-32-random-characters",
)
PROOF_PREFIX = "JWT_CONFIG_PROOF="


_IMPORT_PROBE = r'''
import importlib.util, json, os, sys
from pathlib import Path

backend, copied_config = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(backend))
import app
dotenv = copied_config.parents[1] / ".env"
before_file = dotenv.read_bytes() if dotenv.exists() else None
before_jwt = ("JWT_SECRET" in os.environ, os.environ.get("JWT_SECRET"))
proof = dict(import_succeeded=False, runtime_error=False, error_mentions_jwt=False,
             jwt_preserved=False, token_roundtrip=False, distinct_jti=False)
spec = importlib.util.spec_from_file_location("app.config", copied_config)
module = importlib.util.module_from_spec(spec)
sys.modules["app.config"] = module
app.config = module
try:
    spec.loader.exec_module(module)
except Exception as error:
    proof["runtime_error"] = type(error) is RuntimeError
    proof["error_mentions_jwt"] = "JWT_SECRET" in str(error)
else:
    proof["import_succeeded"] = True
    proof["jwt_preserved"] = module.JWT_SECRET == before_jwt[1]
    first = module.verify_token(module.create_access_token("synthetic-user", 901, "admin"))
    second = module.verify_token(module.create_access_token("synthetic-user", 901, "admin"))
    proof["token_roundtrip"] = (
        first["sub"] == "synthetic-user" and first["user_id"] == 901
        and first["role"] == "admin" and first["exp"] > first["iat"]
        and isinstance(first.get("jti"), str) and bool(first["jti"])
    )
    proof["distinct_jti"] = first.get("jti") != second.get("jti")
finally:
    after_file = dotenv.read_bytes() if dotenv.exists() else None
    proof["jwt_environment_unchanged"] = before_jwt == (
        "JWT_SECRET" in os.environ, os.environ.get("JWT_SECRET"))
    proof["dotenv_unchanged"] = before_file == after_file
    proof["dotenv_exists"] = dotenv.exists()
print("JWT_CONFIG_PROOF=" + json.dumps(proof, sort_keys=True))
'''


def _import_config(tmp_path, *, secret, environment, existing_dotenv):
    backend = Path(__file__).resolve().parents[1]
    isolated = tmp_path / "isolated" / "backend"
    app_dir = isolated / "app"
    app_dir.mkdir(parents=True)
    copied_config = app_dir / "config.py"
    copied_config.write_bytes((backend / "app" / "config.py").read_bytes())
    if existing_dotenv:
        (isolated / ".env").write_text("SYNTHETIC_UNRELATED_SETTING=unchanged\n", encoding="utf-8")

    # Do not inherit application secrets, dotenv paths, provider credentials or
    # database URLs from either the caller or pytest's bootstrap environment.
    env = {name: os.environ[name] for name in (
        "PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH",
    ) if name in os.environ}
    env.update(
        PYTHONPATH=str(backend), PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8",
        TMPDIR=str(tmp_path), TMP=str(tmp_path), TEMP=str(tmp_path),
        USE_POSTGRESQL="false", DATABASE_PATH=str(isolated / "synthetic.sqlite"),
        VECTOR_DB_PATH=str(isolated / "vectors"), HF_HOME=str(isolated / "hf-cache"),
        ENCRYPTION_KEY=base64.urlsafe_b64encode(b"e" * 32).decode(),
        MODEL_PROVIDER="openai_compat", VLLM_ENABLED="false", MEMORY_LLM_ENABLED="false",
        OPENAI_COMPAT_BASE_URL="https://provider.invalid/v1",
        OPENAI_COMPAT_MODEL="synthetic-model", OPENAI_COMPAT_API_KEY="synthetic-provider-key",
        ASTRBOT_INTEGRATION_TOKEN="synthetic-integration-key-at-least-32-characters",
        MULTIPERSONAL_BACKEND_URL="https://backend.invalid",
        ALLOWED_ORIGINS="https://frontend.invalid", LOG_LEVEL="ERROR",
    )
    if secret is not None:
        env["JWT_SECRET"] = secret
    if environment is not None:
        env["ENVIRONMENT"] = environment
    completed = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE, str(backend), str(copied_config)],
        cwd=isolated, env=env, capture_output=True, text=True, timeout=45,
    )
    # Never echo child output, environment values, tokens or exception text.
    assert completed.returncode == 0, "isolated config probe did not finish"
    lines = [line for line in completed.stdout.splitlines() if line.startswith(PROOF_PREFIX)]
    assert len(lines) == 1, "isolated config probe did not return exactly one proof"
    return json.loads(lines[0][len(PROOF_PREFIX):])


@pytest.mark.parametrize("environment,secret,existing_dotenv", [
    pytest.param(None, None, False, id="unset-environment-missing-key-no-file"),
    pytest.param("development", None, True, id="development-missing-key-existing-file"),
    pytest.param("production", None, False, id="production-missing-key-no-file"),
    pytest.param("development", "", False, id="development-empty-key-no-file"),
    pytest.param("production", "", True, id="production-empty-key-existing-file"),
    pytest.param("development", "short-test-key", True, id="development-short-key-existing-file"),
    pytest.param("production", "short-test-key", False, id="production-short-key-no-file"),
    pytest.param(None, LEGACY_DEFAULT, True, id="unset-environment-legacy-placeholder"),
    pytest.param("development", START_TEMPLATE, True, id="development-startup-template"),
    pytest.param("production", EXAMPLE_TEMPLATE, False, id="production-example-template"),
])
def test_invalid_jwt_import_fails_without_changing_environment_or_dotenv(
    tmp_path, environment, secret, existing_dotenv,
):
    proof = _import_config(tmp_path, secret=secret, environment=environment, existing_dotenv=existing_dotenv)
    assert not proof["import_succeeded"]
    assert proof["runtime_error"] and proof["error_mentions_jwt"]
    assert proof["jwt_environment_unchanged"] and proof["dotenv_unchanged"]
    assert proof["dotenv_exists"] is existing_dotenv


@pytest.mark.parametrize("environment,secret,existing_dotenv", [
    pytest.param(None, VALID_SECRET, False, id="unset-environment-explicit-key"),
    pytest.param("development", VALID_SECRET, True, id="development-explicit-key"),
    pytest.param("production", "  " + VALID_SECRET + "  ", True, id="production-preserves-explicit-key-bytes"),
])
def test_explicit_jwt_import_preserves_key_and_real_token_roundtrip(
    tmp_path, environment, secret, existing_dotenv,
):
    proof = _import_config(tmp_path, secret=secret, environment=environment, existing_dotenv=existing_dotenv)
    assert proof["import_succeeded"] and not proof["runtime_error"]
    assert proof["jwt_preserved"] and proof["token_roundtrip"] and proof["distinct_jti"]
    assert proof["jwt_environment_unchanged"] and proof["dotenv_unchanged"]
    assert proof["dotenv_exists"] is existing_dotenv


def test_explicit_empty_deployment_environment_cannot_borrow_process_jwt(monkeypatch):
    from infra.deployment import validate_deployment_environment

    monkeypatch.setenv("JWT_SECRET", VALID_SECRET)
    result = validate_deployment_environment({})
    assert not result.ok and any("JWT_SECRET" in error for error in result.errors)
    assert os.environ["JWT_SECRET"] == VALID_SECRET


def test_all_published_jwt_placeholders_are_rejected_by_authority():
    from infra.deployment import validate_jwt_secret

    for placeholder in KNOWN_PLACEHOLDERS:
        with pytest.raises(RuntimeError, match="JWT_SECRET"):
            validate_jwt_secret(placeholder)
