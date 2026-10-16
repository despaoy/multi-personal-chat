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



def _isolated_encryption_import_environment(backend, tmp_path):
    # Do not inherit application credentials or paths, including an already
    # populated encryption key from the parent pytest process.
    env = {name: os.environ[name] for name in (
        "PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH",
    ) if name in os.environ}
    env.update(
        PYTHONPATH=str(backend), PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8",
        ENVIRONMENT="development", JWT_SECRET="synthetic-encryption-import-jwt-at-least-48-characters",
        USE_POSTGRESQL="false", DATABASE_PATH=str(tmp_path / "synthetic.sqlite"),
        VECTOR_DB_PATH=str(tmp_path / "vectors"), HF_HOME=str(tmp_path / "hf-cache"),
        TMPDIR=str(tmp_path), TMP=str(tmp_path), TEMP=str(tmp_path),
        MODEL_PROVIDER="openai_compat", VLLM_ENABLED="false", MEMORY_LLM_ENABLED="false",
        OPENAI_COMPAT_BASE_URL="https://provider.invalid/v1",
        OPENAI_COMPAT_MODEL="synthetic-inactive-model", OPENAI_COMPAT_API_KEY="synthetic-provider-key",
        DYNAMIC_CONTEXT_SEMANTIC_REVIEW_ENABLED="false", CONTEXTUAL_MEMORY_SELECTION_ENABLED="false",
        CONTEXTUAL_DECISION_POLICY_ENABLED="false", CAHM_SEMANTIC_MEMORY_ENABLED="false",
        MEMORY_SOURCE_RECALL_ENABLED="false", ASTRBOT_INTEGRATION_TOKEN="synthetic-integration-token-at-least-32-characters",
        MULTIPERSONAL_BACKEND_URL="https://backend.invalid", ALLOWED_ORIGINS="https://frontend.invalid",
        SECURITY_MIDDLEWARE_ENABLED="false", LOG_LEVEL="ERROR",
    )
    assert "ENCRYPTION_KEY" not in env
    return env


def _encryption_import_proof(backend, isolated, env, script):
    completed = subprocess.run(
        [sys.executable, "-c", script, str(backend), str(isolated)], cwd=isolated,
        env=env, capture_output=True, text=True, timeout=45,
    )
    # Never echo child output, credentials, ciphertext or exception messages.
    assert completed.returncode == 0, "isolated encryption import probe did not finish"
    lines = [line for line in completed.stdout.splitlines() if line.startswith("ENCRYPTION_IMPORT_PROOF=")]
    assert len(lines) == 1, "isolated import did not produce one safe proof"
    return json.loads(lines[0].split("=", 1)[1])


def test_legacy_only_first_main_import_publishes_migrated_key_before_startup_snapshot(tmp_path):
    backend = Path(__file__).resolve().parents[1]
    isolated = tmp_path / "isolated-backend"
    isolated.mkdir()
    env = _isolated_encryption_import_environment(backend, tmp_path)
    probe = r"""
import base64, json, os, socket, sys
from pathlib import Path
backend, isolated = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(backend))
# Every application/configuration source is temporary before app.main imports.
from app import env as app_env
from infra import encryption
app_env._BACKEND_ROOT = isolated
canonical, legacy = isolated / ".env", isolated / "infra" / ".env"
encryption._ENV_FILE_PATH, encryption._LEGACY_ENV_FILE_PATH = canonical, legacy
key = b"L" * 32
encoded = base64.urlsafe_b64encode(key).decode("ascii")
configured = b"SYNTHETIC_UNRELATED_SETTING=unchanged\n"
canonical.write_bytes(configured)
legacy.parent.mkdir()
legacy_before = f"ENCRYPTION_KEY={encoded}\n".encode()
legacy.write_bytes(legacy_before)
assert "ENCRYPTION_KEY" not in os.environ
existing = encryption.EncryptionManager(key=key).encrypt("synthetic existing legacy data")
def no_network(*args, **kwargs):
    raise AssertionError("Application import must not connect to a model or external service")
socket.socket.connect = no_network
import app.main as main
import app.config as config
from infra.deployment import validate_deployment_environment
snapshot = main.app.state.runtime_container.startup_env
validation = validate_deployment_environment(snapshot)
proof = dict(
    deployment_valid=validation.ok,
    snapshot_same_verified_key=snapshot.get("ENCRYPTION_KEY") == encoded,
    process_same_verified_key=os.environ.get("ENCRYPTION_KEY") == encoded,
    same_legacy_data=config.encryption_mgr.decrypt(existing) == "synthetic existing legacy data",
    canonical_exact=canonical.read_bytes() == configured + f"ENCRYPTION_KEY={encoded}\n".encode(),
    legacy_unchanged=legacy.read_bytes() == legacy_before,
)
os.environ["ENCRYPTION_KEY"] = base64.urlsafe_b64encode(b"N" * 32).decode("ascii")
proof["snapshot_stays_frozen"] = snapshot.get("ENCRYPTION_KEY") == encoded
main.app.state.readiness_probe.shutdown()
print("ENCRYPTION_IMPORT_PROOF=" + json.dumps(proof, sort_keys=True))
"""
    proof = _encryption_import_proof(backend, isolated, env, probe)
    assert proof == dict(deployment_valid=True, snapshot_same_verified_key=True,
                         process_same_verified_key=True, same_legacy_data=True,
                         canonical_exact=True, legacy_unchanged=True, snapshot_stays_frozen=True)


def test_bootstrap_supplies_explicit_encryption_key_before_config_without_dotenv_write(tmp_path):
    backend = Path(__file__).resolve().parents[1]
    isolated = tmp_path / "isolated-backend"
    isolated.mkdir()
    env = _isolated_encryption_import_environment(backend, tmp_path)
    probe = r"""
import base64, json, os, runpy, sys
from pathlib import Path
backend, isolated = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(backend))
from infra import encryption
canonical, legacy = isolated / ".env", isolated / "infra" / ".env"
encryption._ENV_FILE_PATH, encryption._LEGACY_ENV_FILE_PATH = canonical, legacy
configured = b"SYNTHETIC_UNRELATED_SETTING=unchanged\n"
canonical.write_bytes(configured)
assert "ENCRYPTION_KEY" not in os.environ
runpy.run_path(str(backend / "tests" / "conftest.py"))
credential = os.environ.get("ENCRYPTION_KEY")
import app.config as config
roundtrip = config.encryption_mgr.decrypt(config.encryption_mgr.encrypt("synthetic bootstrap data"))
proof = dict(
    bootstrap_supplied_key=isinstance(credential, str) and bool(credential),
    key_decodes_to_32_bytes=isinstance(credential, str) and len(base64.urlsafe_b64decode(credential.encode("ascii"))) == 32,
    environment_preserved=os.environ.get("ENCRYPTION_KEY") == credential,
    public_roundtrip=roundtrip == "synthetic bootstrap data",
    canonical_unchanged=canonical.read_bytes() == configured,
    no_legacy_created=not legacy.exists(),
    no_extra_config_paths=set(isolated.iterdir()) == {canonical},
)
print("ENCRYPTION_IMPORT_PROOF=" + json.dumps(proof, sort_keys=True))
"""
    proof = _encryption_import_proof(backend, isolated, env, probe)
    assert proof == dict(bootstrap_supplied_key=True, key_decodes_to_32_bytes=True,
                         environment_preserved=True, public_roundtrip=True,
                         canonical_unchanged=True, no_legacy_created=True, no_extra_config_paths=True)
