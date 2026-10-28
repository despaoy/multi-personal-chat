from __future__ import annotations

import base64
import os

import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from infra import encryption


@pytest.fixture
def isolated_key_files(monkeypatch, tmp_path):
    canonical = tmp_path / "backend" / ".env"
    legacy = tmp_path / "backend" / "infra" / ".env"
    monkeypatch.setattr(encryption, "_ENV_FILE_PATH", canonical)
    monkeypatch.setattr(encryption, "_LEGACY_ENV_FILE_PATH", legacy)
    monkeypatch.delenv("ENCRYPTION_KEY", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "development")
    return canonical, legacy


def test_preconfigured_canonical_key_survives_restart_without_changing_file(monkeypatch, isolated_key_files):
    canonical, legacy = isolated_key_files
    key = bytes(range(32))
    encoded = base64.urlsafe_b64encode(key).decode("ascii")
    canonical.parent.mkdir(parents=True)
    configured = f"SYNTHETIC_UNRELATED_SETTING=unchanged\nENCRYPTION_KEY={encoded}\n".encode()
    canonical.write_bytes(configured)
    existing = encryption.EncryptionManager(key=key).encrypt("synthetic existing encrypted data")

    first = encryption.EncryptionManager()
    assert first.decrypt(existing) == "synthetic existing encrypted data"
    ciphertext = first.encrypt("synthetic new encrypted data")
    monkeypatch.delenv("ENCRYPTION_KEY", raising=False)
    restarted = encryption.EncryptionManager()
    assert restarted.decrypt(existing) == "synthetic existing encrypted data"
    assert restarted.decrypt(ciphertext) == "synthetic new encrypted data"
    assert canonical.read_bytes() == configured and not legacy.exists()


def test_legacy_key_is_migrated_without_rotating_data(monkeypatch, isolated_key_files):
    canonical, legacy = isolated_key_files
    key = AESGCM.generate_key(bit_length=256)
    encoded = base64.urlsafe_b64encode(key).decode("ascii")
    legacy.parent.mkdir(parents=True)
    legacy.write_text(f"ENCRYPTION_KEY={encoded}\n", encoding="utf-8")

    migrated = encryption.EncryptionManager()
    ciphertext = migrated.encrypt("legacy secret")

    assert f"ENCRYPTION_KEY={encoded}" in canonical.read_text(encoding="utf-8")
    legacy.unlink()
    monkeypatch.delenv("ENCRYPTION_KEY", raising=False)
    restarted = encryption.EncryptionManager()
    assert restarted.decrypt(ciphertext) == "legacy secret"


@pytest.mark.parametrize("environment", [None, "development", "production"],
                         ids=["unset-environment", "development", "production"])
def test_every_environment_rejects_missing_key_without_generation_or_write(
    monkeypatch, isolated_key_files, environment,
):
    canonical, legacy = isolated_key_files
    if environment is None:
        monkeypatch.delenv("ENVIRONMENT", raising=False)
    else:
        monkeypatch.setenv("ENVIRONMENT", environment)
    # Keep an existing unrelated configuration byte-for-byte, rather than only
    # checking whether a brand-new dotenv would have appeared.
    canonical.parent.mkdir(parents=True)
    configured = b"SYNTHETIC_UNRELATED_SETTING=unchanged\n"
    canonical.write_bytes(configured)
    before_paths = set(canonical.parent.rglob("*"))

    def forbidden(*args, **kwargs):
        raise AssertionError("Missing configuration must neither generate nor persist a key")

    monkeypatch.setattr(AESGCM, "generate_key", forbidden)
    monkeypatch.setattr(encryption.EncryptionManager, "_save_key_to_env", forbidden)
    with pytest.raises(encryption.KeyManagementError, match="必须显式配置 ENCRYPTION_KEY，禁止自动生成"):
        encryption.EncryptionManager()

    assert "ENCRYPTION_KEY" not in os.environ
    assert canonical.read_bytes() == configured and not legacy.exists()
    assert set(canonical.parent.rglob("*")) == before_paths


@pytest.mark.parametrize("value", ["not-a-valid-key", ""], ids=["malformed", "empty"])
def test_invalid_persisted_key_fails_closed(isolated_key_files, value):
    canonical, legacy = isolated_key_files
    canonical.parent.mkdir(parents=True)
    canonical.write_text(f"ENCRYPTION_KEY={value}\n", encoding="utf-8")

    legacy.parent.mkdir(parents=True)
    legacy.write_text(
        "ENCRYPTION_KEY=" + base64.urlsafe_b64encode(bytes(range(32))).decode("ascii") + "\n", encoding="utf-8"
    )
    before = {path: path.read_bytes() for path in (canonical, legacy)}
    with pytest.raises(encryption.KeyManagementError, match="ENCRYPTION_KEY"):
        encryption.EncryptionManager()
    assert all(path.read_bytes() == value for path, value in before.items())
    assert "ENCRYPTION_KEY" not in os.environ


def test_explicit_environment_key_never_reads_unused_key_files(monkeypatch, isolated_key_files):
    canonical, _ = isolated_key_files
    key = bytes(range(32))
    encoded = base64.urlsafe_b64encode(key).decode("ascii")
    monkeypatch.setenv("ENCRYPTION_KEY", encoded)
    reads = []

    def forbidden(path):
        reads.append(path)
        raise PermissionError("synthetic unreadable unused key file")

    monkeypatch.setattr(encryption.EncryptionManager, "_read_key_from_file", staticmethod(forbidden))
    manager = encryption.EncryptionManager()
    ciphertext = encryption.EncryptionManager(key=key).encrypt("synthetic existing encrypted data")
    assert manager.decrypt(ciphertext) == "synthetic existing encrypted data"
    assert manager.decrypt(manager.encrypt("synthetic new data")) == "synthetic new data"
    assert reads == [] and not canonical.exists()


@pytest.mark.parametrize("value", ["not-a-valid-key", ""], ids=["malformed", "empty"])
def test_invalid_environment_key_cannot_read_or_switch_to_valid_file(monkeypatch, isolated_key_files, value):
    canonical, _ = isolated_key_files
    canonical.parent.mkdir(parents=True)
    persisted = b"ENCRYPTION_KEY=" + base64.urlsafe_b64encode(bytes(range(32))) + b"\n"
    canonical.write_bytes(persisted)
    monkeypatch.setenv("ENCRYPTION_KEY", value)
    reads = []

    def forbidden(path):
        reads.append(path)
        raise AssertionError("Invalid explicit key must fail before reading another source")

    monkeypatch.setattr(encryption.EncryptionManager, "_read_key_from_file", staticmethod(forbidden))
    with pytest.raises(encryption.KeyManagementError, match="环境变量"):
        encryption.EncryptionManager()
    assert reads == [] and canonical.read_bytes() == persisted


def test_selected_key_file_read_failure_is_not_hidden_by_legacy_or_generation(monkeypatch, isolated_key_files):
    canonical, legacy = isolated_key_files
    canonical.parent.mkdir(parents=True)
    canonical.write_text("# selected key file\n", encoding="utf-8")
    legacy.parent.mkdir(parents=True)
    legacy.write_text("ENCRYPTION_KEY=" + base64.urlsafe_b64encode(bytes(range(32))).decode("ascii") + "\n", encoding="utf-8")
    before = {path: path.read_bytes() for path in (canonical, legacy)}
    cause = PermissionError("synthetic selected key file unreadable")
    actual = type(canonical).read_text
    reads = []

    def read_text(path, *args, **kwargs):
        reads.append(path)
        if path == canonical:
            raise cause
        return actual(path, *args, **kwargs)

    monkeypatch.setattr(type(canonical), "read_text", read_text)
    with pytest.raises(encryption.KeyManagementError, match="读取密钥文件") as caught:
        encryption.EncryptionManager()
    assert caught.value.__cause__ is cause and reads == [canonical]
    assert all(path.read_bytes() == value for path, value in before.items())


def test_decryption_accepts_only_complete_authenticated_ciphertext():
    manager = encryption.EncryptionManager(key=bytes(range(32)))
    source = "完整合成原文：尚未付款，仅周五执行，出差顺延。"
    ciphertext = manager.encrypt(source)
    assert manager.decrypt(ciphertext) == source
    assert manager.decrypt(manager.encrypt("")) == ""
    prefix = "ENC:AES256GCM:"
    nonce, body, tag = ciphertext.removeprefix(prefix).split(":")
    altered_tag = bytearray(base64.urlsafe_b64decode(tag))
    altered_tag[0] ^= 1
    corrupt = prefix + ":".join([nonce, body, base64.urlsafe_b64encode(altered_tag).decode("ascii")])
    invalid = {
        "plaintext": source,
        "wrong_prefix": ciphertext.replace("AES256GCM", "AES256GCX", 1),
        "invalid_base64": prefix + ":".join([nonce + "$", body, tag]),
        "corrupt_authentication_tag": corrupt,
    }
    accepted = []
    for label, value in invalid.items():
        try:
            manager.decrypt(value)
        except encryption.DecryptionError:
            pass
        else:
            accepted.append(label)
    assert not accepted, f"Invalid ciphertexts accepted as successes: {accepted}"
