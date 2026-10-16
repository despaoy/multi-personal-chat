from __future__ import annotations

import base64

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


def test_development_key_persists_and_survives_restart(monkeypatch, isolated_key_files):
    canonical, _ = isolated_key_files
    first = encryption.EncryptionManager()
    ciphertext = first.encrypt("persistent secret")

    assert canonical.exists()
    assert "ENCRYPTION_KEY=" in canonical.read_text(encoding="utf-8")

    monkeypatch.delenv("ENCRYPTION_KEY", raising=False)
    restarted = encryption.EncryptionManager()
    assert restarted.decrypt(ciphertext) == "persistent secret"


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


def test_production_rejects_missing_encryption_key(monkeypatch, isolated_key_files):
    canonical, _ = isolated_key_files
    monkeypatch.setenv("ENVIRONMENT", "production")

    with pytest.raises(encryption.KeyManagementError, match="必须显式配置"):
        encryption.EncryptionManager()

    assert not canonical.exists()


def test_invalid_persisted_key_fails_closed(isolated_key_files):
    canonical, _ = isolated_key_files
    canonical.parent.mkdir(parents=True)
    canonical.write_text("ENCRYPTION_KEY=not-a-valid-key\n", encoding="utf-8")

    with pytest.raises(encryption.KeyManagementError, match="ENCRYPTION_KEY"):
        encryption.EncryptionManager()


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


def test_invalid_environment_key_cannot_read_or_switch_to_valid_file(monkeypatch, isolated_key_files):
    canonical, _ = isolated_key_files
    canonical.parent.mkdir(parents=True)
    persisted = b"ENCRYPTION_KEY=" + base64.urlsafe_b64encode(bytes(range(32))) + b"\n"
    canonical.write_bytes(persisted)
    monkeypatch.setenv("ENCRYPTION_KEY", "not-a-valid-key")
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
