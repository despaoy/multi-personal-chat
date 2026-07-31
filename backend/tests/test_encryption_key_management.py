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