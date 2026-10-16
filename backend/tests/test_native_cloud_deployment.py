"""Cloud deployments must validate the selected provider without a local URL."""

import base64

import pytest

from infra.deployment import validate_deployment_environment


def _cloud_environment():
    return {
        "ENVIRONMENT": "production",
        "DATABASE_URL": "postgresql://app:fixture@database/app",
        "JWT_SECRET": "j" * 48,
        "ENCRYPTION_KEY": base64.urlsafe_b64encode(b"e" * 32).decode(),
        "ASTRBOT_INTEGRATION_TOKEN": "a" * 48,
        "MULTIPERSONAL_BACKEND_URL": "https://backend.example.test",
        "ALLOWED_ORIGINS": "https://app.example.test",
        "MODEL_PROVIDER": "openai_compat",
        "VLLM_ENABLED": "false",
        "OPENAI_COMPAT_BASE_URL": "https://api.example.test",
        "OPENAI_COMPAT_MODEL": "fixture-cloud-model",
        "OPENAI_COMPAT_API_KEY": "fixture-credential",
    }


def test_complete_cloud_configuration_starts_without_local_model_url():
    result = validate_deployment_environment(_cloud_environment())
    assert result.ok, result.errors
    assert not any("VLLM" in message for message in result.warnings)


@pytest.mark.parametrize("missing", ["OPENAI_COMPAT_BASE_URL", "OPENAI_COMPAT_MODEL", "OPENAI_COMPAT_API_KEY"])
def test_cloud_configuration_requires_its_own_complete_settings(missing):
    env = _cloud_environment()
    del env[missing]
    # A configured local service must not conceal a missing cloud dependency.
    env["VLLM_BASE_URL"] = "http://local-model.example.test"
    result = validate_deployment_environment(env)
    assert not result.ok
    assert any(missing in error for error in result.errors)
    assert "fixture-credential" not in str(result)


@pytest.mark.parametrize("provider", ["vllm", ""])
def test_selected_or_default_local_provider_still_requires_local_url(provider):
    env = _cloud_environment()
    env["MODEL_PROVIDER"] = provider
    result = validate_deployment_environment(env)
    assert not result.ok
    assert any("VLLM_BASE_URL" in error for error in result.errors)



@pytest.mark.parametrize("key,accepted", [
    pytest.param(None, False, id="missing-key"),
    pytest.param("not-a-valid-key", False, id="invalid-key"),
    pytest.param(base64.urlsafe_b64encode(b"e" * 32).decode(), True, id="explicit-valid-key"),
])
def test_development_requires_an_explicit_valid_encryption_key(key, accepted):
    env = _cloud_environment()
    env["ENVIRONMENT"] = "development"
    if key is None:
        env.pop("ENCRYPTION_KEY")
    else:
        env["ENCRYPTION_KEY"] = key
    result = validate_deployment_environment(env)
    assert result.ok is accepted
    encryption_errors = [error for error in result.errors if "ENCRYPTION_KEY" in error]
    assert bool(encryption_errors) is not accepted
    assert not any("auto-generate" in warning for warning in result.warnings)


@pytest.mark.parametrize("representation,accepted", [
    pytest.param("urlsafe", True, id="valid-urlsafe"),
    pytest.param("standard", True, id="valid-standard-base64"),
    pytest.param("punctuation", False, id="illegal-character-inserted"),
    pytest.param("padding", False, id="excess-padding"),
])
def test_encryption_manager_and_deployment_share_strict_configured_key_decoding(
    monkeypatch, tmp_path, representation, accepted,
):
    from infra import encryption

    # These bytes force both encodings to exercise their distinct alphabets;
    # all credentials and ciphertext in this test are explicitly synthetic.
    raw = bytes([251, 255]) * 16
    urlsafe = base64.urlsafe_b64encode(raw).decode("ascii")
    standard = base64.b64encode(raw).decode("ascii")
    assert "-" in urlsafe and "_" in urlsafe
    assert "+" in standard and "/" in standard
    if representation == "standard":
        encoded = standard
    elif representation == "punctuation":
        encoded = urlsafe[:8] + "!" + urlsafe[8:]
    elif representation == "padding":
        encoded = urlsafe + "="
    else:
        assert representation == "urlsafe"
        encoded = urlsafe

    canonical = tmp_path / "backend" / ".env"
    legacy = tmp_path / "backend" / "infra" / ".env"
    monkeypatch.setattr(encryption, "_ENV_FILE_PATH", canonical)
    monkeypatch.setattr(encryption, "_LEGACY_ENV_FILE_PATH", legacy)
    monkeypatch.setenv("ENCRYPTION_KEY", encoded)
    env = _cloud_environment()
    # Pass the exact supplied representation into both production entry points.
    # In particular, do not trim, filter, repair padding or re-encode it here.
    env["ENCRYPTION_KEY"] = encoded
    validation = validate_deployment_environment(env)

    if accepted:
        existing = encryption.EncryptionManager(key=raw).encrypt("synthetic preexisting encrypted data")
        configured = encryption.EncryptionManager()
        assert configured.decrypt(existing) == "synthetic preexisting encrypted data"
        assert validation.ok and validation.errors == []
    else:
        with pytest.raises(encryption.KeyManagementError, match="环境变量") as caught:
            encryption.EncryptionManager()
        assert isinstance(caught.value.__cause__, ValueError)
        assert encoded not in str(caught.value) and encoded not in str(caught.value.__cause__)
        assert validation.ok is False
        assert validation.errors and all("ENCRYPTION_KEY" in error for error in validation.errors)
        assert encoded not in repr(validation)
    assert not canonical.exists() and not legacy.exists()
