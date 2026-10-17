"""An output token count is public configuration; credential protection still applies."""

import pytest

from api import config


def test_known_output_length_is_visible_and_all_credentials_remain_masked():
    keys = ["openaiCompatApiKey", "api_key", "clientSecret", "password", "botToken", "unknownAccessToken"]
    original = {key: "fiction-private-credential-value" for key in keys}
    original.update(
        maxTokens=3072,
        temperature=0.4,
        astrbotIntegrationToken="fiction-server-only",
        astrbotIntegrationTokens="fiction-server-only",
    )
    masked = config._mask_config(original)
    assert masked["maxTokens"] == 3072 and masked["temperature"] == 0.4
    assert all(masked[key] != original[key] and "****" in masked[key] for key in keys)
    assert "astrbotIntegrationToken" not in masked and "astrbotIntegrationTokens" not in masked
    assert "fiction-private-credential-value" not in str(masked)
    assert all(original[key] == "fiction-private-credential-value" for key in keys)


@pytest.mark.asyncio
async def test_get_config_returns_saved_output_length_without_losing_masking(monkeypatch):
    async def snapshot(*args):
        return {"maxTokens": 1536, "openaiCompatApiKey": "fiction-private-credential"}, False

    monkeypatch.setattr(config, "run_db", snapshot)
    result = await config.get_config(current_user={"role": "admin"})
    assert result["config"]["maxTokens"] == 1536 and result["cached"] is False
    assert "fiction-private-credential" not in str(result)
