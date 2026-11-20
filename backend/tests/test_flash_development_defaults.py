"""Flash defaults apply only where users have not already selected settings."""

from db.database import SQLiteDB
from inference.generation_options import read_generation_options
from inference.model_manager import OpenAICompatProvider


def test_new_database_defaults_to_flash_and_preserves_existing_selection(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_COMPAT_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_COMPAT_BASE_URL", raising=False)
    path = tmp_path / "settings.db"
    db = SQLiteDB(path)
    try:
        assert db.config["modelProvider"] == "openai_compat"
        assert db.config["openaiCompatModel"] == ""
        provider = OpenAICompatProvider(database=db)
        assert provider.model == "deepseek-flash"
        assert provider.base_url == "https://api.deepseek.com"
        monkeypatch.setenv("OPENAI_COMPAT_MODEL", "environment-model")
        monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "https://provider.example.invalid/v1")
        provider = OpenAICompatProvider(database=db)
        assert provider.model == "environment-model"
        assert provider.base_url == "https://provider.example.invalid/v1"
        assert float(db.config["topP"]) == 1.0
        assert not db.config["openaiCompatApiKey"]
        db.update_config({"modelProvider": "vllm", "openaiCompatModel": "explicit-model"})
    finally:
        db.close_connection()
    db = SQLiteDB(path)
    try:
        assert db.config["modelProvider"] == "vllm"
        assert db.config["openaiCompatModel"] == "explicit-model"
        assert OpenAICompatProvider(database=db).model == "explicit-model"
    finally:
        db.close_connection()


def test_provider_default_never_overrides_explicit_sampling(monkeypatch):
    monkeypatch.delenv("VLLM_TOP_P", raising=False)
    assert read_generation_options({}, default_top_p=1.0)["topP"] == 1.0
    assert read_generation_options({"topP": 0.8}, default_top_p=1.0)["topP"] == 0.8
    monkeypatch.setenv("VLLM_TOP_P", "0.95")
    assert read_generation_options({}, default_top_p=1.0)["topP"] == 0.95
