"""Flash defaults apply only where users have not already selected settings."""

from db.database import SQLiteDB
from inference.generation_options import read_generation_options


def test_new_database_defaults_to_flash_and_preserves_existing_selection(tmp_path):
    path = tmp_path / "settings.db"
    db = SQLiteDB(path)
    try:
        assert db.config["modelProvider"] == "openai_compat"
        assert db.config["openaiCompatModel"] == "deepseek-flash"
        assert float(db.config["topP"]) == 1.0
        assert not db.config["openaiCompatApiKey"]
        db.update_config({"modelProvider": "vllm", "openaiCompatModel": "explicit-model"})
    finally:
        db.close_connection()
    db = SQLiteDB(path)
    try:
        assert db.config["modelProvider"] == "vllm"
        assert db.config["openaiCompatModel"] == "explicit-model"
    finally:
        db.close_connection()


def test_provider_default_never_overrides_explicit_sampling(monkeypatch):
    monkeypatch.delenv("VLLM_TOP_P", raising=False)
    assert read_generation_options({}, default_top_p=1.0)["topP"] == 1.0
    assert read_generation_options({"topP": 0.8}, default_top_p=1.0)["topP"] == 0.8
    monkeypatch.setenv("VLLM_TOP_P", "0.95")
    assert read_generation_options({}, default_top_p=1.0)["topP"] == 0.95
