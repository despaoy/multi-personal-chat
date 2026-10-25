"""Startup and explicit scans share one evidence-backed metadata reader."""

import json

import pytest

from db.lora_metadata import read_lora_metadata


@pytest.fixture
def adapter(tmp_path):
    (tmp_path / "adapter_config.json").write_text('{"r": 8, "lora_alpha": 16}')
    return tmp_path


def test_model_weights_do_not_imply_training_progress(adapter):
    (adapter / "adapter_model.safetensors").write_bytes(b"synthetic")
    assert read_lora_metadata(adapter) == {"rank": 8, "alpha": 16, "trained_steps": 0, "total_steps": 0}


@pytest.mark.parametrize("filename", ["adapter_config.json", "trainer_state.json"])
@pytest.mark.parametrize("content", ["broken", "[]", "null"])
def test_corrupt_metadata_is_not_replaced_with_defaults(adapter, filename, content):
    (adapter / filename).write_text(content)
    with pytest.raises(ValueError):
        read_lora_metadata(adapter)


def test_final_metadata_uses_latest_numeric_checkpoint(adapter):
    final = adapter / "final"
    final.mkdir()
    (final / "adapter_config.json").write_text('{"r": 4}')
    for name, step in [("checkpoint-2", 2), ("checkpoint-10", 10), ("checkpoint-backup", 99)]:
        checkpoint = adapter / name
        checkpoint.mkdir()
        (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": step, "max_steps": 20}))
    assert read_lora_metadata(final)["trained_steps"] == 10
    (final / "trainer_state.json").write_text('{"global_step": 15, "max_steps": 20}')
    assert read_lora_metadata(final)["trained_steps"] == 15


def test_missing_total_does_not_invent_a_completed_total(adapter):
    (adapter / "trainer_state.json").write_text('{"global_step": 8}')
    assert read_lora_metadata(adapter)["total_steps"] == 0


def test_api_scan_and_database_startup_share_reader():
    from api import loras
    from db import database

    assert loras.read_lora_metadata is database.read_lora_metadata is read_lora_metadata
