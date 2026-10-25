"""Read adapter metadata without fabricating training progress."""

from pathlib import Path

from db.json_fields import read_json_object


def read_lora_metadata(adapter_path: Path) -> dict:
    config_path = adapter_path / "adapter_config.json"
    config = read_json_object(config_path.read_text(encoding="utf-8"), str(config_path))
    state_path = adapter_path / "trainer_state.json"
    if not state_path.exists() and adapter_path.name == "final":
        checkpoints = [
            path
            for path in adapter_path.parent.iterdir()
            if path.is_dir() and path.name.startswith("checkpoint-") and path.name.removeprefix("checkpoint-").isdigit()
        ]
        if checkpoints:
            latest = max(checkpoints, key=lambda path: int(path.name.removeprefix("checkpoint-")))
            state_path = latest / "trainer_state.json"
    state = read_json_object(state_path.read_text(encoding="utf-8"), str(state_path)) if state_path.exists() else {}
    return {
        "rank": config.get("r", 0),
        "alpha": config.get("lora_alpha", 0),
        "trained_steps": state.get("global_step", 0),
        "total_steps": state.get("max_steps", 0),
    }
