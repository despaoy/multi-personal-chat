"""Shared training-task storage contract for SQLite and PostgreSQL."""

import json
from collections.abc import Mapping

from db.json_fields import read_json_object


def training_task_record(data: Mapping) -> dict:
    """Decode persisted configuration without hiding malformed task records."""
    return {
        "task_id": data.get("task_id") or data.get("id"),
        "lora_name": data.get("lora_name", ""),
        "status": data.get("status", "pending"),
        "progress": float(data.get("progress", 0) or 0),
        "error_message": data.get("error_message", ""),
        "config": read_json_object(data.get("config_json"), "training task config"),
        "created_at": data.get("created_at") or data.get("createdAt", ""),
        "updated_at": data.get("updated_at") or data.get("updatedAt", ""),
    }


def training_config_json(config: object) -> str:
    """Validate before writing, so failed configuration updates remain atomic."""
    if not isinstance(config, dict):
        raise ValueError("training task config must be a JSON object")
    return json.dumps(config, ensure_ascii=False)


def training_config_update(data: Mapping) -> dict:
    """Preserve explicit config_json precedence and validate the selected value."""
    if "config_json" in data:
        read_json_object(data["config_json"], "training task config")
        return {"config_json": data["config_json"]}
    if "config" in data:
        return {"config_json": training_config_json(data["config"])}
    return {}
