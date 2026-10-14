"""Shared LoRA naming helpers for database and vLLM contracts."""

from __future__ import annotations

import json
import os
import re

_DEFAULT_ALIASES = {
    "hutao_lora_7b": "hutao",
    "minamo_lora": "minamo",
}
_SAFE_MODEL_NAME = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def get_lora_name_map() -> dict[str, str]:
    aliases = dict(_DEFAULT_ALIASES)
    configured = os.getenv("LORA_SERVED_NAME_MAP", "").strip()
    if configured:
        try:
            parsed = json.loads(configured)
        except json.JSONDecodeError as exc:
            raise ValueError("LORA_SERVED_NAME_MAP must be a JSON object") from exc
        if not isinstance(parsed, dict):
            raise ValueError("LORA_SERVED_NAME_MAP must be a JSON object")
        aliases.update({str(key): str(value) for key, value in parsed.items()})

    for database_name, served_name in aliases.items():
        if not _SAFE_MODEL_NAME.fullmatch(database_name):
            raise ValueError(f"Unsafe LoRA database name: {database_name!r}")
        if not _SAFE_MODEL_NAME.fullmatch(served_name):
            raise ValueError(f"Unsafe LoRA served name: {served_name!r}")
    return aliases


def resolve_lora_served_name(database_name: str) -> str:
    if not _SAFE_MODEL_NAME.fullmatch(database_name):
        raise ValueError("LoRA name contains unsupported characters")
    return get_lora_name_map().get(database_name, database_name)
