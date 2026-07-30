"""Shared LoRA naming helpers for database and vLLM contracts."""

from __future__ import annotations

import json
import logging
import os
import re

logger = logging.getLogger(__name__)

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


def safe_resolve_lora_served_name(database_name: str, check_compatibility: bool = False) -> str:
    """安全解析 LoRA served name：可选兼容性检查，不兼容时降级到 default。

    Args:
        database_name: 数据库中的 LoRA 名称
        check_compatibility: 是否执行兼容性检查（默认 False，避免运行时开销）

    Returns:
        兼容则返回 served name，不兼容则返回 "default"
    """
    if database_name == "default" or not database_name:
        return "default"

    if check_compatibility:
        try:
            from inference.adapter_checker import safe_resolve_lora
            safe_name = safe_resolve_lora(database_name)
            if safe_name == "default":
                return "default"
        except Exception:
            pass  # 检查失败时不阻止，保守地使用原名

    return resolve_lora_served_name(database_name)
