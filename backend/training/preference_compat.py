"""Resolve installed TRL capabilities before expensive model loading."""

from __future__ import annotations

import importlib
import inspect


def resolve_preference_backend(method: str):
    if method not in {"dpo", "orpo"}:
        raise ValueError("preference method must be dpo or orpo")
    trl = importlib.import_module("trl")
    prefix = method.upper()
    try:
        return getattr(trl, f"{prefix}Trainer"), getattr(trl, f"{prefix}Config")
    except (AttributeError, ImportError) as exc:
        raise RuntimeError(
            f"installed TRL does not expose {prefix}; choose a supported method or a tested compatible environment"
        ) from exc


def preference_length_kwargs(config_class, *, max_length: int, max_prompt_length: int) -> dict:
    if not 0 < max_prompt_length < max_length:
        raise ValueError("preference lengths require 0 < max_prompt_length < max_length")
    parameters = inspect.signature(config_class).parameters
    if "max_length" not in parameters:
        raise RuntimeError("installed TRL has an unsupported sequence-length contract")
    result = {"max_length": max_length}
    # Older TRL truncates prompts independently. New DPO versions removed this
    # field, but the caller still validates complete prompt+completion lengths.
    if "max_prompt_length" in parameters:
        result["max_prompt_length"] = max_prompt_length
    return result
