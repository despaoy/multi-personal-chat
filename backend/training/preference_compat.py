"""Resolve installed TRL capabilities before expensive model loading."""

from __future__ import annotations

import importlib
import inspect


def render_preference_record(row, tokenizer):
    """Use the same non-thinking chat template as SFT and persona inference."""
    values = {key: row[key] for key in ("prompt", "chosen", "rejected")}
    if isinstance(values["prompt"], str):
        return values
    from trl.data_utils import maybe_apply_chat_template

    return maybe_apply_chat_template(values, tokenizer, enable_thinking=False)


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


def reference_adapter_kwargs(config_class, *, method: str, adapter_path: str) -> dict:
    """Never depend on a TRL version's implicit PEFT reference behavior."""
    if method != "dpo" or not adapter_path:
        return {}
    parameters = inspect.signature(config_class).parameters
    if not {"model_adapter_name", "ref_adapter_name"} <= parameters.keys():
        raise RuntimeError(
            "installed TRL does not support explicit SFT reference adapters; "
            "use the documented preference training environment"
        )
    return {"model_adapter_name": "default", "ref_adapter_name": "reference"}
