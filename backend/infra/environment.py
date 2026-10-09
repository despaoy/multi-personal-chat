"""Explicit environment values shared by feature configuration readers."""
from collections.abc import Mapping


def read_bool(env: Mapping[str, object], name: str, default: bool = False) -> bool:
    value = str(env.get(name, str(default))).strip().lower()
    if value not in {"1", "true", "yes", "on", "0", "false", "no", "off"}:
        raise ValueError(f"{name} must be an explicit boolean")
    return value in {"1", "true", "yes", "on"}
