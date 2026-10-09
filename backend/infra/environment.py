"""Explicit environment values shared by feature configuration readers."""
from collections.abc import Mapping


def read_bool(env: Mapping[str, object], name: str, default: bool = False) -> bool:
    value = str(env.get(name, str(default))).strip().lower()
    if value not in {"1", "true", "yes", "on", "0", "false", "no", "off"}:
        raise ValueError(f"{name} must be an explicit boolean")
    return value in {"1", "true", "yes", "on"}


def parse_unit_interval(value: object, name: str) -> float:
    message = f"{name} must be a finite number between 0 and 1"
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if isinstance(value, bool) or not 0 <= number <= 1:
        raise ValueError(message)
    return number
