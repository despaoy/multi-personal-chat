"""Decode nullable JSON objects stored in TEXT columns."""

import json


def read_json_object(value: str | None, field: str) -> dict:
    if value is None:
        return {}
    decoded = json.loads(value)
    if not isinstance(decoded, dict):
        raise ValueError(f"{field} must be a JSON object")
    return decoded
