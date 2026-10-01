"""Validate server-owned logical keys protected during atomic claim erasure."""


def protected_keys(value):
    if not isinstance(value, (tuple, list)) or len(value) > 200:
        raise ValueError("Expected at most 200 protected logical keys")
    if any(not isinstance(key, str) or not key.strip() or len(key) > 255 for key in value):
        raise ValueError("Invalid protected logical key")
    return tuple(dict.fromkeys(key.strip() for key in value))
