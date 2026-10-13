"""Memory timestamps shared by source writes, recall and context compilation."""

from datetime import datetime, timedelta, timezone

DEFAULT_MEMORY_TIMEZONE = timezone(timedelta(hours=8))


def observation_clock(value: datetime | None = None) -> datetime:
    observation = value if value is not None else datetime.now(timezone.utc)
    if observation.tzinfo is None or observation.utcoffset() is None:
        raise ValueError("memory observation time must include a timezone")
    return observation.astimezone(timezone.utc)


def parse_memory_timestamp(value: object, field: str) -> datetime | None:
    """Unspecified bounds are optional; malformed stored times are errors.

    Naive historical storage retains its established UTC interpretation.
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            value = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"Memory {field} must be a valid ISO timestamp") from exc
    if not isinstance(value, datetime):
        raise ValueError(f"Memory {field} must be a valid ISO timestamp")
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
