"""Trusted source-time normalization shared by memory write paths."""

from datetime import datetime, timedelta, timezone

DEFAULT_MEMORY_TIMEZONE = timezone(timedelta(hours=8))


def observation_clock(value: datetime | None = None) -> datetime:
    observation = value if value is not None else datetime.now(timezone.utc)
    if observation.tzinfo is None or observation.utcoffset() is None:
        raise ValueError("memory observation time must include a timezone")
    return observation.astimezone(timezone.utc)
