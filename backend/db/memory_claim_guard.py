"""Shared preconditions checked while holding the target's write lock."""

from datetime import datetime


class MemoryClaimConflict(ValueError):
    """A proposal's target no longer supports the proposed mutation."""


def _aware_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None
    # Legacy storage timestamps may be local and lack a timezone. Do not
    # invent their timezone and reject legitimate updates on that assumption.
    return parsed if parsed.utcoffset() is not None else None


def observation_precedes(observed_at: str | None, target_observed_at: str | None) -> bool:
    target_time = _aware_time(target_observed_at)
    incoming_time = _aware_time(observed_at)
    return target_time is not None and incoming_time is not None and incoming_time < target_time


def validate_memory_target(
    relation: str, target_status: str, target_observed_at: str | None, observed_at: str | None,
) -> None:
    if relation not in {"SUPERSEDE", "MERGE", "RETRACT"}:
        return
    if target_status != "active":
        raise MemoryClaimConflict("memory target changed; reload before proposing a mutation")
    if observation_precedes(observed_at, target_observed_at):
        raise MemoryClaimConflict("memory observation precedes target; stale mutation rejected")
