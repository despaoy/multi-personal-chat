"""Speaker identity contract for role-playing chat training records.

Chat roles describe the protocol: the assistant is the trained character and
the user is whoever speaks to that character.  A speaker label is separate
metadata and must never be promoted to a system instruction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


CANONICAL_CHARACTER = "canonical_character"
GENERIC_USER = "generic_user"
ALLOWED_INTERLOCUTOR_KINDS = frozenset({CANONICAL_CHARACTER, GENERIC_USER})


@dataclass(frozen=True)
class SpeakerContract:
    """Resolved record-level speaker identity."""

    kind: str | None
    record_label: str | None
    explicit: bool


def _optional_label(value: Any, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    label = value.strip()
    if not label:
        raise ValueError(f"{field} must not be empty")
    return label


def _legacy_record_label(metadata: Mapping[str, Any]) -> str | None:
    for key in ("context_speaker_label", "interlocutor"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    interlocutors = metadata.get("interlocutors")
    if isinstance(interlocutors, Sequence) and not isinstance(
        interlocutors, (str, bytes)
    ):
        labels = [
            value.strip()
            for value in interlocutors
            if isinstance(value, str) and value.strip()
        ]
        if len(labels) == 1:
            return labels[0]
    return None


def parse_speaker_contract(record: Mapping[str, Any]) -> SpeakerContract:
    """Parse explicit V4 metadata while retaining legacy dataset support."""

    metadata = record.get("metadata")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise ValueError("record metadata must be an object")

    raw_kind = metadata.get("interlocutor_kind")
    explicit = raw_kind is not None
    if raw_kind is not None and not isinstance(raw_kind, str):
        raise ValueError("metadata.interlocutor_kind must be a string")
    if raw_kind is not None and raw_kind not in ALLOWED_INTERLOCUTOR_KINDS:
        raise ValueError(f"unsupported interlocutor_kind: {raw_kind!r}")

    explicit_label = _optional_label(
        metadata.get("interlocutor_label"), field="metadata.interlocutor_label"
    )
    legacy_label = _legacy_record_label(metadata)

    if raw_kind == GENERIC_USER:
        if explicit_label or legacy_label:
            raise ValueError("generic_user records must not contain a named speaker")
        return SpeakerContract(kind=GENERIC_USER, record_label=None, explicit=True)

    if raw_kind == CANONICAL_CHARACTER:
        if not explicit_label:
            raise ValueError("canonical_character records require interlocutor_label")
        if legacy_label and legacy_label != explicit_label:
            raise ValueError(
                "interlocutor_label conflicts with legacy context speaker metadata"
            )
        return SpeakerContract(
            kind=CANONICAL_CHARACTER,
            record_label=explicit_label,
            explicit=True,
        )

    if explicit_label:
        raise ValueError("interlocutor_label requires interlocutor_kind")
    return SpeakerContract(
        kind=CANONICAL_CHARACTER if legacy_label else None,
        record_label=legacy_label,
        explicit=explicit,
    )


def resolve_message_speaker(
    raw_message: Mapping[str, Any],
    *,
    role: str,
    contract: SpeakerContract,
) -> str | None:
    """Resolve one message label, preferring an explicit per-turn identity."""

    message_label = _optional_label(
        raw_message.get("speaker_label"), field="message.speaker_label"
    )
    if message_label and role != "user":
        raise ValueError("speaker_label is only allowed on user messages")
    if message_label and contract.kind == GENERIC_USER:
        raise ValueError("generic_user records must not contain named user turns")
    if role != "user" or contract.kind == GENERIC_USER:
        return None
    return message_label or contract.record_label
