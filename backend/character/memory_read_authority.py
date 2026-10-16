"""Read contracts for stored claims and reviewed source receipts.

These digests are minted from scoped repository reads, never model messages.
They bind a read view to its inputs without comparing a paraphrase to storage.
"""

import hashlib
import json


class MemoryRecordError(ValueError):
    """A memory payload failure identified by a safe schema field."""

    def __init__(self, field: str, message: str) -> None:
        self.field = field
        super().__init__(message)


def record_version(row):
    return hashlib.sha256(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def source_version(row):
    return record_version({name: row[name] for name in ("source_message_id", "body", "observed_at")})


def source_event_ids(row):
    events = row.get("source_event_ids")
    if events is None:
        return ()
    if isinstance(events, str):
        events = json.loads(events)
    if not isinstance(events, (list, tuple, set)):
        raise ValueError("Memory source_event_ids must contain a collection of record IDs")
    return events


def read_versions(row, stored_versions, source_receipts):
    ids = [str(row["id"])]
    ids.extend(str(key) for key in source_event_ids(row) if str(key) in stored_versions)
    ids = tuple(dict.fromkeys(ids))
    claims = tuple((key, stored_versions[key]) for key in ids)
    sources = tuple(
        (key, source_id, source_version(receipt))
        for key in ids
        for source_id, receipt in sorted(source_receipts.get(key, {}).items())
    )
    return claims, sources


def source_record_version(row):
    return record_version({name: row[name] for name in ("source_message_id", "body_sha256", "observed_at")})


def read_source_record_bindings(claims, source_pairs, revisions):
    pairs = tuple((key, source_id) for key, _version in claims for source_id in source_pairs.get(key, ()))
    versions = tuple((key, source_id, source_record_version(revisions[key][source_id]))
                     for key, source_id in pairs if source_id in revisions.get(key, {}))
    return pairs, versions
