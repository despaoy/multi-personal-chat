"""Shared persisted revision and completion format for both DB backends."""

REVISION_KEY = "vector_index_rebuild_revision"
STATUS_KEY = "vector_index_rebuild_status"


def parse_revision(raw):
    revision = int(raw)
    if revision < 0 or str(revision) != str(raw):
        raise ValueError("Invalid knowledge index revision")
    return revision


def complete_status(count, fingerprint, revision):
    revision = parse_revision(revision)
    if count < 0 or not fingerprint or ":" in fingerprint:
        raise ValueError("Invalid knowledge index completion")
    return f"complete:{count}:{fingerprint}:{revision}"


def parse_status(raw):
    """Read current status and the two documented legacy completion formats."""
    if raw in ("", "dirty"):
        return raw, 0, "", -1
    parts = raw.split(":")
    if len(parts) not in {"building": (2,), "complete": (2, 3, 4)}.get(parts[0], ()):
        raise ValueError("Invalid knowledge index status format")
    count = int(parts[1])
    if count < 0:
        raise ValueError("Invalid knowledge index chunk count")
    fingerprint = parts[2] if len(parts) >= 3 else ""
    if len(parts) >= 3 and not fingerprint:
        raise ValueError("Invalid knowledge index fingerprint")
    revision = parse_revision(parts[3]) if len(parts) == 4 else -1
    return parts[0], count, fingerprint, revision
