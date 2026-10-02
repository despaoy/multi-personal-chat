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
