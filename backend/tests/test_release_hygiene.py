"""Regression checks for the read-only release file audit."""

import pytest
from scripts.check_release_hygiene import forbidden_artifact, validate_payload


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        "backend/.env.production",
        "qq_assistant.db",
        "data/a.db-wal",
        ".codex-test-runtime-x/a",
        "local-notes/a.md",
    ],
)
def test_private_artifacts_are_rejected(path):
    assert forbidden_artifact(path)


@pytest.mark.parametrize(
    "path", [".env.example", "docs/README.md", "backend/db/database.py", "archive/p6_rag_pipeline/README.md"]
)
def test_source_and_history_are_allowed(path):
    assert not forbidden_artifact(path)


@pytest.mark.parametrize("path,payload", [("a.py", b"def :"), ("a.json", b"{"), ("a.jsonl", b'{}\n{"')])
def test_invalid_source_is_detected(path, payload):
    with pytest.raises((SyntaxError, ValueError)):
        validate_payload(path, payload)


def test_jsonl_allows_blank_lines():
    validate_payload("a.jsonl", b'{}\n\n{"ok": true}\n')
