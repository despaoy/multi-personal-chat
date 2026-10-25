"""The two documented document-import probes share behavior, not identities."""

from unittest.mock import AsyncMock

import pytest

from evaluation import document_read_versions_probe, original_body_coverage_probe
from evaluation.document_import_probe import parse_args, run_document_import_probe


@pytest.mark.parametrize(
    "entrypoint,seed_calls,session_id",
    [
        (document_read_versions_probe, 1, "stage75-native"),
        (original_body_coverage_probe, 2, "stage76-native"),
    ],
)
async def test_entrypoints_keep_distinct_seed_contracts(monkeypatch, entrypoint, seed_calls, session_id):
    args = parse_args(["--phase", "/unused", "--api-key-file", "/unused-key"])
    runner = AsyncMock(return_value="completed")
    monkeypatch.setattr(entrypoint, "run_document_import_probe", runner)

    assert await entrypoint.run(args) == "completed"
    runner.assert_awaited_once_with(args, expected_seed_primary_calls=seed_calls, session_id=session_id)


def test_shared_cli_preserves_defaults_and_explicit_options():
    required = ["--phase", "/phase", "--api-key-file", "/key"]
    defaults = parse_args(required)
    assert vars(defaults) == {
        "phase": "/phase",
        "api_key_file": "/key",
        "variant": "native-pg",
        "answer_tokens": 2048,
        "expect_primary_blocked": False,
        "reuse_verified_seed": False,
    }
    options = parse_args(
        required
        + [
            "--variant",
            "native-pg-fixed",
            "--answer-tokens",
            "1024",
            "--expect-primary-blocked",
            "--reuse-verified-seed",
        ]
    )
    assert vars(options) == {
        **vars(defaults),
        "variant": "native-pg-fixed",
        "answer_tokens": 1024,
        "expect_primary_blocked": True,
        "reuse_verified_seed": True,
    }
    with pytest.raises(SystemExit) as error:
        parse_args(required + ["--answer-tokens", "4096"])
    assert error.value.code == 2


async def test_shared_probe_rejects_unapproved_phase_before_any_output(tmp_path):
    args = parse_args(["--phase", str(tmp_path), "--api-key-file", "/unused-key"])
    with pytest.raises(AssertionError):
        await run_document_import_probe(args, expected_seed_primary_calls=2, session_id="stage76-native")
    assert list(tmp_path.iterdir()) == []
