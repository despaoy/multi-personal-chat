"""Curated read failures must not become successful partial evidence.

Reuse the existing complete two-source fixture, actual source catalogue,
question binding, review parser, packet admission and generation entry point.
Only the file-read fault and the answer model are synthetic controls; this is
not a real-model semantic evaluation.
"""

import copy
import errno
from html import unescape
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_curated_source_changes as existing

from inference.generation_request import (
    GenerationRequest,
    build_generation_request,
    generate_character_response,
)
from knowledge.curated_task_evidence import exclude_changed_sources, render_curated_tasks


async def arrange_curated_io(tmp_path, monkeypatch):
    """Build real reviewed originals for tests or an old-version preflight.

    Calling the fixture's undecorated factory retains its actual documents,
    original files, extractor and domain validator. ``existing.reviewed``
    supplies explicit synthetic reviewer responses through the real parsers.
    No source fault is installed until this complete receipt exists.
    """
    sources = existing.sources.__wrapped__(tmp_path, monkeypatch)
    state = await existing.reviewed(sources)
    return SimpleNamespace(
        sources=sources,
        request=GenerationRequest(
            message=existing.QUERY,
            retrieval=state,
            context_window_tokens=65536,
            evidence_max_chars=0,
        ),
        original_bytes=tuple(path.read_bytes() for path in sources.paths),
        original_receipt=copy.deepcopy(state.public_curated_review),
    )


def fail_curated_original_read(monkeypatch, rig, failure):
    """Fail only an existing original's actual read, after valid review.

    The second complete original remains readable. No validator, source
    receipt, packet, extraction result or retrieval status is replaced.
    """
    actual_read = Path.read_bytes
    affected = rig.sources.paths[0].resolve()
    reads = []

    def read(path):
        if path.resolve() == affected:
            reads.append(path)
            raise failure
        return actual_read(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    return reads


def observe_curated_build(rig):
    """Return the actual plan or error without assuming the fix is present.

    On an old version the caller can inspect the returned plan's status,
    evidence packets and messages; on a strict version it receives the actual
    propagated exception. This helper never labels either outcome successful.
    """
    try:
        plan = build_generation_request(rig.request)
    except Exception as error:
        return SimpleNamespace(plan=None, error=error)
    return SimpleNamespace(plan=plan, error=None)


def assert_originals_unchanged(rig):
    assert tuple(path.read_bytes() for path in rig.sources.paths) == rig.original_bytes
    assert rig.request.retrieval.public_curated_review == rig.original_receipt


def io_failure(error_type):
    code = errno.EACCES if error_type is PermissionError else errno.EIO
    return error_type(code, "isolated curated original read failure")


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", [render_curated_tasks, exclude_changed_sources], ids=["render", "exclude"])
@pytest.mark.parametrize("error_type", [PermissionError, OSError], ids=["permission", "io"])
async def test_curated_read_error_is_not_an_empty_or_stale_source(tmp_path, monkeypatch, entry, error_type):
    rig = await arrange_curated_io(tmp_path, monkeypatch)
    failure = io_failure(error_type)
    with monkeypatch.context() as fault:
        reads = fail_curated_original_read(fault, rig, failure)
        with pytest.raises(error_type) as caught:
            entry(rig.request.retrieval)
        assert caught.value is failure and reads
    assert_originals_unchanged(rig)


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [PermissionError, OSError], ids=["permission", "io"])
async def test_curated_original_read_error_stops_main_generation_before_model(tmp_path, monkeypatch, error_type):
    rig = await arrange_curated_io(tmp_path, monkeypatch)
    failure = io_failure(error_type)
    calls = []

    async def model(**kwargs):
        calls.append(kwargs)
        return existing.FIRST + "\n\n" + existing.SECOND

    with monkeypatch.context() as fault:
        reads = fail_curated_original_read(fault, rig, failure)
        with pytest.raises(error_type) as caught:
            await generate_character_response(rig.request, model)
        assert caught.value is failure and reads
        assert calls == []
    assert_originals_unchanged(rig)


@pytest.mark.asyncio
async def test_unchanged_curated_originals_still_reach_main_model(tmp_path, monkeypatch):
    rig = await arrange_curated_io(tmp_path, monkeypatch)
    calls = []
    reply = existing.FIRST + "\n\n" + existing.SECOND

    async def model(**kwargs):
        calls.append(kwargs)
        return reply

    result = await generate_character_response(rig.request, model)
    assert len(calls) == 1 and result.model_invoked and result.reply == reply
    wire = unescape("\n".join(message["content"] for message in calls[0]["messages"]))
    assert existing.FIRST in wire and existing.SECOND in wire
    row = render_curated_tasks(result.plan.retrieval)[0]
    assert row["status"] == "related_candidate_admitted"
    assert row["scope"] == "attributed_statement" and row["limitations"] == existing.OLD_LIMITS
    assert len(row["evidence"]) == 2
    assert_originals_unchanged(rig)
