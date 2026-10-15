"""Required official DeepSeek tokenizers fail before provider execution.

No live model or network is used;
only the installed tokenizer and pinned local assets are used when healthy.
"""

import hashlib
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_memory_retry_policy import delivered_memory as delivered_memory

from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig
from inference import token_counting as counter
from inference.context_budget import ReviewContextBudget, estimated_tokens
from inference.generation_request import (
    GenerationRequest,
    build_generation_request,
    generate_character_response,
)


@pytest.fixture(params=[
    ("deepseek-v4-pro", "_TOKENIZER_PATH", "_TOKENIZER_SHA256", "deepseek_v4_pro_bpe"),
    ("deepseek-flash", "_FLASH_TOKENIZER_PATH", "_FLASH_TOKENIZER_SHA256", "deepseek_v41_flash_bpe"),
], ids=["pro", "flash"])
def required_deepseek(request, monkeypatch):
    model, path_field, hash_field, mode = request.param
    monkeypatch.setenv("MODEL_PROVIDER", "openai_compat")
    monkeypatch.setenv("OPENAI_COMPAT_MODEL", model)
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "https://api.deepseek.com")
    counter._deepseek_tokenizer.cache_clear()
    counter._deepseek_text_tokens.cache_clear()
    try:
        yield SimpleNamespace(
            model=model, path_field=path_field, hash_field=hash_field,
            original_path=getattr(counter, path_field), mode=mode,
        )
    finally:
        counter._deepseek_tokenizer.cache_clear()
        counter._deepseek_text_tokens.cache_clear()


@pytest.mark.parametrize("failure, expected_type", [
    ("missing_asset", FileNotFoundError),
    ("hash_mismatch", ValueError),
    ("missing_library", ModuleNotFoundError),
    ("invalid_utf8", UnicodeDecodeError),
    ("invalid_tokenizer", Exception),
])
def test_required_tokenizer_failures_never_become_estimates(
    required_deepseek, tmp_path, monkeypatch, failure, expected_type,
):
    selected = required_deepseek
    if failure == "missing_library":
        # The production loader imports locally, so this tests that real edge.
        monkeypatch.setitem(sys.modules, "tokenizers", None)
    else:
        path = tmp_path / "required-tokenizer.json"
        monkeypatch.setattr(counter, selected.path_field, path)
        if failure == "hash_mismatch":
            data = b"{}"
            assert hashlib.sha256(data).hexdigest() != getattr(counter, selected.hash_field)
            path.write_bytes(data)
        elif failure in {"invalid_utf8", "invalid_tokenizer"}:
            # Pass integrity first, to exercise the actual decode/Rust parser.
            data = b"\xff\xfe" if failure == "invalid_utf8" else b"{"
            path.write_bytes(data)
            monkeypatch.setattr(counter, selected.hash_field, hashlib.sha256(data).hexdigest())

    text = "编号N09R119的件数为14；身份核验失败且材料不全。"
    for operation in (lambda: estimated_tokens(text), counter.token_counter_info):
        with pytest.raises(expected_type) as caught:
            operation()
        # In particular, the Rust parser's generic Exception must not be a
        # hash ValueError, UTF-8 error, unavailable-library error, or assertion.
        assert type(caught.value) is expected_type
        if failure == "hash_mismatch":
            assert "asset integrity" in str(caught.value)


def test_restored_required_asset_recovers_without_clearing_failed_caches(
    required_deepseek, tmp_path, monkeypatch,
):
    from tokenizers import Tokenizer

    selected = required_deepseek
    text = "编号N09R119的件数为14；身份核验失败且材料不全。"
    monkeypatch.setattr(counter, selected.path_field, tmp_path / "temporarily-missing.json")
    with pytest.raises(FileNotFoundError):
        estimated_tokens(text)
    with pytest.raises(FileNotFoundError):
        counter.token_counter_info()
    # No cache_clear here: exceptions must not be cached as usable estimates.
    monkeypatch.setattr(counter, selected.path_field, selected.original_path)
    independent = Tokenizer.from_file(str(selected.original_path))
    expected = len(independent.encode(text, add_special_tokens=False).ids)
    assert estimated_tokens(text) == expected
    info = counter.token_counter_info()
    assert info["mode"] == selected.mode and info["network_requests"] == 0
    assert "unavailable_reason" not in info


async def test_required_asset_failure_stops_review_and_real_generation_before_model(
    required_deepseek, tmp_path, monkeypatch, no_live_context_review,
):
    monkeypatch.setattr(counter, required_deepseek.path_field, tmp_path / "missing-before-model.json")
    messages = [{"role": "user", "content": "核对给定虚构资料，并保留未知处。"}]
    with pytest.raises(FileNotFoundError):
        ReviewContextBudget(8192).fits(messages, 128)
    request = GenerationRequest(
        message=messages[0]["content"], persona_prompt="依据给定资料核对。",
        context_window_tokens=8192, max_tokens=128, apply_prompt_policy=False,
    )
    with pytest.raises(FileNotFoundError):
        build_generation_request(request)
    generate = AsyncMock(return_value="This model must not be called")
    with pytest.raises(FileNotFoundError):
        await generate_character_response(request, generate)
    generate.assert_not_awaited()


async def test_required_asset_failure_blocks_real_memory_job_and_keeps_source(
    required_deepseek, delivered_memory, tmp_path, monkeypatch, no_live_context_review,
):
    rig = delivered_memory
    # This dependency belongs to the writer connection, independently of chat.
    embedding = rig.scheduler._embedding_provider
    await rig.scheduler.shutdown(timeout=1)
    rig.scheduler = MemoryEnrichmentScheduler(
        config=MemoryLlmConfig(True, "https://api.deepseek.com", required_deepseek.model,
                               context_window_tokens=8192, idle_seconds=0),
        completion=rig.completion, embedding_provider=embedding,
    )
    monkeypatch.setattr(counter, required_deepseek.path_field, tmp_path / "missing-writer-tokenizer.json")
    response = await rig.step()
    marker = response["context"]["memory_completion"]
    receipt = marker["semantic_receipt"]
    assert response["memory_completion_state"] == "blocked"
    assert marker["reason"] == "semantic_failure" and marker["attempts"] == 1
    assert marker["completion_outcome"]["memory_enrichment_scheduled"] is True
    assert receipt["status"] == "failed" and receipt["error"] == "FileNotFoundError"
    assert receipt["stage"] == "proposal_generation" and receipt["retryable"] is False
    assert receipt["accepted"] == receipt["persisted"] == 0
    assert receipt["source_capture"] == "recorded"
    rig.completion.complete.assert_not_awaited()
    assert rig.scheduler.status.failed == 1 and not rig.selectable()
    assert not await rig.repo.list_memory_records(rig.snapshot.character_id, rig.snapshot.user_scope)
    assert await rig.repo.list_sources(rig.snapshot.character_id, rig.snapshot.user_scope) == [dict(
        source_message_id=rig.source_id, body=rig.body,
        observed_at=rig.snapshot.received_at.isoformat(timespec="microseconds"),
    )]
    assert response["reply"] == rig.original_reply
