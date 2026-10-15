"""Actual writer configuration owns both memory token budgets.

Draft targeted tests: real local BPE/SQLite/runtime/scheduler behavior, with
explicit synthetic model proposals and no live model or network calls.
"""

import json
from datetime import datetime

import pytest
from test_memory_retry_policy import assert_saved_once
from test_memory_retry_policy import delivered_memory as delivered_memory
from tokenizers import Tokenizer

from character.memory_extractor import extract_memories
from character.memory_llm import (
    MemoryEnrichmentScheduler,
    MemoryLlmConfig,
    build_memory_llm_messages,
    classify_memory_write_mode,
)
from db.schemas import MessageRequest
from inference import token_counting as counter
from inference.context_budget import CONTEXT_SAFETY_MARGIN_TOKENS


@pytest.fixture(autouse=True)
def fresh_tokenizer_caches():
    counter._deepseek_tokenizer.cache_clear()
    counter._deepseek_text_tokens.cache_clear()
    try:
        yield
    finally:
        counter._deepseek_tokenizer.cache_clear()
        counter._deepseek_text_tokens.cache_clear()


async def replace_writer(rig, *, base_url, model, window):
    previous = rig.scheduler
    await previous.shutdown(timeout=1)
    rig.scheduler = MemoryEnrichmentScheduler(
        config=MemoryLlmConfig(
            True, base_url, model, context_window_tokens=window, confidence_threshold=.85,
            queue_size=2, idle_seconds=0,
        ),
        completion=rig.completion,
        embedding_provider=previous._embedding_provider,
    )


def change_delivered_message(rig, message):
    response = rig.stored_response()
    request = MessageRequest.model_validate({**response["context"]["request"], "message": message})
    response["context"]["request"] = request.model_dump()
    rig.write_response(response, "pending")
    rig.body = message


def broken_chat_deepseek(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PROVIDER", "openai_compat")
    monkeypatch.setenv("OPENAI_COMPAT_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setattr(counter, "_TOKENIZER_PATH", tmp_path / "missing-chat-tokenizer.json")


@pytest.mark.parametrize("model, path_field, other_model, other_path_field", [
    ("deepseek-v4-pro", "_TOKENIZER_PATH", "deepseek-flash", "_FLASH_TOKENIZER_PATH"),
    ("deepseek-flash", "_FLASH_TOKENIZER_PATH", "deepseek-v4-pro", "_TOKENIZER_PATH"),
], ids=["pro", "flash"])
def test_explicit_counter_uses_real_model_bpe_and_ignores_later_chat_environment(
    monkeypatch, tmp_path, model, path_field, other_model, other_path_field,
):
    monkeypatch.setenv("MODEL_PROVIDER", "vllm")
    monkeypatch.setenv("OPENAI_COMPAT_MODEL", "unrelated-chat-model")
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "https://unrelated.invalid")
    count = counter.text_token_counter("openai_compat", "https://api.deepseek.com", model)
    independent = Tokenizer.from_file(str(getattr(counter, path_field)))
    first = "1234567890" * 40 + "；身份核验失败且材料不全。"
    assert count(first) == len(independent.encode(first, add_special_tokens=False).ids)

    # Count a new string after changing the environment, so a text-cache hit
    # cannot hide a counter that still reads the global chat configuration.
    monkeypatch.setenv("MODEL_PROVIDER", "openai_compat")
    monkeypatch.setenv("OPENAI_COMPAT_MODEL", other_model)
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setattr(counter, other_path_field, tmp_path / "missing-other-model.json")
    second = "编号Q72：" + "9876543210" * 47 + "；末尾限定必须保留。"
    assert count(second) == len(independent.encode(second, add_special_tokens=False).ids)


async def test_actual_local_writer_saves_despite_broken_global_deepseek_asset(
    delivered_memory, monkeypatch, tmp_path,
):
    rig = delivered_memory
    broken_chat_deepseek(monkeypatch, tmp_path)
    # Construct after the conflicting chat environment is installed. The
    # scheduler must select from this actual local writer configuration.
    await replace_writer(rig, base_url="http://local.invalid/v1", model="fixture", window=8192)
    response = await rig.step()
    await assert_saved_once(rig, response, attempts=1)
    rig.completion.complete.assert_awaited_once()


async def test_flash_writer_rejects_legal_hex_prompt_that_character_estimate_would_admit(
    delivered_memory, monkeypatch,
):
    rig = delivered_memory
    monkeypatch.setenv("MODEL_PROVIDER", "vllm")
    monkeypatch.setenv("OPENAI_COMPAT_MODEL", "unrelated-chat-model")
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://local.invalid/v1")
    prefix = "我本人长期喜欢深蓝色油墨，这是明确的个人偏好。以下只是完整虚构编号资料："
    message = prefix + "".join(f"{i * 2654435761 & 0xffffffff:08x}" for i in range(900))
    assert len(message) < 8000
    change_delivered_message(rig, message)
    actual_count = counter.text_token_counter("openai_compat", "https://api.deepseek.com", "deepseek-flash")
    old_count = counter.text_token_counter("vllm", "http://local.invalid/v1", "fixture")
    wire = build_memory_llm_messages(
        message, tuple(extract_memories(message, reference_time=rig.snapshot.received_at)),
        rig.snapshot.history, (), 2000, .85, write_mode=classify_memory_write_mode(message),
        observed_at=rig.snapshot.received_at,
        context_window_tokens=65536, count_tokens=actual_count,
    )
    independent = Tokenizer.from_file(str(counter._FLASH_TOKENIZER_PATH))
    exact_required = sum(
        len(independent.encode(item["content"], add_special_tokens=False).ids) + 4 for item in wire
    ) + 768 + CONTEXT_SAFETY_MARGIN_TOKENS
    old_required = sum(old_count(item["content"]) + 4 for item in wire) + 768 + CONTEXT_SAFETY_MARGIN_TOKENS
    assert old_required < exact_required
    # The incorrect counter would admit this complete serialized request at
    # equality. Actual Flash BPE must reject it before the synthetic provider.
    window = old_required
    await replace_writer(rig, base_url="https://api.deepseek.com", model="deepseek-flash", window=window)

    response = await rig.step()
    marker = response["context"]["memory_completion"]
    receipt = marker["semantic_receipt"]
    assert response["memory_completion_state"] == "blocked" and marker["attempts"] == 1
    assert receipt["status"] == "failed" and receipt["error"] == "InputBudgetError"
    assert receipt["reason"] == "input_budget" and receipt["retryable"] is False
    assert receipt["stage"] == "proposal_generation" and receipt["source_capture"] == "recorded"
    assert receipt["accepted"] == receipt["persisted"] == 0
    rig.completion.complete.assert_not_awaited()
    assert not await rig.repo.list_memory_records(rig.snapshot.character_id, rig.snapshot.user_scope)
    sources = await rig.repo.list_sources(rig.snapshot.character_id, rig.snapshot.user_scope)
    assert sources == [dict(
        source_message_id=rig.source_id, body=message,
        observed_at=rig.snapshot.received_at.isoformat(timespec="microseconds"),
    )]
    assert not rig.selectable()


@pytest.mark.parametrize("writer", ["local", "flash"])
async def test_source_erasure_uses_actual_writer_budget_without_slicing_or_extra_deletion(
    delivered_memory, monkeypatch, tmp_path, writer,
):
    rig = delivered_memory
    window = 32000
    if writer == "local":
        broken_chat_deepseek(monkeypatch, tmp_path)
        await replace_writer(rig, base_url="http://local.invalid/v1", model="fixture", window=window)
        long_body = "完整长原话：" + "甲" * 20000 + "；末尾限定也必须保留。"
    else:
        monkeypatch.setenv("MODEL_PROVIDER", "vllm")
        monkeypatch.setenv("OPENAI_COMPAT_MODEL", "unrelated-chat-model")
        monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://local.invalid/v1")
        await replace_writer(rig, base_url="https://api.deepseek.com", model="deepseek-flash", window=window)
        long_body = "项目参考编号：" + "0123456789" * 5000 + "；末尾限定也必须保留。"
        independent = Tokenizer.from_file(str(counter._FLASH_TOKENIZER_PATH))
        exact_cost = (
            len(independent.encode(long_body, add_special_tokens=False).ids)
            + len(independent.encode("long", add_special_tokens=False).ids) + 100
        )
        old_count = counter.text_token_counter("vllm", "http://local.invalid/v1", "fixture")
        assert old_count(long_body) + old_count("long") + 100 < window // 2 < exact_cost

    originals = {
        "pet": "我的狗叫松糕。",
        "other": "我周末喜欢参观美术馆。",
        "long": long_body,
    }
    for source_id, body in originals.items():
        assert await rig.repo.capture_source(
            rig.snapshot.character_id, rig.snapshot.user_scope,
            source_message_id=source_id, body=body, observed_at=rig.snapshot.received_at,
        ) == "recorded"
    change_delivered_message(rig, "请把我的狗的名字从记忆里删除。")

    async def synthetic_erasure(messages):
        payload = json.loads(messages[-1]["content"])
        rows = {row["source_id"]: row for row in payload["source_erasure_candidates"]}
        assert set(rows) == {"pet", "other"}
        assert rows["pet"]["text"] == originals["pet"]
        assert rows["other"]["text"] == originals["other"]
        assert payload["source_candidates_complete"] is False
        assert long_body not in str(messages)
        return '{"memories":[],"erase_source_ids":["pet"]}'

    rig.completion.complete.side_effect = synthetic_erasure
    response = await rig.step()
    marker = response["context"]["memory_completion"]
    receipt = marker["semantic_receipt"]
    assert response["memory_completion_state"] == "completed" and marker["attempts"] == 1
    assert receipt["status"] == "erased" and receipt["persisted"] == 1 and receipt["source_erased"] == 1
    assert receipt["source_capture"] == "erase_request"
    coverage = receipt["source_candidate_coverage"]
    assert coverage["candidates"] == 2 and coverage["omitted"] == 1 and coverage["complete"] is False
    rig.completion.complete.assert_awaited_once()
    remaining = await rig.repo.list_sources(rig.snapshot.character_id, rig.snapshot.user_scope)
    assert {row["source_message_id"]: row["body"] for row in remaining} == {
        "other": originals["other"], "long": long_body,
    }
    assert all(datetime.fromisoformat(row["observed_at"]) == rig.snapshot.received_at for row in remaining)
    assert not await rig.repo.list_memory_records(rig.snapshot.character_id, rig.snapshot.user_scope)
    assert not rig.selectable()
