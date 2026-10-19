"""Flash migration must preserve full evidence and safe numeric budgets."""

import json
from html import unescape
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from inference import token_counting as counter
from inference.context_budget import CONTEXT_SAFETY_MARGIN_TOKENS, estimated_tokens
from inference.generation_request import _trim_history_to_budget


@pytest.fixture
def flash(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "openai_compat")
    monkeypatch.setenv("OPENAI_COMPAT_MODEL", "deepseek-flash")
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "https://api.deepseek.com")
    counter._deepseek_tokenizer.cache_clear()
    counter._deepseek_text_tokens.cache_clear()
    yield
    counter._deepseek_tokenizer.cache_clear()
    counter._deepseek_text_tokens.cache_clear()


def test_flash_complete_numeric_request_keeps_whole_required_sources(flash):
    fixtures = Path(__file__).parent / "fixtures"
    golden = json.loads((fixtures / "deepseek_numeric_history_budget_case.json").read_text())
    case = json.loads((fixtures / "deepseek_mixed_subject_history_case.json").read_text())
    original = golden["request"]["messages"]
    count = sum(estimated_tokens(m["content"]) + 4 for m in original)
    assert count > 65536 > golden["measured_counting"]["old_estimate"]
    kept = _trim_history_to_budget(
        original[1:-1],
        available_tokens=65536
        - 2048
        - CONTEXT_SAFETY_MARGIN_TOKENS
        - sum(
            (estimated_tokens(item["content"]) + 4 for item in [original[0], original[-1]])
        ),
    )
    assert 0 < len(kept) < len(original[1:-1]) and kept == original[1:-1][-len(kept) :]
    assert all(kept[i]["role"] == "user" and kept[i + 1]["role"] == "assistant" for i in range(0, len(kept), 2))
    wire = [original[0], *kept, original[-1]]
    tokenizer = Tokenizer.from_file(str(counter._FLASH_TOKENIZER_PATH))
    precise = sum(len(tokenizer.encode(m["content"], add_special_tokens=False).ids) + 4 for m in wire)
    assert precise + 2048 + 512 <= 65536
    whole = unescape(wire[-1]["content"])
    assert case["source_message"] in whole and case["question"] in whole
    assert all(d["content"] in whole for d in case["documents"])
    assert counter.token_counter_info()["mode"] == "deepseek_v41_flash_bpe"


def test_switching_models_cannot_reuse_wrong_asset_or_change_other_provider(flash, monkeypatch):
    flash_hash = counter.token_counter_info()["asset_sha256"]
    text = "1234567890，身份核验失败。"
    flash_count = estimated_tokens(text)
    monkeypatch.setenv("OPENAI_COMPAT_MODEL", "deepseek-v4-pro")
    assert counter.token_counter_info()["mode"] == "deepseek_v4_pro_bpe"
    assert counter.token_counter_info()["asset_sha256"] != flash_hash
    monkeypatch.setenv("OPENAI_COMPAT_MODEL", "deepseek-flash")
    assert estimated_tokens(text) == flash_count
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "https://example.invalid")
    assert counter.token_counter_info() == {"mode": "generic_character_estimate"}
