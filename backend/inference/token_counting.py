"""Count the configured DeepSeek working input with its pinned BPE data."""

import hashlib
import os
from functools import lru_cache, partial
from pathlib import Path
from urllib.parse import urlsplit

_TOKENIZER_SHA256 = "8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf"
_TOKENIZER_REVISION = "3c6b30435c8590933c489be0c5200691559e0576"
_TOKENIZER_PATH = Path(__file__).parent / "tokenizer_data/deepseek-v4-pro-0813.json"
_FLASH_TOKENIZER_SHA256 = "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b"
_FLASH_TOKENIZER_REVISION = "2cba9e42aa026125f3ed06c6d98c1db82f7ca027"
_FLASH_TOKENIZER_PATH = Path(__file__).parent / "tokenizer_data/deepseek-v4.1-flash.json"


def _deepseek_model(provider, base_url, model):
    model = model.strip()
    if (
        provider.strip().lower() == "openai_compat"
        and model in {"deepseek-v4-pro", "deepseek-flash"}
        and urlsplit(base_url).hostname == "api.deepseek.com"
    ):
        return model
    return None


@lru_cache(maxsize=2)
def _deepseek_tokenizer(model="deepseek-v4-pro"):
    """Load local data only; never fetch or execute model repository code."""
    from tokenizers import Tokenizer

    path, expected_hash = (
        (_FLASH_TOKENIZER_PATH, _FLASH_TOKENIZER_SHA256)
        if model == "deepseek-flash"
        else (_TOKENIZER_PATH, _TOKENIZER_SHA256)
    )
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError(f"DeepSeek tokenizer asset integrity mismatch for {model}; restore bundled tokenizer data")
    return Tokenizer.from_str(data.decode("utf-8"))


@lru_cache(maxsize=256)
def _deepseek_text_tokens(text, model="deepseek-v4-pro"):
    return len(_deepseek_tokenizer(model).encode(text, add_special_tokens=False).ids)


def _generic_text_tokens(text):
    non_ascii = sum(1 for char in text if ord(char) > 127)
    return non_ascii + (len(text) - non_ascii + 3) // 4


def text_token_counter(provider, base_url, model):
    """Select once from the resolved connection; counting never rereads config."""
    selected = _deepseek_model(provider, base_url, model)
    return partial(_deepseek_text_tokens, model=selected) if selected else _generic_text_tokens


def serving_text_tokens(text):
    """Environment-selected counting for direct callers without a connection."""
    model = _deepseek_model(
        os.getenv("MODEL_PROVIDER", ""), os.getenv("OPENAI_COMPAT_BASE_URL", ""),
        os.getenv("OPENAI_COMPAT_MODEL", ""),
    )
    return _deepseek_text_tokens(text, model) if model else _generic_text_tokens(text)


def token_counter_info():
    """Safe measurement provenance for tests and evaluation receipts."""
    from inference.request_context import get_request_context

    context = get_request_context()
    if context is None:
        model = _deepseek_model(
            os.getenv("MODEL_PROVIDER", ""), os.getenv("OPENAI_COMPAT_BASE_URL", ""),
            os.getenv("OPENAI_COMPAT_MODEL", ""),
        )
    else:
        model = (_deepseek_model(context.provider_name, context.provider.base_url, context.provider.model)
                 if context.provider is not None else None)
    if not model:
        return {"mode": "generic_character_estimate"}
    _deepseek_tokenizer(model)
    flash = model == "deepseek-flash"
    return {
        "mode": "deepseek_v41_flash_bpe" if flash else "deepseek_v4_pro_bpe",
        "repository": "deepseek-ai/DeepSeek-V4.1-Flash" if flash else "deepseek-ai/DeepSeek-V4-Pro-0813",
        "revision": _FLASH_TOKENIZER_REVISION if flash else _TOKENIZER_REVISION,
        "asset_sha256": _FLASH_TOKENIZER_SHA256 if flash else _TOKENIZER_SHA256,
        "network_requests": 0,
    }
