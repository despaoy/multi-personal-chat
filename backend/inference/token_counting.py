"""Count the configured DeepSeek working input with its pinned BPE data."""

import hashlib
import os
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

_TOKENIZER_SHA256 = "8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf"
_TOKENIZER_REVISION = "3c6b30435c8590933c489be0c5200691559e0576"
_TOKENIZER_PATH = Path(__file__).parent / "tokenizer_data/deepseek-v4-pro-0813.json"


def _uses_deepseek_v4_pro():
    return (
        os.getenv("MODEL_PROVIDER", "").strip().lower() == "openai_compat"
        and os.getenv("OPENAI_COMPAT_MODEL", "").strip() == "deepseek-v4-pro"
        and urlsplit(os.getenv("OPENAI_COMPAT_BASE_URL", "")).hostname == "api.deepseek.com"
    )


@lru_cache(maxsize=1)
def _deepseek_tokenizer():
    """Load local data only; never fetch or execute model repository code."""
    try:
        from tokenizers import Tokenizer

        data = _TOKENIZER_PATH.read_bytes()
        if hashlib.sha256(data).hexdigest() != _TOKENIZER_SHA256:
            return None, "asset_hash_mismatch"
        tokenizer = Tokenizer.from_str(data.decode("utf-8"))
    except (ImportError, OSError, ValueError) as error:
        return None, type(error).__name__
    return tokenizer, ""


@lru_cache(maxsize=256)
def _deepseek_text_tokens(text):
    tokenizer, _reason = _deepseek_tokenizer()
    if tokenizer is None:
        # This byte-level BPE has no text-expanding normalizer. One token per
        # UTF-8 byte is a conservative data-unavailable bound, not the unsafe
        # Latin-character average. No request or required record is truncated.
        return len(text.encode("utf-8"))
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


def serving_text_tokens(text):
    """Preserve local estimates; use current official-host Pro data for cloud."""
    if _uses_deepseek_v4_pro():
        return _deepseek_text_tokens(text)
    non_ascii = sum(1 for char in text if ord(char) > 127)
    return non_ascii + (len(text) - non_ascii + 3) // 4


def token_counter_info():
    """Safe measurement provenance for tests and evaluation receipts."""
    if not _uses_deepseek_v4_pro():
        return {"mode": "generic_character_estimate"}
    tokenizer, reason = _deepseek_tokenizer()
    return {
        "mode": "deepseek_v4_pro_bpe" if tokenizer is not None else "utf8_byte_upper_bound",
        "repository": "deepseek-ai/DeepSeek-V4-Pro-0813",
        "revision": _TOKENIZER_REVISION,
        "asset_sha256": _TOKENIZER_SHA256,
        "unavailable_reason": reason,
        "network_requests": 0,
    }
