"""Shared LLM client + data schema for Kisaki V3 sample regeneration pipeline.

Design:
  - Generator + Judge B (pairwise): DeepSeek (api.deepseek.com)
  - Judge A (5-dim semantic): Qwen (DashScope OpenAI-compatible endpoint)
  - Different model families => can claim "independent multi-Judge committee"
  - Generator and Judge B share DeepSeek: documented self-preference risk

All API keys are read from environment variables (loaded from .env via
python-dotenv). Keys are never logged.

Utilities:
  - atomic_write_json: temp file + rename, prevents corrupted outputs
  - RateLimiter: token-bucket-ish minimum interval between API calls
  - exponential_backoff_retry: 1s -> 2s -> 4s -> 8s, max 4 attempts
  - request_cache: deduplicate identical (sample_spec_id, attempt, role) calls
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND = PROJECT_ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Environment-driven configuration
# ---------------------------------------------------------------------------
# Major-9 fix: model identifiers are now centralized in MODEL_REGISTRY with
# provider, family, and a free-form version note. The registry is written
# to judge_config.json at pipeline start so every run records the exact
# model identities used — required for reproducibility and for the
# "independent multi-Judge committee" claim (different model families).
#
# User-confirmed model assignment (from prior session):
#   Judge A    -> qwen-max       (Qwen family, strong CN character understanding)
#   Judge B    -> deepseek-chat  (DeepSeek family, pairwise reasoning)
#   Generator  -> deepseek-chat  (DeepSeek family)
# User originally referred to Judge B as "deepseek-v4-pro", but no such
# public model identifier exists on the DeepSeek API as of 2026-07; the
# current production model is ``deepseek-chat`` (DeepSeek-V3 lineage).
# This discrepancy is recorded in ``version_note`` so the research report
# can document the actual model used versus the user's intent.

MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "judge_a": {
        "role": "judge_a",
        "provider": "dashscope",
        "model_family": "qwen",
        # Critical fix: pin to fixed snapshot qwen3-max-2026-01-23 instead of
        # the floating qwen-max alias, so the model cannot silently upgrade
        # mid-experiment and break reproducibility.
        "model_id": os.getenv("QWEN_JUDGE_A_MODEL", "qwen3-max-2026-01-23"),
        "base_url": os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "version_note": "qwen3-max-2026-01-23 fixed snapshot (was qwen-max alias); pinned for reproducibility",
    },
    "judge_b": {
        "role": "judge_b",
        "provider": "deepseek",
        "model_family": "deepseek",
        # Critical fix: deepseek-chat was deprecated 2026-07-24; use deepseek-v4-pro
        # for Judge B (pairwise reasoning, stronger).
        "model_id": os.getenv("DEEPSEEK_JUDGE_B_MODEL", "deepseek-v4-pro"),
        "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        "version_note": "deepseek-v4-pro (deepseek-chat deprecated 2026-07-24); pairwise reasoning judge",
    },
    "generator": {
        "role": "generator",
        "provider": "deepseek",
        "model_family": "deepseek",
        # Critical fix: deepseek-chat deprecated; use deepseek-v4-flash for generation
        # (faster, cheaper, sufficient quality for in-distribution samples).
        "model_id": os.getenv("DEEPSEEK_GENERATOR_MODEL", "deepseek-v4-flash"),
        "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        "version_note": "deepseek-v4-flash (deepseek-chat deprecated 2026-07-24); same family as Judge B (self-preference risk documented)",
    },
}

# Backwards-compatible module-level constants (now derived from MODEL_REGISTRY)
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
QWEN_BASE_URL = MODEL_REGISTRY["judge_a"]["base_url"]
QWEN_JUDGE_A_MODEL = MODEL_REGISTRY["judge_a"]["model_id"]

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = MODEL_REGISTRY["generator"]["base_url"]
DEEPSEEK_GENERATOR_MODEL = MODEL_REGISTRY["generator"]["model_id"]
DEEPSEEK_JUDGE_B_MODEL = MODEL_REGISTRY["judge_b"]["model_id"]


def build_judge_config() -> dict[str, Any]:
    """Build a judge_config.json payload recording all model identities.

    Major-9: this payload should be written to ``output_dir/judge_config.json``
    at pipeline start so every run is self-documenting. The
    ``cross_model_committee_note`` field documents whether the two Judges
    are from different model families (true committee) or the same family
    (two-stage discrimination only).

    Major fix (similarity backend): also records the copy-detection
    similarity backend (BGE embedding vs char-ngram fallback) and its
    mode, so formal runs can be validated as using the authoritative
    embedding backend.
    """
    families = {MODEL_REGISTRY["judge_a"]["model_family"], MODEL_REGISTRY["judge_b"]["model_family"]}
    is_cross_family = len(families) > 1

    # Record similarity backend info (may raise in strict mode if BGE
    # is missing — that's the desired fail-fast behaviour).
    try:
        from hard_gate_kisaki_v4 import get_similarity_backend_info  # type: ignore
        similarity_info = get_similarity_backend_info()
    except Exception as e:
        similarity_info = {
            "backend": "unavailable",
            "error": f"{type(e).__name__}: {e}",
            "authoritative": "false",
        }

    return {
        "config_version": 2,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "models": MODEL_REGISTRY,
        "cross_model_committee": is_cross_family,
        "cross_model_committee_note": (
            "Judge A and Judge B are from different model families "
            f"({sorted(families)}) — can be claimed as an independent "
            "multi-Judge committee."
            if is_cross_family
            else "Judge A and Judge B share the same model family — "
            "must be described as 'two-stage discrimination', NOT an "
            "independent committee."
        ),
        "generator_judge_b_same_family": (
            MODEL_REGISTRY["generator"]["model_family"]
            == MODEL_REGISTRY["judge_b"]["model_family"]
        ),
        "self_preference_risk_note": (
            "Generator and Judge B share the same model family; this creates "
            "a known self-preference bias. Mitigated by Judge A (different "
            "family) acting as a cross-check in the two-stage pipeline."
        ),
        "similarity_backend": similarity_info,
    }


class MissingAPIKeyError(RuntimeError):
    """Raised when a required LLM API key is not set in the environment."""


def require_qwen_key() -> str:
    if not QWEN_API_KEY:
        raise MissingAPIKeyError(
            "QWEN_API_KEY is not set. Add it to .env (see .env.example)."
        )
    return QWEN_API_KEY


def require_deepseek_key() -> str:
    if not DEEPSEEK_API_KEY:
        raise MissingAPIKeyError(
            "DEEPSEEK_API_KEY is not set. Add it to .env (see .env.example)."
        )
    return DEEPSEEK_API_KEY


# ---------------------------------------------------------------------------
# Atomic write / progress persistence (B.7 building blocks)
# ---------------------------------------------------------------------------

def atomic_write_json(path: Path, value: Any) -> None:
    """Write JSON via temp file + os.replace for crash-safe atomic writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    # tempfile in same dir to guarantee same-filesystem rename atomicity
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n",
        dir=str(path.parent), prefix=path.name + ".", suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(payload)
        tmp_path = Path(handle.name)
    os.replace(tmp_path, path)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append a single JSON record to a .jsonl file with fsync durability.

    Major fix: previously only called write(), so a crash between the
    write and progress.json commit could lose the line (or leave a
    partial line). Now flushes Python buffers + os.fsync() to push the
    bytes to disk before returning, so the append is durable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            # fsync may fail on some filesystems (e.g. network mounts);
            # flush() is still the important part for Python buffers.
            pass


def read_jsonl_ids(path: Path, id_field: str = "sample_spec_id") -> set[str]:
    """Read a .jsonl file and return the set of sample_spec_ids it contains.

    Used at pipeline resume time to reconcile the JSONL outputs against
    progress.json: any spec_id present in the JSONL but not in
    progress.completed_spec_ids is a "crash between data-write and
    progress-commit" survivor, and must be back-filled into progress so
    it is not re-processed (which would create a duplicate line).
    """
    if not path.exists():
        return set()
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            sid = rec.get(id_field)
            if sid:
                ids.add(str(sid))
        except json.JSONDecodeError:
            # Skip malformed/partial trailing line from a crash
            continue
    return ids


# ---------------------------------------------------------------------------
# Rate limiter + exponential backoff
# ---------------------------------------------------------------------------

class RateLimiter:
    """Minimum-interval rate limiter (thread-safe)."""

    def __init__(self, min_interval_seconds: float = 1.0):
        self.min_interval = max(0.0, float(min_interval_seconds))
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            delta = now - self._last
            if delta < self.min_interval:
                time.sleep(self.min_interval - delta)
            self._last = time.monotonic()


def exponential_backoff_retry(
    func,
    *,
    max_attempts: int = 4,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple = (Exception,),
):
    """Retry ``func`` with exponential backoff (1s -> 2s -> 4s -> 8s)."""
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return func()
        except exceptions as exc:  # noqa: BLE001
            last_exc = exc
            if attempt + 1 >= max_attempts:
                break
            delay = min(max_delay, base_delay * (2 ** attempt))
            # small jitter to avoid thundering herd
            delay = delay * (0.8 + 0.4 * random.random())
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


# ---------------------------------------------------------------------------
# Request cache (deduplicate identical LLM calls within a run)
# ---------------------------------------------------------------------------

def request_cache_key(
    *, role: str, sample_spec_id: str, attempt: int, prompt_hash: str,
) -> str:
    raw = f"{role}|{sample_spec_id}|{attempt}|{prompt_hash}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# OpenAI-compatible LLM call (works for both DeepSeek and Qwen DashScope)
# ---------------------------------------------------------------------------

def _sanitize_log_text(text: str) -> str:
    """Strip anything that looks like an API key before logging."""
    if not text:
        return text
    return re.sub(r"sk-[A-Za-z0-9]{8,}", "sk-***", text)


def call_openai_compatible(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.8,
    top_p: float = 0.9,
    max_tokens: int = 512,
    response_format_json: bool = False,
    timeout: int = 60,
) -> str:
    """Call an OpenAI-compatible chat completions endpoint.

    Uses ``requests`` to avoid a hard dependency on the openai SDK (the
    project's existing DeepSeek generator also uses requests directly).
    """
    import requests  # local import keeps module import cheap

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if response_format_json:
        # Both DeepSeek and DashScope accept response_format={"type": "json_object"}
        payload["response_format"] = {"type": "json_object"}

    url = base_url.rstrip("/") + "/chat/completions"
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if resp.status_code == 429:
        # Surface a recognizable error so backoff retry can kick in
        raise RuntimeError(f"rate_limited status=429 model={model}")
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    # Strip accidental <think>...</think> blocks (some reasoning models emit them)
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    return content


# ---------------------------------------------------------------------------
# Role-specific wrappers (key + endpoint + model resolution)
# ---------------------------------------------------------------------------

def call_generator(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.8,
    max_tokens: int = 4096,
) -> str:
    """Call the DeepSeek generator.

    Smoke fix: default max_tokens raised from 512 to 4096 because
    deepseek-v4-flash is a reasoning model — reasoning_tokens are billed
    against the max_tokens budget, so 512 left no room for visible content.
    Also enable response_format=json_object (the generator prompt always
    asks for a JSON conversations array, so this is safe and improves
    parseability).
    """
    return call_openai_compatible(
        base_url=DEEPSEEK_BASE_URL,
        api_key=require_deepseek_key(),
        model=DEEPSEEK_GENERATOR_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format_json=True,
    )


def call_judge_a(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> str:
    """Call Qwen-Max for 5-dimension semantic scoring (Judge A).

    Smoke fix: max_tokens raised from 800 to 4096. qwen3-max-2026-01-23 is
    a reasoning model — reasoning_tokens are billed against max_tokens, so
    800 could leave no room for the JSON verdict on longer prompts.
    """
    return call_openai_compatible(
        base_url=QWEN_BASE_URL,
        api_key=require_qwen_key(),
        model=QWEN_JUDGE_A_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format_json=True,
    )


def call_judge_b(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> str:
    """Call DeepSeek for pairwise comparison (Judge B).

    Smoke fix: max_tokens raised from 600 to 4096. deepseek-v4-pro is a
    reasoning model — reasoning_tokens consumed the entire 600-token
    budget in the smoke run, producing empty/truncated JSON and sending
    every sample to disputed.
    """
    return call_openai_compatible(
        base_url=DEEPSEEK_BASE_URL,
        api_key=require_deepseek_key(),
        model=DEEPSEEK_JUDGE_B_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format_json=True,
    )


# ---------------------------------------------------------------------------
# Shared data schema (dataclasses -> dict via asdict)
# ---------------------------------------------------------------------------

@dataclass
class SampleSpec:
    """A generation task spec, fixed before any LLM call.

    ``sample_spec_id`` is stable across retries (derived from scene + idx +
    quota_plan_hash). Both the candidate and the v3 negative must share the
    same ``sample_spec_id`` and ``human_dialogue`` for Judge B to do a
    fair same-question A/B comparison.
    """
    sample_spec_id: str
    scene: str
    scene_desc: str
    human_dialogue: list[str]  # human turns (1-3 strings)
    v3_negative_sample_id: str  # corresponding v3 sample id for same-question compare
    reference_ids: list[str] = field(default_factory=list)  # few-shot source sample IDs
    quota_plan_hash: str = ""
    target_length_hint: str = "auto"  # short / mid / long / auto


@dataclass
class GateFailure:
    rule: str
    detail: str


@dataclass
class GateResult:
    passed: bool
    failures: list[GateFailure] = field(default_factory=list)
    disputed_flags: list[str] = field(default_factory=list)  # e.g. ["copy_similarity_only"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failures": [asdict(f) for f in self.failures],
            "disputed_flags": self.disputed_flags,
        }


@dataclass
class JudgeAResult:
    scores: dict[str, Any]  # dim -> 0-10 or "not_applicable"
    evidence: dict[str, str]
    violations: list[str]
    reason: str
    applicable_dims: list[str]
    passed: bool
    raw_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class JudgeBResult:
    prefers_candidate_run1: bool  # True if run1 preferred candidate
    prefers_candidate_run2: bool  # True if run2 preferred candidate
    confidence_run1: float
    confidence_run2: float
    evidence_run1: str
    evidence_run2: str
    final_decision: str  # "passed" | "rejected" | "disputed"
    raw_run1: str = ""
    raw_run2: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Stable sample ID
# ---------------------------------------------------------------------------

def stable_sample_spec_id(
    *, scene: str, idx: int, quota_plan_hash: str,
) -> str:
    """Stable ID: kisaki_v4_<scene_pinyin_or_hash>_<idx>_<qp_hash8>.

    Retry-safe: the same (scene, idx, quota_plan_hash) always yields the
    same ID, so progress.json can dedupe across runs.
    """
    scene_tag = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]", "_", scene)[:20]
    qp_hash8 = quota_plan_hash[:8] if quota_plan_hash else "noqp"
    return f"kisaki_v4_{scene_tag}_{idx:03d}_{qp_hash8}"
