"""Hard-gate checks for Kisaki V4 candidate samples (Task B.3).

Eight code-level gates (no LLM involved). Any failure rejects the sample
before it reaches Judge A. Copy detection uses a *combined* rule (not
"any single metric over threshold rejects"):

  Reject A: longest_common_substring > 20  AND  3-gram Jaccard > 0.30
  Reject B: semantic_similarity > 0.92     AND  3-gram Jaccard > 0.20
  Disputed: only semantic_similarity > 0.85 (no auto-reject, goes to human review)

IMPORTANT: ``longest_common_substring`` returns a *contiguous* substring.
Do NOT confuse with longest common subsequence (which allows gaps).
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND = PROJECT_ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

# Import shared schema
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from kisaki_v4_llm_client import GateFailure, GateResult  # noqa: E402

# ---------------------------------------------------------------------------
# Rule constants
# ---------------------------------------------------------------------------

MAX_ASSISTANT_CHARS = 100

AI_SELF_REFERENCE_PATTERNS: tuple[str, ...] = (
    r"我是\s*A[Ii]",
    r"我是\s*一个\s*A[Ii]",
    r"我是\s*人工智能",
    r"作为\s*A[Ii]",
    r"作为\s*一个\s*A[Ii]",
    r"作为\s*人工智能",
    r"我\s*是\s*语言\s*模型",
    r"我\s*是\s*大\s*语言\s*模型",
    r"我\s*是\s*大\s*模型",
    r"我\s*只是一个\s*程序",
    r"我\s*没有\s*感情",
    r"我\s*无法\s*感受",
    r"我\s*是\s*虚拟\s*角色",
    r"我是一个\s*助手",
    r"作为\s*助手",
    r"我是\s*聊天\s*机器人",
)
AI_SELF_REFERENCE_RE = re.compile("|".join(AI_SELF_REFERENCE_PATTERNS))

# Third-person self-description: "月社妃是...的角色" / "妃是...的人物"
THIRD_PERSON_PATTERNS: tuple[str, ...] = (
    r"月社妃是[^，。]*的?(?:角色|人物|少女|女孩)",
    r"妃是[^，。]*的?(?:角色|人物|少女|女孩)",
    r"月社妃这个角色",
    r"妃这个角色",
    r"月社妃，?她",
    r"妃，?她(?:是|把|将|的)",
)
THIRD_PERSON_RE = re.compile("|".join(THIRD_PERSON_PATTERNS))

META_NARRATIVE_WORDS: tuple[str, ...] = (
    "故事", "作者", "剧本", "出场人物", "规则",
)
ZHENG_YIN_CI = "正因如此"

# Scene types where meta-narrative vocabulary is character-appropriate
# (妃 does talk about "故事/作者" in the original game).
META_NARRATIVE_SCENES: frozenset[str] = frozenset({
    "书籍讨论", "角色设定", "观点讨论", "突发奇想", "回忆故事",
})


# ---------------------------------------------------------------------------
# Gate 1: length
# ---------------------------------------------------------------------------

def check_length(conversations: list[dict[str, Any]], max_chars: int = MAX_ASSISTANT_CHARS) -> list[GateFailure]:
    failures: list[GateFailure] = []
    for i, msg in enumerate(conversations):
        if msg.get("from") != "assistant":
            continue
        value = msg.get("value", "")
        # Strip whitespace for length check (count visible chars)
        if len(value) > max_chars:
            failures.append(GateFailure(
                rule="length",
                detail=f"assistant turn {i}: {len(value)} chars > {max_chars}",
            ))
    return failures


# ---------------------------------------------------------------------------
# Gate 2: AI self-reference
# ---------------------------------------------------------------------------

def check_ai_self_reference(text: str) -> list[GateFailure]:
    if AI_SELF_REFERENCE_RE.search(text):
        return [GateFailure(rule="ai_self_reference", detail="matched AI self-reference pattern")]
    return []


# ---------------------------------------------------------------------------
# Gate 3: third-person self-description
# ---------------------------------------------------------------------------

def check_third_person_self_description(text: str) -> list[GateFailure]:
    if THIRD_PERSON_RE.search(text):
        return [GateFailure(
            rule="third_person_self_description",
            detail="describes 妃 in third person as a character/role",
        )]
    return []


# ---------------------------------------------------------------------------
# Gate 4: repeated opening (same scene)
# ---------------------------------------------------------------------------

def check_repeated_opening(
    candidate_conversations: list[dict[str, Any]],
    passed_samples: list[dict[str, Any]],
    scene: str,
) -> list[GateFailure]:
    """Reject if the candidate's first assistant opening equals an already
    passed sample's opening **in the same scene**.
    """
    candidate_first = ""
    for msg in candidate_conversations:
        if msg.get("from") == "assistant":
            candidate_first = msg.get("value", "").strip()[:20]  # first 20 chars
            break
    if not candidate_first:
        return [GateFailure(rule="repeated_opening", detail="no assistant turn found")]
    for sample in passed_samples:
        if sample.get("scene") != scene:
            continue
        for msg in sample.get("conversations", []):
            if msg.get("from") == "assistant":
                existing = msg.get("value", "").strip()[:20]
                if existing and existing == candidate_first:
                    return [GateFailure(
                        rule="repeated_opening",
                        detail=f"opening '{candidate_first}' already used in scene '{scene}'",
                    )]
                break
    return []


# ---------------------------------------------------------------------------
# Gate 5: meta-narrative word count
# ---------------------------------------------------------------------------

def count_meta_narrative_words(text: str) -> int:
    """Total occurrence count of meta-narrative words (not distinct kinds).

    Critical fix: previously this returned the number of *distinct* words
    that appeared at least once, so "故事故事故事故事" counted as 1.
    Now it returns the total count of occurrences across all words.
    """
    return sum(text.count(w) for w in META_NARRATIVE_WORDS)


def check_meta_narrative(
    assistant_turns: list[str],
    scene: str,
) -> list[GateFailure]:
    """Meta-narrative word count gate.

    - Non-meta scene: any meta word occurrence = fail (count > 0).
    - Meta scene: per-turn limit — each assistant turn may contain at most
      1 meta-narrative word occurrence. A turn with 2+ occurrences fails
      regardless of total sample count.

    Major-3 fix: previously used a fixed sample-level cap of 3, which did
    not enforce the per-turn limit and allowed a single turn to have
    multiple meta words.
    """
    if scene not in META_NARRATIVE_SCENES:
        # Non-meta scene: total occurrences across all turns must be 0
        total = sum(count_meta_narrative_words(turn) for turn in assistant_turns)
        if total > 0:
            return [GateFailure(
                rule="meta_narrative",
                detail=f"non-meta scene '{scene}' contains {total} meta-narrative word occurrence(s)",
            )]
        return []
    # Meta scene: per-turn limit (each turn ≤ 1 occurrence)
    failures: list[GateFailure] = []
    for i, turn in enumerate(assistant_turns):
        turn_count = count_meta_narrative_words(turn)
        if turn_count > 1:
            failures.append(GateFailure(
                rule="meta_narrative",
                detail=f"meta scene '{scene}' assistant turn {i}: "
                       f"{turn_count} meta word occurrences > 1 (per-turn cap exceeded)",
            ))
    return failures


# ---------------------------------------------------------------------------
# Gate 6: '正因如此' count
# ---------------------------------------------------------------------------

def count_zheng_yin_ci(text: str) -> int:
    return text.count(ZHENG_YIN_CI)


def check_zheng_yin_ci(assistant_text: str) -> list[GateFailure]:
    count = count_zheng_yin_ci(assistant_text)
    if count > 1:
        return [GateFailure(
            rule="zheng_yin_ci",
            detail=f"'正因如此' appears {count} times (max 1)",
        )]
    return []


# ---------------------------------------------------------------------------
# Gate 7: original-copy detection (combined rule, not "any over threshold")
# ---------------------------------------------------------------------------

def longest_common_substring(s1: str, s2: str) -> int:
    """Length of the longest CONTIGUOUS common substring.

    Dynamic programming. NOT longest common subsequence (which allows gaps).
    Example:
      longest_common_substring("abcdef", "xbcdyz") -> 3 ("bcd")
      longest_common_subsequence("abcdef", "xbcdyz") -> 4 ("bcdf")  # NOT what we want
    """
    if not s1 or not s2:
        return 0
    m, n = len(s1), len(s2)
    # dp[i][j] = length of longest common suffix of s1[:i] and s2[:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    best = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > best:
                    best = dp[i][j]
            # else: dp[i][j] = 0 (already initialized)
    return best


def ngram_set(text: str, n: int = 3) -> set[str]:
    """Character n-gram set (no whitespace normalization beyond .strip())."""
    text = text.strip()
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def ngram_jaccard(s1: str, s2: str, n: int = 3) -> float:
    a = ngram_set(s1, n)
    b = ngram_set(s2, n)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# Embedding-backed semantic similarity, with safe fallback.
_EMBEDDER = None
_EMBEDDER_LOAD_ATTEMPTED = False
_EMBEDDER_MODEL_PATH = None


def _load_embedder():
    global _EMBEDDER, _EMBEDDER_LOAD_ATTEMPTED, _EMBEDDER_MODEL_PATH
    if _EMBEDDER_LOAD_ATTEMPTED:
        return _EMBEDDER
    _EMBEDDER_LOAD_ATTEMPTED = True
    import os
    model_path = os.getenv(
        "KISAKI_EMBEDDING_MODEL_PATH",
        "./models/bge-small-zh-v1.5",
    )
    _EMBEDDER_MODEL_PATH = model_path
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _EMBEDDER = SentenceTransformer(model_path)
    except Exception:
        _EMBEDDER = None
    return _EMBEDDER


def semantic_similarity(s1: str, s2: str) -> float:
    """Cosine similarity of embeddings.

    Falls back to character 4-gram TF cosine if sentence-transformers is
    unavailable. The fallback is conservative (overestimates similarity for
    short Chinese text), so the combined-rule copy detector treats
    >0.85 as disputed (not auto-reject) — this absorbs fallback noise.
    """
    embedder = _load_embedder()
    if embedder is not None:
        try:
            import numpy as np  # type: ignore
            emb = embedder.encode([s1, s2], normalize_embeddings=True)
            cos = float(np.dot(emb[0], emb[1]))
            return max(0.0, min(1.0, cos))
        except Exception:
            pass
    # Fallback: char 4-gram TF cosine
    return _char_ngram_cosine(s1, s2, n=4)


def _char_ngram_cosine(s1: str, s2: str, n: int = 4) -> float:
    """Char n-gram TF cosine (fallback when no embedding model)."""
    from math import sqrt
    a = ngram_set(s1, n)
    b = ngram_set(s2, n)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    # TF = 1 for each unique n-gram (binary cosine)
    inter = len(a & b)
    return inter / sqrt(len(a) * len(b))


def check_original_copy(
    candidate_assistant_text: str,
    references: list[str],
) -> tuple[list[GateFailure], list[str]]:
    """Combined-rule copy detection.

    Returns (failures, disputed_flags).
      Reject A: LCS > 20  AND  3-gram Jaccard > 0.30
      Reject B: cosine > 0.92  AND  3-gram Jaccard > 0.20
      Disputed: only cosine > 0.85 (no auto-reject)
    """
    failures: list[GateFailure] = []
    disputed: list[str] = []
    for i, ref in enumerate(references):
        if not ref or not candidate_assistant_text:
            continue
        lcs = longest_common_substring(candidate_assistant_text, ref)
        jac3 = ngram_jaccard(candidate_assistant_text, ref, n=3)
        cos = semantic_similarity(candidate_assistant_text, ref)

        if lcs > 20 and jac3 > 0.30:
            failures.append(GateFailure(
                rule="original_copy_A",
                detail=f"ref[{i}]: LCS={lcs} (>20) AND Jaccard3={jac3:.3f} (>0.30)",
            ))
        if cos > 0.92 and jac3 > 0.20:
            failures.append(GateFailure(
                rule="original_copy_B",
                detail=f"ref[{i}]: cosine={cos:.3f} (>0.92) AND Jaccard3={jac3:.3f} (>0.20)",
            ))
        if cos > 0.85 and not (cos > 0.92 and jac3 > 0.20):
            disputed.append(
                f"ref[{i}]: cosine={cos:.3f} (>0.85 only) -> human review"
            )
    return failures, disputed


# ---------------------------------------------------------------------------
# Gate 8: JSON structure
# ---------------------------------------------------------------------------

def validate_json_structure(candidate: dict[str, Any]) -> list[GateFailure]:
    """Validate conversations schema: list of {from: human|assistant, value: non-empty str}."""
    failures: list[GateFailure] = []
    conversations = candidate.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        failures.append(GateFailure(rule="json_structure", detail="conversations missing or empty"))
        return failures
    has_human = False
    has_assistant = False
    for i, msg in enumerate(conversations):
        if not isinstance(msg, dict):
            failures.append(GateFailure(rule="json_structure", detail=f"msg[{i}] not dict"))
            continue
        sender = msg.get("from")
        value = msg.get("value")
        if sender not in ("human", "assistant"):
            failures.append(GateFailure(rule="json_structure", detail=f"msg[{i}].from='{sender}'"))
        if not isinstance(value, str) or not value.strip():
            failures.append(GateFailure(rule="json_structure", detail=f"msg[{i}].value empty"))
        if sender == "human":
            has_human = True
        elif sender == "assistant":
            has_assistant = True
    if not has_human:
        failures.append(GateFailure(rule="json_structure", detail="no human turn"))
    if not has_assistant:
        failures.append(GateFailure(rule="json_structure", detail="no assistant turn"))
    return failures


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_all_gates(
    candidate: dict[str, Any],
    scene: str,
    references: list[str],
    passed_samples: list[dict[str, Any]],
) -> GateResult:
    """Run all 8 gates; return GateResult with passed/failures/disputed_flags.

    Critical-1 fix: when ``disputed_flags`` is non-empty (e.g. only semantic
    similarity > 0.85 but no hard failure), ``passed`` is now False so the
    sample routes to human review instead of auto-passing.
    """
    failures: list[GateFailure] = []
    disputed_flags: list[str] = []

    # Gate 8 first: structural validation
    failures.extend(validate_json_structure(candidate))
    if failures:
        return GateResult(passed=False, failures=failures, disputed_flags=disputed_flags)

    conversations = candidate.get("conversations", [])
    assistant_turns = [m.get("value", "") for m in conversations if m.get("from") == "assistant"]
    assistant_text = " ".join(assistant_turns)

    # Gates 1-6
    failures.extend(check_length(conversations))
    failures.extend(check_ai_self_reference(assistant_text))
    failures.extend(check_third_person_self_description(assistant_text))
    failures.extend(check_repeated_opening(conversations, passed_samples, scene))
    # Major-3: pass assistant_turns (list) so per-turn limit can be enforced
    failures.extend(check_meta_narrative(assistant_turns, scene))
    failures.extend(check_zheng_yin_ci(assistant_text))

    # Gate 7: copy detection (may add disputed flags even without hard failure)
    copy_failures, copy_disputed = check_original_copy(assistant_text, references)
    failures.extend(copy_failures)
    disputed_flags.extend(copy_disputed)

    # Critical-1: disputed_flags present => NOT passed (routes to human review)
    passed = not failures and not disputed_flags
    return GateResult(passed=passed, failures=failures, disputed_flags=disputed_flags)


# ---------------------------------------------------------------------------
# CLI: test a single sample (for debugging)
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Run hard gates on a single candidate JSON")
    parser.add_argument("candidate_json", type=Path,
                        help="path to a JSON file with {conversations: [...], scene: ...}")
    parser.add_argument("--references-jsonl", type=Path, default=None,
                        help="optional .jsonl file with reference texts (one per line, {text: '...'})")
    args = parser.parse_args()

    candidate = json.loads(args.candidate_json.read_text(encoding="utf-8"))
    scene = candidate.get("scene", "日常场景")
    references: list[str] = []
    if args.references_jsonl and args.references_jsonl.exists():
        for line in args.references_jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                references.append(json.loads(line).get("text", ""))
            except json.JSONDecodeError:
                references.append(line)

    result = run_all_gates(candidate, scene, references, passed_samples=[])
    print(json.dumps({
        "passed": result.passed,
        "failures": [f.__dict__ if hasattr(f, "__dict__") else f for f in result.failures],
        "disputed_flags": result.disputed_flags,
    }, ensure_ascii=False, indent=2))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
