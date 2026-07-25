"""Judge A (5-dim semantic, Qwen) + Judge B (same-question double-order pairwise, DeepSeek).

Judge A:
  - 5 dimensions: 人物一致性 / 语境连贯 / 自然度 / 原作语气 / 事实关系
  - Each dim: 0-10 or "not_applicable"
  - Pass if all applicable dims >= 7
  - Uses Qwen-Max (different model family from Generator => committee claim valid)

Judge B (V2.1 Critical #4 — same-question):
  - Candidate and v3 negative MUST share the same sample_spec_id and human_dialogue
  - Run 1: candidate as A, v3 negative as B
  - Run 2: v3 negative as A, candidate as B
  - Both prefer candidate => passed
  - Both prefer v3 negative => rejected
  - Inconsistent => disputed (forces human review)
  - Uses DeepSeek (same model as Generator: documented self-preference risk)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND = PROJECT_ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from kisaki_v4_llm_client import (  # noqa: E402
    JudgeAResult,
    JudgeBParsed,
    JudgeBResult,
    RateLimiter,
    SampleSpec,
    call_judge_a,
    call_judge_b,
    exponential_backoff_retry,
    hash_prompt,
    request_cache_key,
)


JUDGE_A_DIMENSIONS = ("人物一致性", "语境连贯", "自然度", "原作语气", "事实关系")
# Major-5 fix: the first 4 dims are mandatory core dims — they must be
# present in the Judge A response and must not be "not_applicable".
# Only "事实关系" may be NA (when the reply doesn't touch on facts).
JUDGE_A_MANDATORY_DIMS = ("人物一致性", "语境连贯", "自然度", "原作语气")
JUDGE_A_PASS_THRESHOLD = 7

# Major-1 fix: pilot confidence threshold for Judge B. Any run with
# confidence below this routes the sample to disputed (cannot trust the
# verdict). 0.6 is the pre-calibration default; the final value can only
# be tuned after the 30-sample calibration with human agreement data.
JUDGE_B_PILOT_CONFIDENCE_THRESHOLD = 0.6

# Major-2 fix: Judge B 4-dim scoring validation constants.
JUDGE_B_DIMENSIONS = ("人物一致性", "原作语气", "元叙事控制", "自然度")
# When preferred=A but A's total score is more than this many points
# below B's total, the verdict is self-contradictory -> disputed.
JUDGE_B_CONTRADICTION_TOTAL_GAP = 3.0
# When preferred=tie, any single-dimension gap larger than this means
# the scores do not actually support a tie -> disputed.
JUDGE_B_TIE_DIM_GAP = 2.0

# Major-3 fix: score-derived decision to eliminate position bias.
# smoke12_v2 showed first_position_preference_rate=1.0 — the judge
# mechanically selects position A regardless of content, and its scores
# are also position-biased (score_contradiction_rate=0.0 means scores
# agree with the biased preferred). Trusting either ``preferred`` or
# single-run scores cannot fix this.
#
# Solution: average the candidate's scores across both orderings.
# Run1: candidate=A (position A, inflated by bias)
# Run2: candidate=B (position B, deflated by bias)
# Averaging cancels the position bias. The decision is then derived
# from the averaged totals:
#   candidate_avg > negative_avg + TIE_GAP => passed
#   candidate_avg < negative_avg - TIE_GAP => rejected
#   otherwise => disputed (genuinely close, not a bias artifact)
JUDGE_B_SCORE_DERIVED_TIE_GAP = 1.0


# ---------------------------------------------------------------------------
# Judge A: 5-dimension semantic scoring (Qwen-Max)
# ---------------------------------------------------------------------------

JUDGE_A_SYSTEM_PROMPT = """你是月社妃对话样本的语义评审官。请对候选回复进行 5 维度评分。

【5 个维度】
1. 人物一致性: 是否符合月社妃的人物画像（克制理智、锋利戏谑、对琉璃的情感、元叙事视角、对"温柔世界"的怀疑）
2. 语境连贯: 是否与用户提问紧密相关，无脱节
3. 自然度: 是否自然流畅，无生硬拼接
4. 原作语气: 是否使用了妃的典型用词与语气（如"因此/哎呀/那么/呼呼呼/——破折号"）
5. 事实关系: 是否不编造训练材料之外的事实（人物关系/世界观）。若回复不涉及事实，标注 "not_applicable"

【评分标准】
- 9-10: 优秀，完全符合原作风格
- 7-8: 合格，基本符合但有微小瑕疵
- 5-6: 临界，存在明显偏差
- 0-4: 不合格，严重偏离人物

【输出要求】
严格输出 JSON（不要 markdown 代码块），结构：
{
  "scores": {"人物一致性": 8, "语境连贯": 9, "自然度": 7, "原作语气": 8, "事实关系": "not_applicable"},
  "evidence": {"人物一致性": "回复体现了妃的克制与锋利", ...},
  "violations": ["如有违反硬约束的问题，列出"],
  "reason": "总体评价（一句话）"
}"""


def build_judge_a_prompt(
    candidate: dict[str, Any],
    scene: str,
    reference_passages: list[str],
) -> str:
    """Build the Judge A user prompt."""
    conv_lines = []
    for msg in candidate.get("conversations", []):
        sender = "用户" if msg.get("from") == "human" else "妃(候选)"
        conv_lines.append(f"  {sender}: {msg.get('value', '')}")
    conv_text = "\n".join(conv_lines)

    ref_text = ""
    if reference_passages:
        ref_lines = []
        for i, ref in enumerate(reference_passages[:3]):
            ref_lines.append(f"  原作参考{i + 1}: {ref[:200]}")
        ref_text = "\n【原作片段参考（仅供人物画像校准，不要求候选复制）】\n" + "\n".join(ref_lines)

    return f"""【场景】{scene}

【候选对话】
{conv_text}{ref_text}

请按 5 维度评分。"""


def parse_judge_a_response(raw: str) -> JudgeAResult:
    """Parse Judge A JSON response. Tolerates markdown fences and missing fields."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        # Return a fail-closed result with the parse error
        return JudgeAResult(
            scores={},
            evidence={},
            violations=[f"json_parse_error: {e}"],
            reason=f"failed to parse Judge A response: {raw[:200]}",
            applicable_dims=[],
            passed=False,
            raw_response=raw,
        )

    scores = parsed.get("scores", {}) or {}
    evidence = parsed.get("evidence", {}) or {}

    # Major fix: violations field type validation.
    # - If the model returns violations as a non-list (e.g. a string),
    #   that itself is a contract violation -> fail-closed.
    # - Coerce valid lists to list[str]; any non-string entry is stringified.
    raw_violations = parsed.get("violations", [])
    violations_type_error = False
    if raw_violations is None:
        violations = []
    elif isinstance(raw_violations, list):
        violations = [str(v) if not isinstance(v, str) else v for v in raw_violations]
    else:
        # Non-list violations field -> contract violation
        violations = [f"violations_field_not_list: {type(raw_violations).__name__}"]
        violations_type_error = True

    reason = parsed.get("reason", "")

    # Major-5 fix: mandatory core dims must be present and not NA.
    # If any mandatory dim is missing or NA, the result cannot pass.
    missing_mandatory = [
        dim for dim in JUDGE_A_MANDATORY_DIMS
        if dim not in scores or scores[dim] is None or scores[dim] == "not_applicable"
    ]
    if missing_mandatory:
        violations.append(
            f"missing_mandatory_dims: {missing_mandatory} "
            f"(all of {list(JUDGE_A_MANDATORY_DIMS)} must be scored)"
        )

    # Determine applicable dims (exclude "not_applicable")
    applicable_dims: list[str] = []
    for dim in JUDGE_A_DIMENSIONS:
        val = scores.get(dim)
        if val is not None and val != "not_applicable":
            applicable_dims.append(dim)

    # Major fix: violations participate in the pass decision.
    # violations represents hard-constraint breaches reported by the Judge
    # (e.g. "character breaks the fourth wall", "fabricated lore"). Any
    # non-empty violations list => passed=False, regardless of dimension
    # scores. A high score with a violation is a contradiction that must
    # fail-closed. This also covers the violations_type_error case.
    has_violations = len(violations) > 0

    # Pass requires ALL of:
    #   (1) no violations (incl. no type error and no missing mandatory dims)
    #   (2) at least one applicable dim
    #   (3) all applicable dims >= threshold
    passed = (
        not has_violations
        and not missing_mandatory
        and bool(applicable_dims)
        and all(
            isinstance(scores.get(dim), (int, float))
            and scores[dim] >= JUDGE_A_PASS_THRESHOLD
            for dim in applicable_dims
        )
    )

    return JudgeAResult(
        scores=scores,
        evidence=evidence,
        violations=violations,
        reason=reason,
        applicable_dims=applicable_dims,
        passed=passed,
        raw_response=raw,
    )


def judge_a(
    candidate: dict[str, Any],
    scene: str,
    reference_passages: list[str],
    *,
    cache_dir: Path | None = None,
    sample_spec_id: str = "",
    attempt: int = 0,
) -> JudgeAResult:
    """Run Judge A on a candidate."""
    prompt = build_judge_a_prompt(candidate, scene, reference_passages)
    messages = [
        {"role": "system", "content": JUDGE_A_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    cache_key = request_cache_key(
        role="judge_a",
        sample_spec_id=sample_spec_id,
        attempt=attempt,
        prompt_hash=hash_prompt(prompt),
    )
    if cache_dir is not None:
        cache_file = cache_dir / f"judge_a_{cache_key}.json"
        if cache_file.exists():
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            return parse_judge_a_response(cached["raw"])

    raw = exponential_backoff_retry(
        lambda: call_judge_a(messages, temperature=0.0, max_tokens=4096),
        max_attempts=4,
        base_delay=1.0,
    )

    if cache_dir is not None:
        cache_file = cache_dir / f"judge_a_{cache_key}.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(
            json.dumps({"raw": raw}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return parse_judge_a_response(raw)


# ---------------------------------------------------------------------------
# Judge B: same-question double-order pairwise (DeepSeek)
# ---------------------------------------------------------------------------

JUDGE_B_SYSTEM_PROMPT = """你是月社妃对话样本的成对比较评审官。你将看到两个回复（A 和 B），它们回答的是**相同的用户提问**。A 和 B 的展示顺序是随机的，不暗示哪个是候选或旧样本。

【比较维度】
1. 人物一致性: 哪个更符合月社妃的人物画像（克制理智、锋利戏谑、对琉璃的情感、元叙事视角）
2. 原作语气: 哪个更贴近原作用词（因此/哎呀/呼呼呼/——破折号）
3. 元叙事控制: 哪个避免了过度元叙事（"故事/作者/剧本"过载是 v3 负例的典型问题）
4. 自然度: 哪个更自然流畅

【评审流程】
请先在 scores 字段中对 A 和 B 分别给出四个维度的 0-10 分，再据此输出 preferred。
先评分后选择可以避免先入为主的位置偏好。

【重要】
- A 和 B 的顺序是随机打乱的，A 在前不代表 A 更好
- 若两者质量接近，选择更贴近原作风格的
- 若两者都差，仍需选出相对较好的一个
- 仅当两者在各维度得分都非常接近（差值 ≤ 1 分）且确实无法分辨高下时，才输出 "tie"
- 不要为了回避判断而滥用 tie

【输出要求】
严格输出 JSON（不要 markdown 代码块）：
{
  "scores": {
    "A": {"人物一致性": 0-10, "原作语气": 0-10, "元叙事控制": 0-10, "自然度": 0-10},
    "B": {"人物一致性": 0-10, "原作语气": 0-10, "元叙事控制": 0-10, "自然度": 0-10}
  },
  "preferred": "A" 或 "B" 或 "tie",
  "confidence": 0.0-1.0,
  "evidence": "一句话说明选择理由",
  "reason": "更详细的对比分析"
}"""


def build_judge_b_prompt(
    response_a: list[dict[str, Any]],
    response_b: list[dict[str, Any]],
    human_dialogue: list[str],
) -> str:
    """Build the Judge B user prompt for one ordering.

    Note: response_a / response_b are the two answers to compare; their
    assignment to the "A"/"B" labels is randomized by the caller (judge_b
    runs two orders), and the system prompt explicitly tells the judge the
    order is random and carries no candidate/negative identity hint.
    """
    def format_resp(resp: list[dict[str, Any]]) -> str:
        lines = []
        for msg in resp:
            sender = "用户" if msg.get("from") == "human" else "妃"
            lines.append(f"    {sender}: {msg.get('value', '')}")
        return "\n".join(lines)

    human_text = "\n".join(f"  用户: {turn}" for turn in human_dialogue)

    return f"""【同题提问】
{human_text}

【回复 A】
{format_resp(response_a)}

【回复 B】
{format_resp(response_b)}

请先对 A 和 B 分别给出四个维度的评分，再选出更贴近月社妃原作风格的一个；若两者确实质量接近且无法分辨，可选 tie。"""


def parse_judge_b_response(raw: str, candidate_is_a: bool) -> JudgeBParsed:
    """Parse one Judge B run into a structured ``JudgeBParsed``.

    Major-2 refactor: previously returned a 5-tuple
    ``(prefers_candidate, confidence, evidence, parse_ok, is_tie)`` that
    could not carry scores or contradiction info. Now returns a dataclass
    so the caller (``judge_b``) can route disputed verdicts with full
    context: low confidence, missing/invalid scores, or a
    preferred-vs-scores contradiction all surface as distinct reasons.

    Routing rules captured here:
    - ``parse_ok=False``: JSON parse failed, ``preferred`` invalid,
      ``confidence`` missing/non-numeric/out-of-range, OR ``scores``
      missing/malformed. Caller routes to ``disputed`` (never rejected).
    - ``low_confidence=True``: confidence < pilot threshold (0.6).
    - ``score_contradiction``: non-empty when ``preferred`` disagrees
      with the score totals (e.g. preferred=A but B's total is higher by
      more than ``JUDGE_B_CONTRADICTION_TOTAL_GAP``), or when ``tie`` is
      claimed but a single-dim gap exceeds ``JUDGE_B_TIE_DIM_GAP``.
    - ``is_tie=True``: judge returned ``preferred: "tie"``; does not
      favour the candidate. Caller routes to ``disputed``.
    """
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    text = text.strip()

    def _fail(evidence: str) -> JudgeBParsed:
        return JudgeBParsed(
            parse_ok=False, prefers_candidate=False, confidence=0.0,
            evidence=evidence, is_tie=False,
        )

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        return _fail(f"json_parse_error: {e}; raw={raw[:200]}")

    preferred = str(parsed.get("preferred", "")).strip().upper()
    if preferred not in ("A", "B", "TIE"):
        return _fail(f"invalid_preferred_value: {preferred!r}; raw={raw[:200]}")

    # Confidence: must be numeric in [0, 1].
    raw_conf = parsed.get("confidence")
    if raw_conf is None:
        return _fail(f"missing_confidence; raw={raw[:200]}")
    try:
        confidence = float(raw_conf)
    except (TypeError, ValueError):
        return _fail(f"invalid_confidence_type: {raw_conf!r}; raw={raw[:200]}")
    if not (0.0 <= confidence <= 1.0):
        return _fail(f"confidence_out_of_range: {confidence}; raw={raw[:200]}")

    # evidence: coerce to string (does not fail parse on its own)
    raw_evidence = parsed.get("evidence", "")
    if isinstance(raw_evidence, str):
        evidence = raw_evidence
    else:
        evidence = json.dumps(raw_evidence, ensure_ascii=False)

    # Major-2 fix: validate scores.A and scores.B (4 dims, 0-10 numeric).
    # Missing/invalid scores -> parse_ok=False -> disputed. The prompt
    # explicitly requires scores, so their absence is a contract breach.
    raw_scores = parsed.get("scores") or {}
    if not isinstance(raw_scores, dict):
        return _fail(f"scores_not_object: {type(raw_scores).__name__}; raw={raw[:200]}")
    scores_a_raw = raw_scores.get("A")
    scores_b_raw = raw_scores.get("B")
    if not isinstance(scores_a_raw, dict) or not isinstance(scores_b_raw, dict):
        return _fail(
            f"scores_A_B_missing_or_not_object; "
            f"A_type={type(scores_a_raw).__name__}, "
            f"B_type={type(scores_b_raw).__name__}; raw={raw[:200]}"
        )
    scores_a: dict[str, float] = {}
    scores_b: dict[str, float] = {}
    for dim in JUDGE_B_DIMENSIONS:
        va = scores_a_raw.get(dim)
        vb = scores_b_raw.get(dim)
        if va is None or vb is None:
            return _fail(f"scores_missing_dim: {dim}; raw={raw[:200]}")
        try:
            fa = float(va)
            fb = float(vb)
        except (TypeError, ValueError):
            return _fail(
                f"scores_dim_not_numeric: {dim} "
                f"A={va!r} B={vb!r}; raw={raw[:200]}"
            )
        if not (0.0 <= fa <= 10.0) or not (0.0 <= fb <= 10.0):
            return _fail(
                f"scores_dim_out_of_range: {dim} A={fa} B={fb}; raw={raw[:200]}"
            )
        scores_a[dim] = fa
        scores_b[dim] = fb

    # Low-confidence flag (Major-1). Does not fail parse, but caller
    # routes low-confidence verdicts to disputed.
    low_confidence = confidence < JUDGE_B_PILOT_CONFIDENCE_THRESHOLD

    # Major-2 fix: detect preferred-vs-scores contradiction.
    total_a = sum(scores_a.values())
    total_b = sum(scores_b.values())
    contradiction = ""
    if preferred == "A" and (total_b - total_a) > JUDGE_B_CONTRADICTION_TOTAL_GAP:
        contradiction = (
            f"preferred=A but B total {total_b:.1f} exceeds A total "
            f"{total_a:.1f} by {total_b - total_a:.1f} "
            f"(> {JUDGE_B_CONTRADICTION_TOTAL_GAP})"
        )
    elif preferred == "B" and (total_a - total_b) > JUDGE_B_CONTRADICTION_TOTAL_GAP:
        contradiction = (
            f"preferred=B but A total {total_a:.1f} exceeds B total "
            f"{total_b:.1f} by {total_a - total_b:.1f} "
            f"(> {JUDGE_B_CONTRADICTION_TOTAL_GAP})"
        )
    elif preferred == "TIE":
        # For a tie, no single dim should differ by more than the tie gap.
        for dim in JUDGE_B_DIMENSIONS:
            gap = abs(scores_a[dim] - scores_b[dim])
            if gap > JUDGE_B_TIE_DIM_GAP:
                contradiction = (
                    f"preferred=tie but dim '{dim}' gap {gap:.1f} "
                    f"(A={scores_a[dim]}, B={scores_b[dim]}) "
                    f"exceeds tie threshold {JUDGE_B_TIE_DIM_GAP}"
                )
                break
        # Also reject "tie" when both runs have very low total scores
        # (judge dodged the question rather than finding them close).
        if not contradiction and total_a < 12.0 and total_b < 12.0:
            contradiction = (
                f"preferred=tie but both totals low (A={total_a:.1f}, "
                f"B={total_b:.1f}, both <12) — tie is not justified"
            )

    if preferred == "TIE":
        return JudgeBParsed(
            parse_ok=True, prefers_candidate=False, confidence=confidence,
            evidence=evidence, is_tie=True, low_confidence=low_confidence,
            scores_a=scores_a, scores_b=scores_b,
            score_contradiction=contradiction,
        )

    prefers_candidate = (preferred == "A") if candidate_is_a else (preferred == "B")
    return JudgeBParsed(
        parse_ok=True, prefers_candidate=prefers_candidate, confidence=confidence,
        evidence=evidence, is_tie=False, low_confidence=low_confidence,
        scores_a=scores_a, scores_b=scores_b,
        score_contradiction=contradiction,
    )


def verify_same_question_for_judge_b(
    candidate: dict[str, Any],
    negative: dict[str, Any],
    sample_spec_id: str,
) -> None:
    """Critical: candidate and v3 negative must share the same human dialogue."""
    cand_human = [m.get("value", "") for m in candidate.get("conversations", []) if m.get("from") == "human"]
    neg_human = [m.get("value", "") for m in negative.get("conversations", []) if m.get("from") == "human"]
    if cand_human != neg_human:
        raise ValueError(
            f"Judge B same-question violated for {sample_spec_id}: "
            f"candidate.human={cand_human} != negative.human={neg_human}"
        )


def judge_b(
    candidate: dict[str, Any],
    v3_negative: dict[str, Any],
    sample_spec_id: str,
    *,
    cache_dir: Path | None = None,
    attempt: int = 0,
    rate_limiter: RateLimiter | None = None,
) -> JudgeBResult:
    """Run Judge B double-order pairwise comparison.

    Major-1 fix: low-confidence verdicts (either run below pilot
    threshold 0.6) route to ``disputed`` — a passed verdict on a low-
    confidence double-order is not trustworthy.

    Major-2 fix: ``parse_judge_b_response`` now validates the 4-dim
    scores for A and B, and flags contradictions (preferred disagrees
    with score totals, or tie claimed with large dim gaps). Any
    contradiction also routes to ``disputed``.

    Rate-limit fix: an optional ``rate_limiter`` is now honoured between
    the two API calls so the DeepSeek provider is not hit back-to-back
    (which previously could trigger 429s and waste retry budget).
    """
    # Pre-flight: enforce same-question constraint
    verify_same_question_for_judge_b(candidate, v3_negative, sample_spec_id)

    cand_conversations = candidate.get("conversations", [])
    neg_conversations = v3_negative.get("conversations", [])
    # Extract the shared human dialogue (use negative's as canonical)
    human_dialogue = [m.get("value", "") for m in neg_conversations if m.get("from") == "human"]

    # Run 1: candidate as A, negative as B
    prompt1 = build_judge_b_prompt(cand_conversations, neg_conversations, human_dialogue)
    messages1 = [
        {"role": "system", "content": JUDGE_B_SYSTEM_PROMPT},
        {"role": "user", "content": prompt1},
    ]
    cache_key1 = request_cache_key(
        role="judge_b_run1", sample_spec_id=sample_spec_id, attempt=attempt,
        prompt_hash=hash_prompt(prompt1),
    )
    if cache_dir is not None:
        cache_file1 = cache_dir / f"judge_b_run1_{cache_key1}.json"
        if cache_file1.exists():
            raw1 = json.loads(cache_file1.read_text(encoding="utf-8"))["raw"]
        else:
            if rate_limiter is not None:
                rate_limiter.wait()
            raw1 = exponential_backoff_retry(
                lambda: call_judge_b(messages1, temperature=0.0, max_tokens=4096),
                max_attempts=4, base_delay=1.0,
            )
            cache_file1.parent.mkdir(parents=True, exist_ok=True)
            cache_file1.write_text(json.dumps({"raw": raw1}, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        if rate_limiter is not None:
            rate_limiter.wait()
        raw1 = exponential_backoff_retry(
            lambda: call_judge_b(messages1, temperature=0.0, max_tokens=4096),
            max_attempts=4, base_delay=1.0,
        )

    # Run 2: negative as A, candidate as B
    # Major fix: honour the rate limiter between the two Judge B calls.
    prompt2 = build_judge_b_prompt(neg_conversations, cand_conversations, human_dialogue)
    messages2 = [
        {"role": "system", "content": JUDGE_B_SYSTEM_PROMPT},
        {"role": "user", "content": prompt2},
    ]
    cache_key2 = request_cache_key(
        role="judge_b_run2", sample_spec_id=sample_spec_id, attempt=attempt,
        prompt_hash=hash_prompt(prompt2),
    )
    if cache_dir is not None:
        cache_file2 = cache_dir / f"judge_b_run2_{cache_key2}.json"
        if cache_file2.exists():
            raw2 = json.loads(cache_file2.read_text(encoding="utf-8"))["raw"]
        else:
            if rate_limiter is not None:
                rate_limiter.wait()
            raw2 = exponential_backoff_retry(
                lambda: call_judge_b(messages2, temperature=0.0, max_tokens=4096),
                max_attempts=4, base_delay=1.0,
            )
            cache_file2.parent.mkdir(parents=True, exist_ok=True)
            cache_file2.write_text(json.dumps({"raw": raw2}, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        if rate_limiter is not None:
            rate_limiter.wait()
        raw2 = exponential_backoff_retry(
            lambda: call_judge_b(messages2, temperature=0.0, max_tokens=4096),
            max_attempts=4, base_delay=1.0,
        )

    # Parse both runs.
    p1 = parse_judge_b_response(raw1, candidate_is_a=True)
    p2 = parse_judge_b_response(raw2, candidate_is_a=False)

    # Position-bias telemetry (tie does not count as a position-A preference).
    first_pos_a_run1 = bool(p1.parse_ok and (not p1.is_tie) and p1.prefers_candidate)
    first_pos_a_run2 = bool(p2.parse_ok and (not p2.is_tie) and (not p2.prefers_candidate))

    # Major-3: pre-initialize score-derived fields so they exist even
    # when the decision falls into a fail-closed branch (parse failure,
    # low confidence, contradiction) before the averaging block runs.
    candidate_avg: dict[str, float] = {}
    negative_avg: dict[str, float] = {}
    score_gap = 0.0

    # Major-3 fix: score-derived decision to eliminate position bias.
    #
    # smoke12_v2 showed first_position_preference_rate=1.0 — the judge
    # always selects position A, and its scores agree with that biased
    # selection (score_contradiction_rate=0.0). The old decision logic
    # (both runs prefer candidate => passed) could never agree because
    # Run1 picks position-A=candidate and Run2 picks position-A=negative,
    # producing a disputed verdict on every sample.
    #
    # New approach: derive the decision from the AVERAGE of the
    # candidate's scores across both orderings. Position bias inflates
    # the candidate's scores in Run1 (candidate at A) and deflates them
    # in Run2 (candidate at B). Averaging cancels the bias:
    #
    #   candidate_avg[dim] = (p1.scores_a[dim] + p2.scores_b[dim]) / 2
    #   negative_avg[dim]  = (p1.scores_b[dim] + p2.scores_a[dim]) / 2
    #
    # The decision is then based on the averaged totals, NOT on the
    # judge's ``preferred`` field. This makes the verdict symmetric:
    # swapping candidate/negative labels does not change the decision.
    #
    # Fail-closed to disputed on any of:
    #   - parse failure (JSON / preferred / confidence / scores invalid)
    #   - any low-confidence run (Major-1)
    #   - any preferred-vs-scores contradiction (Major-2)
    # Only when both runs parse cleanly, are non-low-confidence,
    # non-contradictory, and have valid scores do we derive the decision
    # from averaged scores.
    if not p1.parse_ok or not p2.parse_ok:
        decision = "disputed"
    elif p1.low_confidence or p2.low_confidence:
        decision = "disputed"
    elif p1.score_contradiction or p2.score_contradiction:
        decision = "disputed"
    else:
        # Both runs parsed cleanly with valid, non-contradictory scores.
        # Average the candidate's scores across both orderings to cancel
        # position bias.
        candidate_avg = {}
        negative_avg = {}
        for dim in JUDGE_B_DIMENSIONS:
            # Run1: candidate=A → candidate scores in p1.scores_a
            # Run2: candidate=B → candidate scores in p2.scores_b
            c_scores = [p1.scores_a.get(dim, 0.0), p2.scores_b.get(dim, 0.0)]
            # Run1: negative=B → negative scores in p1.scores_b
            # Run2: negative=A → negative scores in p2.scores_a
            n_scores = [p1.scores_b.get(dim, 0.0), p2.scores_a.get(dim, 0.0)]
            candidate_avg[dim] = sum(c_scores) / 2.0
            negative_avg[dim] = sum(n_scores) / 2.0

        candidate_total = sum(candidate_avg.values())
        negative_total = sum(negative_avg.values())
        score_gap = candidate_total - negative_total

        if score_gap > JUDGE_B_SCORE_DERIVED_TIE_GAP:
            decision = "passed"
        elif score_gap < -JUDGE_B_SCORE_DERIVED_TIE_GAP:
            decision = "rejected"
        else:
            decision = "disputed"

    return JudgeBResult(
        prefers_candidate_run1=p1.prefers_candidate,
        prefers_candidate_run2=p2.prefers_candidate,
        confidence_run1=p1.confidence,
        confidence_run2=p2.confidence,
        evidence_run1=p1.evidence,
        evidence_run2=p2.evidence,
        final_decision=decision,
        raw_run1=raw1,
        raw_run2=raw2,
        is_tie_run1=p1.is_tie,
        is_tie_run2=p2.is_tie,
        first_position_a_run1=first_pos_a_run1,
        first_position_a_run2=first_pos_a_run2,
        low_confidence_run1=p1.low_confidence,
        low_confidence_run2=p2.low_confidence,
        scores_a_run1=p1.scores_a,
        scores_b_run1=p1.scores_b,
        scores_a_run2=p2.scores_a,
        scores_b_run2=p2.scores_b,
        score_contradiction_run1=p1.score_contradiction,
        score_contradiction_run2=p2.score_contradiction,
        candidate_avg_scores=candidate_avg,
        negative_avg_scores=negative_avg,
        score_derived_gap=score_gap,
    )


# ---------------------------------------------------------------------------
# CLI: judge a single candidate (debug)
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Run Judge A and/or Judge B on a candidate (debug)")
    parser.add_argument("--candidate-json", type=Path, required=True,
                        help="candidate JSON with {conversations, scene, sample_spec_id}")
    parser.add_argument("--v3-negative-jsonl", type=Path, default=None,
                        help="v3_negative_pool.jsonl for Judge B same-question compare")
    parser.add_argument("--skip-judge-b", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()

    candidate = json.loads(args.candidate_json.read_text(encoding="utf-8"))
    scene = candidate.get("scene", "日常场景")
    sample_spec_id = candidate.get("sample_spec_id", "")

    # Judge A
    print("=== Judge A ===")
    a_result = judge_a(candidate, scene, reference_passages=[],
                       cache_dir=args.cache_dir, sample_spec_id=sample_spec_id)
    print(json.dumps(a_result.to_dict(), ensure_ascii=False, indent=2))

    if args.skip_judge_b:
        return 0 if a_result.passed else 1

    # Judge B
    if not args.v3_negative_jsonl or not args.v3_negative_jsonl.exists():
        print("=== Judge B skipped (no v3_negative_jsonl) ===")
        return 0 if a_result.passed else 1

    # Find the matching v3 negative by sample_spec_id
    negative = None
    with args.v3_negative_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["sample_spec_id"] == sample_spec_id:
                negative = rec
                break
    if negative is None:
        print(f"=== Judge B skipped (no v3 negative for {sample_spec_id}) ===")
        return 0 if a_result.passed else 1

    print("\n=== Judge B ===")
    b_result = judge_b(candidate, negative, sample_spec_id, cache_dir=args.cache_dir)
    print(json.dumps(b_result.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
