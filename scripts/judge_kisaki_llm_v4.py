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
    JudgeBResult,
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
    violations = list(parsed.get("violations", []) or [])
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

    # Pass requires: (1) no missing mandatory dims, (2) all applicable dims ≥ threshold
    passed = (
        not missing_mandatory
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
        lambda: call_judge_a(messages, temperature=0.0, max_tokens=800),
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

JUDGE_B_SYSTEM_PROMPT = """你是月社妃对话样本的成对比较评审官。你将看到两个回复（A 和 B），它们回答的是**相同的用户提问**。

【比较维度】
1. 人物一致性: 哪个更符合月社妃的人物画像（克制理智、锋利戏谑、对琉璃的情感、元叙事视角）
2. 原作语气: 哪个更贴近原作用词（因此/哎呀/呼呼呼/——破折号）
3. 元叙事控制: 哪个避免了过度元叙事（"故事/作者/剧本"过载是 v3 负例的典型问题）
4. 自然度: 哪个更自然流畅

【重要】
- 不要因为位置前后给高分（A 在前不代表 A 更好）
- 若两者质量接近，选择更贴近原作风格的
- 若两者都差，仍需选出相对较好的一个

【输出要求】
严格输出 JSON（不要 markdown 代码块）：
{
  "preferred": "A" 或 "B",
  "confidence": 0.0-1.0,
  "evidence": "一句话说明选择理由",
  "reason": "更详细的对比分析"
}"""


def build_judge_b_prompt(
    response_a: list[dict[str, Any]],
    response_b: list[dict[str, Any]],
    human_dialogue: list[str],
) -> str:
    """Build the Judge B user prompt for one ordering."""
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

请比较 A 和 B，选出更贴近月社妃原作风格的一个。"""


def parse_judge_b_response(raw: str, candidate_is_a: bool) -> tuple[bool, float, str, bool]:
    """Parse one Judge B run.

    Returns (prefers_candidate, confidence, evidence, parse_ok).
    Major-6 fix: parse failure is now surfaced as a separate ``parse_ok``
    flag instead of being silently mapped to ``prefers_candidate=False``,
    which previously caused Judge B to treat JSON formatting glitches as
    a candidate rejection (double-reject => rejected). Callers should
    route parse failures to ``disputed`` (human review), not ``rejected``.
    """
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        # Return parse_ok=False so caller can route to disputed instead of
        # silently treating this as a candidate loss.
        return False, 0.0, f"json_parse_error: {e}; raw={raw[:200]}", False

    preferred = str(parsed.get("preferred", "")).strip().upper()
    if preferred not in ("A", "B"):
        # Invalid preferred value — also a parse failure (route to disputed)
        return False, 0.0, f"invalid_preferred_value: {preferred!r}; raw={raw[:200]}", False

    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    evidence = parsed.get("evidence", "")

    prefers_candidate = (preferred == "A") if candidate_is_a else (preferred == "B")
    return prefers_candidate, confidence, evidence, True


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
) -> JudgeBResult:
    """Run Judge B double-order pairwise comparison."""
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
            raw1 = exponential_backoff_retry(
                lambda: call_judge_b(messages1, temperature=0.0, max_tokens=600),
                max_attempts=4, base_delay=1.0,
            )
            cache_file1.parent.mkdir(parents=True, exist_ok=True)
            cache_file1.write_text(json.dumps({"raw": raw1}, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        raw1 = exponential_backoff_retry(
            lambda: call_judge_b(messages1, temperature=0.0, max_tokens=600),
            max_attempts=4, base_delay=1.0,
        )

    # Run 2: negative as A, candidate as B
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
            raw2 = exponential_backoff_retry(
                lambda: call_judge_b(messages2, temperature=0.0, max_tokens=600),
                max_attempts=4, base_delay=1.0,
            )
            cache_file2.parent.mkdir(parents=True, exist_ok=True)
            cache_file2.write_text(json.dumps({"raw": raw2}, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        raw2 = exponential_backoff_retry(
            lambda: call_judge_b(messages2, temperature=0.0, max_tokens=600),
            max_attempts=4, base_delay=1.0,
        )

    # Parse: run1 candidate_is_a=True, run2 candidate_is_a=False
    # Major-6 fix: parse failure now surfaces as parse_ok=False; the judge
    # routes such cases to "disputed" (human review) instead of silently
    # treating them as candidate losses (which would double-reject and
    # wrongly kill the sample).
    prefers_cand_1, conf1, evid1, parse_ok1 = parse_judge_b_response(raw1, candidate_is_a=True)
    prefers_cand_2, conf2, evid2, parse_ok2 = parse_judge_b_response(raw2, candidate_is_a=False)

    # Decide: any parse failure => disputed (infrastructure issue, not a
    # genuine quality signal). Otherwise apply the double-order rule.
    if not parse_ok1 or not parse_ok2:
        decision = "disputed"
    elif prefers_cand_1 and prefers_cand_2:
        decision = "passed"
    elif (not prefers_cand_1) and (not prefers_cand_2):
        decision = "rejected"
    else:
        decision = "disputed"

    return JudgeBResult(
        prefers_candidate_run1=prefers_cand_1,
        prefers_candidate_run2=prefers_cand_2,
        confidence_run1=conf1,
        confidence_run2=conf2,
        evidence_run1=evid1,
        evidence_run2=evid2,
        final_decision=decision,
        raw_run1=raw1,
        raw_run2=raw2,
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
