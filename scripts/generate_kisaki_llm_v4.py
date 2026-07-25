"""Kisaki V4 candidate generator (Task B.4).

For each SampleSpec:
  1. Retrieve 2-4 few-shot records from the training-side pool (same scene)
  2. Retrieve the matching v3 negative (same sample_spec_id) — the candidate
     must answer the **same human dialogue** so Judge B can do a fair
     same-question A/B comparison
  3. Build a prompt: character profile + few-shot + negative-to-avoid +
     quota_state + retry feedback (if any)
  4. Call DeepSeek (Generator)
  5. Parse the JSON response into a conversations array

Same-question constraint (V2.1 Critical #4):
  The candidate's human_dialogue MUST equal the v3 negative's human_dialogue.
  This is enforced before generation and again before Judge B comparison.
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
    SampleSpec,
    call_generator,
    exponential_backoff_retry,
    hash_prompt,
    request_cache_key,
)
from generate_kisaki_llm_dialogues_v3 import CHARACTER_DESC, SCENES  # noqa: E402

FEW_SHOT_POOL_PATH = (
    BACKEND / "data" / "character_dialogues" / "experiments" / "v3" / "llm_v4_judged" / "few_shot_pool.jsonl"
)
NEGATIVE_POOL_PATH = (
    BACKEND / "data" / "character_dialogues" / "experiments" / "v3" / "llm_v4_judged" / "v3_negative_pool.jsonl"
)
# SCENES in generate_kisaki_llm_dialogues_v3.py is a list of 2-tuples (scene, desc).
SCENE_DESC_MAP = {scene: desc for scene, desc in SCENES}


# ---------------------------------------------------------------------------
# Pool loaders (cached)
# ---------------------------------------------------------------------------

_FEW_SHOT_BY_SCENE: dict[str, list[dict[str, Any]]] | None = None
_NEGATIVE_BY_SPEC_ID: dict[str, dict[str, Any]] | None = None


def _load_few_shot_pool(path: Path = FEW_SHOT_POOL_PATH) -> dict[str, list[dict[str, Any]]]:
    global _FEW_SHOT_BY_SCENE
    if _FEW_SHOT_BY_SCENE is not None:
        return _FEW_SHOT_BY_SCENE
    by_scene: dict[str, list[dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_scene.setdefault(rec["scene_tag"], []).append(rec)
    _FEW_SHOT_BY_SCENE = by_scene
    return _FEW_SHOT_BY_SCENE


def _load_negative_pool(path: Path = NEGATIVE_POOL_PATH) -> dict[str, dict[str, Any]]:
    global _NEGATIVE_BY_SPEC_ID
    if _NEGATIVE_BY_SPEC_ID is not None:
        return _NEGATIVE_BY_SPEC_ID
    by_id: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_id[rec["sample_spec_id"]] = rec
    _NEGATIVE_BY_SPEC_ID = by_id
    return _NEGATIVE_BY_SPEC_ID


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def _keyword_overlap_score(human_dialogue: list[str], candidate_human_turns: list[str]) -> int:
    """Cheap relevance score: count of shared content keywords.

    Uses a simple character-bigram overlap on the joined human turns to avoid
    pulling in heavyweight embedding deps. Deterministic and reproducible.
    """
    if not human_dialogue or not candidate_human_turns:
        return 0
    q_text = "".join(human_dialogue)
    c_text = "".join(candidate_human_turns)
    if not q_text or not c_text:
        return 0
    q_grams = {q_text[i:i + 2] for i in range(len(q_text) - 1)}
    c_grams = {c_text[i:i + 2] for i in range(len(c_text) - 1)}
    return len(q_grams & c_grams)


def retrieve_few_shots(
    scene: str,
    k: int = 3,
    *,
    human_dialogue: list[str] | None = None,
    exclude_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Retrieve k few-shot records for a scene.

    Major-7 fix: previously this returned the first k records by sample_id
    sort, which always picked the same few-shot set per scene — creating a
    new anchoring/templating bias. Now it ranks candidates by keyword
    overlap with the current ``human_dialogue`` (when provided) and applies
    source diversification so the top-k come from different source files
    when possible.

    Behaviour:
      - If ``human_dialogue`` is provided, candidates are scored by
        character-bigram overlap with the candidate's human turns.
      - Ties are broken by sample_id for determinism.
      - ``exclude_ids`` (e.g. already-used reference IDs from a previous
        attempt) are filtered out to vary few-shots across retries.
      - Source diversification: greedily pick the next-best candidate from
        a *different* source_file before allowing repeats, so the top-k
        span multiple original chapters (reduces single-chapter anchoring).
    """
    pool = _load_few_shot_pool()
    candidates = pool.get(scene, [])
    if not candidates:
        # Fallback: pull from any scene (prefer 角色设定 which is the largest bucket)
        candidates = pool.get("角色设定", [])
        if not candidates:
            # Last resort: flatten
            candidates = [rec for scene_recs in pool.values() for rec in scene_recs]

    if exclude_ids:
        candidates = [r for r in candidates if r["sample_id"] not in exclude_ids]
    if not candidates:
        return []

    # Score by relevance (default 0 when human_dialogue not provided)
    scored: list[tuple[int, str, dict[str, Any]]] = []
    for rec in candidates:
        cand_human = [
            m.get("value", "")
            for m in rec.get("conversations", [])
            if m.get("from") == "human"
        ]
        score = _keyword_overlap_score(human_dialogue or [], cand_human)
        scored.append((score, rec["sample_id"], rec))

    # Sort: score desc, then sample_id asc for determinism
    scored.sort(key=lambda t: (-t[0], t[1]))

    # Source diversification: greedily prefer new source_file, fall back to
    # already-seen ones only when no new source remains.
    selected: list[dict[str, Any]] = []
    used_sources: set[str] = set()
    remaining = list(scored)
    while len(selected) < k and remaining:
        # First pass: pick the best candidate from a new source
        pick_idx = None
        for i, (_, _, rec) in enumerate(remaining):
            src = rec.get("source_file") or ""
            if src not in used_sources:
                pick_idx = i
                break
        # If no new-source candidate, just take the best remaining
        if pick_idx is None:
            pick_idx = 0
        _, _, picked = remaining.pop(pick_idx)
        selected.append(picked)
        used_sources.add(picked.get("source_file") or "")

    return selected


def retrieve_negative(sample_spec_id: str) -> dict[str, Any] | None:
    """Retrieve the v3 negative sharing this sample_spec_id."""
    pool = _load_negative_pool()
    return pool.get(sample_spec_id)


# ---------------------------------------------------------------------------
# Same-question constraint enforcement
# ---------------------------------------------------------------------------

def verify_same_question(spec: SampleSpec, negative: dict[str, Any]) -> None:
    """Critical: candidate and v3 negative must share the same human dialogue."""
    if spec.human_dialogue != negative["human_dialogue"]:
        raise ValueError(
            f"same-question constraint violated for {spec.sample_spec_id}: "
            f"spec.human_dialogue={spec.human_dialogue} != "
            f"negative.human_dialogue={negative['human_dialogue']}"
        )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

JSON_OUTPUT_INSTRUCTION = """请严格输出以下 JSON 结构（不要输出任何其他内容，不要 markdown 代码块标记）：
{
  "conversations": [
    {"from": "human", "value": "<与负例完全相同的用户提问>"},
    {"from": "assistant", "value": "<妃的回复，≤100字符>"}
  ]
}

约束：
1. human 的 value 必须与下面"同题提问"完全一致（逐字符相同）
2. assistant 的 value ≤100 字符
3. 不得出现"我是AI/作为AI/我是一个助手"等 AI 自称
4. 不得以第三人称描述妃（如"月社妃是...的角色"）
5. 非元叙事场景不得使用"故事/作者/剧本/出场人物/规则"等元叙事词
6. "正因如此"最多出现 1 次
7. 不得直接复制原作片段"""


def build_generator_prompt(
    spec: SampleSpec,
    few_shots: list[dict[str, Any]],
    negative: dict[str, Any],
    quota_state: dict[str, Any] | None = None,
    retry_feedback: str = "",
) -> str:
    """Build the full generator prompt."""
    scene_desc = SCENE_DESC_MAP.get(spec.scene, "")
    human_dialogue_str = "\n".join(
        f"  用户: {turn}" for turn in spec.human_dialogue
    )

    # Few-shot block (2-4 records)
    few_shot_blocks: list[str] = []
    for fs in few_shots:
        conv_lines = []
        for msg in fs.get("conversations", []):
            sender = "用户" if msg["from"] == "human" else "妃"
            conv_lines.append(f"    {sender}: {msg['value']}")
        few_shot_blocks.append(
            f"  【原作参考 {fs['sample_id']}】\n" + "\n".join(conv_lines)
        )
    few_shot_text = "\n".join(few_shot_blocks) if few_shot_blocks else "  (无可用原作参考)"

    # Negative block (same-question v3 sample to avoid)
    neg_conv_lines = []
    for msg in negative.get("conversations", []):
        sender = "用户" if msg["from"] == "human" else "妃(v3负例)"
        neg_conv_lines.append(f"    {sender}: {msg['value']}")
    neg_problems = ", ".join(negative.get("problem_tags", []))
    negative_text = (
        f"  【v3 负例（同题，需避免其问题: {neg_problems}）】\n"
        + "\n".join(neg_conv_lines)
    )

    # Quota state (e.g. "本批次元叙事已超限，不得使用元叙事词")
    quota_hint = ""
    if quota_state:
        hints: list[str] = []
        if quota_state.get("meta_narrative_over_limit"):
            hints.append("本批次元叙事样本已超限，不得使用'故事/作者/剧本/出场人物/规则'等元叙事词")
        if quota_state.get("zheng_yin_ci_over_limit"):
            hints.append("本批次'正因如此'已超限，不得使用该词")
        if quota_state.get("laughter_needs_diversity"):
            hints.append("本批次笑声多样性不足，可使用'呼呼呼/噗噗/呵呵'等不同变体（若自然）")
        if quota_state.get("sharp_expression_under_limit"):
            hints.append("本批次锋利表达不足，可适当使用'恕我拒绝/没有那个必要/你疯了吗'等（若自然）")
        if hints:
            quota_hint = "\n  【本批次配额提示】\n    " + "\n    ".join(hints)

    # Retry feedback (from prior hard-gate or Judge failure)
    feedback_text = ""
    if retry_feedback:
        feedback_text = f"\n  【上轮失败反馈，请避免】\n    {retry_feedback}"

    prompt = f"""{CHARACTER_DESC}

【本次任务】
场景: {spec.scene}（{scene_desc}）
同题提问（必须逐字符复用以下 human 内容）:
{human_dialogue_str}

【原作风格参考（模仿风格，但不得直接复制内容）】
{few_shot_text}

【v3 负例（同题，请避免其问题）】
{negative_text}
{quota_hint}{feedback_text}

{JSON_OUTPUT_INSTRUCTION}"""
    return prompt


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_generator_response(raw: str, spec: SampleSpec) -> dict[str, Any]:
    """Parse the generator's JSON response into a candidate record.

    Strips accidental markdown code fences and extracts the conversations array.
    Verifies the same-question constraint: human turns must match spec.human_dialogue.
    """
    text = raw.strip()
    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"generator returned non-JSON: {e}; raw[:200]={raw[:200]!r}")

    conversations = parsed.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise ValueError("generator response missing conversations array")

    # Enforce same-question: human turns must match spec.human_dialogue
    human_turns = [m.get("value", "") for m in conversations if m.get("from") == "human"]
    if human_turns != spec.human_dialogue:
        # Force-overwrite human turns to match spec (generator may have paraphrased)
        # This is the safest way to guarantee same-question for Judge B.
        human_idx = 0
        for msg in conversations:
            if msg.get("from") == "human":
                if human_idx < len(spec.human_dialogue):
                    msg["value"] = spec.human_dialogue[human_idx]
                    human_idx += 1
        # If generator produced fewer human turns than spec, that's a real failure
        if human_idx < len(spec.human_dialogue):
            raise ValueError(
                f"generator produced {human_idx} human turns, spec requires {len(spec.human_dialogue)}"
            )

    return {
        "sample_spec_id": spec.sample_spec_id,
        "scene": spec.scene,
        "scene_desc": SCENE_DESC_MAP.get(spec.scene, ""),
        "conversations": conversations,
        "reference_ids": spec.reference_ids,
        "v3_negative_sample_id": spec.v3_negative_sample_id,
    }


# ---------------------------------------------------------------------------
# Public entry: generate_one_candidate
# ---------------------------------------------------------------------------

def generate_one_candidate(
    spec: SampleSpec,
    *,
    quota_state: dict[str, Any] | None = None,
    retry_feedback: str = "",
    cache_dir: Path | None = None,
    attempt: int = 0,
) -> dict[str, Any]:
    """Generate a single candidate. Raises on unrecoverable failure.

    Major-7: few-shot retrieval is now relevance-based (keyword overlap
    with ``spec.human_dialogue``) and source-diversified. On retries
    (attempt > 0) the previous reference_ids are excluded so the model
    sees a fresh few-shot set instead of re-anchoring to the same samples.
    """
    # 1. Retrieve few-shots and negative
    # On retry, exclude reference_ids used in the previous attempt to vary
    # the anchor and reduce templating. spec.reference_ids holds the IDs
    # used in attempt 0; on attempt>0 we drop them.
    exclude_ids: set[str] | None = None
    if attempt > 0 and spec.reference_ids:
        exclude_ids = set(spec.reference_ids)
    few_shots = retrieve_few_shots(
        spec.scene,
        k=3,
        human_dialogue=spec.human_dialogue,
        exclude_ids=exclude_ids,
    )
    # Update spec.reference_ids so the candidate record reflects what was
    # actually used (for provenance and downstream copy detection).
    spec.reference_ids = [fs["sample_id"] for fs in few_shots]

    negative = retrieve_negative(spec.sample_spec_id)
    if negative is None:
        raise ValueError(
            f"no v3 negative found for sample_spec_id={spec.sample_spec_id}; "
            "build_v3_negative_pool.py must run first"
        )
    # 2. Enforce same-question constraint (pre-flight)
    verify_same_question(spec, negative)

    # 3. Build prompt
    prompt = build_generator_prompt(spec, few_shots, negative, quota_state, retry_feedback)
    messages = [{"role": "user", "content": prompt}]

    # 4. Check request cache (deduplicate identical calls)
    cache_key = request_cache_key(
        role="generator",
        sample_spec_id=spec.sample_spec_id,
        attempt=0,  # attempts handled by pipeline via retry_feedback
        prompt_hash=hash_prompt(prompt),
    )
    if cache_dir is not None:
        cache_file = cache_dir / f"gen_{cache_key}.json"
        if cache_file.exists():
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            return cached["candidate"]

    # 5. Call DeepSeek with exponential backoff
    raw = exponential_backoff_retry(
        lambda: call_generator(messages, temperature=0.8, max_tokens=512),
        max_attempts=4,
        base_delay=1.0,
    )

    # 6. Parse
    candidate = parse_generator_response(raw, spec)

    # 7. Cache
    if cache_dir is not None:
        cache_file = cache_dir / f"gen_{cache_key}.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(
            json.dumps({"candidate": candidate, "raw": raw}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return candidate


# ---------------------------------------------------------------------------
# CLI: generate for one spec_id (debug)
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Generate one V4 candidate (debug)")
    parser.add_argument("--sample-spec-id", required=True,
                        help="kisaki_v3neg_<scene>_<idx> from v3_negative_pool.jsonl")
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()

    negative = retrieve_negative(args.sample_spec_id)
    if negative is None:
        print(json.dumps({"error": f"sample_spec_id not found: {args.sample_spec_id}"},
                         ensure_ascii=False))
        return 2

    spec = SampleSpec(
        sample_spec_id=negative["sample_spec_id"],
        scene=negative["scene"],
        scene_desc=SCENE_DESC_MAP.get(negative["scene"], ""),
        human_dialogue=negative["human_dialogue"],
        v3_negative_sample_id=negative["v3_sample_id"],
        reference_ids=[],
        quota_plan_hash="v3neg",
    )
    candidate = generate_one_candidate(spec, cache_dir=args.cache_dir)
    print(json.dumps(candidate, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
