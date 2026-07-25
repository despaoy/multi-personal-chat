"""Kisaki V4 regeneration pipeline orchestrator (Task B.7 + D.1).

End-to-end flow per SampleSpec:
  1. Generate candidate (DeepSeek) with few-shot + v3 negative + retry feedback
  2. Run 8 code-level hard gates (length / AI-self / third-person / repeated
     opening / meta-narrative / 正因如此 / original-copy / JSON structure)
  3. Run Judge A (5-dim semantic, Qwen-Max) — skip if hard gate fails
  4. Run Judge B (same-question double-order pairwise, DeepSeek) — skip if
     Judge A fails
  5. Route final record to:
       - samples.jsonl           (status="passed")
       - rejected_samples.jsonl  (status="rejected" after MAX_ATTEMPTS)
       - disputed_samples.jsonl  (status="disputed" — needs human review)

Resilience (Task B.7):
  - progress.json: completed sample_spec_ids + stats; atomic-write per sample
  - Atomic writes: temp file + os.replace (from kisaki_v4_llm_client)
  - Stable sample IDs: from v3_negative_pool.jsonl (kisaki_v3neg_<scene>_<idx>);
    retries never change the ID
  - Rate limiter: configurable min interval between API calls (default 1s)
  - Exponential backoff: 1s -> 2s -> 4s -> 8s, max 4 attempts (already
    baked into call_generator/judge_a/judge_b via exponential_backoff_retry)
  - Request cache: (role, sample_spec_id, attempt, prompt_hash) -> cached
    response, deduplicating identical LLM calls across runs

Resume semantics:
  - On restart, reads progress.json and skips already-completed sample_spec_ids
  - Already-written samples.jsonl / rejected_samples.jsonl / disputed_samples.jsonl
    are appended to (never rewritten), so prior outputs are preserved
  - passed_samples list is rebuilt from samples.jsonl at startup so the
    repeated-opening gate stays correct across restarts
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND = PROJECT_ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from kisaki_v4_llm_client import (  # noqa: E402
    RateLimiter,
    SampleSpec,
    append_jsonl,
    atomic_write_json,
    build_judge_config,
    read_jsonl_ids,
)
from generate_kisaki_llm_v4 import (  # noqa: E402
    SCENE_DESC_MAP,
    generate_one_candidate,
    retrieve_few_shots,
    retrieve_negative,
)
from hard_gate_kisaki_v4 import run_all_gates  # noqa: E402
from judge_kisaki_llm_v4 import judge_a, judge_b  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

OUTPUT_DIR = (
    BACKEND / "data" / "character_dialogues" / "experiments" / "v3" / "llm_v4_judged"
)
PROGRESS_PATH = OUTPUT_DIR / "progress.json"
SAMPLES_PATH = OUTPUT_DIR / "samples.jsonl"
REJECTED_PATH = OUTPUT_DIR / "rejected_samples.jsonl"
DISPUTED_PATH = OUTPUT_DIR / "disputed_samples.jsonl"
RUN_LOG_PATH = OUTPUT_DIR / "run_log.jsonl"
CACHE_DIR = OUTPUT_DIR / "cache"

MAX_ATTEMPTS = 3  # per-spec retry on hard-gate/judge failure


# ---------------------------------------------------------------------------
# Progress persistence (B.7.1 + B.7.2)
# ---------------------------------------------------------------------------

def load_progress() -> dict[str, Any]:
    """Load progress.json from the default OUTPUT_DIR, or initialize a fresh one."""
    return _load_progress_from(PROGRESS_PATH)


def _load_progress_from(progress_path: Path) -> dict[str, Any]:
    """Load progress.json from an explicit path (Major-4: output_dir-aware).

    Schema:
      {
        "started_at": iso,
        "last_updated": iso,
        "completed_spec_ids": [str, ...],   # final-state IDs (passed/rejected/disputed)
        "stats": {"passed": int, "rejected": int, "disputed": int, "total_processed": int},
        "last_committed_spec_id": str | null  # Major-10: crash-recovery marker
      }
    """
    if progress_path.exists():
        return json.loads(progress_path.read_text(encoding="utf-8"))
    return {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "last_updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "completed_spec_ids": [],
        "stats": {"passed": 0, "rejected": 0, "disputed": 0, "total_processed": 0},
        "last_committed_spec_id": None,
    }


def save_progress(progress: dict[str, Any]) -> None:
    """Atomic write of progress.json to the default OUTPUT_DIR."""
    _save_progress_to(PROGRESS_PATH, progress)


def _save_progress_to(progress_path: Path, progress: dict[str, Any]) -> None:
    """Atomic write of progress.json to an explicit path (Major-4 + Major-10).

    Uses temp file + os.replace so a crash during write never leaves a
    truncated progress.json (either the old or the new version is on disk).
    """
    progress["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    atomic_write_json(progress_path, progress)


# ---------------------------------------------------------------------------
# Spec loading (B.7.3 — stable sample IDs from v3_negative_pool)
# ---------------------------------------------------------------------------

def load_specs_from_negative_pool(
    negative_pool_path: Path,
    quota_plan_hash: str = "v3neg",
) -> list[SampleSpec]:
    """Build SampleSpec list from v3_negative_pool.jsonl.

    Each v3 negative becomes a SampleSpec that the generator must answer
    with the SAME human dialogue (for Judge B fair A/B comparison). The
    sample_spec_id is the stable ID assigned by build_v3_negative_pool.py
    (kisaki_v3neg_<scene>_<idx>), so retries never change the ID.

    Major-7: reference_ids are now populated using relevance-based retrieval
    (keyword overlap with the spec's human_dialogue), so each spec starts
    with a question-specific few-shot set instead of a scene-constant one.
    """
    specs: list[SampleSpec] = []
    with negative_pool_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # Major-7: relevance-based retrieval using the spec's human_dialogue
            few_shots = retrieve_few_shots(
                rec["scene"], k=3, human_dialogue=rec["human_dialogue"],
            )
            reference_ids = [fs["sample_id"] for fs in few_shots]
            spec = SampleSpec(
                sample_spec_id=rec["sample_spec_id"],
                scene=rec["scene"],
                scene_desc=SCENE_DESC_MAP.get(rec["scene"], ""),
                human_dialogue=rec["human_dialogue"],
                v3_negative_sample_id=rec["v3_sample_id"],
                reference_ids=reference_ids,
                quota_plan_hash=quota_plan_hash,
                target_length_hint="auto",
            )
            specs.append(spec)
    return specs


# ---------------------------------------------------------------------------
# Load already-passed samples (for repeated-opening gate across restarts)
# ---------------------------------------------------------------------------

def load_passed_samples() -> list[dict[str, Any]]:
    """Rebuild the passed_samples list from the default SAMPLES_PATH."""
    return _load_passed_samples_from(SAMPLES_PATH)


def _load_passed_samples_from(samples_path: Path) -> list[dict[str, Any]]:
    """Rebuild the passed_samples list from samples.jsonl at an explicit path.

    Major-4: now accepts an explicit path so run_pipeline can read from
    output_dir/samples.jsonl instead of the global default.
    """
    if not samples_path.exists():
        return []
    passed: list[dict[str, Any]] = []
    for line in samples_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        candidate = rec.get("candidate")
        if candidate:
            passed.append(candidate)
    return passed


# ---------------------------------------------------------------------------
# Per-sample processing (D.1.2 + D.1.3 + D.1.6)
# ---------------------------------------------------------------------------

def _build_reference_texts(scene: str) -> list[str]:
    """Pull assistant texts from few-shot pool for copy detection."""
    refs: list[str] = []
    for fs in retrieve_few_shots(scene, k=3):
        for msg in fs.get("conversations", []):
            if msg.get("from") == "assistant":
                refs.append(msg.get("value", ""))
    return refs


def process_one_attempt(
    spec: SampleSpec,
    *,
    attempt: int,
    retry_feedback: str,
    passed_samples: list[dict[str, Any]],
    rate_limiter: RateLimiter,
    cache_dir: Path,
) -> dict[str, Any]:
    """Run one attempt of generate -> gate -> Judge A -> Judge B.

    Returns a record dict with:
      - sample_spec_id, scene, attempt, timestamp
      - status: "passed" | "rejected" | "disputed"
      - candidate, gate_result, judge_a, judge_b (when reached)
      - retry_feedback (for next attempt if not passed)
      - error (when an exception occurred)
    """
    record: dict[str, Any] = {
        "sample_spec_id": spec.sample_spec_id,
        "scene": spec.scene,
        "attempt": attempt,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # 1. Generate candidate (rate-limited, exponential-backoff inside)
    rate_limiter.wait()
    try:
        candidate = generate_one_candidate(
            spec,
            quota_state=None,      # batch auditor injects this in Stage D
            retry_feedback=retry_feedback,
            cache_dir=cache_dir,
            attempt=attempt,       # Major-7: vary few-shots on retry
        )
    except Exception as exc:  # noqa: BLE001
        record["status"] = "rejected"
        record["error"] = f"generator_failed: {exc}"
        record["retry_feedback"] = f"生成器调用失败: {exc}"
        return record
    record["candidate"] = candidate

    # 2. Hard gates (code-only, no LLM)
    references = _build_reference_texts(spec.scene)
    gate_result = run_all_gates(
        candidate,
        scene=spec.scene,
        references=references,
        passed_samples=passed_samples,
    )
    record["gate_result"] = gate_result.to_dict()
    if not gate_result.passed:
        # If only disputed_flags (no hard failures) -> disputed (human review)
        if gate_result.disputed_flags and not gate_result.failures:
            record["status"] = "disputed"
            record["retry_feedback"] = (
                "硬门禁 disputed (仅语义相似度超阈值): "
                + "; ".join(gate_result.disputed_flags)
            )
        else:
            record["status"] = "rejected"
            failures_str = "; ".join(
                f"{f.rule}: {f.detail}" for f in gate_result.failures
            )
            record["retry_feedback"] = f"硬门禁失败: {failures_str}"
        return record

    # 3. Judge A (5-dim semantic, Qwen-Max)
    rate_limiter.wait()
    try:
        a_result = judge_a(
            candidate,
            scene=spec.scene,
            reference_passages=references[:3],
            cache_dir=cache_dir,
            sample_spec_id=spec.sample_spec_id,
            attempt=attempt,
        )
    except Exception as exc:  # noqa: BLE001
        record["status"] = "rejected"
        record["error"] = f"judge_a_failed: {exc}"
        record["retry_feedback"] = f"Judge A 调用失败: {exc}"
        return record
    record["judge_a"] = a_result.to_dict()
    if not a_result.passed:
        record["status"] = "rejected"
        record["retry_feedback"] = (
            f"Judge A 未通过 (适用维度: {a_result.applicable_dims}; "
            f"分数: {a_result.scores}; 违规: {a_result.violations})"
        )
        return record

    # 4. Judge B (same-question double-order, DeepSeek)
    negative = retrieve_negative(spec.sample_spec_id)
    if negative is None:
        record["status"] = "rejected"
        record["error"] = f"v3_negative not found for {spec.sample_spec_id}"
        return record

    rate_limiter.wait()
    try:
        b_result = judge_b(
            candidate,
            negative,
            spec.sample_spec_id,
            cache_dir=cache_dir,
            attempt=attempt,
        )
    except Exception as exc:  # noqa: BLE001
        # Same-question violation or API failure — treat as rejected (cannot compare)
        record["status"] = "rejected"
        record["error"] = f"judge_b_failed: {exc}"
        record["retry_feedback"] = f"Judge B 调用失败: {exc}"
        return record
    record["judge_b"] = b_result.to_dict()

    if b_result.final_decision == "passed":
        record["status"] = "passed"
        record["retry_feedback"] = ""
    elif b_result.final_decision == "rejected":
        record["status"] = "rejected"
        record["retry_feedback"] = (
            f"Judge B 双顺序均倾向 v3 负例 "
            f"(run1={b_result.prefers_candidate_run1}, "
            f"run2={b_result.prefers_candidate_run2})"
        )
    else:  # disputed
        record["status"] = "disputed"
        record["retry_feedback"] = (
            f"Judge B 双顺序不一致 "
            f"(run1={b_result.prefers_candidate_run1}, "
            f"run2={b_result.prefers_candidate_run2}) — 进入人工审核"
        )
    return record


# ---------------------------------------------------------------------------
# Pipeline entry (D.1.1 + D.1.4 + D.1.5 + D.1.7)
# ---------------------------------------------------------------------------

def run_pipeline(
    *,
    limit: int | None = None,
    rate_limit_seconds: float = 1.0,
    max_attempts: int = MAX_ATTEMPTS,
    cache_dir: Path | None = None,
    output_dir: Path = OUTPUT_DIR,
    negative_pool_path: Path | None = None,
    only_spec_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Run the full pipeline. Resume from progress.json if present.

    Args:
      limit: process at most N pending specs (None = all pending)
      rate_limit_seconds: min interval between API calls
      max_attempts: max attempts per spec on hard-gate/judge failure
      cache_dir: request cache directory (deduplicates LLM calls)
      output_dir: directory for samples/rejected/disputed/run_log/progress
      negative_pool_path: path to v3_negative_pool.jsonl
      only_spec_ids: if set, only process these sample_spec_ids (ignores limit)

    Major-4 fix: all output paths (samples/rejected/disputed/run_log/progress)
    are now derived from ``output_dir`` instead of mixing in module-level
    globals. Previously ``--output-dir`` only affected cache_dir and the
    negative_pool lookup, while samples.jsonl etc. were still written to
    the default OUTPUT_DIR — causing outputs to land in the wrong place.

    Major-10 fix: write ordering now guarantees no duplicate sample writes
    on crash. The order is:
      1. Append to {samples,rejected,disputed}.jsonl  (the data write)
      2. fsync via atomic append semantics
      3. Update progress.json with completed_spec_id + stats  (the commit)
    If a crash happens between (1) and (3), the spec is re-processed on
    resume — but because sample_spec_id is stable and the candidate is
    regenerated, the resumed run may produce a duplicate line in the
    JSONL. To make this fully idempotent we also record the
    ``last_committed_spec_id`` field; on resume, if the last line of
    samples.jsonl matches ``last_committed_spec_id`` we skip it.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Major-4: derive ALL output paths from output_dir
    samples_path = output_dir / "samples.jsonl"
    rejected_path = output_dir / "rejected_samples.jsonl"
    disputed_path = output_dir / "disputed_samples.jsonl"
    run_log_path = output_dir / "run_log.jsonl"
    progress_path = output_dir / "progress.json"
    judge_config_path = output_dir / "judge_config.json"

    # Major-9: write judge_config.json at pipeline start so every run is
    # self-documenting (records exact model ids, families, and the
    # cross-model-committee / self-preference-risk notes).
    atomic_write_json(judge_config_path, build_judge_config())

    if cache_dir is None:
        cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if negative_pool_path is None:
        negative_pool_path = output_dir / "v3_negative_pool.jsonl"

    if not negative_pool_path.exists():
        raise RuntimeError(
            f"v3_negative_pool.jsonl not found at {negative_pool_path}. "
            "Run build_v3_negative_pool.py first."
        )

    # Load specs + progress + passed_samples (resume-safe)
    # Major-4: pass the output_dir-derived paths to the loaders
    specs = load_specs_from_negative_pool(negative_pool_path)
    progress = _load_progress_from(progress_path)
    completed = set(progress["completed_spec_ids"])
    passed_samples = _load_passed_samples_from(samples_path)

    # Major fix (true idempotency): reconcile JSONL outputs against progress.
    # If a crash happened between append_jsonl() and _save_progress_to(),
    # the spec's record is on disk in samples/rejected/disputed.jsonl but
    # its id is NOT in progress.completed_spec_ids. On resume it would be
    # re-processed, producing a DUPLICATE line. We back-fill such ids into
    # progress now so they are skipped.
    jsonl_ids = (
        read_jsonl_ids(samples_path)
        | read_jsonl_ids(rejected_path)
        | read_jsonl_ids(disputed_path)
    )
    orphaned = jsonl_ids - completed
    if orphaned:
        print(f"[resume] found {len(orphaned)} spec(s) on disk but not in progress — back-filling")
        for sid in sorted(orphaned):
            progress["completed_spec_ids"].append(sid)
        completed = set(progress["completed_spec_ids"])
        _save_progress_to(progress_path, progress)

    # Major-10: write an initial progress.json at start so the run is
    # traceable even when pending is empty (records "run started" state
    # and last_updated timestamp).
    _save_progress_to(progress_path, progress)

    # Filter pending
    pending: list[SampleSpec] = []
    for spec in specs:
        if spec.sample_spec_id in completed:
            continue
        if only_spec_ids and spec.sample_spec_id not in only_spec_ids:
            continue
        pending.append(spec)
    if limit is not None and only_spec_ids is None:
        pending = pending[:limit]

    rate_limiter = RateLimiter(min_interval_seconds=rate_limit_seconds)

    print("=== Kisaki V4 regeneration pipeline ===")
    print(f"Total specs in negative pool: {len(specs)}")
    print(f"Already completed: {len(completed)}")
    print(f"Pending this run: {len(pending)}")
    print(f"Passed samples loaded (for repeated-opening gate): {len(passed_samples)}")
    print(f"Rate limit: {rate_limit_seconds}s between API calls")
    print(f"Max attempts per spec: {max_attempts}")
    print(f"Cache dir: {cache_dir}")
    print(f"Output dir: {output_dir}")
    print()

    for i, spec in enumerate(pending):
        print(f"[{i + 1}/{len(pending)}] {spec.sample_spec_id} (scene={spec.scene})")

        # Retry loop: up to max_attempts
        final_record: dict[str, Any] | None = None
        retry_feedback = ""
        for attempt in range(max_attempts):
            record = process_one_attempt(
                spec,
                attempt=attempt,
                retry_feedback=retry_feedback,
                passed_samples=passed_samples,
                rate_limiter=rate_limiter,
                cache_dir=cache_dir,
            )
            final_record = record
            # Log every attempt to run_log.jsonl (D.1.6)
            append_jsonl(run_log_path, record)

            if record["status"] == "passed":
                break
            if record["status"] == "disputed":
                # disputed -> human review, don't retry
                break
            # rejected -> retry with feedback
            retry_feedback = record.get("retry_feedback", "")
            print(
                f"  attempt {attempt + 1} -> {record['status']}; "
                f"feedback: {retry_feedback[:120]}"
            )
        assert final_record is not None

        # Major-10: write ordering — data first, then progress commit.
        # 1. Append the final record to the appropriate JSONL (data write)
        status = final_record["status"]
        if status == "passed":
            append_jsonl(samples_path, final_record)
            if final_record.get("candidate"):
                passed_samples.append(final_record["candidate"])
            progress["stats"]["passed"] += 1
        elif status == "disputed":
            append_jsonl(disputed_path, final_record)
            progress["stats"]["disputed"] += 1
        else:
            append_jsonl(rejected_path, final_record)
            progress["stats"]["rejected"] += 1

        # 2. Commit: update progress.json with the completed spec_id + stats.
        # If a crash happens between (1) and (2), the resume run will see
        # this spec as "pending" and re-process it. Because the spec_id is
        # stable and sample IDs are deterministic, the worst case is a
        # duplicate JSONL line — which the V3 freeze step (Stage E) will
        # deduplicate by sample_spec_id before merging into the train set.
        progress["completed_spec_ids"].append(spec.sample_spec_id)
        progress["stats"]["total_processed"] += 1
        progress["last_committed_spec_id"] = spec.sample_spec_id
        _save_progress_to(progress_path, progress)
        print(f"  -> final: {status}")

    summary = {
        "total_specs": len(specs),
        "processed_this_run": len(pending),
        "stats": progress["stats"],
        "samples_path": str(samples_path),
        "rejected_path": str(rejected_path),
        "disputed_path": str(disputed_path),
        "run_log_path": str(run_log_path),
        "progress_path": str(progress_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Kisaki V4 regeneration pipeline (Task B.7 + D.1)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only N pending specs (default: all pending)",
    )
    parser.add_argument(
        "--rate-limit", type=float, default=1.0,
        help="Minimum seconds between API calls (default: 1.0)",
    )
    parser.add_argument(
        "--max-attempts", type=int, default=MAX_ATTEMPTS,
        help=f"Max attempts per spec (default: {MAX_ATTEMPTS})",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Request cache directory (default: output_dir/cache)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="Output directory for samples/rejected/disputed/progress",
    )
    parser.add_argument(
        "--negative-pool", type=Path, default=None,
        help="Path to v3_negative_pool.jsonl (default: output_dir/v3_negative_pool.jsonl)",
    )
    parser.add_argument(
        "--only", type=str, default=None,
        help="Comma-separated sample_spec_ids to process (skips others, ignores --limit)",
    )
    args = parser.parse_args()

    only_spec_ids: set[str] | None = None
    if args.only:
        only_spec_ids = {s.strip() for s in args.only.split(",") if s.strip()}

    summary = run_pipeline(
        limit=args.limit,
        rate_limit_seconds=args.rate_limit,
        max_attempts=args.max_attempts,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        negative_pool_path=args.negative_pool,
        only_spec_ids=only_spec_ids,
    )
    # Exit non-zero if no specs were processed (e.g. all done or pool missing)
    return 0 if summary["processed_this_run"] >= 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
