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
    get_few_shots_by_ids,
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


def _recompute_stats_from_jsonl(
    samples_path: Path,
    rejected_path: Path,
    disputed_path: Path,
) -> dict[str, int]:
    """Rebuild progress.stats by counting lines in the three JSONL files.

    Other-fix: after a crash-resume back-fill, the in-memory stats could
    be stale (e.g. stats.passed=5 but samples.jsonl has 6 lines because
    a record was appended but progress wasn't updated). Counting from
    disk guarantees the displayed numbers match the actual outputs.
    """
    def _count(path: Path) -> int:
        if not path.exists():
            return 0
        return sum(
            1 for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    passed = _count(samples_path)
    rejected = _count(rejected_path)
    disputed = _count(disputed_path)
    return {
        "passed": passed,
        "rejected": rejected,
        "disputed": disputed,
        "total_processed": passed + rejected + disputed,
    }


# ---------------------------------------------------------------------------
# Per-sample processing (D.1.2 + D.1.3 + D.1.6)
# ---------------------------------------------------------------------------

def _build_reference_texts(scene: str) -> list[str]:
    """Pull assistant texts from few-shot pool for copy detection.

    Legacy fallback: retrieves by scene without human_dialogue scoring.
    Only used when a candidate has no ``reference_ids`` field (e.g. old
    cached candidates). New candidates always go through
    ``_build_reference_texts_from_candidate`` so Hard Gate and Judge A
    compare against the *exact* passages the Generator was anchored on.
    """
    refs: list[str] = []
    for fs in retrieve_few_shots(scene, k=3):
        for msg in fs.get("conversations", []):
            if msg.get("from") == "assistant":
                refs.append(msg.get("value", ""))
    return refs


def _build_reference_texts_from_candidate(candidate: dict[str, Any]) -> list[str]:
    """Pull assistant texts from the few-shot pool using the candidate's
    actual ``reference_ids`` — guaranteeing Generator / Hard Gate / Judge A
    all see the same evidence.

    Consistency fix: previously ``_build_reference_texts(spec.scene)`` ran a
    fresh ``retrieve_few_shots(scene, k=3)`` without the human_dialogue
    scoring and exclude_ids that the Generator used, so Hard Gate could
    check copy against a *different* set of passages than the ones that
    actually shaped the candidate. This variant reads by ID instead.

    Other-fix: ``strict=True`` is now passed so a missing reference id
    aborts the run instead of silently running copy detection against
    a degraded (possibly empty) reference set. Old cached candidates
    without ``reference_ids`` are rejected — formal Pilot must not reuse
    such caches because their few-shot evidence cannot be reconstructed.
    """
    reference_ids: list[str] = candidate.get("reference_ids") or []
    if not reference_ids:
        # Other-fix: do NOT silently fall back to scene retrieval. A
        # candidate without reference_ids is a stale cache entry whose
        # few-shot evidence is unrecoverable; the formal pipeline must
        # regenerate it.
        raise RuntimeError(
            "candidate has no reference_ids — refusing to run Hard Gate "
            "on a stale cache entry whose few-shot evidence cannot be "
            "reconstructed. Delete the cache for this sample_spec_id and "
            "rerun so the Generator emits a fresh candidate with "
            "reference_ids."
        )
    refs: list[str] = []
    for fs in get_few_shots_by_ids(reference_ids, strict=True):
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
    # Consistency fix: read references by the candidate's actual
    # reference_ids so Hard Gate and Judge A see the same few-shot
    # passages the Generator was anchored on (not a fresh retrieve).
    references = _build_reference_texts_from_candidate(candidate)
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
            rate_limiter=rate_limiter,
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

def _get_git_info() -> dict[str, str]:
    """Return current git commit + branch (best-effort, never raises)."""
    import subprocess
    info = {"commit": "unknown", "branch": "unknown", "dirty": "unknown"}
    try:
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            text=True, timeout=5, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001
        pass
    try:
        info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=PROJECT_ROOT,
            text=True, timeout=5, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001
        pass
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
            text=True, timeout=5, stderr=subprocess.DEVNULL,
        ).strip()
        info["dirty"] = "true" if status else "false"
    except Exception:  # noqa: BLE001
        pass
    return info


def _sha256_file(path: Path) -> str:
    """SHA256 of a file's bytes (for input provenance in run_manifest)."""
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_run_manifest(
    *,
    output_dir: Path,
    negative_pool_path: Path,
    quota_plan_path: Path | None,
    only_spec_ids: set[str] | None,
    limit: int | None,
    max_attempts: int,
    rate_limit_seconds: float,
    pending: list[SampleSpec],
) -> dict[str, Any]:
    """Build a run_manifest.json recording commit, args, input hashes, and
    the selected sample_spec_ids for this run.

    Provenance fix: the smoke run had no manifest, so it was impossible to
    tell after the fact which max_attempts was used, which git commit the
    code came from, or whether the input pool had been modified. This
    manifest captures all of that so a calibration run can be audited and
    reproduced.
    """
    from generate_kisaki_llm_v4 import FEW_SHOT_POOL_PATH

    input_files: dict[str, Any] = {
        "negative_pool": {
            "path": str(negative_pool_path),
            "sha256": _sha256_file(negative_pool_path) if negative_pool_path.exists() else "",
        },
        "few_shot_pool": {
            "path": str(FEW_SHOT_POOL_PATH),
            "sha256": _sha256_file(FEW_SHOT_POOL_PATH) if FEW_SHOT_POOL_PATH.exists() else "",
        },
    }
    if quota_plan_path is not None and quota_plan_path.exists():
        input_files["quota_plan"] = {
            "path": str(quota_plan_path),
            "sha256": _sha256_file(quota_plan_path),
        }

    return {
        "manifest_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git": _get_git_info(),
        "command_args": {
            "limit": limit,
            "max_attempts": max_attempts,
            "rate_limit_seconds": rate_limit_seconds,
            "output_dir": str(output_dir),
            "negative_pool_path": str(negative_pool_path),
            "quota_plan_path": str(quota_plan_path) if quota_plan_path else None,
            "only_spec_ids": sorted(only_spec_ids) if only_spec_ids else None,
        },
        "input_files": input_files,
        "selected_sample_ids": [s.sample_spec_id for s in pending],
        "selected_count": len(pending),
        "note": (
            "Models and similarity backend are recorded separately in "
            "judge_config.json; this manifest captures git/args/inputs/"
            "selected_ids for run-level provenance."
        ),
    }


def _build_run_summary(
    *,
    run_log_path: Path,
    samples_path: Path,
    rejected_path: Path,
    disputed_path: Path,
    stats: dict[str, int],
    run_started_at: str,
    total_specs: int,
    processed_this_run: int,
    initial_manifest_path: Path,
    judge_config_path: Path,
    expected_processed: int | None,
) -> dict[str, Any]:
    """Build run_summary.json with final stats + Judge B bias diagnostics.

    Major-6 fix: scans ``run_log.jsonl`` to aggregate the Judge B
    telemetry fields (``first_position_a_run1/2``, ``is_tie_run1/2``,
    ``low_confidence_run1/2``, ``score_contradiction_run1/2``,
    ``final_decision``) into rates. These rates tell the operator
    whether Judge B needs prompt revision before the holdout run:

      - ``first_position_preference_rate``: high (>0.7) => strong
        position bias, prompt must be revised.
      - ``tie_rate``: high (>0.3) => judge is dodging; consider
        sharpening the prompt or lowering the tie dim-gap threshold.
      - ``judge_b_parse_failure_rate``: high (>0.1) => response format
        is unstable; consider stronger JSON mode enforcement.
      - ``low_confidence_rate``: high (>0.3) => judge is uncertain;
        calibration may need a different model or clearer rubric.
      - ``score_contradiction_rate``: high (>0.1) => judge is
        inconsistent between scores and preferred; prompt rubric needs
        tightening.

    Only samples that REACHED Judge B are counted in the denominator
    (samples rejected at Hard Gate / Judge A do not have Judge B
    results and would skew the rates if included).

    Major-1 fix: ``acceptance_check`` block records the smoke gate
    criteria (processed count, scene coverage, parse failures, BGE
    authoritativeness, duplicate IDs, disputed rate). ``run_pipeline``
    uses ``all_passed`` to decide the process exit code so a smoke run
    that produces 12 rejected/disputed samples no longer exits 0.
    """
    # Scan run_log.jsonl for the final attempt of each sample_spec_id
    # that reached Judge B.
    final_by_spec: dict[str, dict[str, Any]] = {}
    if run_log_path.exists():
        for line in run_log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = rec.get("sample_spec_id")
            if not sid:
                continue
            # Keep the latest attempt per spec (final decision)
            final_by_spec[sid] = rec

    judge_b_reached = [
        rec for rec in final_by_spec.values()
        if rec.get("judge_b") is not None
    ]
    jb_count = len(judge_b_reached)

    def _rate(pred) -> float:
        if jb_count == 0:
            return 0.0
        return sum(1 for r in judge_b_reached if pred(r.get("judge_b", {}))) / jb_count

    # Minor-4 fix: first_position_preference_rate is now computed per-run
    # (run1_rate + run2_rate) / 2, NOT as "either run preferred A". The
    # old `r1 or r2` rule inflated the rate: a sample where only one run
    # preferred A counted as 1.0, masking asymmetric bias. Per-run rates
    # expose whether the bias is stable across both orderings or only
    # affects one (e.g. only when candidate is at A).
    def _first_pos_a_run1(jb: dict[str, Any]) -> bool:
        return bool(jb.get("first_position_a_run1", False))

    def _first_pos_a_run2(jb: dict[str, Any]) -> bool:
        return bool(jb.get("first_position_a_run2", False))

    # parse failure: final_decision is disputed AND evidence contains a
    # parse-error marker. We approximate by checking if either run's
    # evidence starts with a known parse-failure prefix.
    parse_failures = 0
    for r in judge_b_reached:
        jb = r.get("judge_b", {})
        ev1 = str(jb.get("evidence_run1", ""))
        ev2 = str(jb.get("evidence_run2", ""))
        if any(ev1.startswith(p) for p in (
            "json_parse_error", "invalid_preferred_value",
            "missing_confidence", "invalid_confidence_type",
            "confidence_out_of_range", "scores_not_object",
            "scores_A_B_missing_or_not_object", "scores_missing_dim",
            "scores_dim_not_numeric", "scores_dim_out_of_range",
        )):
            parse_failures += 1
        elif any(ev2.startswith(p) for p in (
            "json_parse_error", "invalid_preferred_value",
            "missing_confidence", "invalid_confidence_type",
            "confidence_out_of_range", "scores_not_object",
            "scores_A_B_missing_or_not_object", "scores_missing_dim",
            "scores_dim_not_numeric", "scores_dim_out_of_range",
        )):
            parse_failures += 1

    # Count final decisions among Judge-B-reached samples
    jb_passed = sum(1 for r in judge_b_reached if r.get("judge_b", {}).get("final_decision") == "passed")
    jb_rejected = sum(1 for r in judge_b_reached if r.get("judge_b", {}).get("final_decision") == "rejected")
    jb_disputed = sum(1 for r in judge_b_reached if r.get("judge_b", {}).get("final_decision") == "disputed")

    # Major-1: acceptance_check — smoke gate criteria.
    # These are objective, machine-checkable pass conditions. The smoke
    # run must satisfy ALL of them before the operator can move on to
    # the 30-sample calibration. Calibration runs may relax the
    # disputed-rate threshold (calibration is where the threshold is
    # *tuned*), but the other criteria apply universally.
    #
    # processed_count: unique sample_spec_ids that produced a final
    #   record in any of the three JSONL files. Uses final_by_spec
    #   (dedup by spec_id) so retries do not inflate the count.
    # scenes_covered: distinct scene values among those specs.
    # duplicate_ids: True if any spec_id appears more than once across
    #   the three JSONL files (indicates a resume bug or a crash mid-
    #   write that the back-fill did not catch).
    # bge_authoritative: re-reads judge_config.json and checks the
    #   similarity_backend fields. A smoke run on fallback backend
    #   cannot pass acceptance (results are non-authoritative).
    processed_count = len(final_by_spec)
    scenes_covered = sorted({
        rec.get("scene", "")
        for rec in final_by_spec.values()
        if rec.get("scene")
    })
    scenes_covered_count = len(scenes_covered)

    # Duplicate-id detection across the three JSONL files.
    seen_ids: list[str] = []
    for p in (samples_path, rejected_path, disputed_path):
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = rec.get("sample_spec_id")
            if sid:
                seen_ids.append(sid)
    id_counts: dict[str, int] = {}
    for sid in seen_ids:
        id_counts[sid] = id_counts.get(sid, 0) + 1
    duplicate_ids = sorted([sid for sid, c in id_counts.items() if c > 1])
    no_duplicate_ids = len(duplicate_ids) == 0

    # BGE authoritativeness from judge_config.json (written at start).
    bge_authoritative = False
    bge_backend = ""
    if judge_config_path.exists():
        try:
            jcfg = json.loads(judge_config_path.read_text(encoding="utf-8"))
            sim = jcfg.get("similarity_backend", {}) or {}
            bge_backend = sim.get("backend", "")
            bge_authoritative = (
                sim.get("backend") == "bge_embedding"
                and sim.get("authoritative") == "true"
            )
        except (json.JSONDecodeError, OSError):
            bge_authoritative = False

    # disputed_rate: among samples that reached Judge B, the fraction
    # routed to disputed. High disputed rate means the judges cannot
    # agree and calibration cannot produce reliable agreement data.
    disputed_rate = (jb_disputed / jb_count) if jb_count else 0.0

    # Major-1 fix: judge_b_reached_rate prevents a false-green smoke
    # gate. If all 12 samples are rejected at Hard Gate / Judge A,
    # jb_count=0 and disputed_rate=0/0=0.0, which would satisfy
    # disputed_rate_ok without Judge B ever running. Such a run proves
    # nothing about the pairwise judge — it only proves the generator
    # or Hard Gate is broken. Require at least 50% of processed samples
    # to reach Judge B so the disputed_rate and position-bias telemetry
    # are statistically meaningful.
    judge_b_reached_rate = (jb_count / processed_count) if processed_count else 0.0
    judge_b_reached_ok = judge_b_reached_rate >= 0.5

    processed_exactly = (
        expected_processed is not None
        and processed_count == expected_processed
    )
    processed_at_least_one = processed_count > 0
    scenes_covered_ok = scenes_covered_count >= 6
    parse_failures_zero = parse_failures == 0
    disputed_rate_ok = disputed_rate <= 0.25

    acceptance_check = {
        "processed_count": processed_count,
        "expected_count": expected_processed,
        "processed_exactly": processed_exactly,
        "processed_at_least_one": processed_at_least_one,
        "scenes_covered": scenes_covered,
        "scenes_covered_count": scenes_covered_count,
        "scenes_covered_ok": scenes_covered_ok,
        "parse_failure_count": parse_failures,
        "parse_failures_zero": parse_failures_zero,
        "bge_backend": bge_backend,
        "bge_authoritative": bge_authoritative,
        "duplicate_ids": duplicate_ids,
        "no_duplicate_ids": no_duplicate_ids,
        "judge_b_reached_count": jb_count,
        "judge_b_reached_rate": round(judge_b_reached_rate, 4),
        "judge_b_reached_ok": judge_b_reached_ok,
        "disputed_rate": round(disputed_rate, 4),
        "disputed_rate_ok": disputed_rate_ok,
        # Smoke gate: ALL conditions must hold. processed_exactly is
        # only checked when expected_processed is known (quota plan or
        # --limit); otherwise processed_at_least_one is the floor.
        # Major-1: judge_b_reached_ok is required so a run where all
        # samples died at Hard Gate / Judge A cannot pass the gate.
        "all_passed": (
            (processed_exactly if expected_processed is not None else processed_at_least_one)
            and scenes_covered_ok
            and parse_failures_zero
            and bge_authoritative
            and no_duplicate_ids
            and judge_b_reached_ok
            and disputed_rate_ok
        ),
        "smoke_gate_criteria": (
            "processed_exactly (when expected known); scenes_covered >= 6; "
            "parse_failures == 0; bge_authoritative == true; "
            "no_duplicate_ids; judge_b_reached_rate >= 0.5; "
            "disputed_rate <= 0.25"
        ),
    }

    return {
        "summary_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_started_at": run_started_at,
        "run_ended_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_specs_in_pool": total_specs,
        "processed_this_run": processed_this_run,
        "final_stats": stats,
        "output_files": {
            "samples": str(samples_path),
            "rejected": str(rejected_path),
            "disputed": str(disputed_path),
            "run_log": str(run_log_path),
            "initial_manifest": str(initial_manifest_path),
        },
        "judge_b_diagnostics": {
            "samples_reached_judge_b": jb_count,
            "passed": jb_passed,
            "rejected": jb_rejected,
            "disputed": jb_disputed,
            # Minor-4: per-run rates expose asymmetric bias. The averaged
            # rate is kept for backward compatibility but the per-run
            # numbers are authoritative for bias diagnosis.
            "first_position_preference_rate_run1": round(_rate(_first_pos_a_run1), 4),
            "first_position_preference_rate_run2": round(_rate(_first_pos_a_run2), 4),
            "first_position_preference_rate": round(
                (_rate(_first_pos_a_run1) + _rate(_first_pos_a_run2)) / 2.0, 4
            ),
            "tie_rate": round(_rate(lambda jb: jb.get("is_tie_run1") or jb.get("is_tie_run2")), 4),
            "low_confidence_rate": round(_rate(lambda jb: jb.get("low_confidence_run1") or jb.get("low_confidence_run2")), 4),
            "score_contradiction_rate": round(_rate(lambda jb: jb.get("score_contradiction_run1") or jb.get("score_contradiction_run2")), 4),
            "parse_failure_rate": round(parse_failures / jb_count if jb_count else 0.0, 4),
            "parse_failure_count": parse_failures,
        },
        "acceptance_check": acceptance_check,
        "interpretation_notes": (
            "first_position_preference_rate > 0.7 => strong position bias; "
            "revise Judge B prompt. tie_rate > 0.3 => judge dodging; "
            "sharpen rubric. parse_failure_rate > 0.1 => JSON mode unstable. "
            "low_confidence_rate > 0.3 => judge uncertain; consider model "
            "swap. score_contradiction_rate > 0.1 => scores vs preferred "
            "inconsistent; tighten rubric. acceptance_check.all_passed "
            "must be true before proceeding to 30-sample calibration."
        ),
    }


def run_pipeline(
    *,
    limit: int | None = None,
    rate_limit_seconds: float = 1.0,
    max_attempts: int = MAX_ATTEMPTS,
    cache_dir: Path | None = None,
    output_dir: Path = OUTPUT_DIR,
    negative_pool_path: Path | None = None,
    only_spec_ids: set[str] | None = None,
    quota_plan_path: Path | None = None,
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
      quota_plan_path: if set, read quota_plan.json and restrict pending to
        the plan's selected sample_spec_ids (stratified sampling across
        scenes). Combines with --limit (limit further caps the count) and
        --only (only_spec_ids takes precedence over the plan).

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
    # Major-3 fix: build_judge_config(strict_similarity=True) now RAISES
    # RuntimeError if the BGE model cannot be loaded in strict mode
    # (the default for Pilot/formal runs). This aborts the pipeline
    # before any external API call, so no DeepSeek/Qwen budget is spent
    # on a run whose copy-detection results would be non-authoritative.
    atomic_write_json(judge_config_path, build_judge_config(strict_similarity=True))

    # Major-3 fix: explicit post-write authoritative check. Even if
    # build_judge_config somehow succeeded (e.g. BGE loaded but returned
    # an unexpected backend identifier), we re-read the written config
    # and refuse to proceed unless backend==bge_embedding and
    # authoritative=="true". This is the formal-run gate.
    written_cfg = json.loads(judge_config_path.read_text(encoding="utf-8"))
    sim_info = written_cfg.get("similarity_backend", {})
    if sim_info.get("backend") != "bge_embedding" or sim_info.get("authoritative") != "true":
        raise RuntimeError(
            "Major-3 fail-fast: similarity backend is not authoritative. "
            f"backend={sim_info.get('backend')!r}, "
            f"authoritative={sim_info.get('authoritative')!r}. "
            "Pilot/formal runs require the BGE embedding model. "
            "Set KISAKI_SIMILARITY_BACKEND=fallback ONLY for local dev "
            "and call build_judge_config(strict_similarity=False) explicitly."
        )

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
        # Other-fix: recompute stats from the three JSONL files instead
        # of trusting the in-memory counters. A crash mid-run could leave
        # progress.stats stale (e.g. stats says passed=5 but samples.jsonl
        # has 6 lines). Re-counting keeps the displayed numbers accurate.
        progress["stats"] = _recompute_stats_from_jsonl(
            samples_path, rejected_path, disputed_path,
        )
        completed = set(progress["completed_spec_ids"])
        _save_progress_to(progress_path, progress)

    # Major-10: write an initial progress.json at start so the run is
    # traceable even when pending is empty (records "run started" state
    # and last_updated timestamp).
    _save_progress_to(progress_path, progress)

    # Load quota plan (stratified scene sampling) if provided.
    # Major-5 fix: the plan now carries ``ordered_sample_spec_ids``
    # (interleaved across scenes by round-robin) so the run processes
    # samples in a truly stratified order rather than following the
    # negative-pool file order (which clusters same-scene samples).
    # When a plan is present, ``--limit`` is applied to the ordered
    # list (preserving stratification) rather than the raw pool order.
    quota_plan_ordered_ids: list[str] | None = None
    quota_plan_ids: set[str] | None = None
    if quota_plan_path is not None:
        if not quota_plan_path.exists():
            raise RuntimeError(
                f"quota_plan.json not found at {quota_plan_path}. "
                "Run build_kisaki_v4_quota_plan.py first."
            )
        plan = json.loads(quota_plan_path.read_text(encoding="utf-8"))
        quota_plan_ids = set()
        for scene_block in plan.get("scenes", {}).values():
            for sid in scene_block.get("sample_spec_ids", []):
                quota_plan_ids.add(str(sid))
        # Major-5: prefer the interleaved order if the plan provides it;
        # otherwise fall back to the unordered set (legacy plans).
        quota_plan_ordered_ids = plan.get("ordered_sample_spec_ids")
        if quota_plan_ordered_ids:
            quota_plan_ordered_ids = [str(s) for s in quota_plan_ordered_ids]
        print(f"[quota_plan] loaded {len(quota_plan_ids)} spec ids from {quota_plan_path}")

    # Filter pending.
    # Major-5: when a quota plan with ``ordered_sample_spec_ids`` is
    # present, iterate that list (not the raw pool) so the run order is
    # the interleaved plan order. ``--limit`` then caps the stratified
    # list instead of breaking stratification by slicing the pool.
    pending: list[SampleSpec] = []
    if quota_plan_ordered_ids is not None:
        specs_by_id = {s.sample_spec_id: s for s in specs}
        for sid in quota_plan_ordered_ids:
            if sid in completed:
                continue
            if only_spec_ids and sid not in only_spec_ids:
                continue
            spec = specs_by_id.get(sid)
            if spec is not None:
                pending.append(spec)
        # --limit caps the stratified list (still preserves scene order)
        if limit is not None:
            pending = pending[:limit]
    else:
        for spec in specs:
            if spec.sample_spec_id in completed:
                continue
            # only_spec_ids takes precedence over quota_plan (explicit override)
            if only_spec_ids and spec.sample_spec_id not in only_spec_ids:
                continue
            if quota_plan_ids is not None and spec.sample_spec_id not in quota_plan_ids:
                continue
            pending.append(spec)
        if limit is not None and only_spec_ids is None:
            pending = pending[:limit]

    # Provenance fix: write run_manifest BEFORE processing so the run is
    # auditable even if it crashes mid-way. Captures git commit, command
    # args, input file hashes, and the selected sample_spec_ids.
    # Major-6 fix: ``run_manifest.json`` is the IMMUTABLE initial record.
    # On resume (run_manifest.json already exists), we instead append a
    # ``run_manifest_resume_<timestamp>.json`` so the original run's
    # provenance is never overwritten. The resume manifest records only
    # what changed (pending this resume, timestamp, resume_reason).
    manifest = _build_run_manifest(
        output_dir=output_dir,
        negative_pool_path=negative_pool_path,
        quota_plan_path=quota_plan_path,
        only_spec_ids=only_spec_ids,
        limit=limit,
        max_attempts=max_attempts,
        rate_limit_seconds=rate_limit_seconds,
        pending=pending,
    )
    initial_manifest_path = output_dir / "run_manifest.json"
    if initial_manifest_path.exists():
        resume_stamp = time.strftime("%Y%m%dT%H%M%S")
        resume_path = output_dir / f"run_manifest_resume_{resume_stamp}.json"
        atomic_write_json(resume_path, {
            **manifest,
            "manifest_kind": "resume",
            "resumed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "initial_manifest_path": str(initial_manifest_path),
        })
        print(f"[resume] initial run_manifest.json preserved; wrote {resume_path.name}")
    else:
        atomic_write_json(initial_manifest_path, manifest)

    rate_limiter = RateLimiter(min_interval_seconds=rate_limit_seconds)
    run_started_at = time.strftime("%Y-%m-%dT%H:%M:%S")

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

    # Major-6 fix: write run_summary.json with run-level aggregates.
    # Unlike run_manifest.json (immutable, written before processing),
    # run_summary.json is written at the END and captures final stats,
    # timing, and Judge B bias diagnostics (first-position preference
    # rate, tie rate, parse failure rate, low-confidence rate). These
    # metrics feed directly into the calibration decision: if
    # first_position_preference_rate is high, Judge B prompt needs
    # revision before the holdout run.
    #
    # Major-2 fix: expected_processed must reflect the TOTAL expected
    # count from the initial run, not the current resume's pending
    # count. Using len(pending) here would make the acceptance gate
    # fail on every resume: e.g. initial run had 12 specs, processed 8,
    # crashed; resume has pending=4, so expected_processed=4 but
    # processed_count=12 (cumulative) -> processed_exactly=False forever.
    # We read selected_count from the immutable initial_manifest.json
    # so the expected total stays constant across resumes.
    expected_processed_total: int | None = None
    if initial_manifest_path.exists():
        try:
            initial_manifest = json.loads(
                initial_manifest_path.read_text(encoding="utf-8")
            )
            expected_processed_total = initial_manifest.get("selected_count")
        except (json.JSONDecodeError, OSError):
            pass
    if expected_processed_total is None:
        # Fallback: no manifest (e.g. legacy run). Use this run's pending
        # count so the gate still has a target, but log a warning.
        expected_processed_total = len(pending) if pending else None

    run_summary = _build_run_summary(
        run_log_path=run_log_path,
        samples_path=samples_path,
        rejected_path=rejected_path,
        disputed_path=disputed_path,
        stats=progress["stats"],
        run_started_at=run_started_at,
        total_specs=len(specs),
        processed_this_run=len(pending),
        initial_manifest_path=initial_manifest_path,
        judge_config_path=judge_config_path,
        expected_processed=expected_processed_total,
    )
    atomic_write_json(output_dir / "run_summary.json", run_summary)

    summary = {
        "total_specs": len(specs),
        "processed_this_run": len(pending),
        "stats": progress["stats"],
        "samples_path": str(samples_path),
        "rejected_path": str(rejected_path),
        "disputed_path": str(disputed_path),
        "run_log_path": str(run_log_path),
        "progress_path": str(progress_path),
        "run_summary_path": str(output_dir / "run_summary.json"),
        "acceptance_check": run_summary.get("acceptance_check", {}),
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
    parser.add_argument(
        "--quota-plan", type=Path, default=None,
        help="Path to quota_plan.json (stratified scene sampling). "
             "Restricts pending to the plan's selected sample_spec_ids.",
    )
    parser.add_argument(
        "--strict-acceptance", action="store_true",
        help="Exit non-zero if acceptance_check.all_passed is false. "
             "Use this for smoke runs where the run must pass the gate "
             "(processed count, scene coverage, parse failures, BGE "
             "authoritativeness, no duplicate IDs, disputed_rate <= 0.25) "
             "before the operator can proceed to calibration.",
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
        quota_plan_path=args.quota_plan,
    )
    # Major-1 fix: exit code reflects whether the run actually processed
    # samples (not the trivially-true `processed_this_run >= 0`).
    # --strict-acceptance further requires acceptance_check.all_passed,
    # so a smoke run that emits 12 rejected/disputed samples exits
    # non-zero and cannot be mistaken for a green smoke gate.
    if summary["processed_this_run"] == 0:
        return 1
    if args.strict_acceptance:
        acceptance = summary.get("acceptance_check", {}) or {}
        if not acceptance.get("all_passed", False):
            print(
                "[strict-acceptance] acceptance_check.all_passed is false; "
                f"failures: processed_count={acceptance.get('processed_count')}, "
                f"expected_count={acceptance.get('expected_count')}, "
                f"scenes_covered={acceptance.get('scenes_covered_count')}, "
                f"parse_failures={acceptance.get('parse_failure_count')}, "
                f"bge_authoritative={acceptance.get('bge_authoritative')}, "
                f"duplicate_ids={acceptance.get('duplicate_ids')}, "
                f"disputed_rate={acceptance.get('disputed_rate')}"
            )
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
