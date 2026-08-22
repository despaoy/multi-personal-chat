"""Build a deterministic R1V4 paired blind-review package."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


REQUIRED_PROVENANCE_FIELDS = (
    "dataset_sha256",
    "dataset_id",
    "dataset_status",
    "dataset_role",
    "prompt_policy_version",
    "generation",
)
REVIEW_DIMENSIONS = (
    "character_consistency",
    "factual_correctness",
    "context_memory",
    "situational_decision",
    "safety",
)


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"report must be a JSON object: {path}")
    return data


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_report(report: dict[str, Any], label: str) -> None:
    if report.get("schema_version") != 3:
        raise ValueError(f"{label}: benchmark schema_version must be 3")
    if report.get("mock") is not False:
        raise ValueError(f"{label}: blind review requires mock=false")
    if not isinstance(report.get("model"), str) or not report["model"].strip():
        raise ValueError(f"{label}: model identity is missing")
    provenance = report.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"{label}: provenance is missing")
    for field in REQUIRED_PROVENANCE_FIELDS:
        if provenance.get(field) in (None, "", {}):
            raise ValueError(f"{label}: provenance.{field} is missing")
    samples = report.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{label}: samples must be a non-empty list")


def _turns(row: dict[str, Any], label: str) -> tuple[list[str], list[str]]:
    turns = row.get("turns")
    responses = row.get("turn_responses")
    if not isinstance(turns, list) or not all(isinstance(item, str) and item for item in turns):
        raise ValueError(f"{label}: turns must contain non-empty strings")
    if not isinstance(responses, list) or len(responses) != len(turns):
        raise ValueError(f"{label}: turn_responses must align one-to-one with turns")
    if not all(isinstance(item, str) and item for item in responses):
        raise ValueError(f"{label}: turn_responses must contain non-empty strings")
    if row.get("response") != responses[-1]:
        raise ValueError(f"{label}: response must equal the final turn response")
    if row.get("error"):
        raise ValueError(f"{label}: generated sample contains an error")
    return list(turns), list(responses)


def _paired_rows(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    baseline_samples = baseline["samples"]
    candidate_samples = candidate["samples"]
    baseline_ids = [row.get("id") for row in baseline_samples]
    candidate_ids = [row.get("id") for row in candidate_samples]
    if len(baseline_ids) != len(set(baseline_ids)):
        raise ValueError("baseline: duplicate sample IDs")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate: duplicate sample IDs")
    if baseline_ids != candidate_ids:
        raise ValueError("sample IDs or request order do not match")

    pairs = []
    contract_fields = (
        "category",
        "cluster_id",
        "interlocutor",
        "prompt",
        "turns",
        "turn_rubrics",
        "expected_behavior",
        "rubric",
    )
    for index, (left, right) in enumerate(zip(baseline_samples, candidate_samples, strict=True)):
        sample_id = left.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"sample at index {index} has no ID")
        for field in contract_fields:
            if left.get(field) != right.get(field):
                raise ValueError(f"{sample_id}: paired field differs: {field}")
        _turns(left, f"baseline/{sample_id}")
        _turns(right, f"candidate/{sample_id}")
        pairs.append((left, right))
    return pairs


def _blank_scorecard() -> dict[str, None]:
    return {dimension: None for dimension in REVIEW_DIMENSIONS}


def _markdown(review: dict[str, Any]) -> str:
    lines = [
        "# 月社妃 E1 匿名语义审核",
        "",
        "请只阅读本文件，不要打开 `blind_key.json`。先独立完成全部判断，再锁定结果。",
        "",
        "每题填写：总体选择 `A`、`B`、`tie`（相当）或 `invalid`（题目无效）。五项评分使用 0-2：0=明显不满足，1=部分满足，2=充分满足；不适用写 `N/A`。",
        "",
        "重点检查人物一致性、事实正确性、多轮记忆、情景决策和安全性。回答更长不等于更好。",
        "",
    ]
    for index, sample in enumerate(review["samples"], start=1):
        lines.extend(
            [
                f"## {index}. {sample['id']} [{sample['category']}]",
                "",
                f"对话对象：{sample.get('interlocutor') or '未指定'}",
                "",
            ]
        )
        for turn_index, prompt in enumerate(sample["turns"]):
            lines.extend(
                [
                    f"### 第 {turn_index + 1} 轮",
                    "",
                    f"用户：{prompt}",
                    "",
                    f"A：{sample['candidate_A']['turn_responses'][turn_index]}",
                    "",
                    f"B：{sample['candidate_B']['turn_responses'][turn_index]}",
                    "",
                ]
            )
        expected = sample.get("expected_behavior") or {}
        required = "；".join(expected.get("required_behaviors", [])) or "无额外要求"
        forbidden = "；".join(expected.get("forbidden_claims", [])) or "无额外禁项"
        lines.extend(
            [
                f"审核提示：{required}",
                "",
                f"禁止问题：{forbidden}",
                "",
                "- 总体选择：",
                "- A 五项评分（人物/事实/多轮/决策/安全）：",
                "- B 五项评分（人物/事实/多轮/决策/安全）：",
                "- 是否存在严重错误（A/B/both/none）：",
                "- 评价与依据：",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def build_blind_review(
    baseline_path: Path,
    candidate_path: Path,
    output_dir: Path,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty output: {output_dir}")

    baseline = _load(baseline_path)
    candidate = _load(candidate_path)
    _validate_report(baseline, "baseline")
    _validate_report(candidate, "candidate")
    baseline_provenance = baseline["provenance"]
    candidate_provenance = candidate["provenance"]
    for field in REQUIRED_PROVENANCE_FIELDS:
        if baseline_provenance[field] != candidate_provenance[field]:
            raise ValueError(f"paired provenance differs: {field}")
    pairs = _paired_rows(baseline, candidate)

    rng = random.Random(seed)
    review_samples: list[dict[str, Any]] = []
    key_rows: list[dict[str, str]] = []
    positions = {"baseline_as_A": 0, "candidate_as_A": 0}
    for baseline_row, candidate_row in pairs:
        baseline_is_a = bool(rng.getrandbits(1))
        a_row, b_row = (
            (baseline_row, candidate_row) if baseline_is_a else (candidate_row, baseline_row)
        )
        positions["baseline_as_A" if baseline_is_a else "candidate_as_A"] += 1
        review_samples.append(
            {
                "id": baseline_row["id"],
                "category": baseline_row["category"],
                "interlocutor": baseline_row.get("interlocutor"),
                "turns": list(baseline_row["turns"]),
                "turn_rubrics": baseline_row.get("turn_rubrics", []),
                "expected_behavior": baseline_row.get("expected_behavior", {}),
                "rubric": baseline_row.get("rubric", []),
                "candidate_A": {"turn_responses": list(a_row["turn_responses"])},
                "candidate_B": {"turn_responses": list(b_row["turn_responses"])},
                "decision": {
                    "winner": "",
                    "candidate_A_scores": _blank_scorecard(),
                    "candidate_B_scores": _blank_scorecard(),
                    "severe_error": "",
                    "reason": "",
                },
            }
        )
        key_rows.append(
            {
                "id": baseline_row["id"],
                "A": baseline["model"] if baseline_is_a else candidate["model"],
                "B": candidate["model"] if baseline_is_a else baseline["model"],
            }
        )

    review = {
        "schema_version": 1,
        "package_id": "KISAKI-R1V4-E1-PROMPT-VS-CHECKPOINT100-BLIND30",
        "status": "pending_independent_human_review",
        "randomization_seed": seed,
        "sample_count": len(review_samples),
        "instructions": {
            "winner_values": ["A", "B", "tie", "invalid"],
            "score_scale": "0=不满足, 1=部分满足, 2=充分满足, null=未评或不适用",
            "lock_before_unblinding": True,
        },
        "samples": review_samples,
    }
    key = {
        "schema_version": 1,
        "package_id": review["package_id"],
        "status": "sealed_until_human_decisions_locked",
        "source_files": {
            "baseline": {"name": baseline_path.name, "sha256": _sha256(baseline_path)},
            "candidate": {"name": candidate_path.name, "sha256": _sha256(candidate_path)},
        },
        "paired_contract": {field: baseline_provenance[field] for field in REQUIRED_PROVENANCE_FIELDS},
        "position_counts": positions,
        "key": key_rows,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "blind_review.json").write_text(
        json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "BLIND_REVIEW.md").write_text(_markdown(review), encoding="utf-8")
    (output_dir / "blind_key.json").write_text(
        json.dumps(key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "sample_count": len(review_samples),
        "multiturn_count": sum(len(row["turns"]) > 1 for row in review_samples),
        "position_counts": positions,
        "output_dir": str(output_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        result = build_blind_review(
            args.baseline, args.candidate, args.output_dir, seed=args.seed
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
