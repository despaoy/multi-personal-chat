import json
from pathlib import Path

import pytest

from scripts.build_kisaki_r1v4_blind_review import build_blind_review


def _report(model: str, *, mock: bool = False) -> dict:
    samples = []
    for index in range(30):
        turn_count = 3 if index < 7 else 1
        turns = [f"问题 {index}-{turn}" for turn in range(turn_count)]
        response_prefix = "甲" if model in {"prompt-only", "base"} else "乙"
        responses = [f"{response_prefix}回答 {index}-{turn}" for turn in range(turn_count)]
        samples.append(
            {
                "id": f"sample-{index:02d}",
                "category": "multiturn" if turn_count > 1 else "persona",
                "cluster_id": f"cluster-{index}",
                "interlocutor": "测试对象",
                "prompt": turns[0],
                "turns": turns,
                "turn_responses": responses,
                "response": responses[-1],
                "turn_rubrics": [],
                "expected_behavior": {"required_behaviors": ["保持一致"]},
                "rubric": [],
                "error": "",
            }
        )
    return {
        "schema_version": 3,
        "mock": mock,
        "model": model,
        "provenance": {
            "dataset_sha256": "dataset-hash",
            "dataset_id": "dev30",
            "dataset_status": "derived_development_subset",
            "dataset_role": "development_checkpoint_selection",
            "prompt_policy_version": "3.3.0",
            "generation": {"temperature": 0.0, "max_tokens": 256},
        },
        "samples": samples,
    }


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_blind_review_preserves_all_samples_and_multiturn_content(tmp_path):
    baseline_path = tmp_path / "prompt_only.json"
    candidate_path = tmp_path / "checkpoint-100.json"
    _write(baseline_path, _report("prompt-only"))
    _write(candidate_path, _report("checkpoint-100"))

    result = build_blind_review(baseline_path, candidate_path, tmp_path / "blind", seed=42)
    review_text = (tmp_path / "blind/blind_review.json").read_text(encoding="utf-8")
    review = json.loads(review_text)
    key = json.loads((tmp_path / "blind/blind_key.json").read_text(encoding="utf-8"))

    assert result["sample_count"] == 30
    assert result["multiturn_count"] == 7
    assert len(review["samples"]) == len(key["key"]) == 30
    assert all(
        len(row["turns"])
        == len(row["candidate_A"]["turn_responses"])
        == len(row["candidate_B"]["turn_responses"])
        for row in review["samples"]
    )
    assert result["position_counts"]["baseline_as_A"] > 0
    assert result["position_counts"]["candidate_as_A"] > 0
    for forbidden in ("prompt-only", "checkpoint-100", "adapter_path", "model_path"):
        assert forbidden not in review_text
    assert key["source_files"]["baseline"]["sha256"]


def test_blind_review_is_deterministic_and_refuses_overwrite(tmp_path):
    baseline_path = tmp_path / "a.json"
    candidate_path = tmp_path / "b.json"
    _write(baseline_path, _report("base"))
    _write(candidate_path, _report("candidate"))
    first = tmp_path / "first"
    second = tmp_path / "second"
    build_blind_review(baseline_path, candidate_path, first, seed=42)
    build_blind_review(baseline_path, candidate_path, second, seed=42)

    assert (first / "blind_review.json").read_bytes() == (second / "blind_review.json").read_bytes()
    with pytest.raises(ValueError, match="refusing to overwrite"):
        build_blind_review(baseline_path, candidate_path, first, seed=42)


@pytest.mark.parametrize("failure", ["mock", "generation", "order", "turn_count"])
def test_blind_review_rejects_unpaired_or_invalid_inputs(tmp_path, failure):
    baseline = _report("base")
    candidate = _report("candidate")
    if failure == "mock":
        candidate["mock"] = True
    elif failure == "generation":
        candidate["provenance"]["generation"]["temperature"] = 0.7
    elif failure == "order":
        candidate["samples"][0], candidate["samples"][1] = (
            candidate["samples"][1],
            candidate["samples"][0],
        )
    else:
        candidate["samples"][0]["turn_responses"].pop()
    baseline_path = tmp_path / "a.json"
    candidate_path = tmp_path / "b.json"
    _write(baseline_path, baseline)
    _write(candidate_path, candidate)

    with pytest.raises(ValueError):
        build_blind_review(baseline_path, candidate_path, tmp_path / "blind")
