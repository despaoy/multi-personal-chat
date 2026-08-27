"""P4C 运行器的人工进度保护与调用统计口径回归测试。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from knowledge.game_rag.models import ReviewStatus


@pytest.fixture(scope="module")
def runner_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_scene_metadata_candidates.py"
    spec = importlib.util.spec_from_file_location("p4c_runner_for_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _decision(scene_id: str, status: ReviewStatus):
    return SimpleNamespace(scene_id=scene_id, review_status=status)


def test_finalize_reuses_existing_review_instead_of_recreating(runner_module, tmp_path, monkeypatch):
    review_path = tmp_path / "scene_metadata_review.json"
    review_path.write_text("{}", encoding="utf-8")
    existing = SimpleNamespace(
        review_status="draft",
        scene_decisions=[_decision("scene_a", ReviewStatus.needs_review)],
    )
    expected_report = SimpleNamespace(review_doc=existing)

    monkeypatch.setattr(runner_module, "load_scene_metadata_review", lambda path: existing)
    monkeypatch.setattr(runner_module, "validate_scene_metadata_review", lambda review, bundle: [])
    monkeypatch.setattr(
        runner_module,
        "create_scene_metadata_review",
        lambda *args, **kwargs: pytest.fail("已有审核文档时不得重建空白文档"),
    )
    monkeypatch.setattr(
        runner_module,
        "merge_candidates_into_review",
        lambda bundle, review, state, on_conflict: expected_report,
    )

    result = runner_module._prepare_review_for_finalize(object(), object(), review_path)

    assert result is expected_report


def test_finalize_rejects_top_level_approved_review(runner_module, tmp_path, monkeypatch):
    review_path = tmp_path / "scene_metadata_review.json"
    review_path.write_text("{}", encoding="utf-8")
    existing = SimpleNamespace(review_status="approved", scene_decisions=[])
    monkeypatch.setattr(runner_module, "load_scene_metadata_review", lambda path: existing)
    monkeypatch.setattr(runner_module, "validate_scene_metadata_review", lambda review, bundle: [])

    with pytest.raises(ValueError, match="已整体 approved"):
        runner_module._prepare_review_for_finalize(object(), object(), review_path)


def test_finalize_rejects_changes_to_approved_scene_set(runner_module, tmp_path, monkeypatch):
    review_path = tmp_path / "scene_metadata_review.json"
    review_path.write_text("{}", encoding="utf-8")
    existing = SimpleNamespace(review_status="draft", scene_decisions=[])
    changed = SimpleNamespace(
        review_doc=SimpleNamespace(
            review_status="draft",
            scene_decisions=[_decision("scene_a", ReviewStatus.approved)],
        )
    )
    monkeypatch.setattr(runner_module, "load_scene_metadata_review", lambda path: existing)
    monkeypatch.setattr(runner_module, "validate_scene_metadata_review", lambda review, bundle: [])
    monkeypatch.setattr(runner_module, "merge_candidates_into_review", lambda *args, **kwargs: changed)

    with pytest.raises(ValueError, match="approved 场景集合"):
        runner_module._prepare_review_for_finalize(object(), object(), review_path)


def test_call_stats_distinguish_invocation_and_lifetime(runner_module, tmp_path, monkeypatch):
    output_path = tmp_path / "model_call_stats.json"
    monkeypatch.setattr(runner_module, "OUT_DIR", tmp_path)
    monkeypatch.setattr(runner_module, "CALL_STATS_PATH", output_path)
    monkeypatch.setattr(runner_module, "_utc_now", lambda: "2026-01-01T00:00:00+00:00")
    client = SimpleNamespace(stats={"calls": 2, "failures": 0})
    bundle = SimpleNamespace(scenes=[SimpleNamespace(source=SimpleNamespace(line_start=1, line_end=151))])
    state = SimpleNamespace(scene_states=[SimpleNamespace(attempts=5)])

    runner_module._write_call_stats(client, "run", bundle, state)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["client_stats"]["calls"] == 2
    assert payload["lifetime_state_stats"] == {
        "total_attempts": 5,
        "minimum_required_calls": 2,
        "excess_retry_calls": 3,
    }
