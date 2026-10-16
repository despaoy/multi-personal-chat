"""Offline scores distinguish failed work from legitimate abstention."""

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from experiments.rag_answer_ablation import RAGAnswerAblation

QUERY = "虚构远岚登记需要什么材料？"
SOURCE = "虚构远岚登记需本人身份证原件与受理号；预约不免原件。"


@pytest.fixture
def rig(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(dict(questions=[dict(id="case", question_type="synthetic", question=QUERY, expected_doc_ids=[])]))
    )
    helper = SimpleNamespace(
        compute_confidence=lambda rows: rows[0]["score"] if rows else 0.0,
        build_citations=lambda rows: [dict(source_id=row["id"]) for row in rows],
    )
    monkeypatch.setattr("knowledge.rag_helper.get_rag_helper", lambda: helper)
    return SimpleNamespace(dataset=dataset, runner=RAGAnswerAblation(dataset, "仅依据给定资料回答。"))


@pytest.mark.parametrize(
    "error", [OSError("index unavailable"), TimeoutError("index timeout"), ValueError("invalid index row")]
)
def test_unanswerable_source_failure_is_not_a_correct_abstention(rig, error):
    rig.runner._retrieve_rows = Mock(side_effect=error)
    generate = Mock(side_effect=AssertionError("Failed retrieval must not generate"))
    result = rig.runner.run_variant("confidence_abstain", generate)
    row = result.samples[0]
    assert result.failures == 1 and result.unanswerable_abstention_accuracy == 0.0
    assert row["status"] == "error" and row["abstained"] is None
    assert row["confidence"] is None and row["reformulated"] is None
    assert row["cited_doc_ids"] == row["retrieved_doc_ids"] == [] and row["error"]
    rig.runner._retrieve_rows.assert_called_once_with(QUERY)
    generate.assert_not_called()


def test_legitimate_empty_library_is_a_correct_business_abstention(rig):
    rig.runner._retrieve_rows = Mock(return_value=[])
    generate = Mock(side_effect=AssertionError("No evidence must abstain"))
    result = rig.runner.run_variant("confidence_abstain", generate)
    assert result.failures == 0 and result.unanswerable_abstention_accuracy == 1.0
    assert result.samples[0]["status"] == "abstained" and result.samples[0]["abstained"] is True
    assert result.samples[0]["confidence"] == 0.0 and result.samples[0]["error"] == ""
    generate.assert_not_called()


def test_generation_failure_preserves_completed_retrieval_without_claiming_answer_citations(rig):
    payload = json.loads(rig.dataset.read_text())
    payload["questions"][0]["expected_doc_ids"] = ["current"]
    rig.dataset.write_text(json.dumps(payload))
    documents = [dict(id="current", title="现行材料", content=SOURCE, score=0.9)]
    rig.runner._retrieve_rows = Mock(return_value=documents)
    generate = Mock(side_effect=RuntimeError("controlled answer failure"))
    result = rig.runner.run_variant("confidence_abstain", generate)
    row = result.samples[0]
    assert result.failures == 1 and result.citation_hit_rate == 0.0
    assert row["status"] == "error" and row["abstained"] is None and row["answer"] == ""
    assert row["confidence"] == 0.9 and row["retrieved_doc_ids"] == ["current"] and row["cited_doc_ids"] == []
    generate.assert_called_once()
    assert SOURCE in generate.call_args.args[0][-1]["content"]


def test_answerable_success_is_retained(rig):
    payload = json.loads(rig.dataset.read_text())
    payload["questions"][0]["expected_doc_ids"] = ["current"]
    rig.dataset.write_text(json.dumps(payload))
    rig.runner._retrieve_rows = Mock(return_value=[dict(id="current", title="现行材料", content=SOURCE, score=0.9)])
    generate = Mock(return_value=SOURCE)
    result = rig.runner.run_variant("confidence_abstain", generate)
    assert result.failures == 0 and result.citation_hit_rate == 1.0
    assert result.samples[0]["status"] == "answered" and result.samples[0]["abstained"] is False
    assert result.samples[0]["answer"] == SOURCE
    generate.assert_called_once()


@pytest.mark.parametrize("documents", [[], [dict(id="one", title="的", content="", score=0.1)]])
def test_unchanged_keyword_baseline_performs_no_second_retrieval(rig, documents):
    rig.runner._retrieve_rows = Mock(return_value=documents)
    result = rig.runner._retrieve("corrective_rag", "的")
    rig.runner._retrieve_rows.assert_called_once_with("的")
    assert result["abstained"] and result["reformulated"] is False


@pytest.mark.parametrize("success", [False, True])
def test_explicit_changed_baseline_retrieves_once_more_without_mutating_inputs(rig, success):
    first = [dict(id="old", title="permit", content="document requirement", score=0.1)]
    second = [dict(id="current", title="permit", content=SOURCE, score=0.9 if success else 0.1)]
    original = deepcopy([first, second])
    rig.runner._retrieve_rows = Mock(side_effect=[first, second])
    result = rig.runner._retrieve("corrective_rag", "entry")
    assert rig.runner._retrieve_rows.call_count == 2 and result["reformulated"] is True
    assert result["original_query"] == "entry" and result["reformulated_query"].startswith("entry ")
    assert result["abstained"] is (not success) and [first, second] == original


@pytest.mark.parametrize("threshold", [-0.1, 1.1, True, None, float("nan"), float("inf"), "bad"])
def test_invalid_threshold_is_rejected_before_evaluation(tmp_path, threshold):
    with pytest.raises(ValueError, match="RAGAnswerAblation.threshold"):
        RAGAnswerAblation(tmp_path / "not-read.json", "synthetic", threshold=threshold)


@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_boundary_threshold_is_preserved(tmp_path, threshold):
    assert RAGAnswerAblation(tmp_path / "not-read.json", "synthetic", threshold=threshold).threshold == threshold


def test_low_confidence_preserves_retrieved_sources_without_claiming_answer_citations(rig):
    payload = json.loads(rig.dataset.read_text())
    payload["questions"][0]["expected_doc_ids"] = ["current"]
    rig.dataset.write_text(json.dumps(payload))
    rig.runner._retrieve_rows = Mock(return_value=[dict(id="current", title="现行材料", content=SOURCE, score=0.1)])
    generate = Mock(side_effect=AssertionError("Low confidence must not generate"))
    result = rig.runner.run_variant("confidence_abstain", generate)
    row = result.samples[0]
    assert result.failures == 0 and result.over_abstention_rate == 1.0
    assert result.citation_hit_rate == 0.0
    assert row["status"] == "abstained" and row["abstained"] is True and row["confidence"] == 0.1
    assert row["retrieved_doc_ids"] == ["current"] and row["cited_doc_ids"] == []
    generate.assert_not_called()


@pytest.mark.parametrize("answer", [None, "", " \n\t", ["unexpected structured value"]])
def test_invalid_model_answer_cannot_score_as_successful_citation(rig, answer):
    payload = json.loads(rig.dataset.read_text())
    payload["questions"][0]["expected_doc_ids"] = ["current"]
    rig.dataset.write_text(json.dumps(payload))
    rig.runner._retrieve_rows = Mock(return_value=[dict(id="current", title="现行材料", content=SOURCE, score=0.9)])
    result = rig.runner.run_variant("confidence_abstain", Mock(return_value=answer))
    row = result.samples[0]
    assert result.failures == 1 and result.citation_hit_rate == 0.0
    assert row["status"] == "error" and row["abstained"] is None
    assert row["retrieved_doc_ids"] == ["current"] and row["cited_doc_ids"] == []
    assert "non-empty answer text" in row["error"]


def test_explicit_mock_uses_complete_gold_answer_without_calling_a_model(rig):
    payload = json.loads(rig.dataset.read_text())
    payload["questions"][0].update(expected_doc_ids=["current"], gold_answer=SOURCE)
    rig.dataset.write_text(json.dumps(payload))
    rig.runner._retrieve_rows = Mock(return_value=[dict(id="current", title="现行材料", content=SOURCE, score=0.9)])
    generate = Mock(side_effect=AssertionError("Explicit mock must not call a model"))
    result = rig.runner.run_variant("confidence_abstain", generate, mock=True)
    assert result.mock and result.failures == 0 and result.citation_hit_rate == 1.0
    assert result.samples[0]["status"] == "answered" and result.samples[0]["answer"] == SOURCE
    generate.assert_not_called()


def test_invalid_mock_gold_is_not_converted_into_a_successful_text_answer(rig):
    payload = json.loads(rig.dataset.read_text())
    payload["questions"][0].update(expected_doc_ids=["current"], gold_answer=None)
    rig.dataset.write_text(json.dumps(payload))
    rig.runner._retrieve_rows = Mock(return_value=[dict(id="current", title="现行材料", content=SOURCE, score=0.9)])
    generate = Mock(side_effect=AssertionError("Explicit mock must not call a model"))
    result = rig.runner.run_variant("confidence_abstain", generate, mock=True)
    assert result.mock and result.failures == 1 and result.citation_hit_rate == 0.0
    assert result.samples[0]["status"] == "error" and result.samples[0]["answer"] == ""
    generate.assert_not_called()
