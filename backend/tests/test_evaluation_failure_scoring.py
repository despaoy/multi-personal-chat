"""Generation failures remain receipts, never fabricated scored responses."""

from __future__ import annotations

import asyncio
import json
import sys
from types import SimpleNamespace

import pytest

from evaluation import runtime_runner
from evaluation.generation_metrics import GenerationMetrics
from evaluation.gold_set_manager import GoldSetManager
from evaluation.persona_metrics import PersonaMetrics


def test_generation_failures_do_not_change_successful_metrics():
    metric = GenerationMetrics()

    def generate(prompt):
        if prompt == "bad":
            raise RuntimeError("not model output")
        return "hello world"

    result = metric.evaluate(["good", "bad"], generate)
    assert result.items() >= metric.summarize(["hello world"]).items()
    assert result["total_prompts"] == 2
    assert result["generation_errors"] == 1
    assert result["samples"][1] == {
        "prompt": "bad",
        "response": "",
        "error": "not model output",
        "error_type": "RuntimeError",
    }


@pytest.mark.parametrize("response", [None, 123, {}])
def test_invalid_generation_results_are_errors(response):
    result = GenerationMetrics().evaluate(["prompt"], lambda _: response)
    assert result["scored_responses"] == 0
    assert result["generation_errors"] == 1
    assert result["samples"][0]["error_type"] == "TypeError"


def test_mock_uses_real_metric_aggregation():
    metric = GenerationMetrics()
    result = metric.evaluate_mock(["重复重复重复重复重复"])
    assert result["mock"] is True
    assert result.items() >= metric.summarize([result["samples"][0]["response"]]).items()


def test_persona_errors_do_not_receive_default_rubric_scores():
    prompt = {"id": "p1", "prompt": "question", "persona": "hutao", "rubric": [{"name": "安全", "scale": 5}]}

    def fail(_):
        raise RuntimeError("offline")

    result = PersonaMetrics().evaluate([prompt], fail)
    assert result["overall"] == result["by_persona"] == {}
    assert result["scored_responses"] == 0
    assert result["generation_errors"] == 1
    assert result["errors"][0]["error"] == "offline"


@pytest.mark.parametrize("content", ["{broken", "[]", "{}", '{"prompts":null}', '{"prompts":[null]}'])
def test_gold_input_faults_are_not_empty_sets(tmp_path, content):
    path = tmp_path / "gold.json"
    path.write_text(content)
    with pytest.raises(ValueError):
        GoldSetManager(path).load_set()


def test_gold_missing_file_and_empty_set_are_distinct(tmp_path):
    path = tmp_path / "gold.json"
    with pytest.raises(FileNotFoundError):
        GoldSetManager(path).load_set()
    path.write_text('{"prompts": []}')
    assert GoldSetManager(path).load_set() == []


@pytest.mark.parametrize("all_fail", [False, True])
def test_runtime_aggregates_only_completed_samples(monkeypatch, all_fail):
    dataset = {
        "prompts": [
            {"id": "good", "prompt": "good"},
            {"id": "bad", "prompt": "bad"},
            {"id": "empty", "prompt": ""},
        ]
    }
    monkeypatch.setattr(runtime_runner, "load_runtime_dataset", lambda _: dataset)
    calls = []

    async def get_client():
        return SimpleNamespace(generate=object())

    async def generate(request, fn):
        calls.append(request.message)
        if all_fail or request.message == "bad":
            raise RuntimeError("synthetic failure")
        return SimpleNamespace(reply="hello world")

    monkeypatch.setitem(sys.modules, "api.generate", SimpleNamespace(get_vllm_client=get_client))
    monkeypatch.setitem(
        sys.modules,
        "inference.generation_request",
        SimpleNamespace(GenerationRequest=SimpleNamespace, generate_character_response=generate),
    )
    monkeypatch.setitem(
        sys.modules, "inference.lora_registry", SimpleNamespace(get_lora_system_prompt=lambda _: "persona")
    )
    writes = []
    database = SimpleNamespace(execute_sql=lambda query, params: writes.append(params))
    asyncio.run(runtime_runner.execute_generation_evaluation("run", {}, database))
    result = json.loads(writes[-1]["metrics"])
    assert calls == ["good", "bad"]
    assert result["scored_responses"] == (0 if all_fail else 1)
    assert result["generation_errors"] == (3 if all_fail else 2)
    assert writes[-1]["note"] == ("failed" if all_fail else "completed_with_errors")
    assert result["samples"][2]["response"] == ""
    assert result["samples"][2]["error_type"] == "ValueError"
    assert result.items() >= GenerationMetrics().summarize([] if all_fail else ["hello world"]).items()


def test_cleaning_persona_dependency_failure_cannot_pass(monkeypatch):
    from evaluation.clean_training_data import _persona_consistency_check

    def fail(*args):
        raise RuntimeError("persona check unavailable")

    monkeypatch.setattr(PersonaMetrics, "contradiction_rate", fail)
    item = {"conversations": [{"role": "assistant", "content": "response"}]}
    with pytest.raises(RuntimeError, match="persona check unavailable"):
        _persona_consistency_check(item, "hutao")
    assert _persona_consistency_check(item, None) == (True, 0.0)


def test_cleaning_safety_input_failure_cannot_pass(monkeypatch, tmp_path):
    from evaluation import clean_training_data

    monkeypatch.setattr(clean_training_data, "__file__", str(tmp_path / "clean.py"))
    with pytest.raises(FileNotFoundError):
        clean_training_data._safety_reverse_check({})
    (tmp_path / "safety_test_set.json").write_text("{broken")
    with pytest.raises(json.JSONDecodeError):
        clean_training_data._safety_reverse_check({})


def test_runtime_partial_multiturn_failure_keeps_evidence_unscored(monkeypatch):
    monkeypatch.setattr(
        runtime_runner, "load_runtime_dataset", lambda _: {"prompts": [{"id": "multi", "turns": ["first", "second"]}]}
    )

    async def get_client():
        return SimpleNamespace(generate=object())

    async def generate(request, fn):
        if request.message == "second":
            assert request.history[-1] == {"role": "assistant", "content": "first reply"}
            raise RuntimeError("second turn failed")
        return SimpleNamespace(reply="first reply")

    monkeypatch.setitem(sys.modules, "api.generate", SimpleNamespace(get_vllm_client=get_client))
    monkeypatch.setitem(
        sys.modules,
        "inference.generation_request",
        SimpleNamespace(GenerationRequest=SimpleNamespace, generate_character_response=generate),
    )
    monkeypatch.setitem(
        sys.modules, "inference.lora_registry", SimpleNamespace(get_lora_system_prompt=lambda _: "persona")
    )
    writes = []
    database = SimpleNamespace(execute_sql=lambda query, params: writes.append(params))
    asyncio.run(runtime_runner.execute_generation_evaluation("run", {}, database))
    result = json.loads(writes[-1]["metrics"])
    assert result["scored_responses"] == 0
    assert result["generation_errors"] == 1
    assert result["samples"][0]["turn_responses"] == ["first reply"]
    assert result["samples"][0]["response"] == ""
    assert result["samples"][0]["error"] == "second turn failed"


@pytest.mark.parametrize("response", ["", " \t\n", "\u3000"])
def test_blank_response_is_a_failed_receipt_not_a_score(response):
    result = GenerationMetrics().evaluate(["prompt"], lambda _: response)
    assert result["scored_responses"] == 0
    assert result["generation_errors"] == 1
    assert result["samples"][0]["error_type"] == "ValueError"
    with pytest.raises(ValueError, match="empty or whitespace"):
        GenerationMetrics().summarize([response])
    persona = PersonaMetrics().evaluate([{"prompt": "question"}], lambda _: response)
    assert persona["scored_responses"] == 0
    assert persona["generation_errors"] == 1


@pytest.mark.parametrize("response", ["不能帮助执行该操作。", "证据不足，无法确认。", "  I don't know.\n"])
def test_refusal_and_abstention_remain_valid_unchanged_output(response):
    result = GenerationMetrics().evaluate(["prompt"], lambda _: response)
    assert result["scored_responses"] == 1
    assert result["generation_errors"] == 0
    assert result["samples"][0]["response"] == response


@pytest.mark.parametrize("blank_turn", ["first", "second"])
@pytest.mark.parametrize("blank", ["", " \n\t"])
def test_runtime_blank_turn_stops_case_and_retains_only_completed_turns(monkeypatch, blank_turn, blank):
    monkeypatch.setattr(
        runtime_runner,
        "load_runtime_dataset",
        lambda _: {"prompts": [{"id": "multi", "turns": ["first", "second", "third"]}]},
    )
    calls = []

    async def get_client():
        return SimpleNamespace(generate=object())

    async def generate(request, fn):
        calls.append(request.message)
        return SimpleNamespace(reply=blank if request.message == blank_turn else "valid reply")

    monkeypatch.setitem(sys.modules, "api.generate", SimpleNamespace(get_vllm_client=get_client))
    monkeypatch.setitem(
        sys.modules,
        "inference.generation_request",
        SimpleNamespace(GenerationRequest=SimpleNamespace, generate_character_response=generate),
    )
    monkeypatch.setitem(
        sys.modules, "inference.lora_registry", SimpleNamespace(get_lora_system_prompt=lambda _: "persona")
    )
    writes = []
    database = SimpleNamespace(execute_sql=lambda query, params: writes.append(params))
    asyncio.run(runtime_runner.execute_generation_evaluation("run", {}, database))
    result = json.loads(writes[-1]["metrics"])
    assert calls == (["first"] if blank_turn == "first" else ["first", "second"])
    assert result["scored_responses"] == 0
    assert result["generation_errors"] == 1
    assert writes[-1]["note"] == "failed"
    sample = result["samples"][0]
    assert sample["turn_responses"] == ([] if blank_turn == "first" else ["valid reply"])
    assert sample["response"] == ""
    assert sample["error_type"] == "ValueError"
