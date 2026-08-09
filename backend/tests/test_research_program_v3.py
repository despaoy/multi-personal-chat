import json
from pathlib import Path

import pytest

from experiments.quantization_benchmark import QuantizationBenchmark, QuantizationConfig
from experiments.rag_ablation import RAGAblation
from evaluation import character_benchmark_v3

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH = PROJECT_ROOT / "backend" / "data" / "character_dialogues" / "experiments" / "research"


def test_prompt_v2_corrects_relationship_without_banning_natural_laughter():
    prompt = (PROJECT_ROOT / "backend" / "data" / "character_dialogues" / "kisaki_system_prompt_v2.txt").read_text(encoding="utf-8")
    assert "亲生哥哥" in prompt
    assert "义妹" not in prompt
    assert "哈哈" in prompt
    assert "不输出 [文档ID]" in prompt


def test_runtime_lora_registry_uses_canonical_kisaki_prompt():
    from inference.lora_registry import get_lora_system_prompt

    prompt = (
        PROJECT_ROOT
        / "backend"
        / "data"
        / "character_dialogues"
        / "kisaki_system_prompt_v2.txt"
    ).read_text(encoding="utf-8").strip()
    assert get_lora_system_prompt("kisaki") == prompt
    assert get_lora_system_prompt("test-lora-highperf") == prompt


def test_registry_v3_separates_design_from_runtime_and_forbids_formal_mock():
    registry = json.loads((RESEARCH / "research_program_registry_v3.json").read_text(encoding="utf-8"))
    assert registry["schema_version"] == 3
    assert registry["authoritative"] is True
    assert registry["formal_result_policy"]["mock_allowed"] is False
    assert [item["id"] for item in registry["research"]] == ["R0", "R1", "R2", "R3", "R4", "S1"]


def test_rag_v2_has_registered_30_15_15_split_and_is_not_formal_yet():
    dataset = json.loads((RESEARCH / "kisaki_rag_eval_v2_candidates.json").read_text(encoding="utf-8"))
    counts = {}
    for question in dataset["questions"]:
        counts[question["question_type"]] = counts.get(question["question_type"], 0) + 1
    assert counts == {"single_evidence": 30, "multi_evidence": 15, "unanswerable": 15}
    assert dataset["formal_use_allowed"] is False
    with pytest.raises(ValueError, match="reviewed and frozen"):
        RAGAblation(str(RESEARCH / "kisaki_rag_eval_v2_candidates.json"), formal=True)._dataset()


def test_quantization_benchmark_marks_mock_and_uses_real_latency_percentiles():
    bench = QuantizationBenchmark(warmup_requests=5, repeats=3, concurrency_levels=(1, 4, 8))
    mock = bench.benchmark_model_mock(QuantizationConfig("bf16", "", "bf16"))
    assert mock.mock is True
    assert mock.prompt_sha256
    summary = bench._summarize([
        {"ok": True, "e2e_latency_ms": 100, "ttft_ms": 20, "inter_token_latency_ms": 5, "decode_tokens_per_s": 50},
        {"ok": True, "e2e_latency_ms": 300, "ttft_ms": 40, "inter_token_latency_ms": 7, "decode_tokens_per_s": 40},
        {"ok": False, "error": "timeout"},
    ])
    assert summary["completed_requests"] == 2
    assert summary["failed_requests"] == 1
    assert summary["mean_ttft_ms"] == 30
    assert summary["p95_latency_ms"] == 290


def test_r4_config_is_dpo_only_and_hard_blocked_before_human_review():
    config = json.loads((RESEARCH / "preference_alignment_config_v3.json").read_text(encoding="utf-8"))
    assert config["method"] == "dpo"
    assert config["orpo_enabled"] is False
    assert config["minimum_human_approved_pairs"] == 100
    assert config["q_a_lora_status"] == "not_implemented"
    assert config["status"] == "blocked_on_human_review"


def test_r4_dpo_starts_from_a_trainable_sft_adapter_and_uses_registered_lr():
    from training.preference_trainer import PreferenceTrainingConfig

    config = PreferenceTrainingConfig.from_dict({"learning_rate": 5e-7})
    assert config.learning_rate == 5e-7
    source = (PROJECT_ROOT / "backend" / "training" / "preference_trainer.py").read_text(encoding="utf-8")
    assert "prepare_model_for_kbit_training(model)" in source
    assert "is_trainable=True" in source

def test_multiturn_benchmark_inserts_assistant_replies(monkeypatch):
    observed = []

    def fake_call(base_url, model, messages, generation, timeout):
        observed.append([dict(message) for message in messages])
        return f"reply-{len(observed)}", 5.0, ""

    monkeypatch.setattr(character_benchmark_v3, "_call", fake_call)
    response, latency, error, context = character_benchmark_v3._call_conversation(
        "http://test", "model", "system", ["turn-1", "turn-2", "turn-3"], {}, 1
    )
    assert response == "reply-3"
    assert context == ["reply-1", "reply-2"]
    assert latency == 15.0
    assert error == ""
    assert observed[1][-2:] == [
        {"role": "assistant", "content": "reply-1"},
        {"role": "user", "content": "turn-2"},
    ]

def test_citation_contract_includes_stable_source_id():
    from knowledge.rag_helper import RAGHelper

    helper = object.__new__(RAGHelper)
    citations = helper.build_citations([
        {"id": "doc-1", "title": "chapter", "content": "evidence", "score": 0.8}
    ])
    assert citations[0]["source_id"] == "doc-1"
    assert citations[0]["source_title"] == "chapter"

def test_s1_router_supports_project_personas_and_keeps_modes_separate():
    from inference.lora_router import LoRARouter, RouteTarget

    config = {
        "enabled": True,
        "mode": "rule",
        "default_adapter": "default",
        "persona_keywords": {"kisaki": ["月社妃"], "minamo": ["水菜萌"]},
        "persona_adapters": {"kisaki": "kisaki", "minamo": "minamo"},
    }
    router = LoRARouter(config)
    decision = router.route("请让月社妃回答")
    assert decision.target == RouteTarget.PERSONA_ADAPTER.value
    assert decision.adapter_name == "kisaki"
    assert LoRARouter({**config, "mode": "manual"}).route("月社妃").fallback is True
    intent = LoRARouter({**config, "mode": "intent"}).route("查资料", (True, 0.9, "kisaki"))
    assert intent.target == RouteTarget.RAG_REQUIRED.value
    detector_router = LoRARouter({**config, "mode": "intent"})
    detector_router._intent_detector = lambda _: (True, "knowledge intent", "kisaki")
    detected = detector_router.route("请检索原作资料")
    assert detected.target == RouteTarget.RAG_REQUIRED.value
    assert detected.confidence == 1.0
    explicit_reason = detector_router.route("请检索原作资料", (True, "matched rule", "kisaki"))
    assert explicit_reason.target == RouteTarget.RAG_REQUIRED.value
    assert explicit_reason.confidence == 1.0


def test_s1_dataset_keeps_external_roles_out_of_formal_generalization_claims():
    dataset = json.loads((RESEARCH / "system_routing_eval_v1.json").read_text(encoding="utf-8"))
    assert dataset["count"] == 80
    minamo_hutao = [row for row in dataset["cases"] if row.get("expected_adapter") in {"minamo", "hutao"}]
    assert len(minamo_hutao) == 40
    assert all(row["external_demo_only"] for row in minamo_hutao)

def test_router_config_migration_adds_personas_without_overwriting_admin_values():
    from api.router import _normalize_config

    migrated = _normalize_config({
        "persona_adapters": {"kisaki": "custom-kisaki"},
        "persona_keywords": {"hutao": ["custom keyword"]},
    })
    assert migrated["persona_adapters"]["kisaki"] == "custom-kisaki"
    assert migrated["persona_adapters"]["minamo"] == "minamo"
    assert migrated["persona_keywords"]["hutao"] == ["custom keyword"]
    assert "月社妃" in migrated["persona_keywords"]["kisaki"]


def test_formal_r2_refuses_silent_reranker_fallback():
    from types import SimpleNamespace

    runner = RAGAblation(formal=True)
    runner._rag_helper = SimpleNamespace(enable_reranking=False, reranker=None)
    with pytest.raises(RuntimeError, match="Cross-Encoder"):
        runner.variant_hybrid_reranker("question", 5)
