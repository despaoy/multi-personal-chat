"""Focused contracts for the balanced CAHM evaluator."""

from __future__ import annotations

import json
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from character.memory_llm import MemoryLlmConfig
from experiments.evaluate_cahm_balanced import (
    DEFAULT_DATASET,
    RETRIEVAL_VARIANTS,
    _evaluate_relations,
    _evaluate_retrieval_variant,
    _load_jsonl,
    _render_markdown,
    _write_report,
    evaluate,
)


class _ConstantEmbedding:
    model_id = "constant-test-embedding"
    dimension = 3

    def embed_texts(self, texts):
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vector = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
        vector /= np.linalg.norm(vector)
        return np.vstack([vector for _ in texts])

    def embed_query(self, query):
        return self.embed_texts([query])[0]


class _SequenceCompletion:
    def __init__(self, responses):
        self.responses = list(responses)
        self.closed = False

    async def complete(self, messages):
        assert messages[-1]["role"] == "user"
        return self.responses.pop(0)

    async def close(self):
        self.closed = True


def _response(**memory):
    return json.dumps({"memories": [memory]}, ensure_ascii=False)


def test_balanced_gold_is_valid_and_covers_both_tasks():
    rows = _load_jsonl(DEFAULT_DATASET)

    assert len(rows) == 40
    assert len({row["id"] for row in rows}) == len(rows)
    assert sum(row["task"] == "relation" for row in rows) == 20
    assert sum(row["task"] == "retrieval_v2" for row in rows) == 20


@pytest.mark.asyncio
async def test_relation_metrics_are_na_without_memory_llm():
    report = await _evaluate_relations(
        [{"id": "r1", "task": "relation", "message": "我叫小明", "gold_operation": "ADD"}],
        None,
    )

    assert report["evaluated"] is False
    assert report["operation_accuracy"] is None
    assert report["operation_macro_f1"] is None
    assert report["target_accuracy"] is None
    assert report["status_accuracy"] is None
    assert report["failures"] == []


@pytest.mark.asyncio
async def test_relation_operation_target_status_macro_f1_and_failures_use_validated_output():
    cases = [
        {
            "id": "add-wrong",
            "task": "relation",
            "message": "我叫小明。",
            "existing_memories": [],
            "gold_operation": "ADD",
            "gold_status": "active",
        },
        {
            "id": "supersede-correct",
            "task": "relation",
            "message": "补全方向暂时不做了，我最近改做点云识别。",
            "existing_memories": [
                {
                    "id": "102",
                    "memory_key": "goal_点云补全",
                    "memory_type": "shared_event",
                    "content": "用户正在进行或准备：点云补全",
                    "status": "active",
                }
            ],
            "gold_operation": "SUPERSEDE",
            "gold_target_memory_id": "102",
            "gold_status": "active",
        },
    ]
    completion = _SequenceCompletion(
        [
            _response(
                kind="name",
                value="小明",
                content="用户说自己叫小明",
                evidence="我叫小明",
                confidence=0.99,
                operation="NOOP",
                attributed_to="user",
            ),
            _response(
                kind="goal",
                value="点云识别",
                content="用户最近改做点云识别",
                evidence="补全方向暂时不做了，我最近改做点云识别",
                confidence=0.99,
                operation="SUPERSEDE",
                target_memory_id="102",
                target_memory_key="goal_点云补全",
                attributed_to="user",
            ),
        ]
    )
    config = MemoryLlmConfig(enabled=True, base_url="http://memory.invalid", model="test")

    report = await _evaluate_relations(cases, config, completion=completion)

    assert completion.closed is True
    assert report["evaluated"] is True
    assert report["successfully_processed_cases"] == 2
    assert report["operation_accuracy"] == pytest.approx(0.5)
    assert report["operation_macro_f1"] == pytest.approx(1 / 3)
    assert report["target_evaluated_cases"] == 1
    assert report["target_accuracy"] == pytest.approx(1.0)
    assert report["status_evaluated_cases"] == 2
    assert report["status_accuracy"] == pytest.approx(0.5)
    assert [item["id"] for item in report["failures"]] == ["add-wrong"]
    assert report["failures"][0]["raw_llm_output"]


def _retrieval_cases():
    return [
        {
            "id": "lifecycle",
            "task": "retrieval_v2",
            "query": "我喜欢什么咖啡？",
            "gold_ids": ["active"],
            "records": [
                {
                    "memory_id": "active",
                    "memory_status": "active",
                    "memory_key": "preference_咖啡",
                    "memory_type": "user_fact",
                    "content": "用户说喜欢拿铁咖啡",
                    "importance": 0.9,
                },
                {
                    "id": "pending",
                    "status": "pending",
                    "memory_key": "preference_咖啡_pending",
                    "memory_type": "user_fact",
                    "content": "用户可能喜欢摩卡咖啡",
                    "importance": 0.8,
                },
                {
                    "id": "superseded",
                    "status": "superseded",
                    "memory_key": "preference_咖啡_old",
                    "memory_type": "user_fact",
                    "content": "用户以前喜欢黑咖啡",
                    "importance": 0.8,
                },
                {
                    "id": "retracted",
                    "status": "retracted",
                    "memory_key": "preference_咖啡_wrong",
                    "memory_type": "user_fact",
                    "content": "用户说喜欢冰咖啡",
                    "importance": 0.8,
                },
            ],
        },
        {
            "id": "proof",
            "task": "retrieval_v2",
            "query": "为什么推荐我无咖啡因饮料？",
            "gold_ids": ["proof"],
            "require_evidence": True,
            "records": [
                {
                    "id": "proof",
                    "status": "active",
                    "memory_key": "preference_含咖啡因",
                    "memory_type": "user_fact",
                    "content": "用户说不喜欢含咖啡因的饮料",
                    "importance": 0.9,
                    "evidence_json": '["我平时不碰含咖啡因的东西"]',
                    "source_message_ids_json": '["msg-proof"]',
                }
            ],
        },
    ]


@pytest.mark.asyncio
async def test_retrieval_ablation_reports_real_lifecycle_leakage_and_evidence_completeness():
    provider = _ConstantEmbedding()
    legacy = await _evaluate_retrieval_variant(_retrieval_cases(), RETRIEVAL_VARIANTS[0], provider)
    balanced = await _evaluate_retrieval_variant(_retrieval_cases(), RETRIEVAL_VARIANTS[1], provider)

    assert legacy["unsupported_service_switches"] == []
    assert legacy["lifecycle_leakage"]["pending_count"] == 1
    assert legacy["lifecycle_leakage"]["superseded_count"] == 1
    assert legacy["lifecycle_leakage"]["retracted_count"] == 1
    assert legacy["wrong_memory_injection_rate"] > 0.0
    assert legacy["evidence_completeness"] == pytest.approx(0.0)

    assert balanced["recall_at_5"] == pytest.approx(1.0)
    assert balanced["wrong_memory_injection_rate"] == pytest.approx(0.0)
    assert balanced["lifecycle_leakage"]["pending_count"] == 0
    assert balanced["lifecycle_leakage"]["superseded_count"] == 0
    assert balanced["lifecycle_leakage"]["retracted_count"] == 0
    assert balanced["evidence_completeness"] == pytest.approx(1.0)
    assert balanced["average_retrieval_latency_ms"] >= 0.0


@pytest.mark.asyncio
async def test_full_report_writes_json_and_markdown_without_proxy_llm_metrics(monkeypatch):
    monkeypatch.setenv("MEMORY_LLM_ENABLED", "false")
    args = SimpleNamespace(
        dataset=DEFAULT_DATASET,
        memory_llm_base_url="",
        memory_llm_model="",
        memory_llm_api_key="",
        memory_llm_timeout=2.0,
        memory_llm_confidence_threshold=0.85,
        min_hybrid_score=0.35,
        candidate_limit=100,
    )

    report = await evaluate(args, embedding_provider=_ConstantEmbedding())
    output = DEFAULT_DATASET.parents[1] / "results" / f"_balanced_evaluator_test_{uuid4().hex}.json"
    markdown_path = output.with_suffix(".md")
    try:
        json_path, markdown_path = _write_report(report, output)
        persisted = json.loads(json_path.read_text(encoding="utf-8"))
        markdown = markdown_path.read_text(encoding="utf-8")
        assert persisted["relation"]["operation_accuracy"] is None
        assert set(persisted["retrieval"]) == {"legacy_hybrid", "balanced_default"}
        assert "Operation macro-F1" in markdown
        assert "N/A" in markdown
        assert _render_markdown(report) == markdown
    finally:
        output.unlink(missing_ok=True)
        markdown_path.unlink(missing_ok=True)
