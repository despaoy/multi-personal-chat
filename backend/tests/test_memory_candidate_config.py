"""Evaluation reports the candidate scope used by actual retrieval."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from experiments import evaluate_cahm


@pytest.mark.parametrize("repository_failure", [False, True])
async def test_report_matches_uncapped_candidate_reads(tmp_path, monkeypatch, repository_failure):
    rows = [dict(id=str(i), memory_key="preference_红茶", memory_type="user_fact",
                 content="用户说喜欢红茶", importance=0.9, confidence=1.0,
                 status="active", updated_at="2026-10-09T00:00:00+00:00") for i in range(101)]
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(json.dumps(dict(task="retrieval", id="all-candidates", category="preference",
                                      query="我喜欢红茶吗？", records=rows, gold_ids=[str(i) for i in range(101)])), encoding="utf-8")
    provider = SimpleNamespace(model_id="explicit-test-embedding",
                               embed_texts=lambda texts: np.ones((len(texts), 2), dtype=np.float32))
    monkeypatch.setattr(evaluate_cahm, "get_default_embedding_provider", lambda: provider)
    reads = []
    original = evaluate_cahm._CaseRepository.list_memory_records
    async def read(self, character_id, user_scope, limit=30, *, include_inactive=False):
        if repository_failure:
            raise RuntimeError("synthetic repository failure")
        records = await original(self, character_id, user_scope, limit=limit, include_inactive=include_inactive)
        reads.append((limit, len(records)))
        return records
    monkeypatch.setattr(evaluate_cahm._CaseRepository, "list_memory_records", read)
    args = SimpleNamespace(dataset=dataset, memory_llm_base_url="", memory_llm_model="", min_hybrid_score=0.35)
    if repository_failure:
        with pytest.raises(RuntimeError, match="synthetic repository failure"):
            await evaluate_cahm.evaluate(args)
        return
    report = await evaluate_cahm.evaluate(args)
    assert report["configuration"]["semantic_candidate_limit"] is None
    assert reads == [(None, 101)] * 4
    assert all(group["injected_memories"] > 0 and not group["retrieval_failures"] for group in report["groups"]), [(g["name"], g["injected_memories"], g["retrieval_failures"]) for g in report["groups"]]


async def test_evaluation_repository_filters_before_limit_and_includes_requested_history():
    repo = evaluate_cahm._CaseRepository([dict(id=1, status="superseded"), dict(id=2, status="active")])
    assert [row["id"] for row in await repo.list_memory_records("c", None, limit=1)] == [2]
    assert [row["id"] for row in await repo.list_memory_records("c", None, limit=None, include_inactive=True)] == [1, 2]
