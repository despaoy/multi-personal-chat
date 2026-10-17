"""Sample generation must use evidence and complete before publishing."""

from __future__ import annotations

import asyncio
import json
import re
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from db import adapter
from knowledge import intent_trainer as trainer


class SyntheticGenerator:
    def __init__(self, *, failure_call=None, reply=None):
        self.calls = []
        self.failure_call = failure_call
        self.reply = reply

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == self.failure_call:
            raise OSError("synthetic-private-provider-detail")
        if self.reply is not None:
            return self.reply
        return "\n".join(f"合成问题{len(self.calls):02d}-{i:02d}" for i in range(12))


@pytest.fixture
def rig(monkeypatch, tmp_path):
    bases = [{"id": 1, "name": "合成角色"}, {"id": 2, "name": "合成装备"}]
    docs = {
        1: [{"title": "角色事实", "content": "测试角色于上午八点到达；她不携带红色物品。"}],
        2: [{"title": "装备事实", "content": "测试装备重两公斤，仅在下雨时使用，不属于测试角色。"}],
    }
    load_documents = Mock(side_effect=lambda *, knowledge_base_id, limit: docs[knowledge_base_id][:limit])
    database = SimpleNamespace(get_knowledge_bases=Mock(return_value=bases), get_knowledge_documents=load_documents)
    monkeypatch.setattr(adapter, "db", database)
    monkeypatch.setattr(trainer, "SAMPLES_DIR", tmp_path)
    monkeypatch.setattr(trainer, "_generation_status", {"running": False, "stage": "", "message": "", "progress": 0})
    monkeypatch.setattr(trainer, "_cancel_flag", threading.Event())
    monkeypatch.setattr(trainer, "_training_logs", [])
    model = SyntheticGenerator()
    get_client = AsyncMock(return_value=model)
    monkeypatch.setattr(trainer, "_get_vllm_client", get_client)
    path = tmp_path / "samples.json"
    original = {"existing": ["已经审查过的合成问题"]}
    path.write_text(json.dumps(original, ensure_ascii=False), encoding="utf-8")
    return SimpleNamespace(
        bases=bases, docs=docs, db=database, model=model, get_client=get_client, path=path, original=original
    )


def run_generation(*, ids=None, positives=4, negatives=5):
    return asyncio.run(trainer.generate_samples([1, 2] if ids is None else ids, positives, negatives, "synthetic-lora"))


def assert_failed_without_publication(rig):
    assert trainer.get_generation_status()["stage"] == "error"
    assert trainer.get_generation_status()["running"] is False
    assert json.loads(rig.path.read_text(encoding="utf-8")) == rig.original


def test_complete_generation_uses_source_and_publishes_exact_requested_counts(rig):
    result = run_generation()
    assert result == {"success": True, "total": 13, "stats": {"合成角色": 4, "合成装备": 4, "none": 5}}
    saved = json.loads(rig.path.read_text(encoding="utf-8"))
    assert {key: len(value) for key, value in saved.items()} == result["stats"]
    assert len(rig.model.calls) == 12
    assert rig.db.get_knowledge_documents.call_args_list[0].kwargs == {"knowledge_base_id": 1, "limit": 30}
    for call in rig.model.calls[:3]:
        assert rig.docs[1][0]["content"] in call["messages"][1]["content"]
    for call in rig.model.calls[3:6]:
        assert rig.docs[2][0]["content"] in call["messages"][1]["content"]
    for call in rig.model.calls:
        assert call["lora_name"] == "synthetic-lora"
        assert call["temperature"] == 0.9
        assert call["max_tokens"] == 1024
    assert trainer.get_generation_status()["stage"] == "done"


def test_zero_negatives_skips_negative_model_calls(rig):
    result = run_generation(ids=[1], positives=1, negatives=0)
    assert result["stats"] == {"合成角色": 1, "none": 0}
    assert len(rig.model.calls) == 3
    assert json.loads(rig.path.read_text(encoding="utf-8"))["none"] == []


@pytest.mark.parametrize("docs", [[], [{"title": "只有标题", "content": ""}], [{"content": " \n\t "}]])
def test_missing_source_fails_before_model_and_does_not_guess_from_name(rig, docs):
    rig.docs[1] = docs
    with pytest.raises(trainer.SampleGenerationError, match="没有可用文档内容"):
        run_generation()
    rig.get_client.assert_not_awaited()
    assert rig.model.calls == []
    assert_failed_without_publication(rig)


@pytest.mark.parametrize("bases,ids", [([], [1]), ([{"id": 1, "name": "合成角色"}], [1, 2])])
def test_missing_selected_knowledge_base_does_not_generate_partial_selection(rig, bases, ids):
    rig.db.get_knowledge_bases.return_value = bases
    with pytest.raises(trainer.SampleGenerationError):
        run_generation(ids=ids)
    rig.get_client.assert_not_awaited()
    rig.db.get_knowledge_documents.assert_not_called()
    assert_failed_without_publication(rig)


@pytest.mark.parametrize("component", ["database", "documents", "client"])
def test_dependency_error_retains_cause_and_redacts_private_detail(rig, component, caplog):
    error = OSError("synthetic-private-dependency-detail")
    if component == "database":
        rig.db.get_knowledge_bases.side_effect = error
    elif component == "documents":
        rig.db.get_knowledge_documents.side_effect = error
    else:
        rig.get_client.side_effect = error
    with pytest.raises(trainer.SampleGenerationError) as caught:
        run_generation()
    assert caught.value.__cause__ is error
    assert "synthetic-private" not in str(caught.value)
    assert "synthetic-private" not in trainer.get_generation_status()["message"]
    assert "synthetic-private" not in caplog.text
    assert rig.model.calls == []
    assert_failed_without_publication(rig)


@pytest.mark.parametrize("failure_call,stage", [(1, "generate_kb"), (4, "generate_neg"), (8, "generate_hard_neg")])
def test_model_batch_failure_stops_without_retry_or_template_replacement(rig, failure_call, stage, caplog):
    rig.model.failure_call = failure_call
    with pytest.raises(trainer.SampleGenerationError) as caught:
        run_generation(ids=[1], positives=3, negatives=4)
    assert len(rig.model.calls) == failure_call
    assert isinstance(caught.value.__cause__, OSError)
    assert stage in str(caught.value)
    assert "synthetic-private" not in str(caught.value) + caplog.text
    assert_failed_without_publication(rig)


@pytest.mark.parametrize("reply", [None, "", "   ", {"text": "结构错误"}, 123])
def test_empty_or_nontext_output_is_explicit_failure(rig, reply):
    rig.model.generate = AsyncMock(return_value=reply)
    with pytest.raises(trainer.SampleGenerationError, match="非空文本"):
        run_generation()
    assert rig.model.generate.await_count == 1
    assert_failed_without_publication(rig)


def test_duplicate_samples_cannot_satisfy_requested_count(rig):
    rig.model.reply = "重复的合成问题\n重复的合成问题"
    with pytest.raises(trainer.SampleGenerationError, match="有效去重样本不足"):
        run_generation(positives=4)
    assert len(rig.model.calls) == 3
    assert_failed_without_publication(rig)


def test_combined_negative_pool_must_meet_count_after_deduplication(rig, monkeypatch):
    monkeypatch.setattr(trainer, "_generate_negative_samples", AsyncMock(return_value=["相同负例甲", "相同负例乙"]))
    monkeypatch.setattr(
        trainer, "_generate_hard_negatives_per_kb", AsyncMock(return_value={"合成角色": ["相同负例甲", "相同负例乙"]})
    )
    with pytest.raises(trainer.SampleGenerationError, match="有效去重样本不足"):
        run_generation(ids=[1], negatives=4)
    assert_failed_without_publication(rig)


def test_save_failure_never_reports_done(rig, monkeypatch):
    error = OSError("synthetic-private-write-detail")
    monkeypatch.setattr(trainer, "save_samples", Mock(side_effect=error))
    with pytest.raises(trainer.SampleGenerationError) as caught:
        run_generation()
    assert caught.value.__cause__ is error
    assert "save" in str(caught.value)
    assert_failed_without_publication(rig)


def test_async_cancellation_preserves_cancellation_and_original_samples(rig):
    rig.model.generate = AsyncMock(side_effect=asyncio.CancelledError)
    with pytest.raises(asyncio.CancelledError):
        run_generation()
    assert trainer.get_generation_status()["stage"] == "cancelled"
    assert trainer.get_generation_status()["running"] is False
    assert json.loads(rig.path.read_text(encoding="utf-8")) == rig.original


def test_cooperative_cancellation_stops_before_next_batch(rig):
    original_generate = rig.model.generate

    async def cancel_after_reply(**kwargs):
        reply = await original_generate(**kwargs)
        trainer._cancel_flag.set()
        return reply

    rig.model.generate = cancel_after_reply
    assert run_generation() == {"error": "已取消"}
    assert len(rig.model.calls) == 1
    assert trainer.get_generation_status()["stage"] == "cancelled"
    assert json.loads(rig.path.read_text(encoding="utf-8")) == rig.original


def test_sample_selection_preserves_model_order_and_exact_count():
    assert trainer._take_samples(["问题丙", "问题甲", "问题丙", "问题乙"], 2) == ["问题丙", "问题甲"]


def test_disabled_vllm_is_not_an_empty_client(monkeypatch):
    from app import config
    from inference import vllm_client

    monkeypatch.setattr(config, "is_vllm_enabled", lambda: False)
    get_client = AsyncMock()
    monkeypatch.setattr(vllm_client, "get_vllm_client", get_client)
    with pytest.raises(trainer.SampleGenerationError, match="启用并配置 vLLM"):
        asyncio.run(trainer._get_vllm_client())
    get_client.assert_not_awaited()


def test_vllm_initialization_error_is_not_replaced_with_none(monkeypatch):
    from app import config
    from inference import vllm_client

    monkeypatch.setattr(config, "is_vllm_enabled", lambda: True)
    error = OSError("synthetic-private-client-detail")
    monkeypatch.setattr(vllm_client, "get_vllm_client", AsyncMock(side_effect=error))
    with pytest.raises(OSError) as caught:
        asyncio.run(trainer._get_vllm_client())
    assert caught.value is error


def test_nondivisible_requested_count_is_budgeted_without_automatic_shortfall(rig):
    async def exact_batch(**kwargs):
        rig.model.calls.append(kwargs)
        count = int(re.search(r"生成(\d+)条", kwargs["messages"][1]["content"]).group(1))
        return "\n".join(f"合成问题{len(rig.model.calls):02d}-{i:03d}" for i in range(count))

    rig.model.generate = exact_batch
    result = run_generation(ids=[1], positives=100, negatives=0)
    assert result["stats"] == {"合成角色": 100, "none": 0}
    assert len(rig.model.calls) == 3
    assert all("生成34条" in call["messages"][1]["content"] for call in rig.model.calls)


def test_cancel_signal_cannot_relabel_provider_error_as_successful_cancellation(rig):
    error = OSError("synthetic-private-concurrent-error")

    async def fail_after_cancel(**kwargs):
        trainer._cancel_flag.set()
        raise error

    rig.model.generate = fail_after_cancel
    with pytest.raises(trainer.SampleGenerationError) as caught:
        run_generation()
    assert caught.value.__cause__ is error
    assert_failed_without_publication(rig)
