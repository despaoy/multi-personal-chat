"""Complete dialogue batches, explicit failure and exact generation counts.

The real ASGI route and batch parser run. Research is an explicit valid empty
result and the model returns complete synthetic messages, never paid output.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from app.runtime import RuntimeContainer
from db.database import SQLiteDB
from provider_fixtures import cloud_connection
from fastapi import FastAPI, HTTPException

from api import training
from app.dependencies import get_current_admin
from inference import model_manager
from inference.openai_protocol import ProviderResponseError

DESCRIPTION = "完全虚构角色青汐研究员；保留预约条件，不声称已经办理。"
MARKER = "SYNTHETIC_DIALOGUE_OUTPUT_PRIVATE_MARKER"
ENDPOINT = "/api/training/generate-dialogues"
# Independent expected current 80-group distribution and six-group batch cap.
PLAN_80 = ((4, 1), (6, 2), (2, 2), (6, 3), (6, 3), (6, 3),
           (6, 4), (6, 4), (6, 4), (6, 5), (6, 5), (2, 5),
           (6, 6), (4, 6), (6, 7), (2, 8))


def complete_prompt_batch(prompt, ordinal):
    """Check the actual complete JSON example before using it as test input."""
    header = re.search(r"请生成(\d+)组对话数据，每组恰好(\d+)轮对话。", prompt)
    assert header is not None
    count, turns = map(int, header.groups())
    section = prompt.split("【输出格式】", 1)[1]
    start = section.index("[")
    examples, _end = json.JSONDecoder().raw_decode(section[start:])
    assert isinstance(examples, list) and len(examples) == count
    for example in examples:
        assert isinstance(example, dict) and set(example) == {"conversations"}
        messages = example["conversations"]
        assert isinstance(messages, list) and len(messages) == 2 * turns
        for index, message in enumerate(messages):
            assert isinstance(message, dict) and set(message) == {"role", "content"}
            assert message["role"] == ("user" if index % 2 == 0 else "assistant")
            assert isinstance(message["content"], str) and message["content"].strip()
    batch = deepcopy(examples)
    for item_index, item in enumerate(batch):
        for message_index, message in enumerate(item["conversations"]):
            message["content"] = (
                f"  第{ordinal}批第{item_index + 1}组第{message_index // 2 + 1}轮{message['role']}："
                f"仅供合成，必须预约；不能免除原件。{MARKER}\n  "
            )
    return count, turns, batch


def mapped_messages(batch):
    return [[{"from": "human" if message["role"] == "user" else "gpt", "value": message["content"]}
             for message in item["conversations"]] for item in batch]


@pytest.fixture
def rig(monkeypatch, tmp_path):
    state = dict(is_generating=False, cancel_requested=False, progress=0, total=0,
                 batch_num=0, total_batches=0, generated_count=0, new_dialogues=[],
                 all_generated_dialogues=[], started_at=None)
    monkeypatch.setattr(training, "generation_state", state)
    search = AsyncMock(return_value="")
    monkeypatch.setattr(training, "search_character_info", search)
    context = SimpleNamespace(state=state, search=search, batches=[], requested=[], defect=None, fail_at=1)

    async def model(**kwargs):
        ordinal = len(context.requested) + 1
        count, turns, batch = complete_prompt_batch(kwargs["prompt"], ordinal)
        assert kwargs["session_history"] == []
        assert kwargs["max_tokens_override"] == (2048 if turns <= 2 else 4096 if turns <= 5 else 6144)
        context.requested.append((count, turns))
        context.batches.append(deepcopy(batch))
        if context.defect and ordinal == context.fail_at:
            if context.defect == "provider_rejection":
                raise ProviderResponseError(context.upstream_status)
            if context.defect == "provider_exception":
                raise RuntimeError("SYNTHETIC_UPSTREAM_PRIVATE_FAILURE")
            if context.defect == "provider_http_error":
                raise HTTPException(status_code=503, detail="合成模型服务不可用")
            if context.defect == "extra_item":
                batch.append(deepcopy(batch[0]))
            elif context.defect == "extra_turn":
                batch[0]["conversations"].extend(deepcopy(batch[0]["conversations"][:2]))
            elif context.defect == "missing_message":
                batch[0]["conversations"].pop()
            elif context.defect == "wrong_role":
                batch[0]["conversations"][1]["role"] = "user"
            elif context.defect == "non_string_content":
                batch[0]["conversations"][0]["content"] = {"private": MARKER}
            elif context.defect == "non_text":
                return batch, 0.25
            raw = json.dumps(batch, ensure_ascii=False)
            if context.defect == "wrapped":
                raw = "解释：" + MARKER + "\n" + raw + "\n结束解释"
            elif context.defect == "fenced":
                raw = "```json\n" + raw + "\n```"
            return raw, 0.25
        return json.dumps(batch, ensure_ascii=False), 0.25

    manager = SimpleNamespace(
        _current_provider=SimpleNamespace(value="openai_compat"),
        async_generate=AsyncMock(side_effect=model),
        generate=Mock(side_effect=AssertionError("Synchronous model generation is forbidden")),
    )
    provider = cloud_connection("fixture-dialogue-model")
    provider.async_generate = manager.async_generate
    manager.get_current_provider = lambda: provider
    monkeypatch.setattr(model_manager, "get_model_manager", lambda: manager)
    context.manager = manager
    app = FastAPI()
    app.state.runtime_container = RuntimeContainer(
        db=SQLiteDB(tmp_path / "dialogue-accounting.sqlite"), is_pg_mode=lambda: False)
    app.include_router(training.router)
    app.dependency_overrides[get_current_admin] = lambda: {"id": 901, "role": "admin"}
    context.app = app
    return context


async def post(rig, count=1, **changes):
    payload = dict(character_description=DESCRIPTION, num_dialogues=count, style="合成谨慎风格")
    payload.update(changes)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=rig.app), base_url="http://fixture.invalid") as client:
        return await client.post(ENDPOINT, json=payload)


def assert_released(rig):
    assert rig.state["is_generating"] is False and rig.state["cancel_requested"] is False
    rig.manager.generate.assert_not_called()


async def test_eighty_dialogues_finish_exact_distribution_with_complete_prompt_examples_and_content(rig):
    response = await post(rig, 80)
    assert response.status_code == 200
    data = response.json()
    assert tuple(rig.requested) == PLAN_80
    assert rig.manager.async_generate.await_count == len(PLAN_80)
    rig.search.assert_awaited_once_with(DESCRIPTION)
    assert data["success"] is True and data["cancelled"] is False and data["total"] == 80
    assert data["cost_time"] == 4.0
    expected = [messages for batch in rig.batches for messages in mapped_messages(batch)]
    assert [item["conversations"] for item in data["dialogues"]] == expected
    assert Counter(item["turns"] for item in data["dialogues"]) == {1: 4, 2: 8, 3: 18, 4: 18, 5: 14, 6: 10, 7: 6, 8: 2}
    assert all(len(item["conversations"]) == 2 * item["turns"] for item in data["dialogues"])
    assert rig.state["all_generated_dialogues"] == data["dialogues"]
    assert rig.state["generated_count"] == 80 and rig.state["progress"] == 100
    assert rig.state["batch_num"] == rig.state["total_batches"] == 16
    assert_released(rig)


async def test_second_bad_batch_preserves_first_complete_dialogue_without_quality_retries(rig, caplog):
    rig.defect, rig.fail_at = "missing_message", 2
    response = await post(rig, 2)
    assert response.status_code == 502
    detail = response.json()["detail"]
    partial = detail.pop("partial_result")
    assert detail == dict(
        error="dialogue_batch_invalid", stage="dialogue_validation", reason="wrong_turn_count",
        batch=2, expected_count=1, target_turns=2, completed_count=1, requested_count=2,
        message="模型返回的对话批次不符合约定，请检查模型服务及输出。",
    )
    assert rig.requested == [(1, 1), (1, 2)] and rig.manager.async_generate.await_count == 2
    assert len(rig.state["all_generated_dialogues"]) == 1
    assert rig.state["all_generated_dialogues"][0]["conversations"] == mapped_messages(rig.batches[0])[0]
    assert rig.state["new_dialogues"] == rig.state["all_generated_dialogues"]
    assert rig.state["generated_count"] == 1 and rig.state["total"] == 2 and rig.state["progress"] == 50
    assert partial == dict(dialogues=rig.state["all_generated_dialogues"], total=1, cost_time=0.5)
    assert rig.batches[1][0]["conversations"][0]["content"] not in response.text
    assert MARKER not in caplog.text
    assert_released(rig)


@pytest.mark.parametrize("defect,status,message", [
    ("provider_exception", 500, "对话生成失败，请检查模型服务及后端日志。"),
    ("provider_http_error", 503, "合成模型服务不可用"),
])
async def test_provider_failure_returns_completed_batches_without_polling_or_retry(rig, caplog, defect, status, message):
    rig.defect, rig.fail_at = defect, 2
    response = await post(rig, 2)
    assert response.status_code == status
    assert response.json()["detail"] == dict(
        message=message,
        partial_result=dict(dialogues=rig.state["all_generated_dialogues"], total=1, cost_time=0.25),
    )
    assert len(rig.state["all_generated_dialogues"]) == 1
    assert rig.state["all_generated_dialogues"][0]["conversations"] == mapped_messages(rig.batches[0])[0]
    assert rig.requested == [(1, 1), (1, 2)] and rig.manager.async_generate.await_count == 2
    assert "SYNTHETIC_UPSTREAM_PRIVATE_FAILURE" not in response.text + caplog.text
    assert MARKER not in caplog.text
    assert_released(rig)


async def test_two_complete_batches_still_return_success_without_partial_error(rig):
    response = await post(rig, 2)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True and data["cancelled"] is False
    assert data["total"] == 2 and data["cost_time"] == 0.5
    assert [item["conversations"] for item in data["dialogues"]] == [
        mapped_messages(batch)[0] for batch in rig.batches
    ]
    assert rig.requested == [(1, 1), (1, 2)]
    assert_released(rig)


@pytest.mark.parametrize("upstream,fail_at,http_status,kind", [
    (401, 1, 502, "provider_auth_error"),
    (402, 2, 503, "provider_balance_error"),
])
async def test_model_rejection_keeps_provider_reason_and_any_completed_batches(rig, upstream, fail_at, http_status, kind):
    rig.defect, rig.fail_at, rig.upstream_status = "provider_rejection", fail_at, upstream
    response = await post(rig, 2)
    assert response.status_code == http_status
    detail = response.json()["detail"]
    assert detail["kind"] == kind and detail["upstreamStatus"] == upstream
    assert "密钥" in detail["message"] if upstream == 401 else "余额" in detail["message"]
    if fail_at == 1:
        assert "partial_result" not in detail
    else:
        assert detail["partial_result"] == dict(
            dialogues=rig.state["all_generated_dialogues"], total=1, cost_time=0.25,
        )
        assert len(rig.state["all_generated_dialogues"]) == 1
        assert rig.state["all_generated_dialogues"][0]["conversations"] == mapped_messages(rig.batches[0])[0]
    assert rig.manager.async_generate.await_count == fail_at
    assert_released(rig)


@pytest.mark.parametrize("defect,reason", [
    ("wrapped", "invalid_json"), ("fenced", "invalid_json"),
    ("extra_item", "wrong_batch_size"), ("extra_turn", "wrong_turn_count"),
    ("wrong_role", "wrong_role_order"), ("non_string_content", "invalid_content"),
    ("non_text", "non_text"),
])
async def test_invalid_output_fails_whole_batch_without_repair_filtering_truncation_or_retry(rig, caplog, defect, reason):
    rig.defect = defect
    response = await post(rig)
    assert response.status_code == 502
    assert response.json()["detail"] == dict(
        error="dialogue_batch_invalid", stage="dialogue_validation", reason=reason,
        batch=1, expected_count=1, target_turns=1, completed_count=0, requested_count=1,
        message="模型返回的对话批次不符合约定，请检查模型服务及输出。",
    )
    rig.manager.async_generate.assert_awaited_once()
    assert rig.state["all_generated_dialogues"] == rig.state["new_dialogues"] == []
    assert rig.state["generated_count"] == 0 and rig.state["progress"] == 0
    assert MARKER not in response.text and MARKER not in caplog.text
    assert_released(rig)


@pytest.mark.parametrize("changes", [{"num_dialogues": 0}, {"character_description": " \n\t "}])
async def test_invalid_request_is_rejected_before_research_model_or_generation_admission(rig, changes):
    before = deepcopy(rig.state)
    response = await post(rig, **changes)
    assert response.status_code == 422
    rig.search.assert_not_awaited()
    rig.manager.async_generate.assert_not_awaited()
    assert rig.state == before
    assert_released(rig)
