"""Complete recorded epistemic subtask answers must survive output validation."""

import json
from html import unescape
from pathlib import Path

import pytest

from character.output_guard import (
    FACTUAL_TASK_STYLE_DRIFT,
    IGNORED_ADVICE_BOUNDARY,
    UNSUPPORTED_FACTUAL_CLAIM,
    ReplyGuard,
    validate_reply,
)
from inference import token_counting
from inference.context_budget import estimated_tokens
from inference.generation_request import GenerationRequest, generate_character_response

CASE = json.loads((Path(__file__).parent / "fixtures/deepseek_epistemic_guard_case.json").read_text())
FIRST_REPLY = CASE["first_response"]["choices"][0]["message"]["content"]
FACTUAL_PREFIX = FIRST_REPLY.rsplit("\n\n", 1)[0]


@pytest.fixture(autouse=True)
def official_pro(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "openai_compat")
    monkeypatch.setenv("OPENAI_COMPAT_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "https://api.deepseek.com")
    token_counting._deepseek_tokenizer.cache_clear()
    token_counting._deepseek_text_tokens.cache_clear()
    yield
    token_counting._deepseek_tokenizer.cache_clear()
    token_counting._deepseek_text_tokens.cache_clear()


def request(guard=None, mode="lightweight"):
    # This is complete supplied recorded input to an injected generator, not
    # fake semantic MemoryItems or a native retrieval qualification claim.
    messages = CASE["first_request"]["messages"]
    return GenerationRequest(
        message=messages[-1]["content"],
        persona_prompt=messages[0]["content"],
        history=messages[1:-1],
        apply_prompt_policy=False,
        context_window_tokens=65536,
        max_tokens=2048,
        reply_guard=guard or ReplyGuard(factual_task=True),
        reply_guard_mode=mode,
    )


def assert_complete_input(kwargs):
    text = unescape(unescape(kwargs["messages"][-1]["content"]))
    assert CASE["question"] in text and CASE["source_message"] in text
    assert all(d["content"] in text for d in CASE["documents"])
    assert sum(estimated_tokens(m["content"]) + 4 for m in kwargs["messages"]) + 2048 + 512 <= 65536


def test_recorded_original_complete_answer_is_not_role_deflection():
    assert CASE["synthetic"] and CASE["source_request_indices"] == [12, 13]
    assert CASE["first_response"]["usage"]["prompt_tokens"] == 66907
    assert CASE["recorded_guard_fields_absent"]  # Do not fabricate old observations.
    assert "我无法回答" in FIRST_REPLY
    assert "我没有关于自己偏好的依据" in FIRST_REPLY
    assert "不会把公共说明或你的偏好当成我的选择" in FIRST_REPLY
    assert all(d["content"] in unescape(CASE["first_request"]["messages"][-1]["content"]) for d in CASE["documents"])
    assert CASE["source_message"] in unescape(CASE["first_request"]["messages"][-1]["content"])
    assert validate_reply(FIRST_REPLY, ReplyGuard(factual_task=True)) == ()


@pytest.mark.parametrize("mode", ["lightweight", "strict"])
async def test_complete_recorded_first_answer_is_returned_unchanged_once(mode):
    calls = []

    async def captured_generator(**kwargs):
        calls.append(kwargs)
        assert_complete_input(kwargs)
        return FIRST_REPLY

    result = await generate_character_response(request(mode=mode), captured_generator)
    assert len(calls) == 1 and result.model_invoked and result.response_mode == "generated"
    assert result.reply == FIRST_REPLY
    assert not result.guard_violations and not result.guard_retried and not result.guard_fallback
    assert all("【输出校验修正】" not in m["content"] for m in calls[0]["messages"])


@pytest.mark.parametrize(
    "unknown",
    [
        "至于角色本人的方案偏好，我无法回答。目前没有我的偏好记录，不把用户偏好当作角色事实，保留未知。",
        "至于角色本人的方案偏好，我当前无法准确回答。目前没有我的偏好记录，保留未知。",
        "至于角色本人的方案偏好，我不能回答。目前没有我的偏好记录，保留未知。",
        "至于角色本人的方案偏好，我不能准确回答。目前没有我的偏好记录，保留未知。",
        "至于角色本人的方案偏好，我无法处理这个缺少依据的判断。目前没有我的偏好记录，保留未知。",
        "至于角色本人的方案偏好，我当前不能处理这个缺少依据的判断。目前没有我的偏好记录，保留未知。",
    ],
    ids=[
        "cannot-answer",
        "currently-cannot-answer",
        "cannot-answer-negative",
        "cannot-answer-accurately",
        "cannot-handle",
        "currently-cannot-handle",
    ],
)
def test_unknown_subtask_preserves_independent_complete_factual_answer(unknown):
    reply = FACTUAL_PREFIX + "\n\n" + unknown
    assert "全部办理条件" in reply and "全部例外" in reply
    assert "不喜欢代办受理" in reply and "身份核验通过才使用加速受理" in reply
    assert validate_reply(reply, ReplyGuard(factual_task=True)) == ()


@pytest.mark.parametrize("position", ["before", "after"])
def test_abstention_does_not_globally_license_independent_world_deflection(position):
    deflection = "这个问题与我的虚拟世界无关，不是我所擅长的领域。"
    reply = deflection + "\n" + FIRST_REPLY if position == "before" else FIRST_REPLY + "\n" + deflection
    assert validate_reply(reply, ReplyGuard(factual_task=True)) == (FACTUAL_TASK_STYLE_DRIFT,)


def test_abstention_does_not_license_another_unsupported_factual_claim():
    reply = FIRST_REPLY + "\n第七天通常会发放奖励。"
    assert validate_reply(reply, ReplyGuard(factual_task=True, unknown_login_reward=True)) == (
        UNSUPPORTED_FACTUAL_CLAIM,
    )


async def test_real_boundary_retry_preserves_complete_epistemic_answer():
    calls = []

    async def captured_generator(**kwargs):
        calls.append(kwargs)
        assert_complete_input(kwargs)
        return FIRST_REPLY + "\n建议你去散步。" if len(calls) == 1 else FIRST_REPLY

    result = await generate_character_response(
        request(ReplyGuard(factual_task=True, forbid_advice=True)), captured_generator
    )
    assert len(calls) == 2 and result.guard_retried
    assert result.guard_violations == (IGNORED_ADVICE_BOUNDARY,)
    assert not result.guard_post_retry_violations and not result.guard_fallback
    assert result.reply == FIRST_REPLY
    assert calls[0]["messages"][-1] == calls[1]["messages"][-1]
    assert "建议你去散步。" not in str(calls[1]["messages"])
