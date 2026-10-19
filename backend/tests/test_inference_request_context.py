"""Request-owned provider/counter identity with isolated HTTP and synthetic replies.

The existing fixtures supply the real compatible
provider/HTTP adapter and real delivered-memory database/scheduler/service.
The nested-body case isolates the branch protocol; it is not a full narrative
product test. No live model, network, real credentials or semantic assessment.
"""

import asyncio
from contextvars import ContextVar
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException
from test_compiled_cloud_transport import ANSWER, QUERY
from test_compiled_cloud_transport import installed as installed
from test_memory_retry_policy import assert_saved_once, proposal
from test_memory_retry_policy import delivered_memory as delivered_memory
from tokenizers import Tokenizer

from api import generate
from db.schemas import GenerateResponse, MessageRequest
from inference import token_counting
from inference.context_budget import estimated_tokens
from inference.request_context import get_request_context, resolve_request_context, use_request_context
from inference.review_client import get_context_review_client

COUNT_TEXT = "1234567890" * 37 + "；身份核验失败且材料不全，不能补成已通过。"


@pytest.fixture(autouse=True)
def unrelated_chat_environment(monkeypatch):
    """A bound real cloud connection must override this unrelated environment."""
    monkeypatch.setenv("MODEL_PROVIDER", "vllm")
    monkeypatch.setenv("OPENAI_COMPAT_MODEL", "unrelated-environment-model")
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://local.invalid/v1")
    token_counting._deepseek_tokenizer.cache_clear()
    token_counting._deepseek_text_tokens.cache_clear()
    try:
        yield
    finally:
        token_counting._deepseek_tokenizer.cache_clear()
        token_counting._deepseek_text_tokens.cache_clear()


def count_independently(model, text=COUNT_TEXT):
    path = (token_counting._FLASH_TOKENIZER_PATH if model == "deepseek-flash"
            else token_counting._TOKENIZER_PATH)
    return len(Tokenizer.from_file(str(path)).encode(text, add_special_tokens=False).ids)


def own_cloud_model(installed, monkeypatch, model="deepseek-flash"):
    installed.stored["openaiCompatModel"] = model
    counted = Mock(wraps=installed.provider.snapshot)
    monkeypatch.setattr(installed.provider, "snapshot", counted)
    return counted


async def receive(trace="scope", *, character=False):
    return await generate._generate_reply_impl(
        MessageRequest(
            message=QUERY, loraName="default", traceId=trace,
            **({"characterId": "tsukiyashiro_kisaki"} if character else {}),
        ),
        persist_message=False, record_invocation=False, enable_rag=False,
    )


async def test_preparation_default_review_main_answer_and_counter_share_one_snapshot(
    installed, monkeypatch,
):
    snapshots = own_cloud_model(installed, monkeypatch)
    observations = []

    def change_settings_during_first_review():
        if len(installed.calls) == 1:
            installed.stored.update(
                openaiCompatModel="deepseek-v4-pro",
                openaiCompatBaseUrl="https://api.deepseek.com/v1",
                openaiCompatApiKey="fiction-next-key",
            )
            installed.provider._refresh_db_config()

    installed.state.on_request = change_settings_during_first_review

    async def prepare(request, character_id, **kwargs):
        context = get_request_context()
        assert context is not None
        before = estimated_tokens(COUNT_TEXT)
        assert token_counting.token_counter_info()["mode"] == (
            "deepseek_v41_flash_bpe" if context.provider.model == "deepseek-flash" else "deepseek_v4_pro_bpe")
        client = await get_context_review_client()
        assert await client.generate(
            messages=[{"role": "system", "content": "仅审核完整给定资料。"},
                      {"role": "user", "content": QUERY}],
            lora_name=None, temperature=0.0, max_tokens=64, stream=False, enable_thinking=False,
        ) == ANSWER
        in_thread = await asyncio.to_thread(lambda: (get_request_context(), estimated_tokens(COUNT_TEXT)))
        assert in_thread == (context, before)
        assert estimated_tokens(COUNT_TEXT) == before
        observations.append(("prepare", context, before))
        return SimpleNamespace(character_id=character_id, history=(), compiled=None, reply_guard=None)

    compiled = generate._generate_with_retrieval

    async def compile_with_observation(*args, **kwargs):
        observations.append(("main", get_request_context(), estimated_tokens(COUNT_TEXT)))
        return await compiled(*args, **kwargs)

    monkeypatch.setattr(generate, "_prepare_character_turn", prepare)
    monkeypatch.setattr(generate, "_generate_with_retrieval", compile_with_observation)
    first = await receive("first", character=True)
    assert first.model == "deepseek-flash (openai_compat)" and first.reply == ANSWER
    assert snapshots.call_count == 1 and get_request_context() is None
    second = await receive("second", character=True)
    assert second.model == "deepseek-v4-pro (openai_compat)" and second.reply == ANSWER
    assert snapshots.call_count == 2 and get_request_context() is None
    assert len(installed.calls) == 4
    assert [call["body"]["model"] for call in installed.calls] == [
        "deepseek-flash", "deepseek-flash", "deepseek-v4-pro", "deepseek-v4-pro",
    ]
    assert [call["auth"] for call in installed.calls] == [
        "Bearer fiction-key", "Bearer fiction-key", "Bearer fiction-next-key", "Bearer fiction-next-key",
    ]
    for start, model in ((0, "deepseek-flash"), (2, "deepseek-v4-pro")):
        prepare_observation, main_observation = observations[start:start + 2]
        assert prepare_observation[0] == "prepare" and main_observation[0] == "main"
        assert prepare_observation[1] is main_observation[1]
        assert prepare_observation[2] == main_observation[2] == count_independently(model)
    assert observations[0][1] is not observations[2][1]
    for call in installed.calls:
        assert QUERY in str(call["body"]["messages"])


async def test_concurrent_requests_keep_distinct_provider_and_thread_counter_contexts(installed, monkeypatch):
    snapshots = own_cloud_model(installed, monkeypatch)
    compiled = generate._generate_with_retrieval
    entered = {name: asyncio.Event() for name in ("first", "second")}
    release = asyncio.Event()
    observations, cleaned = {}, []

    async def pause_compile(request, *args, **kwargs):
        context = get_request_context()
        observations[request.traceId] = context
        entered[request.traceId].set()
        await release.wait()
        copied, cost = await asyncio.to_thread(lambda: (get_request_context(), estimated_tokens(COUNT_TEXT)))
        assert copied is context and cost == count_independently(context.provider.model)
        return await compiled(request, *args, **kwargs)

    async def invoke(name):
        try:
            return await receive(name)
        finally:
            cleaned.append(get_request_context())

    monkeypatch.setattr(generate, "_generate_with_retrieval", pause_compile)
    first = asyncio.create_task(invoke("first"))
    second = None
    try:
        await asyncio.wait_for(entered["first"].wait(), 2)
        installed.stored["openaiCompatModel"] = "deepseek-v4-pro"
        second = asyncio.create_task(invoke("second"))
        await asyncio.wait_for(entered["second"].wait(), 2)
    finally:
        release.set()
        results = await asyncio.gather(first, *([second] if second is not None else []))
    assert [result.model for result in results] == [
        "deepseek-flash (openai_compat)", "deepseek-v4-pro (openai_compat)",
    ]
    assert observations["first"] is not observations["second"]
    assert snapshots.call_count == 2 and cleaned == [None, None]
    assert {call["body"]["model"] for call in installed.calls} == {"deepseek-flash", "deepseek-v4-pro"}
    assert get_request_context() is None


@pytest.mark.parametrize("cancel", [False, True], ids=["exception", "cancellation"])
async def test_actual_api_failure_or_cancellation_resets_its_own_task_context(installed, monkeypatch, cancel):
    snapshots = own_cloud_model(installed, monkeypatch)
    compiled = generate._generate_with_retrieval
    entered, release = asyncio.Event(), asyncio.Event()
    cleaned = []
    failure = ValueError("synthetic request-scope failure")

    async def fail_compile(*args, **kwargs):
        assert get_request_context() is not None
        entered.set()
        await release.wait()
        raise failure

    async def invoke():
        try:
            return await receive()
        finally:
            # Inspect the actual failed task after _generate_reply_impl unwinds,
            # not just the parent's independently isolated ContextVar value.
            cleaned.append(get_request_context())

    monkeypatch.setattr(generate, "_generate_with_retrieval", fail_compile)
    task = asyncio.create_task(invoke())
    try:
        await asyncio.wait_for(entered.wait(), 2)
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            release.set()
            with pytest.raises(HTTPException) as error:
                await task
            assert error.value.__cause__ is failure
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert cleaned == [None] and get_request_context() is None
    assert installed.calls == []
    monkeypatch.setattr(generate, "_generate_with_retrieval", compiled)
    assert (await receive("healthy-after-failure")).reply == ANSWER
    assert snapshots.call_count == 2 and len(installed.calls) == 1
    assert get_request_context() is None


async def test_nested_api_body_reuses_outer_snapshot_without_claiming_full_narrative_coverage(
    installed, monkeypatch,
):
    snapshots = own_cloud_model(installed, monkeypatch)
    seen = []

    async def isolated_branch_protocol(request, current_user=None, **kwargs):
        context = get_request_context()
        seen.append(context)
        if request.traceId == "outer":
            installed.stored["openaiCompatModel"] = "deepseek-v4-pro"
            installed.provider._refresh_db_config()
            return await generate._generate_reply_impl(
                request.model_copy(update={"traceId": "inner"}), current_user,
                persist_message=False, record_invocation=False, prepared_override=object(),
            )
        assert request.traceId == "inner"
        reply, _ = await context.provider.async_complete(
            [{"role": "user", "content": QUERY}], temperature=0.0, max_tokens=64, top_p=1.0,
        )
        return GenerateResponse(reply=reply, model=context.provider.model, costTime=0, modelInvoked=True)

    monkeypatch.setattr(generate, "_generate_reply_body", isolated_branch_protocol)
    result = await generate._generate_reply_impl(
        MessageRequest(message=QUERY, traceId="outer", branchId="synthetic-branch"),
        persist_message=False, record_invocation=False,
    )
    assert result.reply == ANSWER and result.model == "deepseek-flash"
    assert len(seen) == 2 and seen[0] is seen[1]
    assert snapshots.call_count == 1 and len(installed.calls) == 1
    assert installed.calls[0]["body"]["model"] == "deepseek-flash"
    assert get_request_context() is None


@pytest.mark.parametrize("idle", [False, True], ids=["immediate-worker", "idle-timer-and-worker"])
async def test_real_memory_tasks_detach_only_provider_scope_and_preserve_other_context(
    installed, delivered_memory, monkeypatch, idle,
):
    own_cloud_model(installed, monkeypatch)
    rig = delivered_memory
    rig.scheduler._idle_seconds = .01 if idle else 0
    marker = ContextVar("request_scope_test_marker", default=None)
    provider_observations, timer_observations = [], []

    async def complete(messages):
        provider_observations.append((get_request_context(), marker.get()))
        return proposal(rig.body)

    flush_after_idle = rig.scheduler._flush_after_idle

    async def observe_idle(scope_key):
        timer_observations.append((get_request_context(), marker.get()))
        return await flush_after_idle(scope_key)

    rig.completion.complete.side_effect = complete
    monkeypatch.setattr(rig.scheduler, "_flush_after_idle", observe_idle)
    context = await resolve_request_context(installed.manager)
    token = marker.set("keep-this-context")
    try:
        with use_request_context(context):
            response = await rig.step()
            assert get_request_context() is context
    finally:
        marker.reset(token)
    await assert_saved_once(rig, response, attempts=1)
    assert provider_observations == [(None, "keep-this-context")]
    assert timer_observations == ([(None, "keep-this-context")] if idle else [])
    assert installed.calls == [] and get_request_context() is None
