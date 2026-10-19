"""Whole grounded evidence uses the selected model's actual request budget."""

from types import SimpleNamespace

import pytest

from inference.context_budget import ReviewContextBudget
from inference.provider_context import ProviderContextBudget
from knowledge.grounded_answer.cache import AnswerCache
from knowledge.grounded_answer.packet import EvidencePacketBuilder
from knowledge.grounded_answer.prompt import GroundedPromptBuilder
from knowledge.grounded_answer.service import GroundedAnswerService
from test_grounded_answer import _bundle

pytestmark = pytest.mark.usefixtures("literal_grounded_query")


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("fits", [False, True])
async def test_whole_evidence_history_and_output_share_one_model_window(monkeypatch, streaming, fits):
    # An explicit deterministic counter isolates admission from tokenizer assets.
    monkeypatch.setattr("inference.context_budget.estimated_tokens", len)
    data = _bundle()
    body = "甲" * 7000 + "；但申请尚未批准，周末不办理。"
    data["results"][0]["content"] = body
    data["results"][0]["summary"] = ""
    calls = []

    async def generate(**kwargs):
        calls.append(kwargs)
        assert body in kwargs["messages"][-1]["content"]
        assert kwargs["messages"][1]["content"] == "完整历史"
        return "申请尚未批准，周末不办理。[S1]"

    service = GroundedAnswerService(
        retriever=lambda *args, **kwargs: data, cache=AnswerCache(ttl_seconds=0),
        packet_builder=EvidencePacketBuilder(None),
        prompt_builder=GroundedPromptBuilder(None, context_budget=ReviewContextBudget(16384)),
    )
    options = dict(generate=generate, domain_id="story", history=[{"role": "user", "content": "完整历史"}],
                   max_tokens=512 if fits else 16384)
    events = []

    async def run():
        if streaming:
            async for event in service.answer_stream("申请批准了吗？周末办理吗？", **options):
                events.append(event)
            return next(event.data["answer"] for event in events if event.type == "done")
        return (await service.answer("申请批准了吗？周末办理吗？", **options)).answer

    if fits:
        assert await run() == "申请尚未批准，周末不办理。"
        assert len(calls) == 1 and calls[0]["max_tokens"] == 512
    else:
        with pytest.raises(ValueError, match="model context window"):
            await run()
        assert not calls
        assert [event.type for event in events] == (["meta"] if streaming else [])


def test_deferred_character_limit_requires_an_actual_model_budget():
    with pytest.raises(ValueError, match="model context budget"):
        GroundedPromptBuilder(None)


async def test_default_service_rebinds_and_invalidates_cache_when_provider_budget_changes(monkeypatch):
    from knowledge.grounded_answer import service as module

    state = [ProviderContextBudget(65536, evidence_max_chars=0)]
    monkeypatch.setattr(module, "get_provider_context_budget", lambda: state[0])
    monkeypatch.setattr("knowledge.multiscale_rag.runtime.get_multiscale_rag_service",
                        lambda: SimpleNamespace(config=SimpleNamespace(domain_id="story")))
    module.reset_grounded_answer_service()
    try:
        cloud = await module.get_grounded_answer_service()
        assert cloud.packet_builder.evidence_budget is None
        assert cloud.prompt_builder.context_budget.window_tokens == 65536
        assert await module.get_grounded_answer_service() is cloud
        state[0] = ProviderContextBudget(8192, evidence_max_chars=6000)
        local = await module.get_grounded_answer_service()
        assert local is not cloud and local.cache is not cloud.cache
        assert local.packet_builder.evidence_budget == 6000
        assert local.prompt_builder.context_budget.window_tokens == 8192
    finally:
        module.reset_grounded_answer_service()
