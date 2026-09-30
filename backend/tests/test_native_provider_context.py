from types import SimpleNamespace

import pytest


def selected(name):
    return SimpleNamespace(_current_provider=SimpleNamespace(value=name))


def test_cloud_budget_is_independent_of_local_serving_limit_and_environment_is_unchanged(monkeypatch):
    from inference.provider_context import get_provider_context_budget

    env = dict(VLLM_MAX_MODEL_LEN="4096", OPENAI_COMPAT_CONTEXT_WINDOW_TOKENS="65536")
    before = dict(env)
    cloud = get_provider_context_budget(selected("openai_compat"), env=env)
    local = get_provider_context_budget(selected("vllm"), env=env)
    assert cloud.window_tokens == 65536
    assert (cloud.history_limit, cloud.history_max_chars, cloud.source_max_chars) == (128, 65536, 16384)
    assert cloud.review.window_tokens == 65536 and cloud.review.history_messages == 256
    assert local.window_tokens == 4096 and local.review is None
    assert (local.history_limit, local.history_max_chars, local.source_max_chars) == (24, 16000, 2400)
    assert env == before


def test_cloud_without_explicit_budget_does_not_inherit_large_local_window():
    from inference.provider_context import get_provider_context_budget

    cloud = get_provider_context_budget(selected("openai_compat"), env={"VLLM_MAX_MODEL_LEN": "131072"})
    assert cloud.window_tokens == 8192


@pytest.mark.parametrize("value", ["0", "4096", "1000001", "not-a-number", "65536.5"])
def test_invalid_cloud_budget_is_not_silently_expanded(value):
    from inference.provider_context import get_provider_context_budget

    with pytest.raises(ValueError):
        get_provider_context_budget(selected("openai_compat"), env={"OPENAI_COMPAT_CONTEXT_WINDOW_TOKENS": value})


def test_native_service_propagates_actual_cloud_budget_to_all_reviewers_and_history(monkeypatch):
    from inference import model_manager
    from services.character_context import build_character_context_service

    monkeypatch.setattr(model_manager, "get_model_manager", lambda: selected("openai_compat"))
    monkeypatch.setenv("OPENAI_COMPAT_CONTEXT_WINDOW_TOKENS", "65536")
    for key in [
        "DYNAMIC_CONTEXT_SEMANTIC_REVIEW_ENABLED",
        "CONTEXTUAL_MEMORY_SELECTION_ENABLED",
        "CONTEXTUAL_DECISION_POLICY_ENABLED",
    ]:
        monkeypatch.setenv(key, "true")
    service = build_character_context_service(object())
    assert service._history_limit == 128 and service._history_max_chars == 65536
    budgets = [
        service._semantic_estimator._context_budget,
        service._memory_selector.context_budget,
        service._contextual_policy.context_budget,
    ]
    assert all(b.window_tokens == 65536 and b.history_messages == 256 for b in budgets)


@pytest.mark.asyncio
async def test_native_api_compiler_keeps_complete_cloud_message_and_three_large_prior_turns(monkeypatch):
    from api import generate as api
    from db.schemas import MessageRequest
    from inference import model_manager
    from inference.generation_request import build_generation_request

    monkeypatch.setattr(model_manager, "get_model_manager", lambda: selected("openai_compat"))
    monkeypatch.setenv("OPENAI_COMPAT_CONTEXT_WINDOW_TOKENS", "65536")
    monkeypatch.setenv("VLLM_MAX_MODEL_LEN", "8192")
    query = "完整当前资料。" * 900 + "末尾禁止把虚构经历当作用户真实资料。"
    history = [{"role": "user", "content": f"完整前提{i}。" * 1000} for i in range(3)]
    observed = []

    async def observe(request, generate):
        plan = build_generation_request(request)
        observed.append((request, plan))
        return SimpleNamespace(reply="保留完整内容", plan=plan, model_invoked=True, response_mode="generated")

    monkeypatch.setattr(api, "generate_character_response", observe)
    result = await api._generate_with_retrieval(
        MessageRequest(message=query, history=[]),
        "default",
        runtime_config={"maxTokens": 1024},
        enable_rag=False,
        prepared_character_turn=SimpleNamespace(history=history, compiled=None, reply_guard=None),
        model_generate=object(),
    )
    request, plan = observed[0]
    assert result[0] == "保留完整内容" and request.context_window_tokens == 65536
    assert query in plan.messages[-1]["content"]
    assert all(item in plan.messages for item in history)


@pytest.mark.asyncio
async def test_cloud_window_does_not_remove_generation_overflow_guard(monkeypatch):
    from inference.generation_request import GenerationRequest, build_generation_request
    from inference.provider_context import get_provider_context_budget

    budget = get_provider_context_budget(selected("openai_compat"), env={"OPENAI_COMPAT_CONTEXT_WINDOW_TOKENS": "8192"})
    with pytest.raises(ValueError, match="serving context budget"):
        build_generation_request(
            GenerationRequest(message="完整材料。" * 2000, context_window_tokens=budget.window_tokens)
        )
