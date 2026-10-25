"""Independent complete protocols for evidence dependency admission and routing."""

from services import chat_answer

import copy
import json
from types import SimpleNamespace

import pytest

from character.models import CompiledCharacterContext
from knowledge.retrieval_query_plan import QueryPlanReviewError, RetrievalQueryPlan, plan_retrieval
from knowledge.turn_dependencies import KINDS, parse_dependencies, query_segments

QUERY = "请读取我保存的练琴偏好。下面是完整独立假设，规则是否适用？只读取，不变更记忆。"


def groups():
    return dict(private_memory=[0], current_input=[1], public_knowledge=[], control=[2], unresolved_source=[])


def baseline():
    parsed = parse_dependencies(groups(), QUERY)
    assert parsed.private_context_only and "".join(parsed.segments) == QUERY
    return groups()


def test_full_private_input_and_controls_do_not_certify_saved_facts():
    assert baseline() and len(query_segments(QUERY)) == 3


@pytest.mark.parametrize(
    "defect",
    [
        "missing",
        "duplicate",
        "bool",
        "out_of_range",
        "negative",
        "extra_key",
        "not_list",
    ],
)
def test_incomplete_or_privileged_plans_cannot_suppress_public_retrieval(defect):
    value = copy.deepcopy(baseline())
    if defect == "missing":
        value["current_input"] = []
    elif defect == "duplicate":
        value["private_memory"] = [0, 0]
    elif defect == "bool":
        value["private_memory"] = [False]
    elif defect == "out_of_range":
        value["private_memory"] = [3]
    elif defect == "negative":
        value["private_memory"] = [-1]
    elif defect == "extra_key":
        value["source_grants"] = ["other-user"]
    elif defect == "not_list":
        value["private_memory"] = "0"
    with pytest.raises(ValueError):
        parse_dependencies(value, QUERY)


@pytest.mark.parametrize("kind", ["public_knowledge", "unresolved_source"])
def test_an_extra_external_or_unresolved_task_preserves_retrieval(kind):
    value = baseline()
    value["current_input"] = []
    value[kind] = [1]
    assert not parse_dependencies(value, QUERY).private_context_only


def test_one_sentence_can_depend_on_private_and_public_evidence():
    value = baseline()
    value["public_knowledge"] = [0]
    assert not parse_dependencies(value, QUERY).private_context_only


@pytest.mark.asyncio
async def test_legacy_empty_search_views_are_rejected_as_incomplete(monkeypatch):
    monkeypatch.setenv("RAG_TASK_PLANNER_ENABLED", "true")

    async def review(messages):
        return '{"search_views":[]}'

    with pytest.raises(QueryPlanReviewError, match="query_plan_invalid"):
        await plan_retrieval(QUERY + "仅讨论给定假设。" * 90, reviewer=review)


@pytest.mark.asyncio
async def test_every_full_original_segment_reaches_planner(monkeypatch):
    monkeypatch.setenv("RAG_TASK_PLANNER_ENABLED", "true")
    query = QUERY + "仅讨论给定假设。" * 80

    async def review(messages):
        payload = json.loads(messages[-1]["content"])
        assert payload["query"] == query and "".join(x["text"] for x in payload["segments"]) == query
        value = {kind: [] for kind in KINDS}
        value["private_memory"] = [0]
        value["current_input"] = list(range(1, len(payload["segments"])))
        return json.dumps(dict(dependencies=value, public_partitions=[]))

    result = await plan_retrieval(query, reviewer=review)
    assert result.status == "applied" and result.private_context_only


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["private", "mixed", "unknown", "missing_context", "branch", "legacy"])
async def test_actual_route_uses_dependencies_not_presence_of_one_memory(monkeypatch, mode, provider_budget):
    from db.schemas import MessageRequest
    from knowledge import public_question_binding, retrieval_query_plan

    value = baseline()
    if mode == "mixed":
        value["public_knowledge"] = [0]
    if mode == "unknown":
        value["private_memory"] = []
        value["unresolved_source"] = [0]
    dep = None if mode == "legacy" else parse_dependencies(value, QUERY)
    calls = []

    async def planner(query, *, history, context_budget):
        assert query == QUERY and history == ()
        assert context_budget is provider_budget.review
        return RetrievalQueryPlan(status="applied", dependencies=dep)

    async def retrieve(*args, **kwargs):
        calls.append("public")
        return dict(results=[], citations=[], confidence=0.0, abstained=True)

    async def model(**kwargs):
        calls.append("main")
        assert kwargs["max_tokens"] == 2048
        return "当前没有找到可核对的私人记录；外部事实也不能从假设推断。"

    compiled = CompiledCharacterContext("", "", "", branch_context="fiction" if mode == "branch" else "")
    prepared = None if mode == "missing_context" else SimpleNamespace(compiled=compiled, history=())
    async def no_binding(*args, **kwargs):
        # This routing-only control has no question or source approval.
        return None

    monkeypatch.setattr(public_question_binding, "resolve_question_binding", no_binding)
    monkeypatch.setattr(retrieval_query_plan, "plan_retrieval", planner)
    monkeypatch.setattr(chat_answer, 'retrieve_chat_knowledge', retrieve)
    monkeypatch.setattr(chat_answer, 'get_lora_system_prompt', lambda _: "人物规则")
    if mode == "legacy":

        with pytest.raises(chat_answer.ChatRetrievalError) as error:
            await chat_answer.generate_chat_answer(
                MessageRequest(message=QUERY), None, prepared_character_turn=prepared,
                runtime_config={"maxTokens": 2048, "useKnowledgeBase": True}, model_generate=model,
            )
        assert error.value.http_status == 503 and not calls
        return
    _chat_result_meta = await chat_answer.generate_chat_answer(
        MessageRequest(message=QUERY),
        None,
        prepared_character_turn=prepared,
        runtime_config={"maxTokens": 2048, "useKnowledgeBase": True},
        model_generate=model,
    )
    reply, used, meta = _chat_result_meta.reply, _chat_result_meta.used_rag, _chat_result_meta.metadata
    assert "没有找到" in reply and calls.count("main") == 1
    public_expected = mode in {"mixed", "unknown"}
    assert ("public" in calls) is public_expected
    if not public_expected:
        assert meta["answerMode"] == "personal_context" and not meta["abstained"]
    else:
        assert meta["abstained"]


@pytest.mark.parametrize("kind", ["control", "unresolved_source"])
def test_overlapping_sentence_preserves_controls_and_unresolved_dependencies(kind):
    value = baseline()
    value[kind].append(0)
    plan = parse_dependencies(value, QUERY)
    assert dict(plan.groups)[kind][-1] == 0
    assert plan.private_context_only == (kind == "control")


@pytest.mark.parametrize("indices", [[], [0]])
def test_unknown_dependency_label_is_rejected_instead_of_rewritten(indices):
    value = baseline()
    value["unknown"] = indices
    value.pop("unresolved_source")
    with pytest.raises(ValueError, match="Unexpected dependency groups"):
        parse_dependencies(value, QUERY)


@pytest.fixture(params=["vllm", "openai_compat"], ids=["local-budget", "cloud-budget"])
def provider_budget(monkeypatch, request):
    from types import SimpleNamespace
    from inference.provider_context import get_provider_context_budget

    manager = SimpleNamespace(_current_provider=SimpleNamespace(value=request.param))
    budget = get_provider_context_budget(manager, env={
        "VLLM_MAX_MODEL_LEN": "8192", "OPENAI_COMPAT_CONTEXT_WINDOW_TOKENS": "8192",
    })
    assert budget.window_tokens > 0
    if request.param == "openai_compat":
        assert budget.review.window_tokens > 0
    else:
        assert budget.review is None
    monkeypatch.setattr(chat_answer, "get_provider_context_budget", lambda: budget)
    return budget
