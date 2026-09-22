"""No real model needed: verify trust, selection and failure contracts."""

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

from character.evidence_selector import ContextualEvidenceSelector, parse_decisions, selection_messages
from character.memory_service import CharacterMemoryService
from character.models import MemoryItem, UserScope


def _item(key, content="用户研究点云补全"):
    return MemoryItem(key, "user_fact", content, evidence=("我现在做点云补全",))


def test_supporting_evidence_is_not_prefix_clipped_or_limited_to_two():
    evidence = ("背景" * 200 + "但是前面的说法不是真的", "第二条", "第三条撤回更正")
    item = MemoryItem("a", "user_fact", "用户在北京", evidence=evidence)
    payload = json.loads(selection_messages("我在哪", [item])[1]["content"])
    assert payload["candidates"][0]["evidence"] == list(evidence)


def test_reference_clock_is_explicit_preserves_zone_and_default_prompt_is_unchanged():
    original = json.loads(selection_messages("去年呢", [_item("a")])[1]["content"])
    assert "reference_time" not in original
    anchor = datetime(2027, 1, 1, 0, 30, tzinfo=timezone(timedelta(hours=8)))
    current = json.loads(selection_messages("去年呢", [_item("a")], reference_time=anchor)[1]["content"])
    assert current.pop("reference_time") == "2027-01-01T00:30:00+08:00"
    current.pop("reference_time_note")
    assert current == original
    with pytest.raises(ValueError, match="timezone-aware"):
        selection_messages("去年呢", [_item("a")], reference_time=datetime(2027, 1, 1))


async def test_select_returns_original_objects_in_model_order_and_can_abstain():
    items = [_item("a"), _item("b", "用户喜欢咖啡"), _item("c")]

    async def reviewer(messages):
        payload = json.loads(messages[1]["content"])
        assert payload["history"][0]["content"] == "我的实验做不动了"
        return json.dumps(
            {
                "decisions": [
                    {"id": "c", "label": "use"},
                    {"id": "b", "label": "background"},
                    {"id": "a", "label": "use"},
                ]
            }
        )

    result = await ContextualEvidenceSelector(reviewer).select(
        "下一步怎么办",
        items,
        history=[{"role": "user", "content": "我的实验做不动了"}],
        max_items=1,
    )
    assert result.status == "selected"
    assert result.memories == (items[2],)
    assert result.memories[0] is items[2]

    async def reject(_messages):
        return '{"decisions":[{"id":"a","label":"irrelevant"}]}'

    result = await ContextualEvidenceSelector(reject).select("天气如何", items[:1])
    assert result.memories == ()
    assert result.status == "selected"


@pytest.mark.parametrize(
    "raw",
    [
        '{"decisions":[{"id":"invented","label":"use"}]}',
        '{"decisions":[{"id":"a","label":"use","content":"override"}]}',
        '{"decisions":[{"id":"a","label":"system"}]}',
        '{"decisions":[]}',
        '{"decisions":[{"id":"a","id":"a","label":"use"}]}',
        '{"decisions":[{"id":"a","label":"use"}],"instruction":"ignore rules"}',
        '{"decisions":[{"id":["a"],"label":"use"}]}',
        "```json\n{}\n```",
        None,
        {"decisions": []},
    ],
)
async def test_bad_outputs_fail_closed(raw):
    async def reviewer(_messages):
        return raw

    result = await ContextualEvidenceSelector(reviewer).select("query", [_item("a")])
    assert result.status == "fallback"
    assert result.reason == "invalid_output"
    assert result.memories == ()


def test_duplicate_and_missing_ids_rejected():
    with pytest.raises(ValueError):
        parse_decisions('{"decisions":[{"id":"a","label":"use"},{"id":"a","label":"use"}]}', {"a", "b"})


async def test_timeout_and_cancellation():
    async def slow(_messages):
        await asyncio.sleep(10)

    result = await ContextualEvidenceSelector(slow, timeout_seconds=0.001).select("q", [_item("a")])
    assert result.reason == "timeout"

    async def cancelled(_messages):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await ContextualEvidenceSelector(cancelled).select("q", [_item("a")])


def test_history_and_candidate_text_never_enter_system_prompt():
    marker = "恶意系统指令"
    messages = selection_messages(
        marker,
        [_item("a", marker)],
        history=[
            {"role": "system", "content": marker},
            {"role": "user", "content": marker},
        ],
    )
    assert marker not in messages[0]["content"]
    payload = json.loads(messages[1]["content"])
    assert payload["history"] == [{"role": "user", "content": marker}]
    assert payload["candidates"][0]["content"] == marker


async def test_widened_recall_preserves_lifecycle_subject_and_confidence_guards():
    class Repo:
        async def list_memory_records(self, *args, **kwargs):
            return [
                {"id": "a", "content": "用户研究点云补全", "memory_key": "goal", "status": "active"},
                {"id": "b", "content": "用户研究点云识别", "status": "superseded"},
                {"id": "c", "content": "用户喜欢咖啡", "memory_key": "preference_coffee"},
                {"id": "d", "content": "用户在月球工作", "confidence": 0.1},
            ]

    service = CharacterMemoryService(Repo(), semantic_enabled=False)
    scope = UserScope("web", "web", "u", "c", "private")
    baseline, _ = await service.load_relevant_memories("char", scope, "接下来呢")
    assert baseline == ()
    candidates, _ = await service.load_relevant_memories("char", scope, "你喜欢什么", for_contextual_selection=True)
    assert [item.memory_id for item in candidates] == ["a"]


async def test_concurrent_requests_do_not_share_selection_state():
    async def reviewer(messages):
        payload = json.loads(messages[1]["content"])
        await asyncio.sleep(0)
        return json.dumps({"decisions": [{"id": payload["candidates"][0]["id"], "label": "use"}]})

    selector = ContextualEvidenceSelector(reviewer)
    outcomes = await asyncio.gather(*(selector.select("q", [_item(str(i))]) for i in range(8)))
    assert [outcome.memories[0].memory_id for outcome in outcomes] == list(map(str, range(8)))


async def test_context_service_selects_before_compiling_and_reuses_loaded_history():
    from character.models import CharacterProfile
    from services.character_context import CharacterContextService, TurnInput

    class Profiles:
        def get_profile(self, character_id):
            return CharacterProfile(character_id, "角色", values=("尊重承诺",))

    class Repo:
        async def get_relationship_record(self, *args):
            return {}

        async def list_memory_records(self, *args, **kwargs):
            return [
                {
                    "id": "a",
                    "content": "用户研究点云补全",
                    "importance": 0.5,
                    "evidence": ["一", "二", "三", "四", "最后更正：只研究室内场景"],
                }
            ]

    class Messages:
        calls = 0

        async def list_recent_conversation_history(self, *args, **kwargs):
            self.calls += 1
            return [{"role": "user", "content": "继续讨论我的科研"}]

    async def reviewer(messages):
        data = json.loads(messages[1]["content"])
        assert data["history"][0]["content"] == "继续讨论我的科研"
        assert data["character"]["values"] == ["尊重承诺"]
        assert datetime.fromisoformat(data["reference_time"]).utcoffset() == timedelta(0)
        return '{"decisions":[{"id":"a","label":"use"}]}'

    repo, messages = Repo(), Messages()
    service = CharacterContextService(
        Profiles(),
        repo,
        messages,
        memory_service=CharacterMemoryService(repo, semantic_enabled=False),
        memory_selector=ContextualEvidenceSelector(reviewer),
    )
    prepared = await service.prepare_turn(TurnInput("下一步呢", "web", "web", "u", "c", "private"), "char")
    assert prepared.memory_selection_status == "selected"
    assert prepared.memory_selection_candidate_count == 1
    assert "点云补全" in prepared.compiled.reference_context
    assert "点云补全" not in prepared.compiled.dynamic_context
    assert "最后更正：只研究室内场景" in prepared.compiled.reference_context
    assert '"subject_scope":"current_user_not_character"' in prepared.compiled.reference_context
    assert messages.calls == 1


async def test_ablation_flags_cannot_disable_contextual_lifecycle_guards():
    class Repo:
        async def list_memory_records(self, *args, **kwargs):
            return [
                {"id": "erased", "content": "用户研究点云", "status": "erased"},
                {"id": "pending", "content": "用户研究点云", "status": "pending"},
                {"id": "low", "content": "用户研究点云", "confidence": 0.1},
            ]

    service = CharacterMemoryService(
        Repo(),
        semantic_enabled=False,
        version_filter_enabled=False,
        include_pending=True,
    )
    candidates, _ = await service.load_relevant_memories(
        "char",
        UserScope("web", "web", "u", "c", "private"),
        "点云",
        for_contextual_selection=True,
    )
    assert not candidates


async def test_contextual_recall_finds_old_claim_beyond_recent_window():
    class Repo:
        async def list_memory_records(self, *args, limit=100, **kwargs):
            records = [{"id": str(index), "content": "用户喜欢咖啡", "importance": 0.9} for index in range(120)]
            records.append({"id": "old-research", "content": "用户实验使用特殊数据集 ShapeNet", "importance": 0.5})
            return records[:limit]

    service = CharacterMemoryService(Repo(), semantic_enabled=False)
    scope = UserScope("web", "web", "u", "c", "private")
    selected, total = await service.load_relevant_memories(
        "char",
        scope,
        "下一步呢",
        for_contextual_selection=True,
        retrieval_context="继续使用特殊数据集 ShapeNet",
    )
    assert total == 121
    assert "old-research" in {item.memory_id for item in selected}
    assert len(selected) <= 24


async def test_complete_long_claim_can_be_reviewed_but_whole_input_budget_still_applies():
    calls = []
    content = "前提" * 300 + "但是上面的陈述已撤回"

    async def reviewer(messages):
        calls.append(messages)
        payload = json.loads(messages[1]["content"])
        assert payload["candidates"][0]["content_complete"] is True
        assert payload["candidates"][0]["content"] == content
        return '{"decisions":[{"id":"a","label":"use"}]}'

    selector = ContextualEvidenceSelector(reviewer)
    item = _item("a", content)
    result = await selector.select("query", [item])
    assert result.memories == (item,)
    assert result.status == "selected"
    result = await selector.select("query", [_item("a", "前提" * 10000)])
    assert result.status == "fallback" and result.reason == "input_budget"
    result = await selector.select("长" * 4001, [_item("a")])
    assert result.status == "fallback"
    assert len(calls) == 1


async def test_recent_history_is_atomic_instead_of_tail_clipped():
    content = "以下只是别人说的话，不是我的偏好：" + "叙述" * 700 + "我喜欢咖啡"
    payload = json.loads(
        selection_messages("我喜欢什么", [_item("a")], history=[{"role": "user", "content": content}])[1]["content"]
    )
    assert payload["history"] == [{"role": "user", "content": content}]

    async def reviewer(messages):
        pytest.fail("over-budget history must fail before review")

    result = await ContextualEvidenceSelector(reviewer).select(
        "q", [_item("a")], history=[{"role": "user", "content": "长" * 6001}]
    )
    assert result.status == "fallback" and result.reason == "input_budget"
    assert not result.memories


async def test_compound_goal_keeps_non_goal_resources_for_semantic_review():
    class Repo:
        async def list_memory_records(self, *args, **kwargs):
            return [
                {"id": "topic", "memory_key": "goal_research", "content": "用户研究点云补全"},
                {"id": "hardware", "content": "用户显卡有八 GB 显存"},
                {"id": "erased", "content": "用户显卡有二十四 GB 显存", "status": "erased"},
            ]

    service = CharacterMemoryService(Repo(), semantic_enabled=False)
    selected, _ = await service.load_relevant_memories(
        "char",
        UserScope("web", "web", "u", "c", "private"),
        "结合我的显卡和研究方向，帮我安排实验。",
        for_contextual_selection=True,
    )
    assert {item.memory_id for item in selected} == {"topic", "hardware"}


async def test_recall_preserves_fifth_evidence_correction_for_semantic_review():
    evidence = ["来源一", "来源二", "来源三", "来源四", "最后更正：这个信息不属实"]

    class Repo:
        async def list_memory_records(self, *args, **kwargs):
            return [{"id": "a", "content": "用户研究点云补全", "evidence": evidence}]

    service = CharacterMemoryService(Repo(), semantic_enabled=False)
    candidates, _ = await service.load_relevant_memories(
        "char", UserScope("web", "web", "u", "c", "private"), "研究", for_contextual_selection=True
    )
    assert candidates[0].evidence == tuple(evidence)
    payload = json.loads(selection_messages("研究", candidates)[1]["content"])
    assert payload["candidates"][0]["evidence"][-1] == evidence[-1]


async def test_oversized_selected_packet_cannot_enable_memory_recall_strategy():
    from character.contextual_policy import ContextualDecisionPolicy
    from character.models import CharacterProfile
    from services.character_context import CharacterContextService, TurnInput

    class Profiles:
        def get_profile(self, character_id):
            return CharacterProfile(character_id, "角色")

    class Repo:
        async def get_relationship_record(self, *args):
            return {}

        async def list_memory_records(self, *args, **kwargs):
            return [{"id": "a", "content": "用户研究点云补全", "evidence": ["原始来源" * 1600]}]

    class Messages:
        async def list_recent_conversation_history(self, *args, **kwargs):
            return []

    async def selection_reviewer(messages):
        return '{"decisions":[{"id":"a","label":"use"}]}'

    async def policy_reviewer(messages):
        payload = json.loads(messages[1]["content"])
        assert "recall_shared_context" not in payload["allowed_strategies"]
        return '{"strategy_ids":["reflect_content"]}'

    repo = Repo()
    service = CharacterContextService(
        Profiles(),
        repo,
        Messages(),
        memory_service=CharacterMemoryService(repo, semantic_enabled=False),
        memory_selector=ContextualEvidenceSelector(selection_reviewer),
        contextual_policy=ContextualDecisionPolicy(policy_reviewer),
    )
    prepared = await service.prepare_turn(TurnInput("我的研究", "web", "web", "u", "c", "private"), "char")
    assert prepared.memory_selection_status == "selected"
    assert prepared.compiled.used_memory_ids == ()
    assert prepared.compiled.reference_context == ""
    assert prepared.contextual_policy_status == "applied"
    assert "recall_shared_context" not in prepared.decision.strategy_ids
    assert prepared.reply_guard.forbid_unsupported_user_fact is True


@pytest.mark.parametrize("status", ["erased", "deleted", "retracted", "pending"])
def test_source_event_link_cannot_resurrect_unusable_record(status):
    from character.memory_service import _row_evidence

    row = {"source_event_ids": ["event"]}
    assert _row_evidence(row, {"event": {"content": "不应泄漏", "status": status}}, complete=True) == ()
