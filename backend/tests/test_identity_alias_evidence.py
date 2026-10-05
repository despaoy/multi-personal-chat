"""Complete fictional alias predicates; component tests, not model approval."""

import json
from datetime import datetime, timezone

import pytest

from character.memory_llm import parse_llm_proposals

POSITIVE = "我的常用别名是芷露。芷露只是本人的别名，不是本人的规范姓名。"
NEGATIVE = "小岚不是本人的别名。小岚是朋友兰溪的别名，只属于朋友。"
SOURCE = POSITIVE + NEGATIVE + "请记住，以后这个角色记住本人别名和否定归属，用于同一个角色的其他会话。"


def parse(
    evidence, *, value="芷露", kind="name", source=SOURCE, operation="ADD", target="", existing=(), qualifiers=None
):
    raw = dict(
        kind=kind,
        value=value,
        evidence=evidence,
        content="用户明确声明" + evidence,
        operation=operation,
        target_memory_id="1" if target else "",
        target_memory_key=target,
        confidence=0.97,
        attributed_to="user",
        qualifiers={"certainty": "稳定"} if qualifiers is None else qualifiers,
        scope_level="user_character",
    )
    return parse_llm_proposals(
        json.dumps({"memories": [raw]}, ensure_ascii=False),
        source_message=source,
        existing_memories=existing,
        confidence_threshold=0.85,
    )


def test_alias_name_label_projects_separate_predicate_without_losing_negation():
    (proposal,) = parse(POSITIVE)
    assert proposal.memory.memory_key == "user_alias"
    assert proposal.memory.content == "用户明确说自己的常用别名是芷露"
    assert proposal.evidence == POSITIVE and not proposal.qualifiers
    assert proposal.scope_level == "user_character"


def test_complete_self_alias_exclusion_keeps_named_friend_explanation():
    (proposal,) = parse(NEGATIVE, value="小岚", kind="other_user_fact")
    assert proposal.memory.memory_key == "user_alias_exclusion_小岚"
    assert proposal.memory.content == "用户明确说小岚不是本人的别名"
    assert proposal.evidence == NEGATIVE and not proposal.qualifiers


@pytest.mark.parametrize(
    "evidence,source",
    [
        ("芷露", SOURCE),
        ("我的常用别名是芷露", "我的常用别名是芷露，但只在周末使用这个称呼。"),
        ("我的常用别名是芷露", "我的常用别名是芷露。但仅在周末使用这个称呼。"),
        ("我的常用别名是芷露", "朋友说“我的常用别名是芷露”。"),
    ],
)
def test_clipped_conditional_or_quoted_alias_cannot_become_primary_name(evidence, source):
    assert not parse(evidence, source=source, qualifiers={})


def test_alias_cannot_supersede_primary_name_target():
    current = dict(
        memory_id="1", memory_key="user_name", memory_type="user_fact", content="用户说自己叫青岑", status="active"
    )
    assert not parse(POSITIVE, operation="SUPERSEDE", target="user_name", existing=(current,))


def test_explicit_alias_update_can_only_target_its_own_predicate():
    current = dict(
        memory_id="1",
        memory_key="user_alias",
        memory_type="user_fact",
        content="用户明确说自己的常用别名是若翎",
        status="active",
    )
    (proposal,) = parse(POSITIVE, operation="SUPERSEDE", target="user_alias", existing=(current,))
    assert proposal.memory.memory_key == "user_alias" and proposal.target_memory_key == "user_alias"


def test_actual_condition_and_unknown_labels_are_not_removed():
    assert not parse(POSITIVE, qualifiers={"certainty": "稳定", "condition": "订单已付款"})
    assert not parse(POSITIVE, qualifiers={"certainty": "hypothetical"})


def test_positive_friend_name_is_not_a_negative_self_alias_predicate():
    source = "小岚是朋友兰溪的别名，只属于朋友。"
    assert not parse(source, value="小岚", kind="other_user_fact", source=source, qualifiers={})


def test_ordinary_self_primary_name_keeps_existing_slot():
    (proposal,) = parse("我叫青岑。", value="青岑", source="我叫青岑。请记住，以后这个角色记住。", qualifiers={})
    assert proposal.memory.memory_key == "user_name"


@pytest.mark.asyncio
async def test_normal_scheduler_storage_keeps_primary_and_alias_separate(tmp_path):
    from repositories.character_memory import DatabaseCharacterMemoryRepository

    from character.memory_extractor import extract_memories
    from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig
    from character.models import UserScope
    from db.database import SQLiteDB

    db = SQLiteDB(tmp_path / "alias-component.sqlite")
    repo = DatabaseCharacterMemoryRepository(db)
    scope = UserScope("web", "alias-fixture", "owner", "group-a", "group")
    stamp = datetime.now(timezone.utc)
    db.capture_memory_source(
        character_id="role",
        platform="web",
        adapter="alias-fixture",
        sender_id="owner",
        conversation_type="group",
        conversation_id="group-a",
        source_message_id="primary",
        body="我叫青岑。请记住，以后这个角色记住。",
        observed_at=stamp,
    )
    primary = db.append_character_memory_claim(
        character_id="role",
        platform="web",
        adapter="alias-fixture",
        sender_id="owner",
        conversation_type="group",
        conversation_id="group-a",
        scope_level="user_character",
        memory_type="user_fact",
        memory_key="user_name",
        content="用户说自己叫青岑",
        evidence_json='["我叫青岑。"]',
        source_message_id="primary",
        observed_at=stamp.isoformat(),
    )

    class Completion:
        async def close(self):
            pass

        async def complete(self, messages):
            # Explicit component completion; never a claimed real-model result.
            return json.dumps(
                {
                    "memories": [
                        dict(
                            kind="name",
                            value="芷露",
                            evidence=POSITIVE,
                            confidence=0.97,
                            operation="ADD",
                            attributed_to="user",
                            qualifiers={"certainty": "稳定"},
                            scope_level="user_character",
                        )
                    ]
                },
                ensure_ascii=False,
            )

    scheduler = MemoryEnrichmentScheduler(
        config=MemoryLlmConfig(enabled=True, base_url="http://127.0.0.1:1", model="component"), completion=Completion()
    )
    try:
        receipt = await scheduler.schedule_and_wait(
            repository=repo,
            character_id="role",
            user_scope=scope,
            message=SOURCE,
            rule_hints=extract_memories(SOURCE, reference_time=stamp),
            history=(),
            source_message_id="alias",
            observed_at=datetime.now(timezone.utc),
            timeout_seconds=10,
        )
        assert receipt["status"] == "saved" and receipt["persisted"] == 1
        destination = UserScope("web", "alias-fixture", "owner", "group-b", "group")
        rows = await repo.list_memory_records("role", destination, limit=None)
        assert {row["memory_key"] for row in rows} == {"user_name", "user_alias"}
        assert next(row for row in rows if row["memory_key"] == "user_name")["id"] == primary["id"]
        assert not await repo.list_sources("role", destination, limit=100)
    finally:
        await scheduler.shutdown(timeout=5)


def test_exclusion_index_can_include_its_own_contiguous_friend_explanation():
    (proposal,) = parse(
        NEGATIVE, value="小岚是朋友兰溪的别名", kind="other_user_fact", qualifiers={"certainty": "明确"}
    )
    assert proposal.memory.memory_key == "user_alias_exclusion_小岚"
    assert proposal.memory.content == "用户明确说小岚不是本人的别名"
    assert proposal.evidence == NEGATIVE


def test_unrelated_friend_index_cannot_supply_self_alias_ownership():
    assert not parse(NEGATIVE, value="兰溪", kind="other_user_fact", qualifiers={})
