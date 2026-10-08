import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from character.erasure_authority import partial_erasure_plan
from character.memory_llm import (
    MemoryEnrichmentScheduler,
    MemoryLlmConfig,
    build_memory_llm_messages,
    is_memory_erasure_request,
    parse_llm_proposals,
)
from character.models import UserScope
from db.database import SQLiteDB
from db.memory_claim_guard import MemoryClaimConflict
from repositories.character_memory import DatabaseCharacterMemoryRepository

COURSE = "event_海庭鹤林工坊纸版压印课预约确认"
INK = "preference_深蓝色油墨进行纸版压印"
ERASE = "请从长期记忆中彻底删除我本人的海庭鹤林工坊纸版压印课预约确认。"
KEEP = "不要忘掉我对深蓝色油墨进行纸版压印的长期个人偏好。"


def source():
    return json.loads((Path(__file__).parent / "fixtures/deepseek_memory_selective_erasure_cases.json").read_text())[
        "cases"
    ][-1]["message"]


def records():
    return (
        dict(
            id=4,
            memory_type="shared_event",
            memory_key=COURSE,
            content="用户原话事件记录（完整内容见证据）",
            status="active",
        ),
        dict(
            id=5,
            memory_type="user_fact",
            memory_key=INK,
            content="用户喜欢深蓝色油墨进行纸版压印，这是明确且长期的个人偏好。",
            status="active",
        ),
    )


def response(target=4, operation="ERASE", evidence=ERASE, **extras):
    key = COURSE if target == 4 else INK
    raw = dict(
        kind="shared_event" if target == 4 else "like",
        value="海庭鹤林工坊纸版压印课预约确认" if target == 4 else "深蓝色油墨进行纸版压印",
        content="",
        evidence=evidence,
        confidence=0.98,
        operation=operation,
        target_memory_id=str(target),
        target_memory_key=key,
        attributed_to="user",
    )
    raw.update(extras)
    return json.dumps({"memories": [raw]}, ensure_ascii=False)


def test_full_original_request_authorizes_only_named_course_and_freezes_kept_logical_key():
    assert is_memory_erasure_request(source())
    plan = partial_erasure_plan(source(), records())
    assert plan.valid and plan.allowed_ids == ("4",) and plan.protected_ids == ("5",) and plan.protected_keys == (INK,)
    parsed = parse_llm_proposals(response(), source_message=source(), existing_memories=records())
    assert len(parsed) == 1 and parsed[0].target_memory_id == "4" and parsed[0].protected_memory_keys == (INK,)


@pytest.mark.parametrize("operation", ["ERASE", "RETRACT", "MERGE", "SUPERSEDE", "COEXIST"])
def test_model_cannot_mutate_explicit_retained_target_using_positive_other_item_evidence(operation):
    parsed = parse_llm_proposals(response(5, operation), source_message=source(), existing_memories=records())
    assert all(item.operation == "NOOP" and item.memory is None for item in parsed)


@pytest.mark.parametrize(
    "evidence", [KEEP, KEEP.replace("不要", ""), "忘掉", "我对深蓝色油墨进行纸版压印的长期个人偏好"]
)
def test_clipped_or_negative_retention_evidence_cannot_authorize_erasure(evidence):
    assert not parse_llm_proposals(response(evidence=evidence), source_message=source(), existing_memories=records())


@pytest.mark.parametrize(
    "message",
    [
        "如果要从长期记忆中删除海庭鹤林工坊纸版压印课预约确认。不要忘掉深蓝色油墨进行纸版压印。",
        "朋友说请从长期记忆中删除海庭鹤林工坊纸版压印课预约确认。不要忘掉深蓝色油墨进行纸版压印。",
        "“请从长期记忆中删除海庭鹤林工坊纸版压印课预约确认。”不要忘掉深蓝色油墨进行纸版压印。",
        "不要忘掉海庭鹤林工坊纸版压印课预约确认。",
        ERASE + "不要删除聊天历史和长期记忆。" + KEEP,
    ],
)
def test_non_authorizing_or_ambiguous_instructions_keep_model_erasure_denied(message):
    assert not is_memory_erasure_request(message)
    assert not parse_llm_proposals(response(), source_message=message, existing_memories=records())


def test_unknown_retained_object_and_same_object_conflict_authorize_no_target():
    unknown = ERASE + "不要忘掉我所有的其他资料。"
    plan = partial_erasure_plan(unknown, records())
    assert plan.unresolved_protection and plan.allowed_ids == ()
    assert not parse_llm_proposals(response(), source_message=unknown, existing_memories=records())
    conflict = ERASE + "不要忘掉海庭鹤林工坊纸版压印课预约确认。"
    assert partial_erasure_plan(conflict, records()).allowed_ids == ()
    assert not parse_llm_proposals(response(), source_message=conflict, existing_memories=records())


def test_model_cannot_add_duplicate_kept_preference_or_supply_its_own_protection():
    evidence = "我喜欢深蓝色油墨进行纸版压印。"
    message = source() + evidence
    raw = json.loads(response(5, "ADD", evidence, target_memory_id="", target_memory_key="", protected_memory_keys=[]))
    assert not parse_llm_proposals(json.dumps(raw), source_message=message, existing_memories=records())
    parsed = parse_llm_proposals(
        response(protected_memory_keys=[]), source_message=source(), existing_memories=records()
    )
    assert parsed[0].protected_memory_keys == (INK,)


def test_memory_model_gets_original_complete_text_and_authoritative_target_constraints():
    wire = build_memory_llm_messages(source(), (), (), records(), 10000, 0.8)
    payload = json.loads(wire[-1]["content"])
    assert payload["current_user_message"] == source()
    plan = payload["partial_erasure_authorization"]
    assert plan["allowed_erase_memory_ids"] == ["4"] and plan["protected_memory_ids"] == ["5"]
    assert plan["erase_source_ids_allowed"] is False
    assert "不是执行结果" in wire[0]["content"]


def test_mixed_authorization_does_not_bypass_sensitive_source_safety():
    assert not parse_llm_proposals(
        response(), source_message=source() + "我的密码是ExampleSecret123456。", existing_memories=records()
    )


@pytest.fixture(scope="module", params=["sqlite", "pg"])
def database(request, tmp_path_factory):
    if request.param == "sqlite":
        db = SQLiteDB(tmp_path_factory.mktemp("partial-erasure") / "records.sqlite")
    else:
        url = os.environ["STAGE38_RETENTION_PG_URL"]
        assert "/stage3_stage38_retention_guards?" in url and "port=25433" in url
        with pytest.MonkeyPatch.context() as env:
            env.setenv("DATABASE_URL", url)
            from db.pg_database import PgDatabase, SyncPgAdapter
        db = SyncPgAdapter(PgDatabase(url))
    yield db
    (db.close if request.param == "pg" else db.close_connection)()


def seed(database, *, dependent=False):
    user = "partial-" + uuid.uuid4().hex
    fields = ("tsukiyashiro_kisaki", "qq", "unit-partial", user, "private", user)
    sid = "shared-" + uuid.uuid4().hex
    stamp = datetime.now(timezone.utc)
    body = "我的课程确认为MB-764-C。我喜欢深蓝色油墨进行纸版压印。"
    assert database.capture_memory_source(*fields, source_message_id=sid, body=body, observed_at=stamp) == "recorded"
    course = database.append_character_memory_claim(
        *fields,
        "shared_event",
        COURSE,
        "用户的课程确认",
        source_message_id=sid,
        observed_at=stamp.isoformat(),
        evidence_json=json.dumps([body]),
    )
    ink = database.append_character_memory_claim(
        *fields,
        "user_fact",
        INK,
        "用户喜欢深蓝色油墨进行纸版压印",
        source_message_id=sid,
        observed_at=stamp.isoformat(),
        relation_type="COEXIST" if dependent else "ADD",
        parent_memory_id=course["id"] if dependent else None,
        evidence_json=json.dumps(["我喜欢深蓝色油墨进行纸版压印。"]),
    )
    return fields, sid, stamp, course, ink


def rows(database, fields):
    return database.list_character_memory_claims(*fields, limit=None, include_inactive=True)


def test_sql_erases_only_course_keeps_ink_and_revokes_shared_original(database):
    fields, sid, stamp, course, ink = seed(database)
    before = [r for r in rows(database, fields) if r["id"] == ink["id"]]
    assert database.erase_character_memories(*fields, memory_key=COURSE, protected_memory_keys=(INK,)) == 1
    assert rows(database, fields) == before
    assert database.list_memory_sources(*fields, limit=100) == []
    assert (
        database.capture_memory_source(*fields, source_message_id=sid, body="old speech", observed_at=stamp) == "stale"
    )


@pytest.mark.parametrize("protected", [COURSE, INK])
def test_atomic_guard_rolls_back_protected_root_or_descendant_and_source_purge(database, protected):
    fields, sid, stamp, course, ink = seed(database, dependent=True)
    before = rows(database, fields)
    sources = database.list_memory_sources(*fields, limit=100)
    with pytest.raises(MemoryClaimConflict, match="retained logical memory"):
        database.erase_character_memories(*fields, memory_key=COURSE, protected_memory_keys=(protected,))
    assert rows(database, fields) == before and database.list_memory_sources(*fields, limit=100) == sources


def test_empty_protection_preserves_existing_recursive_eraser_contract(database):
    fields, sid, stamp, course, ink = seed(database, dependent=True)
    assert database.erase_character_memories(*fields, memory_key=COURSE) == 2
    assert rows(database, fields) == [] and database.list_memory_sources(*fields, limit=100) == []


@pytest.mark.parametrize("invalid", ["not-a-key-batch", ("",), ("x" * 256,), [None]])
def test_invalid_retained_key_constraint_cannot_mutate_storage(database, invalid):
    fields, sid, stamp, course, ink = seed(database)
    before = rows(database, fields)
    sources = database.list_memory_sources(*fields, limit=100)
    with pytest.raises(ValueError):
        database.erase_character_memories(*fields, memory_key=COURSE, protected_memory_keys=invalid)
    assert rows(database, fields) == before and database.list_memory_sources(*fields, limit=100) == sources


async def test_repository_forwards_protected_keys_and_refuses_unsafe_legacy_fallback():
    class Modern:
        def erase_character_memories(self, *args, **kwargs):
            self.kwargs = kwargs
            return 1

    modern = Modern()
    repo = DatabaseCharacterMemoryRepository(modern)
    scope = UserScope("qq", "unit", "u", "u", "private")
    assert await repo.erase_memory("role", scope, memory_key=COURSE, protected_memory_keys=(INK,)) == 1
    assert modern.kwargs["protected_memory_keys"] == (INK,)
    legacy = DatabaseCharacterMemoryRepository(object())
    with pytest.raises(RuntimeError, match="保护保留条目"):
        await legacy.erase_memory("role", scope, memory_id=4, protected_memory_keys=(INK,))


@pytest.mark.parametrize('source_ids', [[], ['mixed-source']])
async def test_actual_scheduler_guards_unlinked_sources_and_passes_kept_keys_to_eraser(source_ids):
    class Repo:
        def __init__(self):
            self.erased = []
            self.raw_erased = []
            self.captured = []

        async def list_memory_records(self, *args, **kwargs):
            return list(records())

        async def capture_source(self, *args, **kwargs):
            self.captured.append(kwargs)
            return "recorded"

        async def search_sources(self, *args, **kwargs):
            return [
                dict(
                    source_message_id="mixed-source",
                    body="课程确认和深蓝色油墨进行纸版压印偏好",
                    observed_at="2026-10-02T00:00:00+00:00",
                )
            ]

        async def list_sources(self, *args, **kwargs):
            return []

        async def erase_unlinked_sources(self, *args, **kwargs):
            self.raw_erased.append(kwargs)
            return 1

        async def erase_memory(self, *args, **kwargs):
            self.erased.append(kwargs)
            return 1

    class Completion:
        async def complete(self, messages):
            self.payload = json.loads(messages[-1]["content"])
            raw = json.loads(response())
            raw["erase_source_ids"] = source_ids
            return json.dumps(raw)

        async def close(self):
            pass

    class Embedding:
        def embed_texts(self, texts):
            return [[1.0, 0.0] for _ in texts]

    repo = Repo()
    completion = Completion()
    worker = MemoryEnrichmentScheduler(
        config=MemoryLlmConfig(enabled=True, base_url="http://unused", model="unit"),
        completion=completion,
        embedding_provider=Embedding(),
    )
    try:
        result = await worker.schedule_and_wait(
            repository=repo,
            character_id="role",
            user_scope=UserScope("qq", "unit", "u", "u", "private"),
            message=source(),
            rule_hints=[],
            source_message_id="new-delete",
            timeout_seconds=2,
        )
        if source_ids:
            assert result['status'] == 'failed' and result['stage'] == 'proposal_validation'
            assert result['accepted'] == result['persisted'] == 0
            assert not repo.erased and not repo.raw_erased
            return
        assert result["status"] == "erased" and result["persisted"] == 1 and result["source_capture"] == "erase_request"
        assert not repo.captured and not repo.raw_erased
        assert repo.erased[0]["memory_key"] == COURSE and repo.erased[0]["protected_memory_keys"] == (INK,)
        assert completion.payload["current_user_message"] == source()
        assert result["source_erasure_policy"] == "claim_targets_only_for_partial_retention"
    finally:
        await worker.shutdown(timeout=1)


def test_repeated_preference_pointer_resolves_only_previously_matched_unique_object():
    message = ERASE + KEEP + "请保留这项油墨偏好。"
    plan = partial_erasure_plan(message, records())
    assert plan.allowed_ids == ("4",) and not plan.unresolved_protection
    wrong = ERASE + KEEP + "请保留这项课程记录。"
    assert partial_erasure_plan(wrong, records()).unresolved_protection


def test_ambiguous_preference_pointer_cannot_choose_between_retained_objects():
    second = dict(
        id=6,
        memory_type="user_fact",
        memory_key="preference_白色油墨进行纸版压印",
        content="用户喜欢白色油墨进行纸版压印",
        status="active",
    )
    message = ERASE + KEEP + "不要忘掉白色油墨进行纸版压印。请保留这项偏好。"
    plan = partial_erasure_plan(message, (*records(), second))
    assert plan.unresolved_protection and plan.allowed_ids == ()
