import copy
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from character.temporal_projection import project_temporal_record
from db.database import SQLiteDB
from db.memory_claim_guard import MemoryClaimConflict
from db.retained_evidence import _outside_quotes
from db.retained_source_observation import historical_fragments

FIXTURE = json.loads((Path(__file__).parent / "fixtures/deepseek_memory_cross_source_erasure_cases.json").read_text())
OLD = FIXTURE["new_native_source"]
BODY = FIXTURE["unlinked_historical_source"]
VALUE = "深蓝色油墨进行纸版压印"
INK = "preference_" + VALUE
COURSE = FIXTURE["retained_memory_key"]
PREFERENCE = "我喜欢" + VALUE + "，这是我明确且长期的个人偏好"
QUOTE = "“" + PREFERENCE + "。”"


@pytest.mark.parametrize("opening,closing", [("“", "”"), ("「", "」"), ("『", "』"), ('"', '"')])
def test_literal_own_history_keeps_all_remaining_characters_and_exact_offsets(opening, closing):
    body = BODY.replace(QUOTE, opening + PREFERENCE + "。" + closing)
    fragments, spans = historical_fragments(body, {VALUE}, _outside_quotes)
    assert fragments == body.split(opening + PREFERENCE + "。" + closing)
    assert [body[a:z] for a, z in spans] == fragments
    assert FIXTURE["retained_recheck_prefix"] in fragments[0]
    assert VALUE not in "".join(fragments)


@pytest.mark.parametrize(
    "body",
    [
        "朋友说：“" + PREFERENCE + "。”我的课程有效。",
        "我引用我以前的原话：“我不喜欢" + VALUE + "。”课程有效。",
        "我引用我以前的原话：“" + PREFERENCE + "，课程回执MC-845-D。”课程有效。",
        "我引用我以前的原话：“" + PREFERENCE + "。”其实朋友才是说这句话的人。",
        "我引用我以前的原话：“" + PREFERENCE + "。课程有效。",
        "我引用我以前的原话：“" + PREFERENCE + "。”我的" + VALUE + "记录还有第二份。",
        "“我引用我以前的原话：『" + PREFERENCE + "。』”课程有效。",
    ],
)
def test_unknown_roles_compounds_incomplete_quotes_and_remaining_copies_cannot_be_projected(body):
    with pytest.raises(MemoryClaimConflict):
        historical_fragments(body, {VALUE}, _outside_quotes)


def test_repeated_owned_quotes_preserve_original_source_order():
    body = BODY + "另一次引用我以前的原话：“" + PREFERENCE + "。”课程没有取消。"
    fragments, spans = historical_fragments(body, {VALUE}, _outside_quotes)
    assert [body[a:z] for a, z in spans] == fragments
    assert len(fragments) == 3 and VALUE not in "".join(fragments)


@pytest.mark.parametrize("memory_type", ["shared_event", "user_fact"])
def test_fragment_read_view_never_hydrates_complete_raw_source_or_promotes_current_fact(memory_type):
    fragments, spans = historical_fragments(BODY, {VALUE}, _outside_quotes)
    stamp = datetime.now(timezone.utc).isoformat()
    row = dict(
        memory_type=memory_type,
        memory_key="user_name",
        content="用户叫错误名字",
        status="active",
        observed_at=stamp,
        valid_from="2040-01-01T00:00:00+00:00",
        valid_to="2041-01-01T00:00:00+00:00",
        evidence=fragments,
        source_message_ids=["actual-original"],
        metadata={
            "qualifiers": {},
            "temporal_provenance": {
                "version": 1,
                "producer": "source_erasure_projection",
                "validity_authority": "unspecified",
            },
        },
    )
    before = copy.deepcopy(row)
    view = project_temporal_record(row, {"actual-original": {"body": BODY, "observed_at": stamp}})
    assert view["temporal_mode"] == "observation" and view["temporal_observed_at"] == stamp
    assert view["valid_from"] == stamp and not view["valid_to"]
    assert view["evidence"] == fragments and VALUE not in json.dumps(view, ensure_ascii=False)
    assert FIXTURE["retained_recheck_prefix"] in view["retrieval_content"] and row == before


@pytest.fixture(scope="module", params=["sqlite", "pg"])
def database(request, tmp_path_factory):
    if request.param == "sqlite":
        db = SQLiteDB(tmp_path_factory.mktemp("cross-source-fragments") / "records.sqlite")
    else:
        url = os.environ["STAGE42_FRAGMENTS_PG_URL"]
        assert "/stage3_stage42_fragment_guards?" in url and "port=25433" in url
        with pytest.MonkeyPatch.context() as env:
            env.setenv("DATABASE_URL", url)
            from db.pg_database import PgDatabase, SyncPgAdapter
        db = SyncPgAdapter(PgDatabase(url))
    yield db
    (db.close if request.param == "pg" else db.close_connection)()


def new_fields():
    owner = "fragment-" + uuid.uuid4().hex
    return ("tsukiyashiro_kisaki", "qq", "unit-fragment", owner, "private", owner)


def speech(db, fields, body=BODY):
    sid = uuid.uuid4().hex
    stamp = datetime.now(timezone.utc)
    assert db.capture_memory_source(*fields, source_message_id=sid, body=body, observed_at=stamp) == "recorded"
    return sid, stamp


def seed(db, fields=None, body=BODY):
    fields = fields or new_fields()
    sid, stamp = speech(db, fields, OLD)
    course = db.append_character_memory_claim(
        *fields,
        "shared_event",
        COURSE,
        "用户原话事件记录（完整内容见证据）",
        evidence_json=json.dumps([OLD]),
        metadata_json=json.dumps(
            {
                "content_semantics": "quoted_source",
                "speaker_role": "user",
                "described_subject": "not_resolved",
                "attributed_to": "user",
                "qualifiers": {},
            }
        ),
        source_message_id=sid,
        observed_at=stamp.isoformat(),
    )
    ink = db.append_character_memory_claim(
        *fields,
        "user_fact",
        INK,
        "用户喜欢" + VALUE,
        evidence_json=json.dumps([PREFERENCE]),
        metadata_json=json.dumps({"attributed_to": "user"}),
        source_message_id=sid,
        observed_at=stamp.isoformat(),
    )
    later, later_stamp = speech(db, fields, body)
    return fields, course, ink, sid, later, later_stamp


def rows(db, fields):
    return db.list_character_memory_claims(*fields, limit=None, include_inactive=True)


def test_sql_unlinked_source_retained_as_new_partial_observation_with_real_identity_and_clock(database):
    fields, course, ink, original, later, stamp = seed(database)
    assert database.erase_character_memories(*fields, memory_key=INK, protected_memory_keys=(COURSE,)) == 1
    current = rows(database, fields)
    assert len(current) == 2 and all(x["id"] != ink["id"] for x in current)
    (fragment,) = [x for x in current if x["memory_key"].startswith("source_fragment_")]
    expected, spans = historical_fragments(BODY, {VALUE}, _outside_quotes)
    assert json.loads(fragment["evidence_json"]) == expected
    assert fragment["source_message_id"] == later and json.loads(fragment["source_message_ids_json"]) == [later]
    assert fragment["observed_at"] == stamp.isoformat() and fragment["memory_type"] == "shared_event"
    metadata = json.loads(fragment["metadata_json"])
    assert metadata["content_semantics"] == "quoted_source" and metadata["described_subject"] == "not_resolved"
    assert metadata["erasure_evidence_projection"] == dict(
        version=1,
        kind="original_source_fragments",
        sources=[dict(source_message_id=later, spans=spans)],
        complete_original_source=False,
    )
    assert VALUE not in json.dumps(current, ensure_ascii=False)
    assert next(x for x in current if x["id"] == course["id"])["source_message_id"] == original
    assert database.list_memory_sources(*fields, limit=100) == []
    assert database.capture_memory_source(*fields, source_message_id=later, body=BODY, observed_at=stamp) == "stale"
    assert database.erase_character_memories(*fields, memory_key=INK, protected_memory_keys=(COURSE,)) == 0
    assert rows(database, fields) == current


def test_sql_unseparable_historical_quote_rolls_back_target_projection_source_and_owner_fence(database):
    body = BODY.replace(QUOTE, "“" + PREFERENCE + "，我的课程回执MC-845-D。”")
    fields, *_ = seed(database, body=body)
    before = rows(database, fields)
    sources = database.list_memory_sources(*fields, limit=100)
    with pytest.raises(MemoryClaimConflict):
        database.erase_character_memories(*fields, memory_key=INK, protected_memory_keys=(COURSE,))
    assert rows(database, fields) == before and database.list_memory_sources(*fields, limit=100) == sources
    speech(database, fields, "独立的新核对记录，课程仍有效。")


def test_sql_other_owner_character_and_conversation_never_produce_cross_scope_fragments(database):
    fields, *_ = seed(database)
    other_owner = new_fields()
    other_character = ("other-character", *fields[1:])
    other_conversation = (*fields[:5], "other-conversation")
    others = [other_owner, other_character, other_conversation]
    for other in others:
        speech(database, other)
    assert database.erase_character_memories(*fields, memory_key=INK, protected_memory_keys=(COURSE,)) == 1
    for other in others:
        assert not rows(database, other)
    assert len(database.list_memory_sources(*other_owner, limit=100)) == 1


def test_sql_previously_fenced_unlinked_source_never_becomes_a_new_observation(database):
    fields = new_fields()
    speech(database, fields)
    sid, stamp = speech(database, fields, "我叫旧记录。")
    database.append_character_memory_claim(
        *fields,
        "user_fact",
        "user_name",
        "用户叫旧记录",
        evidence_json=json.dumps(["我叫旧记录。"]),
        source_message_id=sid,
        observed_at=stamp.isoformat(),
    )
    assert database.erase_character_memories(*fields, memory_key="user_name") == 1
    # New receipt and genuine new source are captured after the first fence;
    # the older unrelated raw source must remain excluded, despite its body.
    sid, stamp = speech(database, fields, OLD)
    database.append_character_memory_claim(
        *fields,
        "user_fact",
        INK,
        "用户喜欢" + VALUE,
        evidence_json=json.dumps([PREFERENCE]),
        source_message_id=sid,
        observed_at=stamp.isoformat(),
    )
    assert database.erase_character_memories(*fields, memory_key=INK) == 1
    assert rows(database, fields) == []


def test_sql_failure_after_actual_fragment_and_link_insert_rolls_back_every_write(database, monkeypatch):
    from db import memory_source

    fields, *_ = seed(database)
    before = rows(database, fields)
    sources = database.list_memory_sources(*fields, limit=100)
    original_revoke = memory_source.revoke_plan

    def fail_after_revocation(scope, memory_ids, **kwargs):
        yield from original_revoke(scope, memory_ids, **kwargs)
        raise MemoryClaimConflict("isolated test failure after actual fragment insertion and revocation")

    monkeypatch.setattr(memory_source, "revoke_plan", fail_after_revocation)
    with pytest.raises(MemoryClaimConflict):
        database.erase_character_memories(*fields, memory_key=INK, protected_memory_keys=(COURSE,))
    assert rows(database, fields) == before and database.list_memory_sources(*fields, limit=100) == sources


def test_sql_candidate_overflow_from_real_recorded_sources_is_an_atomic_conflict(database):
    fields, *_ = seed(database)
    for _ in range(200):
        speech(database, fields)
    before = rows(database, fields)
    sources = database.list_memory_sources(*fields, limit=200)
    with pytest.raises(MemoryClaimConflict, match="candidate bound"):
        database.erase_character_memories(*fields, memory_key=INK, protected_memory_keys=(COURSE,))
    assert rows(database, fields) == before and database.list_memory_sources(*fields, limit=200) == sources
