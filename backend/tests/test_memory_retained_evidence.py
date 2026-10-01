import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from db.database import SQLiteDB
from db.memory_claim_guard import MemoryClaimConflict
from db.retained_evidence import project_retained

COURSE = "event_海庭鹤林工坊纸版压印课预约确认"
INK = "preference_深蓝色油墨进行纸版压印"
BODY = json.loads((Path(__file__).parent / "fixtures/deepseek_memory_evidence_erasure_cases.json").read_text())[
    "qq_owner_prerequisite"
]["full_original_message"]
PREFERENCE = "我喜欢深蓝色油墨进行纸版压印，这是我明确且长期的个人偏好。"


def row(evidence, *, content="用户原话事件记录（完整内容见证据）", quoted=True):
    meta = {"attributed_to": "user", "speaker_role": "user", "described_subject": "not_resolved", "qualifiers": {}}
    if quoted:
        meta["content_semantics"] = "quoted_source"
    return dict(
        id=4,
        memory_key=COURSE,
        content=content,
        evidence_json=json.dumps(evidence, ensure_ascii=False),
        metadata_json=json.dumps(meta, ensure_ascii=False),
    )


def erased(evidence=(PREFERENCE,)):
    return [
        dict(
            id=5,
            memory_key=INK,
            content="用户喜欢深蓝色油墨进行纸版压印",
            evidence_json=json.dumps(evidence, ensure_ascii=False),
            metadata_json="{}",
        )
    ]


def source(body=BODY):
    return dict(body=body, state="recorded", source_message_id="actual-unit-shared-source")


def test_original_full_shared_evidence_is_replaced_only_by_exact_remaining_fragments():
    original = row([BODY])
    patch = project_retained(original, source(), erased())
    assert patch
    expected = BODY.split(PREFERENCE)
    assert json.loads(patch["evidence_json"]) == expected and all(x in BODY for x in expected)
    assert "深蓝色" not in json.dumps(patch, ensure_ascii=False)
    assert all(
        x in "".join(expected) for x in ["MB-764-C", "2026年12月5日周六", "尚未参加课程", "没有出发", "只描述我"]
    )
    meta = json.loads(patch["metadata_json"])
    assert meta["content_semantics"] == "quoted_source" and meta["described_subject"] == "not_resolved"
    proof = meta["erasure_evidence_projection"]
    assert proof["complete_original_source"] is False and proof["kind"] == "original_source_fragments"
    assert [BODY[a:z] for a, z in proof["sources"][0]["spans"]] == expected
    assert original == row([BODY])


def test_independent_normalized_sibling_remains_unchanged_when_full_quote_topic_is_erased():
    independent = row([PREFERENCE], content="用户喜欢深蓝色油墨进行纸版压印", quoted=False)
    deleted = [dict(id=4, memory_key=COURSE, evidence_json=json.dumps([BODY]), metadata_json="{}")]
    assert project_retained(independent, source(), deleted) is None


@pytest.mark.parametrize(
    "snippet", ["深蓝色油墨", "色油墨进行纸版压印，这是我明确且长期的个人偏好。"]
)
def test_partial_statement_cannot_silently_remove_retained_context(snippet):
    with pytest.raises(MemoryClaimConflict, match="complete source statement"):
        project_retained(row([BODY]), source(), erased([snippet]))


@pytest.mark.parametrize("body", [None, "not the actual original source"])
def test_absent_or_forged_original_source_cannot_authorize_projection(body):
    with pytest.raises(MemoryClaimConflict):
        project_retained(row([BODY]), source(body), erased())



def test_unrelated_evidence_from_a_different_source_is_preserved_in_order():
    other = "我后来另外收到工具维修确认，编号WX-204。"
    patch = project_retained(row([other, BODY]), source(), erased())
    assert json.loads(patch["evidence_json"]) == [other, *BODY.split(PREFERENCE)]


def test_short_quoted_display_copy_is_purged_as_well_as_evidence():
    patch = project_retained(
        row([BODY], content="用户原话事件记录：" + json.dumps(BODY, ensure_ascii=False)), source(), erased()
    )
    assert "深蓝色" not in patch["content"] and "片段" in patch["content"]


def test_quoted_display_without_separable_evidence_rolls_back():
    with pytest.raises(MemoryClaimConflict, match="no separable evidence"):
        project_retained(row([], content=BODY), source(), erased())


@pytest.mark.parametrize("field", ["content", "metadata_json"])
def test_an_assertion_or_arbitrary_metadata_copy_cannot_be_silently_rewritten(field):
    retained = row([BODY], quoted=False)
    if field == "content":
        retained["content"] = "用户的课程资料还包含深蓝色油墨进行纸版压印"
    else:
        retained["metadata_json"] = json.dumps({"qualifiers": {"copy": PREFERENCE}}, ensure_ascii=False)
    with pytest.raises(MemoryClaimConflict):
        project_retained(retained, source(), erased())


def test_sentence_inside_a_quotation_is_not_a_source_boundary():
    body = "课程回执MB-764-C。朋友说：“" + PREFERENCE + "”这并非我的偏好。"
    with pytest.raises(MemoryClaimConflict, match="complete source statement"):
        project_retained(row([body]), source(body), erased())


def test_repeated_complete_statements_are_all_removed_without_reordering():
    body = "课程回执MB-764-C。" + PREFERENCE + "尚未参加课程。" + PREFERENCE + "没有出发。"
    patch = project_retained(row([body]), source(body), erased())
    assert json.loads(patch["evidence_json"]) == ["课程回执MB-764-C。", "尚未参加课程。", "没有出发。"]


@pytest.mark.parametrize("bad", ["{}", "[3]", '[""]', "not-json"])
def test_invalid_retained_evidence_is_not_repaired_into_authority(bad):
    record = row([BODY])
    record["evidence_json"] = bad
    with pytest.raises(MemoryClaimConflict):
        project_retained(record, source(), erased())


@pytest.fixture(scope="module", params=["sqlite", "pg"])
def database(request, tmp_path_factory):
    if request.param == "sqlite":
        db = SQLiteDB(tmp_path_factory.mktemp("retained-evidence") / "records.sqlite")
    else:
        url = os.environ["STAGE39_EVIDENCE_PG_URL"]
        assert "/stage3_stage39_evidence_guards?" in url and "port=25433" in url
        with pytest.MonkeyPatch.context() as env:
            env.setenv("DATABASE_URL", url)
            from db.pg_database import PgDatabase, SyncPgAdapter
        db = SyncPgAdapter(PgDatabase(url))
    yield db
    (db.close if request.param == "pg" else db.close_connection)()


def seed(db, *, ambiguous=False, extra=False):
    owner = "evidence-" + uuid.uuid4().hex
    fields = ("tsukiyashiro_kisaki", "qq", "unit-evidence", owner, "private", owner)
    sid = "shared-" + uuid.uuid4().hex
    stamp = datetime.now(timezone.utc)
    assert db.capture_memory_source(*fields, source_message_id=sid, body=BODY, observed_at=stamp) == "recorded"
    course = db.append_character_memory_claim(
        *fields,
        "shared_event",
        COURSE,
        row([BODY])["content"],
        source_message_id=sid,
        observed_at=stamp.isoformat(),
        evidence_json=json.dumps([BODY]),
        metadata_json=row([BODY])["metadata_json"],
    )
    ink = db.append_character_memory_claim(
        *fields,
        "user_fact",
        INK,
        "用户喜欢深蓝色油墨进行纸版压印",
        source_message_id=sid,
        observed_at=stamp.isoformat(),
        evidence_json=json.dumps(["深蓝色油墨" if ambiguous else PREFERENCE]),
    )
    if extra:
        db.append_character_memory_claim(
            *fields,
            "shared_event",
            "event_另一份独立工具确认",
            "用户原话记录",
            source_message_id=sid,
            observed_at=stamp.isoformat(),
            evidence_json=json.dumps([BODY]),
            metadata_json=json.dumps({"qualifiers": {"copy": PREFERENCE}}, ensure_ascii=False),
        )
    return fields, sid, stamp, course, ink


def records(db, fields):
    return db.list_character_memory_claims(*fields, limit=None, include_inactive=True)


def test_sql_inverse_erasure_preserves_course_and_purges_its_copied_preference_atomically(database):
    fields, sid, stamp, course, ink = seed(database)
    old = [x for x in records(database, fields) if x["id"] == course["id"]][0]
    assert database.erase_character_memories(*fields, memory_key=INK, protected_memory_keys=(COURSE,)) == 1
    remaining = records(database, fields)
    assert len(remaining) == 1
    current = remaining[0]
    assert current["id"] == course["id"] and "深蓝色" not in json.dumps(current, ensure_ascii=False)
    assert json.loads(current["evidence_json"]) == BODY.split(PREFERENCE)
    assert all(
        current.get(k) == old.get(k) for k in old if k not in {"content", "evidence_json", "metadata_json", "metadata"}
    )
    assert database.list_memory_sources(*fields, limit=100) == []
    assert database.capture_memory_source(*fields, source_message_id=sid, body=BODY, observed_at=stamp) == "stale"


@pytest.mark.parametrize("failure", ["ambiguous", "metadata"])
def test_sql_failed_projection_rolls_back_target_retained_row_source_and_fence(database, failure):
    fields, sid, stamp, course, ink = seed(database, ambiguous=failure == "ambiguous", extra=failure == "metadata")
    before = records(database, fields)
    sources = database.list_memory_sources(*fields, limit=100)
    with pytest.raises(MemoryClaimConflict):
        database.erase_character_memories(*fields, memory_key=INK, protected_memory_keys=(COURSE,))
    assert records(database, fields) == before and database.list_memory_sources(*fields, limit=100) == sources
    assert (
        database.capture_memory_source(
            *fields,
            source_message_id="pre-erasure-" + uuid.uuid4().hex,
            body="实际之前接收的独立原话",
            observed_at=stamp,
        )
        == "recorded"
    )


def test_sql_shared_source_cleanup_does_not_touch_another_actual_owner(database):
    fields, sid, stamp, course, ink = seed(database)
    other, _, _, _, _ = seed(database)
    before = records(database, other)
    sources = database.list_memory_sources(*other, limit=100)
    assert database.erase_character_memories(*fields, memory_key=INK, protected_memory_keys=(COURSE,)) == 1
    assert records(database, other) == before and database.list_memory_sources(*other, limit=100) == sources


def test_sql_evidence_cleanup_runs_for_plain_authorized_erasure_without_optional_retention_keys(database):
    fields, sid, stamp, course, ink = seed(database)
    assert database.erase_character_memories(*fields, memory_key=INK) == 1
    assert "深蓝色" not in json.dumps(records(database, fields), ensure_ascii=False)
