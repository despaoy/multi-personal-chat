import copy
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from db.database import SQLiteDB
from db.memory_claim_guard import MemoryClaimConflict
from db.retained_evidence import project_retained

FIXTURE = json.loads((Path(__file__).parent / "fixtures/deepseek_memory_coupled_erasure_cases.json").read_text())
BODY = FIXTURE["new_native_source"]
VALUE = "深蓝色油墨进行纸版压印"
INK = "preference_" + VALUE
COURSE = "event_海庭鹤林工坊纸版压印课预约确认"
EVIDENCE = "我喜欢" + VALUE + "，这是我明确且长期的个人偏好"
CUT = "并且" + EVIDENCE + "。"


def retained(body=BODY):
    return dict(
        id=6,
        memory_key=COURSE,
        content="用户原话事件记录（完整内容见证据）",
        evidence_json=json.dumps([body], ensure_ascii=False),
        metadata_json=json.dumps(
            {
                "content_semantics": "quoted_source",
                "speaker_role": "user",
                "described_subject": "not_resolved",
                "attributed_to": "user",
                "qualifiers": {},
            }
        ),
    )


def erased(evidence=EVIDENCE, **changes):
    return [
        dict(
            dict(
                id=7,
                memory_key=INK,
                content="用户喜欢" + VALUE + "，这是其明确且长期的个人偏好。",
                evidence_json=json.dumps([evidence], ensure_ascii=False),
                metadata_json=json.dumps(
                    {"attributed_to": "user", "qualifiers": {"preference_strength": "明确且长期"}}
                ),
            ),
            **changes,
        )
    ]


def source(body=BODY):
    return dict(body=body, state="recorded", source_message_id=FIXTURE["native_source_message_id"])


@pytest.mark.parametrize("evidence", [EVIDENCE, BODY, BODY.split("。")[0] + "。", "我喜欢" + VALUE])
def test_real_stored_subsentence_and_broad_citations_remove_only_preference_not_course_facts(evidence):
    original = retained()
    before = copy.deepcopy(original)
    patch = project_retained(original, source(), erased(evidence))
    assert json.loads(patch["evidence_json"]) == BODY.split(CUT)
    text = "".join(json.loads(patch["evidence_json"]))
    assert all(
        value in text
        for value in [
            "MC-845-D",
            "2026年12月5日周六",
            "预约仍然有效",
            "尚未参加课程也没有出发",
            "没有联系工坊取消预约",
            "我本人刚收到",
        ]
    )
    assert VALUE not in json.dumps(patch, ensure_ascii=False) and original == before
    metadata = json.loads(patch["metadata_json"])
    assert metadata["described_subject"] == "not_resolved" and metadata["content_semantics"] == "quoted_source"
    projection = metadata["erasure_evidence_projection"]
    assert projection["complete_original_source"] is False
    assert [BODY[a:z] for a, z in projection["sources"][0]["spans"]] == BODY.split(CUT)


@pytest.mark.parametrize("connector", ["并且", "而且", "同时", "另外", "此外", "也", ""])
def test_literal_clause_connectors_are_removed_with_the_preference_not_with_neighboring_facts(connector):
    body = "我本人的课程回执为MC-845-D，" + connector + "我喜欢" + VALUE + "。预约有效且尚未参加。"
    patch = project_retained(retained(body), source(body), erased("我喜欢" + VALUE))
    assert json.loads(patch["evidence_json"]) == ["我本人的课程回执为MC-845-D，", "预约有效且尚未参加。"]


def test_preference_first_keeps_a_later_explicitly_owned_course_in_the_same_sentence():
    body = "我喜欢" + VALUE + "，我的课程回执是MC-845-D且预约有效。"
    patch = project_retained(retained(body), source(body), erased(body))
    assert json.loads(patch["evidence_json"]) == ["，我的课程回执是MC-845-D且预约有效。"]


@pytest.mark.parametrize(
    "evidence", ["深蓝色油墨", "色油墨进行纸版压印", "我喜欢不存在的油墨", "NOT_AN_ORIGINAL_SOURCE"]
)
def test_unknown_or_partially_grounded_old_evidence_cannot_authorize_new_source_cuts(evidence):
    with pytest.raises(MemoryClaimConflict):
        project_retained(retained(), source(), erased(evidence))


@pytest.mark.parametrize(
    "field,value",
    [
        ("content", "朋友喜欢" + VALUE),
        ("metadata_json", json.dumps({"attributed_to": "friend"})),
        ("metadata_json", json.dumps({"attributed_to": "user", "content_semantics": "quoted_source"})),
    ],
)
def test_a_quoted_or_third_party_record_cannot_become_a_normalized_user_preference_cut(field, value):
    with pytest.raises(MemoryClaimConflict):
        project_retained(retained(), source(), erased(**{field: value}))


@pytest.mark.parametrize(
    "template",
    [
        "朋友说：“我喜欢{value}。”课程MC-845-D。",
        "我的课程MC-845-D，我不喜欢{value}。",
        "课程MC-845-D，我喜欢{value}但回执已经作废。",
    ],
)
def test_quotes_negation_and_unknown_compound_predicates_remain_conflicts(template):
    body = template.format(value=VALUE)
    with pytest.raises(MemoryClaimConflict):
        project_retained(retained(body), source(body), erased(body))


def test_repeated_literal_preference_predicates_are_purged_while_other_source_order_is_preserved():
    body = "我的课程回执MC-845-D。我喜欢" + VALUE + "，课程尚未参加。我喜欢" + VALUE + "，也没有出发。"
    patch = project_retained(retained(body), source(body), erased(body))
    assert json.loads(patch["evidence_json"]) == ["我的课程回执MC-845-D。", "，课程尚未参加。", "，也没有出发。"]


@pytest.fixture(scope="module", params=["sqlite", "pg"])
def database(request, tmp_path_factory):
    if request.param == "sqlite":
        db = SQLiteDB(tmp_path_factory.mktemp("coupled-clause") / "records.sqlite")
    else:
        url = os.environ["STAGE40_CLAUSE_PG_URL"]
        assert "/stage3_stage40_clause_guards?" in url and "port=25433" in url
        with pytest.MonkeyPatch.context() as env:
            env.setenv("DATABASE_URL", url)
            from db.pg_database import PgDatabase, SyncPgAdapter
        db = SyncPgAdapter(PgDatabase(url))
    yield db
    (db.close if request.param == "pg" else db.close_connection)()


def seed(db, evidence=EVIDENCE, invalid=False):
    owner = "clause-" + uuid.uuid4().hex
    fields = ("tsukiyashiro_kisaki", "qq", "unit-clause", owner, "private", owner)
    sid = "shared-" + uuid.uuid4().hex
    stamp = datetime.now(timezone.utc)
    assert db.capture_memory_source(*fields, source_message_id=sid, body=BODY, observed_at=stamp) == "recorded"
    r = retained()
    course = db.append_character_memory_claim(
        *fields,
        "shared_event",
        COURSE,
        r["content"],
        evidence_json=r["evidence_json"],
        metadata_json=r["metadata_json"],
        source_message_id=sid,
        observed_at=stamp.isoformat(),
    )
    e = erased(evidence)[0]
    ink = db.append_character_memory_claim(
        *fields,
        "user_fact",
        INK,
        e["content"],
        evidence_json=e["evidence_json"],
        metadata_json=e["metadata_json"],
        source_message_id=sid,
        observed_at=stamp.isoformat(),
    )
    if invalid:
        db.append_character_memory_claim(
            *fields,
            "user_fact",
            "other_fact",
            "用户仍有" + VALUE,
            source_message_id=sid,
            observed_at=stamp.isoformat(),
            evidence_json=json.dumps([BODY]),
        )
    return fields, sid, stamp, course, ink


def rows(db, fields):
    return db.list_character_memory_claims(*fields, limit=None, include_inactive=True)


@pytest.mark.parametrize("evidence", [EVIDENCE, BODY])
def test_sql_real_subsentence_and_whole_multi_fact_evidence_both_preserve_all_course_fields(database, evidence):
    fields, sid, stamp, course, ink = seed(database, evidence)
    old = next(x for x in rows(database, fields) if x["id"] == course["id"])
    assert database.erase_character_memories(*fields, memory_key=INK, protected_memory_keys=(COURSE,)) == 1
    current = rows(database, fields)
    assert len(current) == 1 and current[0]["id"] == course["id"]
    current = current[0]
    assert json.loads(current["evidence_json"]) == BODY.split(CUT)
    assert all(
        current.get(k) == old.get(k) for k in old if k not in {"content", "evidence_json", "metadata_json", "metadata"}
    )
    assert database.list_memory_sources(*fields, limit=100) == []
    assert database.capture_memory_source(*fields, source_message_id=sid, body=BODY, observed_at=stamp) == "stale"


def test_sql_a_remaining_assertion_conflict_rolls_back_all_clause_evidence_and_revocation(database):
    fields, sid, stamp, course, ink = seed(database, invalid=True)
    before = rows(database, fields)
    sources = database.list_memory_sources(*fields, limit=100)
    with pytest.raises(MemoryClaimConflict):
        database.erase_character_memories(*fields, memory_key=INK, protected_memory_keys=(COURSE,))
    assert rows(database, fields) == before and database.list_memory_sources(*fields, limit=100) == sources
    assert (
        database.capture_memory_source(
            *fields,
            source_message_id="pre-erasure-" + uuid.uuid4().hex,
            body="实际此前收到的独立原话",
            observed_at=stamp,
        )
        == "recorded"
    )
