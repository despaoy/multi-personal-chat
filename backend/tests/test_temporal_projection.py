"""Unknown model lifetimes remain source references, never present fact slots."""

import json
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from character.context_builder import _complete_memory_evidence_packet, _memory_evidence_packet, _memory_is_injectable
from character.evidence_selector import selection_messages
from character.memory_extractor import extract_memories
from character.memory_service import CharacterMemoryService, _HistoricalWindow, _is_historical_record
from character.models import CompiledCharacterContext, MemoryItem, UserScope
from character.temporal_projection import project_temporal_record
from character.temporal_provenance import model_temporal_provenance
from inference.memory_response import read_memory_fields, render_memory_response

OBSERVED = datetime(2026, 9, 27, 8, tzinfo=timezone.utc)
NOW = OBSERVED + timedelta(minutes=1)


@pytest.mark.parametrize('text,key,content', [
    ('更正一下，我现在在天文馆工作，不在原来的单位了。', 'user_workplace', '用户说自己在天文馆工作'),
    ('我在档案馆工作，不在以前的工作地点了。', 'user_workplace', '用户说自己在档案馆工作'),
    ('纠正一下，我住在柳州，不在原来的住处了。', 'user_residence', '用户说自己居住在柳州'),
    ('更正一下，我的专业是地质学，不是经济学。', 'user_major', '用户说自己的专业是地质学'),
    ('纠正一下，我叫云岚，不是云林。', 'user_name', '用户说自己叫云岚'),
])
def test_complete_scalar_correction_is_an_asserted_state(text, key, content):
    row = record(text, '用户自然摘要', key)
    before = deepcopy(row)
    view = project_temporal_record(row)
    assert view['temporal_mode'] == 'asserted_state'
    assert view['content'] == content
    assert view['evidence'] == [text]
    assert row == before


@pytest.mark.parametrize('text', [
    '更正一下，我明年在天文馆工作，不在原来的单位了。',
    '更正一下，如果调岗，我在天文馆工作，不在原来的单位了。',
    '更正一下，我在天文馆工作，但这只是设想。',
    '更正一下，我在天文馆工作，不在原来的单位了吗？',
    '更正一下，我妹妹在天文馆工作，不在原来的单位了。',
    '更正一下，我在天文馆工作，她不在原来的单位了。',
    '更正一下，我在天文馆工作，不在原来的住处了。',
    '更正一下，我在天文馆工作，不在原来的单位了，下周离职。',
])
def test_correction_view_does_not_drop_unknown_scope_or_negation(text):
    assert project_temporal_record(record(text, key='user_workplace'))['temporal_mode'] == 'observation'


def record(evidence, content="用户临时在外地", key="user_location", **extra):
    base = dict(id="m1", memory_key=key, memory_type="user_fact", content=content,
                status="active", relation_type="ADD", confidence=.96,
                observed_at=OBSERVED.isoformat(), valid_from=OBSERVED.isoformat(), valid_to=OBSERVED.isoformat(),
                evidence=[evidence], source_message_ids=["source1"], metadata={"qualifiers": {},
                "temporal_provenance": model_temporal_provenance(evidence=evidence, observed_at=OBSERVED,
                    proposed_from=OBSERVED.isoformat(), proposed_to=OBSERVED.isoformat())})
    base.update(extra)
    return base


@pytest.mark.parametrize('prefix', ['请记住，', '请记住', '记下：', '请记下一下：'])
@pytest.mark.parametrize('statement', ['我叫云岚。', '我住在宜昌。', '我的专业是海洋学。', '我在档案馆工作。'])
def test_management_wrapper_does_not_downgrade_whole_supported_statement(prefix, statement):
    item, = extract_memories(statement)
    row = record(prefix + statement, '用户自然概括', item.memory_key)
    before = deepcopy(row)
    view = project_temporal_record(row)
    assert view['temporal_mode'] == 'asserted_state'
    assert view['content'] == item.content
    assert view['evidence'] == [prefix + statement]
    assert row == before


@pytest.mark.parametrize('statement', [
    '请记住，如果调岗，我住在宜昌。',
    '请记住，我明年住在宜昌。',
    '请记住，我住在宜昌，下周搬走。',
    '请记住，我住在宜昌吗？',
    '不要记住，我住在宜昌。',
    '请记住，“我住在宜昌”是示例。',
    '朋友让我记住，我住在宜昌。',
    '请记住，我妹妹住在宜昌。',
    '请记住，我住在宜昌，但这只是设想。',
])
def test_management_wrapper_preserves_scope_and_qualification_boundaries(statement):
    assert project_temporal_record(record(statement, key='user_residence'))['temporal_mode'] == 'observation'


@pytest.mark.asyncio
async def test_wrapped_name_survives_recall_and_storage_status_rendering():
    row = record('请记住，我叫云岚。', '用户叫云岚', 'user_name')
    _, _, context = await recall([row], '你现在还保存着我的姓名吗？')
    assert dict(context.memory_field_presence)['name'] is True
    assert '有你的姓名记录' in render_memory_response('你现在还保存着我的姓名吗？', context)
    assert '云岚' in render_memory_response('我叫什么名字？', context)


@pytest.mark.parametrize("text,content,key,expected", [
    ("我现在住在合肥。", "用户说自己居住在合肥", "user_residence", "asserted_state"),
    ("我不喜欢芹菜。", "用户说不喜欢芹菜", "preference_芹菜", "asserted_state"),
    ("我明年住在合肥。", "用户说自己居住在合肥", "user_residence", "observation"),
    ("去年我住在合肥。", "用户说自己居住在合肥", "user_residence", "observation"),
    ("如果调岗，我住在合肥。", "用户说自己居住在合肥", "user_residence", "observation"),
    ("我住在合肥到年底。", "用户说自己居住在合肥", "user_residence", "observation"),
    ("我不吃芹菜了。", "用户说不喜欢芹菜", "preference_芹菜", "observation"),
])
def test_model_date_presence_and_value_cannot_determine_assertion_applicability(text, content, key, expected):
    views = []
    for start, end in [("", ""), (OBSERVED.isoformat(), ""),
                       ("", "2030-01-01T00:00:00+08:00"),
                       ("2030-01-01T00:00:00+08:00", "2020-01-01T00:00:00+08:00")]:
        row = record(text, content, key, valid_from=start, valid_to=end)
        row["metadata"]["temporal_provenance"] = model_temporal_provenance(
            evidence=text, observed_at=OBSERVED, proposed_from=start, proposed_to=end)
        before = deepcopy(row)
        view = project_temporal_record(row)
        assert view.get("temporal_mode", "fact") == expected
        assert row == before
        views.append({field: view.get(field) for field in (
            "content", "retrieval_content", "valid_from", "valid_to", "valid_at", "invalid_at",
            "temporal_mode", "temporal_observed_at")})
    assert all(view == views[0] for view in views)


@pytest.mark.asyncio
@pytest.mark.parametrize("text,value", [
    ("我现在住在合肥。", "合肥"),
    ("我明年住在合肥。", None),
])
async def test_missing_dates_follow_same_recall_and_typed_read_contract(text, value):
    row = record(text, "用户说自己居住在合肥", "user_residence", valid_from="", valid_to="")
    row["metadata"]["temporal_provenance"] = model_temporal_provenance(evidence=text, observed_at=OBSERVED)
    items, _, context = await recall([row], "我现在住哪里？")
    assert items
    result, = read_memory_fields("我现在住哪里？", context)
    if value is not None:
        assert result.value == value
        assert "合肥" in render_memory_response("我现在住哪里？", context)
    else:
        assert result.status == "unverified"
        assert render_memory_response("我现在住哪里？", context) is None


@pytest.mark.parametrize("text", ["我叫林溪。", "我来自泉州。", "我现在住在合肥。", "我的专业是地质学。",
                                  "我现在在观测站工作。", "我是大二学生。", "我喜欢咖啡。"])
def test_independent_stable_assertion_uses_observation_not_guessed_expiry(text):
    item, = extract_memories(text)
    row = record(text, item.content, item.memory_key)
    before = deepcopy(row)
    projected = project_temporal_record(row)
    assert projected["temporal_mode"] == "asserted_state"
    assert projected["valid_from"] == OBSERVED.isoformat()
    assert projected["valid_to"] == ""
    assert projected["content"] == row["content"]
    assert row == before  # Read views must not migrate stored records.


@pytest.mark.parametrize("text,content,key", [
    ("我这周在南宁出差，下周回去。", "用户这周在南宁出差", "user_location"),
    ("我现在住在合肥，下周搬走。", "用户说自己居住在合肥", "user_residence"),
    ("今年刚升大三", "用户说自己是大三", "user_study_stage"),
    ("我明年住在合肥", "用户说自己居住在合肥", "user_residence"),
    ("去年我住在合肥", "用户说自己居住在合肥", "user_residence"),
    ("明年我住在合肥", "用户说自己居住在合肥", "user_residence"),
])
def test_unknown_or_partially_supported_statement_is_a_quote(text, content, key):
    row = record(text, content, key)
    view = project_temporal_record(row)
    assert view["temporal_mode"] == "observation"
    assert text in view["content"] and OBSERVED.isoformat() in view["content"]
    assert "不代表当前状态" in view["content"]
    assert view["evidence"] == row["evidence"]
    assert view["valid_to"] == ""


@pytest.mark.parametrize("metadata", [{}, {"origin": "rule_v2"},
    {"temporal_provenance": {"version": 1, "producer": "rule", "validity_authority": "unspecified"}},
    {"temporal_provenance": {"version": 1, "producer": "semantic_memory", "validity_authority": "verified"}},
    {"temporal_provenance": {"version": 1, "producer": "semantic_memory", "validity_authority": []}},
    {"temporal_provenance": {"version": 2, "producer": "semantic_memory", "validity_authority": "unverified"}}])
def test_legacy_and_other_producers_are_not_migrated(metadata):
    row = record("我现在住在合肥。", metadata=metadata)
    assert project_temporal_record(row) is row


class Repo:
    def __init__(self, rows):
        self.rows = rows

    async def list_memory_records(self, *args, **kwargs):
        return deepcopy(self.rows)


async def recall(rows, query, **kwargs):
    service = CharacterMemoryService(Repo(rows), semantic_enabled=False)
    trace = {}
    items, _ = await service.load_relevant_memories("role", UserScope("web", "test", "u", "c", "private"), query,
                                    diagnostics=trace, reference_time=NOW, **kwargs)
    context = CompiledCharacterContext("", "", "", tuple(item.memory_id for item in items),
        memory_status="available", memory_packets=items,
        memory_field_presence=tuple(trace.get("field_presence", {}).items()))
    return items, trace, context


@pytest.mark.asyncio
async def test_zero_width_source_survives_actual_recall_and_compilation():
    row = record("我这周在南宁出差，下周回去。")
    items, trace, _ = await recall([row], "我在南宁出差的安排呢？")
    assert len(items) == 1
    item = items[0]
    assert item.temporal_mode == "observation"
    assert item.valid_from == item.valid_to == ""
    assert item.observed_at == OBSERVED.isoformat()
    assert _memory_is_injectable(item, NOW)
    assert not _memory_is_injectable(item, OBSERVED - timedelta(seconds=1))
    packet = _memory_evidence_packet(item)
    assert "有效期" not in packet and "有效时间未核实" in packet
    complete = json.loads(_complete_memory_evidence_packet(item)[2:])
    assert complete["temporal_mode"] == "observation"
    selected = json.loads(selection_messages("出差安排呢？", items)[-1]["content"])
    assert selected["candidates"][0]["observed_at"] == OBSERVED.isoformat()
    assert trace["temporal_views"]["observation"] == 1


@pytest.mark.asyncio
async def test_observation_is_neither_current_typed_value_nor_proven_absence():
    row = record("我明年住在合肥", "用户说自己居住在合肥", "user_residence")
    items, trace, context = await recall([row], "我现在住哪里？")
    assert items and trace["field_presence"]["residence"] is None
    assert read_memory_fields("我现在住哪里？", context)[0].status == "unverified"
    assert render_memory_response("我现在住哪里？", context) is None
    # Even a consumer that drops storage diagnostics cannot read it as a fact.
    assert read_memory_fields("我现在住哪里？", replace(context, memory_field_presence=()))[0].status == "unverified"


@pytest.mark.asyncio
async def test_stable_residence_read_survives_unverified_temporary_location():
    text = "我现在住在合肥。"
    stable, = extract_memories(text)
    rows = [record(text, stable.content, stable.memory_key),
            record("我这周在南宁出差，下周回去。", id="m2")]
    _, trace, context = await recall(rows, "我现在住哪里？")
    assert trace["field_presence"]["residence"] is True
    assert read_memory_fields("我现在住哪里？", context)[0].value == "合肥"
    assert "合肥" in render_memory_response("我现在住哪里？", context)


@pytest.mark.asyncio
@pytest.mark.parametrize("status,relation", [("retracted", "RETRACT"), ("erased", "ERASE"),
                                               ("pending", "PENDING"), ("superseded", "SUPERSEDE")])
async def test_source_view_cannot_revive_noncurrent_lifecycle(status, relation):
    items, _, _ = await recall([record("我这周在南宁出差", status=status, relation_type=relation)], "南宁出差呢？")
    assert not items


def test_packet_flag_is_enforced_even_with_canonical_content():
    item = MemoryItem("1", "user_fact", "用户说自己居住在合肥", memory_key="user_residence",
                      temporal_mode="observation", observed_at=OBSERVED.isoformat(),
                      evidence=("我现在住在合肥。",), source_message_ids=("source",))
    context = CompiledCharacterContext("", "", "", ("1",), memory_status="available", memory_packets=(item,))
    assert read_memory_fields("我现在住哪里？", context)[0].status == "unverified"


@pytest.mark.asyncio
@pytest.mark.parametrize('source,summary,key,query,expected', [
    ('我刚搬到岳阳了。', '用户搬家到了岳阳', 'user_residence', '我现在住哪里？', '岳阳'),
    ('我的专业是测绘。', '用户在学测绘专业', 'user_major', '我的专业是什么？', '测绘'),
    ('我叫顾舟。', '用户的名字为顾舟', 'user_name', '我叫什么？', '顾舟'),
])
async def test_natural_summary_does_not_erase_source_supported_field(source, summary, key, query, expected):
    row = record(source, summary, key)
    before = deepcopy(row)
    _, _, context = await recall([row], query)
    result, = read_memory_fields(query, context)
    assert result.status == 'known' and result.value == expected
    assert row == before


def test_disagreeing_source_values_are_not_resolved_by_display_summary():
    row = record('我现在住在宜春。', '用户住在宜春', 'user_residence')
    row['evidence'].append('我现在住在镇江。')
    assert project_temporal_record(row)['temporal_mode'] == 'observation'


def test_source_not_model_summary_controls_canonical_read_value():
    row = record('我的专业是测绘。', '用户的专业是音乐', 'user_major')
    view = project_temporal_record(row)
    assert view['content'] == '用户说自己的专业是测绘'
    assert row['content'] == '用户的专业是音乐'


@pytest.mark.parametrize("offset,expected", [(-1, False), (0, True), (1, False)])
def test_historical_source_is_an_observation_not_an_infinite_fact(offset, expected):
    row = project_temporal_record(record("我这周在南宁出差"))
    start = OBSERVED + timedelta(days=offset)
    window = _HistoricalWindow(start=start, end=start + timedelta(days=1))
    assert _is_historical_record(row, window, include_pending=False) is expected


def test_superseded_model_interval_cannot_be_reopened_as_a_stable_history_fact():
    row = project_temporal_record(record("我现在住在合肥。", "用户说自己居住在合肥", "user_residence",
                                         status="superseded"))
    assert row["temporal_mode"] == "observation"


@pytest.mark.parametrize("text", [
    "我住在合肥到年底。", "我住在合肥三个月。", "我在观测站工作直到毕业。",
    "我喜欢咖啡只限这周。", "我住在合肥到2027-01-01。", "我住在合肥到2027年1月1日。",
])
def test_temporal_suffix_consumed_as_rule_value_does_not_prove_stability(text):
    # The old extractor may consume a temporal suffix as part of an entity.
    # Matching its content alone is not independent proof of a lasting state.
    candidates = extract_memories(text)
    assert candidates
    for item in candidates:
        view = project_temporal_record(record(text, item.content, item.memory_key))
        assert view["temporal_mode"] == "observation"


@pytest.mark.asyncio
async def test_explicit_source_review_recovers_model_statement_without_asserting_current_fact():
    row = record("我明年住在合肥", "用户说自己居住在合肥", "user_residence")
    items, trace, _ = await recall([row], "我说过哪些住址？")
    # The existing source-review grammar uses 居住地, not arbitrary synonyms.
    assert trace.get("mention_review") is None
    items, trace, _ = await recall([row], "我说过哪些居住地？")
    assert not items
    assert "我明年住在合肥" in trace["mention_review"]
    assert "时效未核实" in trace["mention_review"]
    assert OBSERVED.isoformat() in trace["mention_review"]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["retracted", "erased", "deleted"])
async def test_source_review_does_not_resurrect_removed_statement(status):
    _, trace, _ = await recall([record("我明年住在合肥", key="user_residence", status=status)],
                               "我说过哪些居住地？")
    assert "我明年住在合肥" not in trace["mention_review"]


def test_semantic_ranking_uses_source_not_presentation_and_caches_that_view():
    class Provider:
        calls = []

        def embed_texts(self, texts):
            self.calls.append(texts)
            return [[1., 0.] for _ in texts]

    provider = Provider()
    service = CharacterMemoryService(Repo([]), semantic_enabled=True, embedding_provider=provider)
    source = "我这周在南宁出差，下周回去。"
    view = project_temporal_record(record(source))
    service._semantic_similarities("出差安排呢？", [view])
    assert provider.calls[0] == ["出差安排呢？", source]
    view["content"] = "展示标签改变：" + view["content"]
    service._semantic_similarities("之前的出差呢？", [view])
    assert provider.calls[1] == ["之前的出差呢？"]


@pytest.mark.asyncio
async def test_lexical_ranking_ignores_temporal_display_boilerplate():
    source = "我住在海边。"
    row = record(source, "非规范摘要", "fact_location")
    service = CharacterMemoryService(Repo([row]), semantic_enabled=False)
    # A presentation-label match alone is not source relevance. The previous
    # view matched this exact added phrase and was incorrectly admitted.
    items, _ = await service.load_relevant_memories(
        "role", UserScope("web", "test", "u", "c", "private"), "时效未核实，不代表当前状态", reference_time=NOW)
    assert not items
