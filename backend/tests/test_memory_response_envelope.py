"""Equivalent JSON containers must preserve candidate validation and atomic parsing."""

import json

import pytest

from character.memory_llm import parse_llm_proposals


def candidate(value="物理学"):
    return dict(kind="major", value=value, evidence="我的专业是物理学", confidence=.95, operation="ADD")


@pytest.mark.parametrize("prefix,suffix", [("", ""), ("```json\n", "\n```")])
def test_bare_candidate_array_matches_envelope(prefix, suffix):
    source = "我的专业是物理学。"
    expected = parse_llm_proposals(json.dumps({"memories": [candidate()]}), source_message=source)
    actual = parse_llm_proposals(prefix + json.dumps([candidate()]) + suffix, source_message=source)
    assert actual == expected
    assert len(actual) == 1


def test_bare_array_does_not_bypass_evidence_validation():
    assert parse_llm_proposals(json.dumps([candidate("考古学")]), source_message="我的专业是物理学。") == []


@pytest.mark.parametrize("raw", ["{}", '{"result": []}', '{"kind":"major"}',
                                '[' + json.dumps(candidate()) + ','])
def test_invalid_or_truncated_container_is_failure_not_no_change(raw):
    with pytest.raises(ValueError):
        parse_llm_proposals(raw, source_message="我的专业是物理学。")


@pytest.mark.parametrize("raw", ['{"memories":[]}', '[]'])
def test_explicit_empty_collection_is_no_change(raw):
    assert parse_llm_proposals(raw, source_message="我的专业是物理学。") == []


@pytest.mark.parametrize('raw', [
    '结果如下：{"memories":[]}', '{"memories":[]} private-tail',
    '{"memories":[]} {"memories":[]}', '```json\n{"memories":[]}',
    '```json\n{"memories":[]}\n``` private-tail',
    '{"memories":[null]}', '{"memories":[17]}', '{"memories":[[]]}',
    '{"memories":["private-candidate"]}', '{"memories":null}',
    '{"memories":[],"memories":[]}',
    '{"memories":[{"confidence":NaN}]}', '{"memories":[{"confidence":Infinity}]}',
    '{"memories":[{"confidence":-Infinity}]}',
])
def test_invalid_envelope_fails_without_salvaging_partial_json(raw):
    with pytest.raises(ValueError) as caught:
        parse_llm_proposals(raw, source_message='我的专业是物理学。')
    assert 'private-' not in str(caught.value)


def test_excess_candidates_fail_instead_of_silent_truncation():
    with pytest.raises(ValueError, match='上限'):
        parse_llm_proposals(json.dumps({'memories': [candidate()] * 5}), source_message='我的专业是物理学。')


@pytest.mark.asyncio
@pytest.mark.parametrize('case', ['valid', 'empty', 'trailing', 'mixed', 'excess', 'confidence', 'low_confidence'])
async def test_scheduler_preserves_source_but_never_writes_partial_invalid_response(tmp_path, case):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from repositories.character_memory import DatabaseCharacterMemoryRepository

    from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig
    from character.models import UserScope
    from db.database import SQLiteDB
    database = SQLiteDB(tmp_path / 'envelope.sqlite')
    repo = DatabaseCharacterMemoryRepository(database)
    scope = UserScope('web', 'envelope-test', 'reader', 'reader', 'private')
    raw = json.dumps({'memories': [candidate()]})
    if case == 'empty':
        raw = '{"memories":[]}'
    elif case == 'trailing':
        raw += ' private-tail'
    elif case == 'mixed':
        raw = json.dumps({'memories': [candidate(), None]})
    elif case == 'excess':
        raw = json.dumps({'memories': [candidate()] * 5})
    elif case == 'confidence':
        raw = json.dumps({'memories': [candidate(), dict(candidate(), confidence=True)]})
    elif case == 'low_confidence':
        raw = json.dumps({'memories': [dict(candidate(), confidence=0.1)]})
    completion = SimpleNamespace(complete=AsyncMock(return_value=raw), close=AsyncMock())
    worker = MemoryEnrichmentScheduler(config=MemoryLlmConfig(True, 'unused', 'fixture'), completion=completion)
    try:
        receipt = await worker.schedule_and_wait(repository=repo, character_id='role', user_scope=scope,
            message='我的专业是物理学。', rule_hints=(), source_message_id='source')
        assert receipt['source_capture'] == 'recorded'
        completion.complete.assert_awaited_once()
        records = await repo.list_memory_records('role', scope)
        if case == 'valid':
            assert receipt['status'] == 'saved' and receipt['persisted'] == len(records) == 1
        elif case in {'empty', 'low_confidence'}:
            assert receipt['status'] == 'no_change' and not records
        else:
            assert receipt['status'] == 'failed' and receipt['stage'] == 'proposal_validation'
            assert receipt['accepted'] == receipt['persisted'] == 0 and not records
            assert receipt['error'] == 'ValueError' and 'private-tail' not in json.dumps(receipt)
    finally:
        await worker.shutdown(timeout=1)
        database.close_connection()


@pytest.mark.parametrize('confidence', [None, True, False, '0.95', 'private-value', [], {}, -0.1, 1.1, 1e100])
def test_invalid_confidence_is_protocol_failure(confidence):
    raw = dict(candidate(), confidence=confidence)
    with pytest.raises(ValueError, match='confidence') as caught:
        parse_llm_proposals(json.dumps({'memories': [raw]}), source_message='我的专业是物理学。')
    assert 'private-value' not in str(caught.value)


@pytest.mark.parametrize('operation', ['ADD', 'NOOP', 'IGNORE'])
def test_missing_confidence_fails_before_business_filtering(operation):
    raw = dict(candidate(), operation=operation)
    del raw['confidence']
    with pytest.raises(ValueError, match='confidence'):
        parse_llm_proposals(json.dumps({'memories': [raw]}), source_message='我的专业是物理学。')


@pytest.mark.parametrize('confidence,accepted', [(0, False), (0.1, False), (0.95, True), (1, True)])
def test_valid_numeric_confidence_preserves_admission(confidence, accepted):
    raw = dict(candidate(), confidence=confidence)
    result = parse_llm_proposals(json.dumps({'memories': [raw]}), source_message='我的专业是物理学。')
    assert bool(result) is accepted
