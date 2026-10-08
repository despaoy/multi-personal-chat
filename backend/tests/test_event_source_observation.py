import json

import pytest

from character.memory_llm import parse_llm_proposals


def event(source, *, evidence=None, summary='用户在泉州学陶艺', value='陶艺', operation='ADD', **extra):
    return parse_llm_proposals(json.dumps(dict(memories=[dict(operation=operation, kind='shared_event',
        value=value, evidence=evidence or source, content=summary, confidence=0.99, **extra)])),
        source_message=source)


@pytest.mark.parametrize('source', ['我妹妹在泉州学陶艺。', '我姐姐喜欢我送她的陶艺作品。',
    '我和姐姐一起学陶艺。', '我自己在学陶艺。', '我同事在学陶艺。', '她在学陶艺。'])
def test_event_keeps_source_instead_of_inventing_speaker_as_actor(source):
    proposal, = event(source)
    assert proposal.evidence == source
    assert proposal.memory.content == '用户原话事件记录：' + json.dumps(source, ensure_ascii=False)
    assert proposal.memory.memory_type == 'shared_event'
    assert proposal.source_observation


def test_short_model_quote_cannot_erase_other_actor_or_later_qualification():
    source = '我妹妹在泉州学陶艺，我自己并没有学陶艺。'
    proposal, = event(source, evidence='在泉州学陶艺')
    assert proposal.evidence == source
    assert '我自己并没有' in proposal.memory.content


def test_long_source_is_not_truncated_or_discarded():
    source = '我妹妹在学陶艺。' + '她给我介绍了各种制作方法。' * 12 + '我自己并没有参与。'
    proposal, = event(source, evidence='我妹妹在学陶艺')
    assert proposal.evidence == source
    assert proposal.memory.content == '用户原话事件记录（完整内容见证据）'


def test_event_speaker_is_provenance_not_model_actor_label():
    source = '我同事在学陶艺。'
    proposal, = event(source, attributed_to='third_party')
    assert proposal.attributed_to == 'user'  # The actual source speaker, not the event actor.
    assert proposal.source_observation
    assert proposal.evidence == source


def test_non_event_fields_are_not_silently_converted_to_quotations():
    source = '我喜欢陶艺。'
    proposal, = parse_llm_proposals(json.dumps(dict(memories=[dict(operation="ADD", kind='like', value='陶艺',
        evidence=source, content='用户喜欢陶艺', confidence=0.99)])), source_message=source)
    assert proposal.memory.content == '用户喜欢陶艺'
    assert not proposal.source_observation


def test_pending_prefix_and_qualifiers_never_cut_off_quoted_tail():
    source = '我可能去参加陶艺活动，' + '具体安排还没确定' * 16 + '但不会独自去。'
    proposal, = event(source, evidence='我可能去参加陶艺活动，具体安排还没确定', operation='PENDING',
                      qualifiers={'condition': '具体安排还没确定'})
    assert proposal.evidence.endswith('但不会独自去。')
    assert proposal.memory.content == '待确认：用户原话事件记录（完整内容见证据）'


def test_event_merge_accumulates_observations_without_erasing_old_details():
    source = '陶艺课增加了拉坯练习。'
    raw = dict(kind='shared_event', value='拉坯练习', evidence=source,
               content='用户参加拉坯练习', confidence=0.99, operation='MERGE',
               target_memory_id='7', target_memory_key='event_陶艺课')
    proposal, = parse_llm_proposals(json.dumps(dict(memories=[raw])), source_message=source,
        existing_memories=({'id': '7', 'memory_type': 'shared_event', 'memory_key': 'event_陶艺课',
                            'status': 'active', 'content': '用户和姐姐一起参加陶艺课'},))
    assert proposal.operation == 'COEXIST'
    assert proposal.target_memory_id == '7'
    assert proposal.source_observation
    assert proposal.evidence == source


@pytest.mark.asyncio
async def test_event_observation_merge_retains_prior_record_and_provenance(tmp_path):
    from test_memory_write_lifecycle import _Completion, _scope

    from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig
    from character.models import MemoryItem
    from db.database import SQLiteDB
    from repositories.character_memory import DatabaseCharacterMemoryRepository

    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / 'events.db'))
    scope = _scope()
    old = await repo.append_claim('kisaki', scope, MemoryItem('', 'shared_event', '以前的陶艺课记录', 0.7),
                                  memory_key='event_陶艺课')
    source = '陶艺课增加了拉坯练习。'
    completion = _Completion(json.dumps(dict(memories=[dict(kind='shared_event', value='拉坯练习',
        evidence=source, confidence=0.99, operation='MERGE', target_memory_id=str(old['id']),
        target_memory_key='event_陶艺课')]), ensure_ascii=False))
    class FixedEmbedding:
        def embed_texts(self, texts):
            return [[1.0] for _ in texts]

    scheduler = MemoryEnrichmentScheduler(config=MemoryLlmConfig(enabled=True, base_url='http://unused',
        model='test'), completion=completion, embedding_provider=FixedEmbedding())
    try:
        assert scheduler.schedule(repository=repo, character_id='kisaki', user_scope=scope,
            message=source, rule_hints=[], source_message_id='new-event')
        assert await scheduler.flush_memory(timeout=3)
        rows = await repo.list_memory_records('kisaki', scope, limit=None, include_inactive=True)
        assert len(rows) == 2 and all(row['status'] == 'active' for row in rows)
        current = next(row for row in rows if row['source_message_id'] == 'new-event')
        assert current['parent_memory_id'] == old['id']
        assert current['supersedes_memory_id'] is None
        assert current['metadata']['content_semantics'] == 'quoted_source'
        assert current['metadata']['described_subject'] == 'not_resolved'
        assert current['evidence'] == [source]
    finally:
        await scheduler.shutdown(timeout=3)
