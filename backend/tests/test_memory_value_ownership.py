"""Model-proposed objects retain polarity and are never guessed from a neighbour."""

import json

import httpx
import pytest

from character.memory_llm import (
    MemoryEnrichmentScheduler, MemoryLlmConfig, MemoryResponseError,
    OpenAICompatibleMemoryCompletion, parse_llm_proposals,
)
from character.models import UserScope
from db.database import SQLiteDB
from repositories.character_memory import DatabaseCharacterMemoryRepository


def candidate(kind, value, evidence, **changes):
    return dict(operation='ADD',kind=kind,value=value,evidence=evidence,confidence=.99,attributed_to='user',**changes)


def parse(source, *rows):
    return parse_llm_proposals(json.dumps({'memories':rows},ensure_ascii=False),source_message=source)


@pytest.mark.parametrize('source',[
    '请记住：我喜欢木工；我不喜欢紫樱茶。',
    '请记住：我不喜欢紫樱茶；我喜欢木工。',
    '请记住：我喜欢木工，但紫樱茶我不喝。',
])
def test_other_objects_negation_cannot_rewrite_a_correct_positive_proposal(source):
    rows=parse(source,candidate('like','木工',source),candidate('dislike','紫樱茶',source))
    by_key={row.memory.memory_key:row for row in rows}
    assert set(by_key)=={'preference_木工','preference_紫樱茶'}
    assert by_key['preference_木工'].memory.content=='用户说喜欢木工'
    assert by_key['preference_紫樱茶'].memory.content=='用户说不喜欢紫樱茶'
    assert all(row.evidence==source for row in rows)


def test_negative_exception_does_not_reverse_the_positive_preference():
    source='我喜欢咖啡，但晚上不喝。'
    (proposal,)=parse(source,candidate('like','咖啡',source,qualifiers={'exception':'晚上不喝'}))
    assert '用户说喜欢咖啡' in proposal.memory.content
    assert '用户说不喜欢咖啡' not in proposal.memory.content
    assert proposal.qualifiers==( ('exception','晚上不喝'), )
    assert proposal.evidence==source


@pytest.mark.parametrize('value',['准备保研面试','正在准备绘画展'])
def test_goal_index_keeps_the_valid_literal_model_object(value):
    source=f'请记住，我{value}。'
    (proposal,)=parse(source,candidate('goal',value,source))
    assert proposal.memory.memory_key=='goal_'+value
    assert value in proposal.memory.content and proposal.evidence==source


@pytest.mark.parametrize('kind,value,source',[
    ('dislike','咖啡因刺激饮品','我喝无咖啡因咖啡，普通咖啡才不喝。'),
    ('like','咖啡','我喜欢红茶。'),
    ('goal','深海航行','我正在准备绘画展。'),
])
def test_unsupported_structured_object_is_an_explicit_failure_not_a_replacement(kind,value,source):
    with pytest.raises(MemoryResponseError) as caught:
        parse(source,candidate(kind,value,source))
    assert caught.value.field=='value'
    assert value not in str(caught.value) and source not in str(caught.value)


@pytest.mark.parametrize('failure',[False,True])
async def test_actual_writer_preserves_both_polarities_or_rejects_the_batch_atomically(tmp_path,failure):
    database=SQLiteDB(tmp_path/'objects.sqlite')
    repository=DatabaseCharacterMemoryRepository(database)
    scope=UserScope('test','value-contract','fiction','fiction','private')
    source='请记住：我喜欢木工；我不喜欢紫樱茶。'
    rows=[candidate('like','木工',source),candidate('dislike','紫樱茶',source)]
    if failure:
        rows.append(candidate('dislike','无证据饮品',source))
    calls=[]
    def respond(request):
        payload=json.loads(request.content)
        calls.append(payload)
        assert json.loads(payload['messages'][-1]['content'])['current_user_message']==source
        return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'memories':rows},ensure_ascii=False)}}],
            'usage':{'prompt_tokens':110,'completion_tokens':30,'total_tokens':140}})
    config=MemoryLlmConfig(True,'https://fixture.invalid/v1','fixture-model',api_key='fiction',idle_seconds=0)
    completion=OpenAICompatibleMemoryCompletion(config)
    completion._client=httpx.AsyncClient(transport=httpx.MockTransport(respond))
    scheduler=MemoryEnrichmentScheduler(config=config,completion=completion)
    try:
        result=await scheduler.schedule_and_wait(repository=repository,character_id='role',user_scope=scope,
            message=source,rule_hints=(),source_message_id='complete-source')
        records=await repository.list_memory_records('role',scope,include_inactive=True)
        usage=database.execute_sql('SELECT * FROM model_invocations')
        assert len(calls)==len(usage)==1
        assert usage[0]['purpose']=='memory_judgement' and usage[0]['totalTokens']==140
        assert usage[0]['usageSource']=='provider'
        assert result['source_capture']=='recorded'
        sources=await repository.list_sources('role',scope)
        assert len(sources)==1 and sources[0]['body']==source
        if failure:
            assert result['status']=='failed' and result['stage']=='proposal_validation'
            assert result['field']=='value' and result['accepted']==result['persisted']==0
            assert usage[0]['errorType']=='MemoryResponseError' and records==[]
        else:
            assert result['status']=='saved' and result['persisted']==len(records)==2
            by_key={row['memory_key']:row for row in records}
            assert by_key['preference_木工']['content']=='用户说喜欢木工'
            assert by_key['preference_紫樱茶']['content']=='用户说不喜欢紫樱茶'
            assert usage[0]['errorType']==''
            assert all(row['source_message_id']=='complete-source' for row in records)
    finally:
        await scheduler.shutdown(timeout=1)
        database.close_connection()


def test_previous_contrast_repair_input_fails_without_replacing_the_model_object():
    source="准确地说，我喝无咖啡因咖啡，普通咖啡才不喝。"
    row=dict(operation="MERGE",kind="dislike",value="含咖啡因的饮料",content="用户不喜欢含咖啡因的饮料",
             evidence=source,confidence=.95,attributed_to="user",target_memory_id="7",target_memory_key="preference_咖啡")
    existing=({"id":7,"memory_key":"preference_咖啡","memory_type":"user_fact","content":"用户说不喜欢咖啡","status":"active"},)
    with pytest.raises(MemoryResponseError) as caught:
        parse_llm_proposals(json.dumps({"memories":[row]},ensure_ascii=False),source_message=source,existing_memories=existing)
    assert caught.value.field=="value"
