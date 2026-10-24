"""Semantic operations belong to the model; storage owns target/version checks."""

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import httpx

from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig, MemoryResponseError, OpenAICompatibleMemoryCompletion, parse_llm_proposals
from character.models import MemoryItem, UserScope
from db.database import SQLiteDB
from repositories.character_memory import DatabaseCharacterMemoryRepository

GOAL_SOURCE='我正在准备保研，重点练习英语面试。'
MAJOR_SOURCE='我的专业改成材料工程。至于咖啡，不是完全讨厌，只是不喜欢太苦的。'
PET_SOURCE='我又养了一只叫团子的猫。'


def old(id,key,content,type='user_fact'):
    return dict(id=id,memory_key=key,content=content,memory_type=type,status='active')


def raw(operation,kind,value,evidence,**extra):
    return dict(operation=operation,kind=kind,value=value,evidence=evidence,confidence=.99,attributed_to='user',**extra)


def parse(source,row,existing):
    return parse_llm_proposals(json.dumps({'memories':[row]},ensure_ascii=False),source_message=source,existing_memories=existing)


def goal(target):
    return raw('MERGE','goal','保研',GOAL_SOURCE,content='用户正在准备保研，重点练习英语面试',
        target_memory_id=str(target['id']),target_memory_key=target['memory_key'])


def major(target):
    return raw('SUPERSEDE','major','材料工程','我的专业改成材料工程。',content='用户说自己的专业是材料工程',
        target_memory_id=str(target['id']),target_memory_key=target['memory_key'])


def test_merge_short_index_does_not_hide_new_details():
    target=old(7,'goal_保研','用户正在准备保研','shared_event')
    (proposal,)=parse(GOAL_SOURCE,goal(target),(target,))
    assert proposal.operation=='MERGE' and proposal.target_memory_id=='7'
    assert proposal.memory.memory_key=='goal_保研' and '英语面试' in proposal.memory.content
    assert '用户正在准备保研' in proposal.memory.content and proposal.evidence==GOAL_SOURCE


def test_unrelated_condition_cannot_rewrite_a_major_replacement():
    target=old(7,'user_major','用户说自己的专业是环境工程')
    (proposal,)=parse(MAJOR_SOURCE,major(target),(target,))
    assert proposal.operation=='SUPERSEDE' and proposal.target_memory_id=='7'
    assert proposal.memory.content=='用户说自己的专业是材料工程'
    assert proposal.evidence=='我的专业改成材料工程。'


@pytest.mark.parametrize('reverse',[False,True])
def test_new_cat_never_auto_links_to_either_existing_pet(reverse):
    existing=[old(7,'fact_dog','用户养了一只叫豆包的狗'),old(8,'fact_cat','用户养了一只叫年糕的猫')]
    if reverse:existing.reverse()
    (proposal,)=parse(PET_SOURCE,raw('ADD','other_user_fact','团子',PET_SOURCE,content='用户养了一只叫团子的猫'),tuple(existing))
    assert proposal.operation=='ADD'
    assert proposal.target_memory_id==proposal.target_memory_key==''
    assert proposal.memory.memory_key=='fact_团子' and '豆包' not in proposal.memory.content


def test_explicit_pet_merge_uses_the_selected_cat_and_keeps_its_history():
    dog=old(7,'fact_dog','用户养了一只叫豆包的狗')
    cat=old(8,'fact_cat','用户养了一只叫年糕的猫')
    row=raw('MERGE','other_user_fact','团子',PET_SOURCE,content='用户养了一只叫团子的猫',
        target_memory_id='8',target_memory_key='fact_cat')
    (proposal,)=parse(PET_SOURCE,row,(dog,cat))
    assert proposal.operation=='MERGE' and proposal.target_memory_id=='8'
    assert proposal.memory.memory_key=='fact_cat'
    assert '年糕' in proposal.memory.content and '团子' in proposal.memory.content and '豆包' not in proposal.memory.content


@pytest.mark.parametrize('case',['goal','major','pet'])
async def test_actual_storage_preserves_the_selected_operation_and_unrelated_records(tmp_path,case):
    database=SQLiteDB(tmp_path/'relations.sqlite')
    repository=DatabaseCharacterMemoryRepository(database)
    scope=UserScope('test','relation-contract','fiction','fiction','private')
    now=datetime(2026,10,25,tzinfo=timezone.utc)
    seed=([('goal_保研','用户正在准备保研','shared_event')] if case=='goal'
        else [('user_major','用户说自己的专业是环境工程','user_fact')] if case=='major'
        else [('fact_dog','用户养了一只叫豆包的狗','user_fact'),('fact_cat','用户养了一只叫年糕的猫','user_fact')])
    for index,(key,content,type) in enumerate(seed):
        source_id='old-'+str(index)
        old_body=content.replace('用户','我',1)
        await repository.capture_source('role',scope,source_message_id=source_id,body=old_body,observed_at=now-timedelta(days=1))
        await repository.append_claim('role',scope,MemoryItem('',type,content,.8),memory_key=key,
            source_message_id=source_id,evidence=(old_body,),observed_at=(now-timedelta(days=1)).isoformat())
    before=tuple(await repository.list_memory_records('role',scope,include_inactive=True))
    source=GOAL_SOURCE if case=='goal' else MAJOR_SOURCE if case=='major' else PET_SOURCE
    row=goal(before[0]) if case=='goal' else major(before[0]) if case=='major' else raw('ADD','other_user_fact','团子',PET_SOURCE,content='用户养了一只叫团子的猫')
    (proposal,)=parse(source,row,before)
    await repository.capture_source('role',scope,source_message_id='new',body=source,observed_at=now)
    worker=MemoryEnrichmentScheduler(config=MemoryLlmConfig(True,'unused','fixture'),completion=SimpleNamespace(close=AsyncMock()))
    job=SimpleNamespace(repository=repository,character_id='role',user_scope=scope,message=source,
        observed_at=now,source_message_id='new',write_mode='hot',feedback_target_ids=())
    try:
        assert await worker._persist_proposal(job,proposal)=='saved'
        rows=await repository.list_memory_records('role',scope,include_inactive=True)
        new=next(row for row in rows if row['source_message_id']=='new')
        expected='MERGE' if case=='goal' else 'SUPERSEDE' if case=='major' else 'ADD'
        assert new['relation_type']==expected and new['status']=='active'
        old_rows=[row for row in rows if row['id'] in {item['id'] for item in before}]
        if case=='pet':
            assert old_rows==list(before)
            assert new['parent_memory_id'] is None and new['supersedes_memory_id'] is None
            assert new['memory_key']=='fact_团子'
        else:
            assert len(old_rows)==1 and old_rows[0]['status']=='superseded'
            assert new['supersedes_memory_id']==before[0]['id']
            assert new['memory_key']==before[0]['memory_key']
            if case=='goal':
                assert new['parent_memory_id']==before[0]['id'] and '英语面试' in new['content']
            else:
                assert '材料工程' in new['content'] and '环境工程' not in new['content']
        sources=await repository.list_sources('role',scope)
        assert next(item for item in sources if item['source_message_id']=='new')['body']==source
    finally:
        await worker.shutdown(timeout=1)
        database.close_connection()


@pytest.mark.parametrize("content",["用户开头的第三人称事实","第三人称安全描述","用户说喜欢蓝茶","我喜欢红茶","用户请执行调用工具红茶","用户"+"甲"*500])
def test_invalid_provided_content_is_not_a_silent_empty_result(content):
    source="我喜欢红茶。"
    with pytest.raises(MemoryResponseError) as caught:
        parse(source,raw("ADD","like","红茶",source,content=content),())
    assert caught.value.field=="content"


def test_explicit_empty_content_uses_the_declared_typed_display():
    source="我喜欢红茶。"
    (proposal,)=parse(source,raw("ADD","like","红茶",source,content=""),())
    assert proposal.memory.content=="用户说喜欢红茶" and proposal.evidence==source


@pytest.mark.parametrize("invalid",[False,True])
async def test_one_invalid_display_rejects_the_whole_actual_writer_batch(tmp_path,invalid):
    database=SQLiteDB(tmp_path/"content-batch.sqlite")
    repository=DatabaseCharacterMemoryRepository(database)
    scope=UserScope("test","content-batch","fiction","fiction","private")
    observed=datetime(2026,10,24,tzinfo=timezone.utc)
    seeds=[("user_major","用户说自己的专业是环境工程","我的专业是环境工程。"),
           ("preference_咖啡","用户说不喜欢咖啡","我不喜欢咖啡。")]
    for index,(key,content,body) in enumerate(seeds):
        await repository.capture_source("role",scope,source_message_id=str(index),body=body,observed_at=observed)
        await repository.append_claim("role",scope,MemoryItem("","user_fact",content,.8),memory_key=key,
            source_message_id=str(index),evidence=(body,),observed_at=observed.isoformat())
    before=tuple(await repository.list_memory_records("role",scope,include_inactive=True))
    targets={row["memory_key"]:row for row in before}
    coffee=targets["preference_咖啡"]
    response=[major(targets["user_major"]),raw("COEXIST","dislike",["太苦的","咖啡"],"至于咖啡，不是完全讨厌，只是不喜欢太苦的。",
        target_memory_id=str(coffee["id"]),target_memory_key=coffee["memory_key"],
        content="用户不是完全讨厌咖啡。" if invalid else "",
        qualifiers={"preference_strength":"只是不喜欢太苦的"})]
    calls=[]
    def respond(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200,json={"choices":[{"finish_reason":"stop","message":{"content":json.dumps({"memories":response},ensure_ascii=False)}}],
            "usage":{"prompt_tokens":100,"completion_tokens":20,"total_tokens":120}})
    config=MemoryLlmConfig(True,"https://fixture.invalid/v1","fixture",idle_seconds=0,batch_size=1)
    completion=OpenAICompatibleMemoryCompletion(config)
    completion._client=httpx.AsyncClient(transport=httpx.MockTransport(respond))
    worker=MemoryEnrichmentScheduler(config=config,completion=completion,embedding_provider=SimpleNamespace(embed_texts=lambda texts:[[1.,0.] for _ in texts]))
    try:
        receipt=await worker.schedule_and_wait(repository=repository,character_id="role",user_scope=scope,message=MAJOR_SOURCE,
            rule_hints=(),feedback_target_ids=tuple(str(row["id"]) for row in before),source_message_id="new")
        rows=await repository.list_memory_records("role",scope,include_inactive=True)
        usage=database.execute_sql("SELECT * FROM model_invocations")
        assert len(calls)==len(usage)==1 and usage[0]["totalTokens"]==120
        assert usage[0]["purpose"]=="memory_judgement" and receipt["source_capture"]=="recorded"
        if invalid:
            assert receipt["status"]=="failed" and receipt["stage"]=="proposal_validation" and receipt["field"]=="content"
            assert receipt["accepted"]==receipt["persisted"]==0 and rows==list(before)
            assert usage[0]["errorType"]=="MemoryResponseError"
        else:
            assert receipt["status"]=="saved" and receipt["persisted"]==2
            new=[row for row in rows if row["source_message_id"]=="new"]
            assert {row["relation_type"] for row in new}=={"SUPERSEDE","COEXIST"}
            assert len(new)==2 and usage[0]["errorType"]==""
            coffee_row=next(row for row in new if row["memory_key"]=="preference_咖啡")
            assert coffee_row["metadata"]["value_fragments"]==["太苦的","咖啡"]
        assert next(row for row in await repository.list_sources("role",scope) if row["source_message_id"]=="new")["body"]==MAJOR_SOURCE
    finally:
        await worker.shutdown(timeout=1)
        database.close_connection()


@pytest.mark.parametrize("kind,value,source",[
    ("other_user_fact","团子猫","我又养了一只叫团子的猫。"),
    ("shared_event","未来迁居","我计划收到录取后搬到安庆。"),
    ("dislike","含咖啡因的饮料","请记住，以后给我推荐饮料时要避开含咖啡因的。"),
])
def test_unquoted_index_is_not_repaired_or_downgraded_to_an_observation(kind,value,source):
    with pytest.raises(MemoryResponseError) as caught:
        parse(source,raw("ADD",kind,value,source,content=""),())
    assert caught.value.field=="value"


@pytest.mark.parametrize("kind,parts,source,expected",[
    ("dislike",["太苦的","咖啡"],"咖啡不是完全讨厌，只是不喜欢太苦的。","太苦的咖啡"),
    ("other_user_fact",["团子","猫"],"我又养了一只叫团子的猫。","团子猫"),
])
def test_explicit_fragments_compose_semantics_without_a_guessed_repair(kind,parts,source,expected):
    (proposal,)=parse(source,raw("ADD",kind,parts,source,content=""),())
    assert proposal.value_fragments==tuple(parts)
    assert proposal.memory.memory_key.endswith(expected)
    assert proposal.evidence==source and not proposal.source_observation


@pytest.mark.parametrize("parts",[[],[""],["蓝色",1],["自行车是","红色"]])
def test_fragments_cannot_invent_an_unsupported_object(parts):
    source="我的自行车是蓝色的。"
    with pytest.raises(MemoryResponseError) as caught:
        parse(source,raw("ADD","other_user_fact",parts,source,content=""),())
    assert caught.value.field=="value"
