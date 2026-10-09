"""Stored zero importance survives an authorized erasure proposal."""
import json

import pytest
from repositories.character_memory import DatabaseCharacterMemoryRepository

from character.memory_llm import parse_llm_proposals
from character.models import MemoryItem, UserScope
from db.database import SQLiteDB


@pytest.mark.parametrize("importance",[0.0,0.5,1.0])
async def test_erasure_preserves_stored_fields(tmp_path,importance):
    repo=DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path/'memory.sqlite'))
    scope=UserScope('web','fixture','fiction','fiction','private')
    content='用户说自己来自或居住在杭州'
    await repo.add_or_update_memory('role',scope,MemoryItem('0','user_fact',content,importance),memory_key='user_location',source_message_id='source')
    records=await repo.list_memory_records('role',scope)
    assert len(records)==1 and records[0]['importance']==importance
    source='请把你记住的我的住址彻底删掉。'
    payload=dict(operation='ERASE',kind='erasure',value='用户要求彻底删除住址信息',evidence=source,confidence=1.0,attributed_to='user',target_memory_id=str(records[0]['id']),target_memory_key='user_location')
    result=parse_llm_proposals(json.dumps(dict(memories=[payload]),ensure_ascii=False),source_message=source,existing_memories=records)
    assert len(result)==1 and result[0].operation=='ERASE'
    assert result[0].memory.importance==importance
    assert result[0].memory.memory_type=='user_fact'
    assert result[0].memory.content==content
    records[0]["importance"]=None
    with pytest.raises(TypeError):
        parse_llm_proposals(json.dumps(dict(memories=[payload]),ensure_ascii=False),source_message=source,existing_memories=records)
