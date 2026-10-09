"""Model text fields cannot silently accept structured or numeric values."""
import json

import pytest

from character.memory_llm import parse_llm_memories


@pytest.mark.parametrize('field',['kind','value','content','target_memory_id','target_memory_key','evidence'])
@pytest.mark.parametrize('invalid',[None,0,False,[],{}])
def test_non_string_text_field_is_explicit_protocol_error(field,invalid):
    row=dict(operation='ADD',kind='like',value='红茶',content='用户喜欢红茶',
             evidence='我喜欢红茶',confidence=.95,attributed_to='user')
    row[field]=invalid
    with pytest.raises(ValueError,match=field):
        parse_llm_memories(json.dumps({'memories':[row]},ensure_ascii=False),source_message='请记住，我喜欢红茶。')

def test_omitted_optional_content_and_target_fields_preserve_valid_add():
    row=dict(operation='ADD',kind='like',value='红茶',evidence='我喜欢红茶',confidence=.95,attributed_to='user')
    memories=parse_llm_memories(json.dumps({'memories':[row]},ensure_ascii=False),source_message='请记住，我喜欢红茶。')
    assert len(memories)==1 and memories[0].content=='用户说喜欢红茶'
