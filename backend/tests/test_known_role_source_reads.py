"""Full five-source reads cannot acquire implicit role-category filters."""
import hashlib
import json
from pathlib import Path

import pytest

from knowledge.rag_helper import QueryExpander
from knowledge.source_expansion import requested_document_titles

root=Path(__file__).parent/'fixtures'
spec=json.loads((root/'deepseek_known_role_source_case.json').read_text())
parent=(root/spec['base_fixture']).read_bytes()
assert hashlib.sha256(parent).hexdigest()==spec['base_fixture_sha256']
case={**json.loads(parent),**spec['case_overrides']}
titles=tuple(x['title'] for x in case['documents'])

def test_full_actual_question_retains_five_titles_and_no_inferred_role_filter():
    assert requested_document_titles(case['question'])==titles
    assert QueryExpander().extract_filters(case['question'])=={}
    assert len(case['bridges'])==9 and len(case['documents'])==5

@pytest.mark.parametrize('query',[
    '请查知识库，读取《甲》《乙》这两份完整资料。请依据《乙》回答角色问题。',
    '请查知识库，分别读取《甲》《乙》这两份完整资料。再引用《甲》和《乙》的原文。',
])
def test_repeated_already_requested_title_does_not_change_read_scope(query):
    assert requested_document_titles(query)==('甲','乙')
    assert QueryExpander().extract_filters(query)=={}

@pytest.mark.parametrize('suffix',[
    '另读取《丙》角色资料。', '不要读取《乙》角色资料。',
    '另提到《乙角色资料。', '另提到乙》角色资料。',
])
def test_unknown_title_exclusions_and_unbalanced_titles_still_defer(suffix):
    query='请查知识库，读取《甲》《乙》这两份完整资料。'+suffix
    assert requested_document_titles(query)==()
    assert QueryExpander().extract_filters(query)=={'category':'角色'}

def test_declared_count_with_complete_modifier_is_checked():
    query=case['question'].replace('这五份完整资料','这四份完整资料',1)
    assert requested_document_titles(query)==()

def test_quoted_background_does_not_become_a_direct_read():
    assert requested_document_titles('背景说读取《甲》《乙》这两份完整资料。角色是什么？')==()
