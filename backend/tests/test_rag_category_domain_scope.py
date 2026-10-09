"""Implicit categories require an unquoted declared domain, not actor words."""
import pytest

from knowledge import rag_helper
from knowledge.rag_helper import DomainProfile, QueryExpander, RAGHelper


@pytest.mark.parametrize("query", [
    "请按青川办理规则比较完整材料和身份审核，角色的经历不是我的事实。",
    "查询申请资料的费用。人物与用户的偏好应分开。",
    "这条是假设条件，角色是否喜欢不代表本人已经办理。",
    "角色有哪些技能？",
    "背景写着‘原神角色’，请回答公共办理规则。",
    "背景写着\"Genshin characters\"，请回答公共办理规则。",
])
def test_incidental_or_quoted_domain_does_not_exclude_public_rules(query):
    assert QueryExpander().extract_filters(query)=={}

@pytest.mark.parametrize("query, expected", [
    ("原神角色有哪些技能？", {"category":"角色"}),
    ("钟离角色的机制是什么？", {"category":"角色"}),
    ("Genshin武器有哪些？", {"category":"武器"}),
    ("原神璃月的角色资料", {"category":"角色", "region":"璃月"}),
])
def test_declared_game_domain_retains_category_inference(query,expected):
    assert QueryExpander().extract_filters(query)==expected

def test_ascii_anchor_is_not_a_substring_grant():
    assert QueryExpander().extract_filters("NotGenshinX角色是什么？")=={}

def test_configured_profile_does_not_borrow_another_domains_category():
    profile=DomainProfile("civic",category_map={"角色":"role"},filter_anchors=["青川办事处"])
    expander=QueryExpander(profiles=[profile])
    assert expander.extract_filters("原神角色技能") == {}
    assert expander.extract_filters("青川办事处角色责任") == {"category":"role"}

def test_unanchored_query_keeps_complete_rule_candidates_and_explicit_filter(monkeypatch):
    class Index:
        cache_generation=1
        def __init__(self):self.filters=[]
        def hybrid_search(self,query,**kwargs):
            self.filters.append(kwargs["filters"])
            return [] if kwargs["filters"] else [dict(id="rule",title="青川办理",content="完整申请和核验条件",score=.8)]
    index=Index()
    monkeypatch.setattr(rag_helper,"get_vector_db",lambda:index)
    helper=RAGHelper()
    assert helper.retrieve_context("请按青川办理规则判断，角色不是用户。",enable_rerank=False,use_cache=False)[0]["id"]=="rule"
    assert all(x is None for x in index.filters)
    index.filters=[]
    assert helper.retrieve_context("原神角色",filters={"knowledge_base_id":7},enable_rerank=False,use_cache=False)==[]
    assert index.filters and all(x=={"knowledge_base_id":7} for x in index.filters)
