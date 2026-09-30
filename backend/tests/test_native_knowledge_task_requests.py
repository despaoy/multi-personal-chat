"""Explicit native knowledge requests must survive personal-task routing."""

import pytest

from knowledge.intent_detector import RAGIntentDetector
from knowledge.query_tasks import requests_explicit_information
from knowledge.task_dependency import local_context_only


@pytest.mark.parametrize(
    "query",
    [
        "请同时回答两项：我的昵称；查知识库回答木刻课程实际开课时间、联系人和预约编号。",
        "查知识库回答青苔工坊周日木刻课程的开课时间。",
        "请查阅知识库，列出青苔工坊木刻课程联系人。",
        "检索资料库里的木刻课程报名截止时间。",
        "搜索数据库中的课程预约编号。",
        "请分别回答课程开课时间和报名截止时间。",
        "请一并回答课程开课时间和联系人。",
    ],
)
def test_direct_information_requests_trigger_native_rag(query):
    assert requests_explicit_information(query)
    assert not local_context_only(query)
    assert RAGIntentDetector().needs_rag(query)[0]


@pytest.mark.parametrize(
    "query",
    [
        "我今天查知识库整理好了课程资料。",
        "我在看查知识库的方法。",
        "我看到“查知识库回答课程规则”这句话。",
        "课程说明写着「检索资料库里的联系人」。",
        "不要查知识库，我只是想聊聊自己的近况。",
        "不用搜索数据库。",
    ],
)
def test_mentions_quoted_and_negated_requests_are_not_direct_information(query):
    assert not requests_explicit_information(query)


@pytest.mark.parametrize("query", ["我的姓名是什么？", "我的专业是什么？"])
def test_closed_personal_reads_still_skip_rag(query):
    assert local_context_only(query)
    assert not RAGIntentDetector().needs_rag(query)[0]
