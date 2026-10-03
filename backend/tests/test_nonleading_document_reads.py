"""Independent synthetic compound-task examples; no native private fixtures."""

from threading import RLock
from types import SimpleNamespace

import pytest

from knowledge.rag_helper import QueryExpander
from knowledge.source_expansion import requested_document_titles


@pytest.mark.parametrize(
    "prefix",
    [
        "请记住我的完整条件。\n从2028年5月1日起，\n我喜欢紫茶且只有登记完成才选择紫茶。\n今天继续遵守原条件。\n",
        "请记住我喜欢紫茶。",
        "今天先记住这一条；\n",
    ],
)
def test_independent_read_after_complete_statement(prefix):
    query = prefix + "读取《紫茶规则》《合成角色偏好》。本轮核对条件及其例外。"
    assert requested_document_titles(query) == ("紫茶规则", "合成角色偏好")
    assert QueryExpander().extract_filters(query) == {}


@pytest.mark.parametrize(
    "query",
    [
        "他说：‘第一条。\n读取《紫茶规则》。’我只是转述。",
        "原话是“请记住条件。读取《紫茶规则》。”",
        "原话是“第一条。读取《紫茶规则》。",
        "示例：`第一条。读取《紫茶规则》。`",
        "请记住条件。不要读取《紫茶规则》。",
        "不要读取《合成角色偏好》。读取《紫茶规则》。",
        "请记住条件。读取《紫茶规则》，不要读取第一份。",
        "请记住条件。读取《紫茶规则》。另读取《合成角色偏好》。",
        "请记住条件。读取《紫茶规则》《合成角色偏好》这三份资料。",
        "请记住条件。背景说读取《紫茶规则》，这里只转述。",
    ],
)
def test_quoted_negative_ambiguous_or_incomplete_tasks_do_not_grant_reads(query):
    assert requested_document_titles(query) == ()


async def test_nonleading_read_resolves_exact_originals_after_semantic_abstention(monkeypatch):
    from api import generate, knowledge
    from knowledge import rag_helper, vector_db
    from knowledge.multiscale_rag import runtime

    docs = [
        dict(id=i, title=t, knowledge_base_id=7, category="演练", content=f"{t}的完整条件、例外及限定。")
        for i, t in enumerate(["紫茶规则", "合成角色偏好"], 1)
    ]
    records = [d | dict(id=f"doc_{d['id']}_chunk_0", document_id=d["id"], chunk_index=0) for d in docs]
    indexed = SimpleNamespace(
        _lock=RLock(),
        cache_generation=11,
        snapshot_validated=True,
        metadata=records,
        _match_filters=lambda record, filters: True,
    )
    monkeypatch.setattr(
        runtime,
        "get_multiscale_rag_service",
        lambda: pytest.fail("Explicit independent reads cannot be replaced by curated lore"),
    )
    monkeypatch.setattr(knowledge, "_ensure_vector_index", lambda: True)
    monkeypatch.setattr(knowledge, "_vector_index_revision", 77)
    monkeypatch.setattr(knowledge, "_get_rebuild_revision", lambda: 77)
    monkeypatch.setattr(vector_db, "get_vector_db", lambda: indexed)
    monkeypatch.setattr(generate, "db", SimpleNamespace(get_knowledge_document=lambda identity: docs[identity - 1]))
    monkeypatch.setenv("CORRECTIVE_RAG_ENABLED", "false")
    query = "请记住我喜欢紫茶。\n读取《紫茶规则》《合成角色偏好》。核对条件及例外。"

    def retrieve(actual, **kwargs):
        assert actual == query and kwargs["filters"] is None
        return dict(results=[], confidence=0.0, abstained=True)

    monkeypatch.setattr(
        rag_helper,
        "get_rag_helper",
        lambda: SimpleNamespace(retrieve_with_citations=retrieve, build_citations=lambda records: []),
    )
    result = await generate._retrieve_rag_bundle(query, 3, None)
    assert result["requested_source_titles"] == ["紫茶规则", "合成角色偏好"]
    assert [p["original_body"] for p in result["original_source_packets"]] == [d["content"] for d in docs]
    assert all(p["original_source_receipt"]["authority_revision"] == 77 for p in result["source_coverage"])


@pytest.mark.parametrize(
    "task,expected",
    [
        ("本轮只核对今天和未来的完整条件、公定规则及例外。", ("紫茶规则", "合成角色偏好")),
        ("本轮只核对第一份资料的条件。", ()),
        ("本轮只核对紫茶规则的条件。", ()),
        ("本轮只读取条件及例外。", ()),
        ("本轮只核对《紫茶规则》的条件。", ()),
    ],
)
def test_complete_read_question_scope_does_not_become_document_exclusion(task, expected):
    query = "请记住我喜欢紫茶。读取《紫茶规则》《合成角色偏好》。" + task
    assert requested_document_titles(query) == expected


def test_identity_condition_is_not_a_document_count_selector():
    query = "请记住我喜欢紫茶。读取《紫茶规则》《合成角色偏好》。本轮只核对公定身份核验条件及其例外。"
    assert requested_document_titles(query) == ("紫茶规则", "合成角色偏好")
