"""Changed document metadata and ordinary generic-RAG freshness only."""

from knowledge import chat_retrieval

from knowledge import index_lifecycle

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from api import knowledge
from db.database import SQLiteDB
from db.schemas import KnowledgeDocumentUpdate


@pytest.fixture
def saved_document(tmp_path, monkeypatch):
    database = SQLiteDB(tmp_path / "knowledge.db")
    old = database.create_knowledge_base("草案库", "old")
    new = database.create_knowledge_base("现行库", "new")
    body = "完整规则。末尾仍需书面确认。"
    row = database.save_knowledge_document(
        dict(
            title="旧标题",
            content=body,
            category="草案",
            knowledge_base_id=old["id"],
        ),
        chunks=[body],
    )
    monkeypatch.setattr(knowledge, "db", database)
    monkeypatch.setattr(index_lifecycle, "db", database)
    monkeypatch.setattr(index_lifecycle, "_vector_index_revision", index_lifecycle._get_rebuild_revision())
    database.commit_knowledge_index_revision(
        index_lifecycle._get_rebuild_revision(), 1, index_lifecycle._compute_chunk_fingerprint()
    )
    return database, row, new


@pytest.mark.parametrize("field,value", [("title", "新标题"), ("category", "现行"), ("knowledge_base_id", "new_base")])
async def test_metadata_change_invalidates_warm_index_without_rewriting_chunks(saved_document, field, value):
    database, row, new = saved_document
    if value == "new_base":
        value = new["id"]
    original = database.get_knowledge_chunks(row["id"])
    fingerprint = index_lifecycle._compute_chunk_fingerprint()
    revision = index_lifecycle._get_rebuild_revision()
    response = await knowledge.update_knowledge_document(
        row["id"], KnowledgeDocumentUpdate(**{field: value}), dict(role="admin")
    )
    assert response["success"] and database.get_knowledge_document(row["id"])[field] == value
    assert database.get_knowledge_document(row["id"])["content"] == row["content"]
    assert database.get_knowledge_chunks(row["id"]) == original
    assert index_lifecycle._compute_chunk_fingerprint() != fingerprint
    assert index_lifecycle._read_rebuild_status()[0] == "dirty" and index_lifecycle._get_rebuild_revision() == revision + 1
    assert index_lifecycle._vector_index_revision is None


async def test_combined_metadata_move_invalidates_once_with_full_original_source(saved_document):
    database, row, new = saved_document
    chunks = database.get_knowledge_chunks(row["id"])
    revision = index_lifecycle._get_rebuild_revision()
    await knowledge.update_knowledge_document(
        row["id"],
        KnowledgeDocumentUpdate(title="新标题", category="现行", knowledge_base_id=new["id"]),
        dict(role="admin"),
    )
    assert index_lifecycle._get_rebuild_revision() == revision + 1
    assert database.get_knowledge_chunks(row["id"]) == chunks
    assert database.get_knowledge_document(row["id"])["content"].endswith("末尾仍需书面确认。")


@pytest.mark.parametrize("change", [{}, dict(title="旧标题"), dict(category="草案"), dict(fileSize=123)])
async def test_unchanged_indexed_metadata_does_not_rebuild_or_drop_evidence(saved_document, change, monkeypatch):
    database, row, new = saved_document
    original = database.get_knowledge_chunks(row["id"])
    revision = index_lifecycle._get_rebuild_revision()
    await knowledge.update_knowledge_document(row["id"], KnowledgeDocumentUpdate(**change), dict(role="admin"))
    from threading import RLock
    from unittest.mock import Mock

    from knowledge import vector_db

    names = {kb["id"]: kb["name"] for kb in database.get_knowledge_bases()}
    metadata = [index_lifecycle._vector_chunk_document(chunk, names) for chunk in database.iter_chunks_with_document()]
    forbidden = Mock(side_effect=AssertionError("unchanged evidence must reuse the index"))
    vector = SimpleNamespace(
        _lock=RLock(), snapshot_validated=True, metadata=metadata, clear_cache=Mock(),
        clear_all=forbidden, add_documents=forbidden,
        get_stats=lambda: dict(total_documents=1, index_size=1, bm25_corpus_size=1),
    )
    monkeypatch.setattr(vector_db, "get_vector_db", lambda: vector)
    assert index_lifecycle._get_rebuild_revision() == revision
    assert index_lifecycle._ensure_vector_index() is True
    forbidden.assert_not_called()
    assert (index_lifecycle._vector_index_revision is not None)
    assert database.get_knowledge_chunks(row["id"]) == original and index_lifecycle._read_rebuild_status()[0] == "complete"


async def test_content_and_metadata_update_use_one_revision_and_complete_new_chunks(saved_document):
    database, row, new = saved_document
    revision = index_lifecycle._get_rebuild_revision()
    await knowledge.update_knowledge_document(
        row["id"],
        KnowledgeDocumentUpdate(title="新标题", content="完整新规则。原规则不再有效，仍须书面确认。"),
        dict(role="admin"),
    )
    assert index_lifecycle._get_rebuild_revision() == revision + 1 and index_lifecycle._vector_index_revision is None
    chunks = database.get_knowledge_chunks(row["id"])
    assert len(chunks) == 1 and chunks[0]["content"] == "完整新规则。原规则不再有效，仍须书面确认。"


async def test_missing_document_does_not_change_index_authority(saved_document):
    revision = index_lifecycle._get_rebuild_revision()
    with pytest.raises(HTTPException) as error:
        await knowledge.update_knowledge_document(9999, KnowledgeDocumentUpdate(title="不存在"), dict(role="admin"))
    assert (
        error.value.status_code == 404
        and index_lifecycle._get_rebuild_revision() == revision
        and (index_lifecycle._vector_index_revision is not None)
    )


async def test_generic_generation_checks_freshness_before_cache(monkeypatch):
    from threading import RLock

    import knowledge.rag_helper as rag_module
    from knowledge import vector_db

    calls = []
    record = dict(id="doc_1_chunk_0", document_id=1, chunk_index=0, knowledge_base_id=7,
                  title="新标题", category="规程", content="末尾纠正。")
    def read_original(identity):
        assert identity == 1
        calls.append("source")
        return {**record, "id": identity}

    monkeypatch.setattr(chat_retrieval.db, "get_knowledge_document", read_original)
    bundle = dict(results=[dict(record, score=0.8, fused_score=0.8, normalized_score=0.8)], citations=[], confidence=0.8, abstained=False)
    indexed = SimpleNamespace(_lock=RLock(), cache_generation=11, snapshot_validated=True, metadata=[record],
                              _match_filters=lambda row, filters: all(row.get(k) == v for k, v in filters.items()))
    monkeypatch.setattr(vector_db, "get_vector_db", lambda: indexed)
    monkeypatch.setattr(index_lifecycle, "_vector_index_revision", 77)
    monkeypatch.setattr(index_lifecycle, "_get_rebuild_revision", lambda: 77)

    def ready():
        calls.append("ensure")
        return True

    def generic(*args, **kwargs):
        calls.append("generic")
        return bundle

    monkeypatch.setattr(index_lifecycle, "_ensure_vector_index", ready)
    monkeypatch.setattr(rag_module, "get_rag_helper", lambda: SimpleNamespace(retrieve_with_citations=generic, build_citations=lambda rows: object.__new__(rag_module.RAGHelper).build_citations(rows)))
    actual = await chat_retrieval.retrieve_chat_knowledge("核对规则", 3, {"knowledge_base_id": 7})
    assert actual["results"] == bundle["results"] and actual["source_coverage"] and calls == ["ensure", "generic", "source"]
    assert actual["source_coverage"][0]["original_source_receipt"]["authority"] == "fresh_knowledge_document_read"



async def test_unready_index_never_returns_stale_cached_sources(monkeypatch):
    import knowledge.rag_helper as rag_module

    monkeypatch.setattr(index_lifecycle, "_ensure_vector_index", lambda: False)

    def forbidden():
        raise AssertionError("Stale helper must not be read")

    monkeypatch.setattr(rag_module, "get_rag_helper", forbidden)
    with pytest.raises(RuntimeError, match="index.*not ready"):
        await chat_retrieval.retrieve_chat_knowledge("核对现行规则", 3, {"knowledge_base_id": 7})


async def test_curated_character_evidence_does_not_depend_on_generic_index(monkeypatch):
    from knowledge.multiscale_rag import runtime
    from knowledge.retrieval_core import query

    bundle = dict(context_text="人物已核对证据", citations=[], retrieval_strategy="multi_scale_character")
    monkeypatch.setattr(
        runtime,
        "get_multiscale_rag_service",
        lambda: SimpleNamespace(config=object(), retrieve_with_citations=lambda *a, **k: bundle),
    )
    monkeypatch.setattr(
        query, "QueryAnalyzer", lambda *a: SimpleNamespace(analyze=lambda *a: SimpleNamespace(matched_domains=[]))
    )

    def forbidden():
        raise AssertionError("Character evidence must keep its own index contract")

    monkeypatch.setattr(index_lifecycle, "_ensure_vector_index", forbidden)
    assert await chat_retrieval.retrieve_chat_knowledge("人物经历", 3, None) is bundle
