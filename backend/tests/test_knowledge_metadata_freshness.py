"""Changed document metadata and ordinary generic-RAG freshness only."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from api import generate, knowledge
from db.database import SQLiteDB
from db.schemas import KnowledgeDocumentUpdate


@pytest.fixture
def saved_document(tmp_path, monkeypatch):
    database = SQLiteDB(tmp_path / "knowledge.db")
    old = database.create_knowledge_base("草案库", "old")
    new = database.create_knowledge_base("现行库", "new")
    row = database.add_knowledge_document(
        dict(
            title="旧标题",
            content="完整规则。末尾仍需书面确认。",
            category="草案",
            knowledge_base_id=old["id"],
            chunkCount=1,
        )
    )
    database.add_knowledge_chunk(dict(documentId=row["id"], chunkIndex=0, content=row["content"]))
    monkeypatch.setattr(knowledge, "db", database)
    monkeypatch.setattr(knowledge, "VECTOR_DB_AVAILABLE", False)
    monkeypatch.setattr(knowledge, "_vector_index_built", True)
    knowledge._write_rebuild_status(
        "complete", 1, knowledge._compute_chunk_fingerprint(), knowledge._get_rebuild_revision()
    )
    return database, row, new


@pytest.mark.parametrize("field,value", [("title", "新标题"), ("category", "现行"), ("knowledge_base_id", "new_base")])
async def test_metadata_change_invalidates_warm_index_without_rewriting_chunks(saved_document, field, value):
    database, row, new = saved_document
    if value == "new_base":
        value = new["id"]
    original = database.get_knowledge_chunks(row["id"])
    fingerprint = knowledge._compute_chunk_fingerprint()
    revision = knowledge._get_rebuild_revision()
    response = await knowledge.update_knowledge_document(
        row["id"], KnowledgeDocumentUpdate(**{field: value}), dict(role="admin")
    )
    assert response["success"] and database.get_knowledge_document(row["id"])[field] == value
    assert database.get_knowledge_document(row["id"])["content"] == row["content"]
    assert database.get_knowledge_chunks(row["id"]) == original
    assert knowledge._compute_chunk_fingerprint() != fingerprint
    assert knowledge._read_rebuild_status()[0] == "dirty" and knowledge._get_rebuild_revision() == revision + 1
    assert not knowledge._vector_index_built


async def test_combined_metadata_move_invalidates_once_with_full_original_source(saved_document):
    database, row, new = saved_document
    chunks = database.get_knowledge_chunks(row["id"])
    revision = knowledge._get_rebuild_revision()
    await knowledge.update_knowledge_document(
        row["id"],
        KnowledgeDocumentUpdate(title="新标题", category="现行", knowledge_base_id=new["id"]),
        dict(role="admin"),
    )
    assert knowledge._get_rebuild_revision() == revision + 1
    assert database.get_knowledge_chunks(row["id"]) == chunks
    assert database.get_knowledge_document(row["id"])["content"].endswith("末尾仍需书面确认。")


@pytest.mark.parametrize("change", [{}, dict(title="旧标题"), dict(category="草案"), dict(fileSize=123)])
async def test_unchanged_indexed_metadata_does_not_rebuild_or_drop_evidence(saved_document, change):
    database, row, new = saved_document
    original = database.get_knowledge_chunks(row["id"])
    revision = knowledge._get_rebuild_revision()
    await knowledge.update_knowledge_document(row["id"], KnowledgeDocumentUpdate(**change), dict(role="admin"))
    assert knowledge._get_rebuild_revision() == revision and knowledge._vector_index_built
    assert database.get_knowledge_chunks(row["id"]) == original and knowledge._read_rebuild_status()[0] == "complete"


async def test_content_and_metadata_update_use_one_revision_and_complete_new_chunks(saved_document):
    database, row, new = saved_document
    revision = knowledge._get_rebuild_revision()
    await knowledge.update_knowledge_document(
        row["id"],
        KnowledgeDocumentUpdate(title="新标题", content="完整新规则。原规则不再有效，仍须书面确认。"),
        dict(role="admin"),
    )
    assert knowledge._get_rebuild_revision() == revision + 1 and not knowledge._vector_index_built
    chunks = database.get_knowledge_chunks(row["id"])
    assert len(chunks) == 1 and chunks[0]["content"] == "完整新规则。原规则不再有效，仍须书面确认。"


async def test_missing_document_does_not_change_index_authority(saved_document):
    revision = knowledge._get_rebuild_revision()
    with pytest.raises(HTTPException) as error:
        await knowledge.update_knowledge_document(9999, KnowledgeDocumentUpdate(title="不存在"), dict(role="admin"))
    assert (
        error.value.status_code == 404
        and knowledge._get_rebuild_revision() == revision
        and knowledge._vector_index_built
    )


@pytest.mark.parametrize("corrective", [False, True])
async def test_generic_generation_checks_freshness_before_cache_or_correction(monkeypatch, corrective):
    from threading import RLock

    import knowledge.corrective_rag as corrective_module
    import knowledge.rag_helper as rag_module
    from knowledge import vector_db

    calls = []
    record = dict(id="doc_1_chunk_0", document_id=1, chunk_index=0, knowledge_base_id=7,
                  title="新标题", category="规程", content="末尾纠正。")
    bundle = dict(results=[record], citations=[], confidence=0.8, abstained=False)
    indexed = SimpleNamespace(_lock=RLock(), cache_generation=11, snapshot_validated=True, metadata=[record],
                              _match_filters=lambda row, filters: all(row.get(k) == v for k, v in filters.items()))
    monkeypatch.setattr(vector_db, "get_vector_db", lambda: indexed)
    monkeypatch.setattr(knowledge, "_vector_index_revision", 77)
    monkeypatch.setattr(knowledge, "_get_rebuild_revision", lambda: 77)
    monkeypatch.setenv("CORRECTIVE_RAG_ENABLED", "true" if corrective else "false")

    def ready():
        calls.append("ensure")
        return True

    def generic(*args, **kwargs):
        calls.append("generic")
        return bundle

    def correction(*args, **kwargs):
        calls.append("corrective")
        return bundle

    monkeypatch.setattr(knowledge, "_ensure_vector_index", ready)
    monkeypatch.setattr(rag_module, "get_rag_helper", lambda: SimpleNamespace(retrieve_with_citations=generic))
    monkeypatch.setattr(
        corrective_module, "get_corrective_rag", lambda: SimpleNamespace(retrieve_with_correction=correction)
    )
    actual = await generate._retrieve_rag_bundle("核对规则", 3, {"knowledge_base_id": 7})
    assert actual is bundle and calls == ["ensure", "corrective" if corrective else "generic"]


async def test_unready_index_never_returns_stale_cached_sources(monkeypatch):
    import knowledge.rag_helper as rag_module

    monkeypatch.setattr(knowledge, "_ensure_vector_index", lambda: False)

    def forbidden():
        raise AssertionError("Stale helper must not be read")

    monkeypatch.setattr(rag_module, "get_rag_helper", forbidden)
    with pytest.raises(RuntimeError, match="index.*not ready"):
        await generate._retrieve_rag_bundle("核对现行规则", 3, {"knowledge_base_id": 7})


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

    monkeypatch.setattr(knowledge, "_ensure_vector_index", forbidden)
    assert await generate._retrieve_rag_bundle("人物经历", 3, None) is bundle
