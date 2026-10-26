"""Original evidence coverage, bounded windows and real public publication."""

from knowledge.index_lifecycle import default_knowledge_index as index_lifecycle

import io
import zipfile

import pytest
from fastapi import UploadFile

from api import knowledge
from db.schemas import KnowledgeDocumentCreate, KnowledgeDocumentUpdate
from knowledge.text_splitter import smart_text_split
from test_index_creation_owner import publication as publication


def assert_slices(source, chunks, size, overlap):
    assert chunks and all(0 < len(chunk) <= size and chunk in source for chunk in chunks)
    assert all(left[-overlap:] == right[:overlap] for left, right in zip(chunks, chunks[1:])) if overlap else True
    assert chunks[0] + "".join(chunk[overlap:] for chunk in chunks[1:]) == source


@pytest.mark.parametrize("source,size,overlap", [
    ("  原文保留空格。\r\n第二行不得修改。\r\n", 600, 100),
    ("长句必须完整保留且不得自动续借" * 90 + "。", 600, 100),
    ("无标点原文" * 90, 60, 0),
    ("甲。乙？丙！\r\n\r\n" * 80, 60, 12),
    ("“引号中的完整约束不允许丢失”" * 70, 80, 20),
    ("中文a🙂\r\n", 1, 0),
    ("甲乙丙丁。戊己庚辛。", 4, 3),
    ("a" * 600, 600, 100),
])
def test_slices_preserve_every_source_character_with_exact_overlap(source, size, overlap):
    assert_slices(source, smart_text_split(source, size, overlap), size, overlap)


@pytest.mark.parametrize("source", ["", " \r\n\t "])
def test_blank_documents_have_no_index_chunks(source):
    assert smart_text_split(source) == []


@pytest.mark.parametrize("size,overlap", [(0, 0), (-1, 0), (True, 0), (12.5, 0),
                                         (10, -1), (10, 10), (10, True), (10, 1.5)])
def test_invalid_window_is_explicit_and_valid_request_still_works(size, overlap):
    with pytest.raises(ValueError):
        smart_text_split("", size, overlap)
    assert smart_text_split("完整来源", 10, 0) == ["完整来源"]


@pytest.mark.parametrize("separator", ["。", "\r\n\r\n"])
def test_readable_boundary_is_used_without_changing_source(separator):
    source = "甲" * 35 + separator + "乙" * 80
    chunks = smart_text_split(source, 60, 10)
    assert chunks[0] == "甲" * 35 + separator
    assert_slices(source, chunks, 60, 10)


@pytest.mark.parametrize('route', ['create', 'update', 'zip', 'scan_root', 'scan_folder'])
@pytest.mark.parametrize('source', ['  完整规定：只有本人携带书面确认才可申请。\r\n未确认不得付款。\r\n', '申请人须本人携带身份证和书面确认且不得自动续借' * 55 + '。\r\n截至10月23日18点。'])
async def test_public_paths_preserve_bounded_evidence_through_real_index(publication, route, source):
    rig = publication
    admin = {'role': 'admin'}
    if route in {'create', 'update'}:
        result = await knowledge.create_knowledge_document(KnowledgeDocumentCreate(title='完整规则', content=source if route == 'create' else '待替换旧正文', knowledge_base_id=rig.kb['id'], folder_id=rig.folder['id']), admin, database=index_lifecycle.db)
        if route == 'update':
            result = await knowledge.update_knowledge_document(result['document']['id'], KnowledgeDocumentUpdate(content=source), admin, database=index_lifecycle.db)
    elif route == 'zip':
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as archive:
            archive.writestr('适用条件/完整规则.txt', source)
        buffer.seek(0)
        result = await knowledge.upload_zip(rig.kb['id'], UploadFile(filename='完整规则.zip', file=buffer), admin, database=index_lifecycle.db)
    else:
        target = rig.root / '完整资料'
        folder = target / '适用条件' if route == 'scan_folder' else target
        folder.mkdir(parents=True)
        (folder / '完整规则.txt').write_bytes(source.encode())
        result = await knowledge.import_scanned_directory('完整资料', rig.kb['id'], admin, database=index_lifecycle.db)
    assert result['success']
    doc, = rig.db.get_knowledge_documents()
    chunks = [row['content'] for row in rig.db.get_knowledge_chunks(doc['id'])]
    assert doc['content'] == source and doc['chunkCount'] == len(chunks)
    assert_slices(source, chunks, 600, 100)
    assert index_lifecycle._ensure_vector_index()
    prefix = f"[{rig.kb['name']}/{doc['category'] or ''}] {doc['title']}: "
    assert [row['content'] for row in rig.vector.metadata] == [prefix + chunk for chunk in chunks]
    assert all((row['document_id'] == doc['id'] and row['knowledge_base_id'] == rig.kb['id'] for row in rig.vector.metadata))
