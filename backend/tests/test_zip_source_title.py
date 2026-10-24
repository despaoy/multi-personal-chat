"""Accepted filename suffix case must not change named-source identity."""

from knowledge import index_lifecycle

import io
import json
import zipfile

import pytest
from fastapi import HTTPException, UploadFile
from test_index_creation_owner import TEXT, publication as native_publication

from api import knowledge
from knowledge.source_expansion import expand_source_context, requested_document_titles

publication = native_publication


async def upload(rig, name, entry, body):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr(entry, body)
    data.seek(0)
    file = UploadFile(data, filename=name)
    try:
        return await knowledge.upload_zip(rig.kb["id"], file, {"role": "admin"})
    finally:
        await file.close()


@pytest.mark.parametrize(
    "archive_name,extension", [("完整资料.zip", "TXT"), ("完整资料.ZIP", "mD"), ("完整资料.ZiP", "JSON")]
)
async def test_case_variants_publish_the_same_exact_named_source(publication, archive_name, extension):
    rig = publication
    body = json.dumps({"完整规则": TEXT}, ensure_ascii=False) if extension == "JSON" else TEXT
    result = await upload(rig, archive_name, "适用条件/登记规则." + extension, body)
    assert result["success"] and result["status"] == "complete" and result["createdDocs"] == 1
    doc = rig.db.get_knowledge_documents()[0]
    assert doc["title"] == "登记规则" and doc["content"] == body
    assert doc["knowledge_base_id"] == rig.kb["id"] and doc["folder_id"] == rig.folder["id"]
    assert doc["category"] == "适用条件" and doc["fileType"] == extension
    assert "".join(row["content"] for row in rig.db.get_knowledge_chunks(doc["id"])) == body
    assert index_lifecycle._ensure_vector_index()
    query = "请读取《登记规则》。"
    assert requested_document_titles(query) == ("登记规则",)
    rejected_semantic_candidates = dict(results=[], abstained=True, confidence=0.0)
    expanded = expand_source_context(
        rejected_semantic_candidates,
        rig.vector,
        expected_generation=rig.vector.cache_generation,
        source_budget_tokens=4096,
        filters={"knowledge_base_id": rig.kb["id"]},
        query=query,
    )
    assert expanded["unresolved_requested_titles"] == [] and expanded["abstained"] is False
    assert expanded["semantic_candidates_discarded"] and expanded["confidence"] == 0.0
    assert expanded["requested_source_scope"] == "original_filter"
    assert expanded["results"] and all(row["document_id"] == doc["id"] for row in expanded["results"])
    assert all(
        row["title"] == "登记规则" and row["retrieval_role"] == "requested_source" for row in expanded["results"]
    )
    assert body in "".join(row["content"] for row in expanded["results"])
    other = rig.db.create_knowledge_base("无关合成库")
    excluded = expand_source_context(
        rejected_semantic_candidates,
        rig.vector,
        expected_generation=rig.vector.cache_generation,
        source_budget_tokens=4096,
        filters={"knowledge_base_id": other["id"]},
        query=query,
    )
    assert excluded["results"] == [] and excluded["abstained"]
    assert excluded["unresolved_requested_titles"] == ["登记规则"]


@pytest.mark.parametrize("failure", ["bad_archive", "traversal"])
async def test_uppercase_archive_keeps_content_and_path_failures_explicit(publication, failure):
    rig = publication
    with pytest.raises(HTTPException) as caught:
        if failure == "traversal":
            await upload(rig, "完整资料.ZIP", "../越界.TXT", TEXT)
        else:
            file = UploadFile(io.BytesIO(b"not a ZIP archive"), filename="无效资料.ZIP")
            try:
                await knowledge.upload_zip(rig.kb["id"], file, {"role": "admin"})
            finally:
                await file.close()
    assert caught.value.status_code == 400
    assert rig.db.get_knowledge_documents() == []
    assert index_lifecycle._get_rebuild_revision() == rig.revision
    rig.add.assert_not_called()
