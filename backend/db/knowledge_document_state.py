"""Normalize one atomic knowledge document publication."""

UPDATABLE_COLUMNS = frozenset(
    {
        "title",
        "content",
        "category",
        "knowledge_base_id",
        "folder_id",
        "sourceType",
        "sourceUrl",
        "fileType",
        "fileSize",
        "chunkCount",
        "updatedAt",
    }
)
INDEXED_COLUMNS = frozenset({"title", "content", "category", "knowledge_base_id", "folder_id", "sourceType"})


def prepare_document_change(existing, document, chunks, now):
    values = {key: value for key, value in document.items() if key in UPDATABLE_COLUMNS and value is not None}
    if "content" in values and chunks is None:
        raise ValueError("Content publication requires its complete replacement chunks")
    if chunks is not None:
        if not isinstance(chunks, list) or any(not isinstance(chunk, str) for chunk in chunks):
            raise TypeError("Replacement chunks must be a list of text strings")
        chunks = list(chunks)
        values["chunkCount"] = len(chunks)
    if existing is None:
        if chunks is None:
            raise ValueError("New document publication requires replacement chunks")
        values = {"title": "", "content": "", "category": "未分类", "sourceType": "text", **values, "createdAt": now}
    dirty = (
        existing is None
        or chunks is not None
        or any(values[key] != existing.get(key) for key in INDEXED_COLUMNS if key in values)
    )
    values["updatedAt"] = now
    return values, chunks, dirty
