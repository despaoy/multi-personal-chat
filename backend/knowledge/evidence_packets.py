"""Whole retrieved document chunks, independent of citation presentation."""

from collections.abc import Mapping, Sequence
from typing import Any


def document_evidence_packets(results: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    packets = []
    for document in results:
        body = document.get("content")
        if not isinstance(body, str) or not body.strip():
            continue
        identity = document.get("id") or document.get("chunk_id")
        ids = (
            [str(identity)]
            if isinstance(identity, (str, int)) and not isinstance(identity, bool) and str(identity).strip()
            else []
        )
        title = document.get("title") or document.get("original_title") or "未命名资料"
        packets.append({"kind": "evidence", "document_ids": ids, "text": f"【检索资料片段: {title}】\n{body}"})
    return tuple(packets)
