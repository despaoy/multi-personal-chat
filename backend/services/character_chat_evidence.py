"""Display snapshots come from admitted sources, never model-authored source labels."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CharacterChatCompletion:
    reply: str
    model: str
    evidence: dict = field(default_factory=dict)


def evidence_snapshot(result):
    retrieval = result.plan.retrieval
    documents = {}
    for item in retrieval.documents:
        identity = str(item.get("id") or item.get("chunk_id") or "")
        if not identity or identity in documents:
            raise ValueError("Retrieved source snapshot has missing or duplicate identity")
        documents[identity] = item
    sources = []
    for citation in result.response_citations:
        identity = str(citation.get("source_id") or citation.get("id") or "")
        if not identity:
            raise ValueError("Answer-used citation has no source identity")
        if identity not in documents:
            raise ValueError("Answer-used citation has no retrieved source snapshot")
        document = documents[identity]
        text = str(document.get("content") or document.get("text") or "")
        sources.append(
            {
                "id": identity,
                "title": str(document.get("title") or document.get("original_title") or identity),
                "excerpt": text[:12000],
                "truncated": len(text) > 12000,
                "line_start": citation.get("line_start"),
                "line_end": citation.get("line_end"),
            }
        )
    return {
        "sources": sources,
        "retrieval_status": retrieval.status,
        "response_mode": result.response_mode,
        "reason": retrieval.reason,
    }
