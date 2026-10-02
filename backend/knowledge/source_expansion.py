"""Add bounded context from the same verified source as generic RAG anchors."""

from inference.context_budget import estimated_tokens

_AUTHORITY_FIELDS = ("id", "document_id", "chunk_index", "knowledge_base_id", "title", "category", "content")


def expand_source_context(bundle, vector_db, *, expected_generation, source_budget_tokens, filters=None):
    """Preserve ranked anchors; siblings are context, not additional rank votes.

    This is an indexed-chunk expansion, not certification of a complete raw
    document. Actual fixed input, memory, output and citation costs are still
    decided by the canonical request budget. Large sources never replace the
    anchors with an indivisible whole-document packet.
    """
    anchors = bundle.get("results") or []
    if bundle.get("abstained") or not anchors:
        return bundle
    if not isinstance(source_budget_tokens, int) or isinstance(source_budget_tokens, bool) or source_budget_tokens <= 0:
        raise ValueError("Invalid source context budget")
    with vector_db._lock:
        if not vector_db.snapshot_validated or vector_db.cache_generation != expected_generation:
            raise RuntimeError("Source index changed during retrieval")
        metadata = {}
        for record in vector_db.metadata:
            identity = record.get("id")
            if not isinstance(identity, str) or identity in metadata:
                raise RuntimeError("Ambiguous indexed source identity")
            metadata[identity] = record
        parents = {}
        selected_ids = set()
        for anchor in anchors:
            identity = anchor.get("id")
            stored = metadata.get(identity)
            if stored is None or any(anchor.get(key) != stored.get(key) for key in _AUTHORITY_FIELDS):
                raise RuntimeError("Retrieved source no longer matches indexed authority")
            if filters and not vector_db._match_filters(stored, filters):
                raise RuntimeError("Retrieved source violates requested scope")
            parent = stored.get("document_id")
            index = stored.get("chunk_index")
            if not isinstance(parent, int) or isinstance(parent, bool) or parent <= 0:
                continue
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                raise RuntimeError("Invalid indexed source position")
            if identity != f"doc_{parent}_chunk_{index}":
                raise RuntimeError("Invalid indexed chunk identity")
            group = (parent, stored.get("knowledge_base_id"), stored.get("title"), stored.get("category"))
            parents.setdefault(group, []).append((identity, index))
            selected_ids.add(identity)
        candidates = []
        for record in metadata.values():
            group = (
                record.get("document_id"),
                record.get("knowledge_base_id"),
                record.get("title"),
                record.get("category"),
            )
            if group not in parents or record["id"] in selected_ids:
                continue
            if filters and not vector_db._match_filters(record, filters):
                continue
            index = record.get("chunk_index")
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                raise RuntimeError("Invalid indexed sibling position")
            if record["id"] != f"doc_{group[0]}_chunk_{index}":
                raise RuntimeError("Invalid indexed sibling identity")
            distance = min(abs(index - selected_index) for _, selected_index in parents[group])
            candidates.append((list(parents).index(group), distance, index, record, group))
        candidates.sort(key=lambda item: item[:3])
        remaining = max(0, source_budget_tokens - sum(estimated_tokens(a["content"]) + 4 for a in anchors))
        added = []
        for _, _, _, record, group in candidates:
            text = record.get("content")
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError("Invalid indexed sibling text")
            cost = estimated_tokens(text) + 4
            if cost > remaining:
                continue
            remaining -= cost
            added.append(
                {
                    **record,
                    "retrieval_role": "source_context",
                    "supporting_document_ids": [identity for identity, _ in parents[group]],
                    "score": 0.0,
                    "normalized_score": 0.0,
                }
            )
        results = [*anchors, *added]
        coverage = []
        for group, supporting in parents.items():
            indexed_ids = [identity for identity, _ in supporting]
            indexed_ids.extend(record["id"] for _, _, _, record, candidate_group in candidates if candidate_group == group)
            retrieved_ids = [r["id"] for r in results if r["id"] in set(indexed_ids)]
            coverage.append({"source_id": f"doc_{group[0]}", "source_title": group[2],
                             "indexed_document_ids": indexed_ids, "retrieved_document_ids": retrieved_ids})
        return {**bundle, "results": results, "source_context_added": len(added), "source_coverage": tuple(coverage)}
