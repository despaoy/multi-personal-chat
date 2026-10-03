"""Add bounded context from the same verified source as generic RAG anchors."""

import re

from inference.context_budget import estimated_tokens

_AUTHORITY_FIELDS = ("id", "document_id", "chunk_index", "knowledge_base_id", "title", "category", "content")


def requested_document_titles(query):
    """Explicit multi-document reads; quoted background does not grant a read.

    Resolve only the leading complete read operator and its contiguous titles.
    Keep the original query untouched for ranking, selection and generation.
    Unknown later title tasks or exclusions defer the entire request.
    """
    match = re.match(
        r"^(?:请)?(?:(?:查|查询|检索)知识库[，,：:]?\s*)?"
        r"(?:逐项比较|逐一比较|逐项核对|分别读取|分别列出|读取|核对|核查|查看|查阅)\s*"
        r"(?:[^《》\n，,。！？!?；;]{1,80}中\s*)?"
        r"(?P<titles>《[^《》\n]{1,200}》(?:\s*(?:[、，,]|和|与|及)?\s*《[^《》\n]{1,200}》)*)",
        str(query or "").strip(),
    )
    if not match:
        return ()
    remainder = str(query or "").strip()[match.end():]
    titles = tuple(dict.fromkeys(re.findall(r"《([^《》\n]{1,200})》", match["titles"])))
    later = re.findall(r"《([^《》\n]{1,200})》", remainder)
    unmatched = re.sub(r"《[^《》\n]{1,200}》", "", remainder)
    if (any(title not in titles for title in later) or "《" in unmatched or "》" in unmatched
            or re.search(r"(?:不要|不必|不用|无需|排除|仅|只|"
                         r"(?:^|[，,。；;：:\n]|请|也|还|另外|同时|并且|但是|但|千万)别)"
                         r"(?:再|逐项|逐一|分别)?(?:读|阅读|读取|查询|检索|比较|核对|列出|提供|查看)", remainder)):
        return ()
    declared = re.match(r"这([一二三四五六七八九十\d]+)份(?:完整)?(?:说明|资料|文档|文件)", remainder)
    if declared:
        value = declared[1]
        digits = {char: number for number, char in enumerate("零一二三四五六七八九")}
        if value.isdecimal():
            count = int(value)
        elif value in digits:
            count = digits[value]
        elif value.count('十') == 1:
            left, right = value.split('十')
            if (left and left not in digits) or (right and right not in digits):
                return ()
            count = (digits[left] if left else 1) * 10 + (digits[right] if right else 0)
        else:
            return ()
        if count != len(titles):
            return ()
    return titles


def expand_source_context(bundle, vector_db, *, expected_generation, source_budget_tokens, filters=None, query=""):
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
        # Resolve explicit document-title references as source data, never as
        # instructions. Only exact, unambiguous titles in the referring source's
        # knowledge base and the original filter scope can add context.
        groups = {}
        for record in metadata.values():
            group = (record.get("document_id"), record.get("knowledge_base_id"),
                     record.get("title"), record.get("category"))
            if filters and not vector_db._match_filters(record, filters):
                continue
            groups.setdefault(group, []).append(record)
        original_groups = set(parents)
        queue = list(parents)
        requested_titles = requested_document_titles(query)
        requested_groups = set()
        unresolved_titles = []
        ambiguous_titles = []
        # A direct user read has independent roots in the original filter scope.
        # Ranked anchors do not define which knowledge bases may contain them.
        for title in requested_titles:
            targets = [group for group in groups if group[2] == title]
            if len(targets) > 1:
                # A direct read requests these authorized originals. Retain
                # each source identity rather than arbitrarily picking a
                # version or discarding independent requested documents.
                ambiguous_titles.append(title)
            if not targets:
                unresolved_titles.append(title)
                continue
            for target in targets:
                requested_groups.add(target)
                if target not in parents:
                    parents[target] = []
                    queue.append(target)
        for source_group in queue:
            for referring in groups.get(source_group, []):
                text = referring.get("content")
                if not isinstance(text, str) or not text.strip():
                    raise RuntimeError("Invalid indexed reference text")
                for title in re.findall(r"《([^《》\n]{1,200})》", text):
                    targets = [group for group in groups
                               if group[1] == source_group[1] and group[2] == title]
                    if len(targets) > 1:
                        raise RuntimeError("Ambiguous indexed document reference")
                    if not targets:
                        continue
                    target = targets[0]
                    if target in original_groups or target == source_group:
                        continue
                    support = (referring["id"], referring.get("chunk_index"))
                    if target not in parents:
                        parents[target] = [support]
                        queue.append(target)
                    elif queue.index(source_group) < queue.index(target) and support not in parents[target]:
                        parents[target].append(support)
        candidates = []
        for record in metadata.values():
            group = (record.get("document_id"), record.get("knowledge_base_id"),
                     record.get("title"), record.get("category"))
            if group not in parents or record["id"] in selected_ids:
                continue
            if filters and not vector_db._match_filters(record, filters):
                continue
            parent, index = record.get("document_id"), record.get("chunk_index")
            if (not isinstance(parent, int) or isinstance(parent, bool) or parent <= 0
                    or not isinstance(index, int) or isinstance(index, bool) or index < 0):
                raise RuntimeError("Invalid indexed sibling position")
            if record["id"] != f"doc_{parent}_chunk_{index}":
                raise RuntimeError("Invalid indexed sibling identity")
            distance = (min(abs(index - selected_index) for _, selected_index in parents[group])
                        if group in original_groups else 0)
            candidates.append((queue.index(group), distance, index, record, group))
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
            extra = {**record, "score": 0.0, "normalized_score": 0.0}
            if group in requested_groups:
                # A directly requested authorized source is an independent
                # task root, not support for a random ranked neighbor.
                extra["retrieval_role"] = "requested_source"
            else:
                extra.update(retrieval_role="source_context",
                             supporting_document_ids=[identity for identity, _ in parents[group]])
            added.append(extra)
        results = [*anchors, *added]
        coverage = []
        for group, _supporting in parents.items():
            indexed_ids = [record["id"] for record in groups[group]]
            indexed_set = set(indexed_ids)
            retrieved_ids = [r["id"] for r in results if r["id"] in indexed_set]
            coverage.append({"source_id": f"doc_{group[0]}", "source_title": group[2],
                             "indexed_document_ids": indexed_ids, "retrieved_document_ids": retrieved_ids})
        result = {**bundle, "results": results, "source_context_added": len(added), "source_coverage": tuple(coverage)}
        if requested_titles:
            result.update(requested_source_titles=list(requested_titles),
                          unresolved_requested_titles=unresolved_titles,
                          ambiguous_requested_titles=ambiguous_titles,
                          requested_source_scope="original_filter")
        return result
