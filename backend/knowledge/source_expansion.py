"""Add bounded context from the same verified source as generic RAG anchors."""

import re

from inference.context_budget import estimated_tokens

_AUTHORITY_FIELDS = ("id", "document_id", "chunk_index", "knowledge_base_id", "title", "category", "content")


def requested_document_titles(query):
    """Resolve independently closed document roots alongside private reads.

    The complete original question still reaches retrieval and generation.
    Legacy single declarations retain their existing scope checks; quoted,
    code, excluded and unresolved requests cannot become partial read grants.
    """
    text = str(query or "").strip()
    if "`" in text:
        return ()
    from db.memory_source_search import resolve_source_read_plan

    plan = resolve_source_read_plan(text)
    if plan.document_titles:
        return plan.document_titles
    from character.quoted_erasure_authority import masked_quotes

    try:
        masked = masked_quotes(text)[0]
    except ValueError:
        return ()
    starts = list(
        re.finditer(
            r"(?:^|[。！？!?；;\n])\s*(?=(?:请)?"
            r"(?:(?:查|查询|检索)知识库[，,：:]?\s*)?"
            r"(?:逐项比较|逐一比较|逐项核对|分别读取|分别列出|读取|核对|核查|查看|查阅)\s*"
            r"(?:[^《》\n，,。！？!?；;]{1,80}中\s*)?《)",
            masked,
        )
    )
    if len(starts) != 1:
        return ()
    start = starts[0].end()
    prefix = masked[:start]
    if "《" in prefix or "》" in prefix:
        return ()  # A preceding unresolved title task must not be guessed away.
    return _leading_document_titles(text[start:])


def _leading_document_titles(query):
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
    remainder = str(query or "").strip()[match.end() :]
    titles = tuple(dict.fromkeys(re.findall(r"《([^《》\n]{1,200})》", match["titles"])))
    return _checked_document_titles(titles, remainder)


def _checked_document_titles(titles, remainder):
    """Validate the unresolved suffix without changing the actual question."""
    later = re.findall(r"《([^《》\n]{1,200})》", remainder)
    unmatched = re.sub(r"《[^《》\n]{1,200}》", "", remainder)
    exclusion_text = remainder
    for task in re.finditer(r"本轮只核对(?P<body>[^。！？!?；;\n]+)", remainder):
        body = task["body"]
        if (
            re.search(r"条件|规则|例外|限制|差异", body)
            and not re.search(
                r"《|》|第|(?:这|那|一|两|三|四|五|六|七|八|九|十|\d+)份|篇|文档|资料|文件|前者|后者|那个|这个", body
            )
            and not any(title in body for title in titles)
        ):
            # This names the questions to answer after reading all originals.
            # Literal titles or document selectors still defer the whole task.
            exclusion_text = exclusion_text.replace(task[0], task[0].replace("只核对", "核对", 1), 1)
    # A closed nonmutating-memory scope declares no document exclusion.
    # Locate it in the unquoted view and edit only this exclusion-check view;
    # ranking and generation still receive the complete original question.
    from character.quoted_erasure_authority import masked_quotes

    try:
        scope_view = masked_quotes(exclusion_text)[0]
    except ValueError:
        return ()
    for task in reversed(
        list(
            re.finditer(
                r"(?:^|[。！？!?；;\n])\s*本轮只读取(?P<scope>[^，,。！？!?；;\n]{0,20})[，,]\s*不新增或删除记忆(?=[，,。！？!?；;\n]|$)",
                scope_view,
            )
        )
    ):
        # This is a generic existing-data scope plus an explicit nonmutation
        # clause. Concrete document selectors or unclosed scopes retain the
        # original exclusion check; quoted scopes were masked above.
        if task["scope"].strip() not in {"", "既有事实", "已有事实", "既有资料", "已有资料", "既有信息", "已有信息"}:
            continue
        a, z = task.span()
        exclusion_text = exclusion_text[:a] + exclusion_text[a:z].replace("只读取", "读取", 1) + exclusion_text[z:]
    if (
        any(title not in titles for title in later)
        or "《" in unmatched
        or "》" in unmatched
        or re.search(
            r"(?:不要|不必|不用|无需|排除|仅|只|"
            r"(?:^|[，,。；;：:\n]|请|也|还|另外|同时|并且|但是|但|千万)别)"
            r"(?:再|逐项|逐一|分别)?(?:读|阅读|读取|查询|检索|比较|核对|列出|提供|查看)",
            exclusion_text,
        )
    ):
        return ()
    declared = re.match(r"这([一二三四五六七八九十\d]+)份(?:完整)?(?:说明|资料|文档|文件)", remainder)
    if declared:
        value = declared[1]
        digits = {char: number for number, char in enumerate("零一二三四五六七八九")}
        if value.isdecimal():
            count = int(value)
        elif value in digits:
            count = digits[value]
        elif value.count("十") == 1:
            left, right = value.split("十")
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
    requested_titles = requested_document_titles(query)
    discard_candidates = bool(bundle.get("abstained") or not anchors)
    if discard_candidates:
        if not requested_titles:
            return bundle
        # The explicit title is a separate exact lookup, not a confidence
        # upgrade for rejected semantic candidates. Resolve it only after
        # validating the same index generation and original filter scope.
        anchors = []
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
        scope_excluded = []
        if requested_titles:
            # Only declared named originals are independent task roots. Validate
            # every ranked anchor before filtering; source corruption cannot be
            # hidden by an unrelated title. Literal references may still recover
            # a ranked neighbor as dependent, zero-vote source context below.
            parents = {group: support for group, support in parents.items() if group[2] in requested_titles}
            scoped = []
            for anchor in anchors:
                group = (
                    anchor.get("document_id"),
                    anchor.get("knowledge_base_id"),
                    anchor.get("title"),
                    anchor.get("category"),
                )
                if group in parents:
                    scoped.append(anchor)
                else:
                    scope_excluded.append(anchor["id"])
            anchors = scoped
            selected_ids = {anchor["id"] for anchor in anchors}
            if not anchors:
                discard_candidates = True
        # Resolve explicit document-title references as source data, never as
        # instructions. Only exact, unambiguous titles in the referring source's
        # knowledge base and the original filter scope can add context.
        groups = {}
        for record in metadata.values():
            group = (
                record.get("document_id"),
                record.get("knowledge_base_id"),
                record.get("title"),
                record.get("category"),
            )
            if filters and not vector_db._match_filters(record, filters):
                continue
            groups.setdefault(group, []).append(record)
        original_groups = set(parents)
        queue = list(parents)
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
        references = {}
        for source_group in queue:
            for referring in groups.get(source_group, []):
                text = referring.get("content")
                if not isinstance(text, str) or not text.strip():
                    raise RuntimeError("Invalid indexed reference text")
                for title in re.findall(r"《([^《》\n]{1,200})》", text):
                    targets = [group for group in groups if group[1] == source_group[1] and group[2] == title]
                    for target in targets:
                        for linked in groups[target]:
                            parent, index = linked.get("document_id"), linked.get("chunk_index")
                            if (
                                not isinstance(parent, int)
                                or isinstance(parent, bool)
                                or parent <= 0
                                or not isinstance(index, int)
                                or isinstance(index, bool)
                                or index < 0
                            ):
                                raise RuntimeError("Invalid indexed sibling position")
                            if linked["id"] != f"doc_{parent}_chunk_{index}":
                                raise RuntimeError("Invalid indexed sibling identity")
                    references[(source_group[0], title)] = dict(
                        referring_source_id=f"doc_{source_group[0]}",
                        referring_source_title=source_group[2],
                        referenced_title=title,
                        lookup_status=(
                            "ambiguous_in_referring_scope"
                            if len(targets) > 1
                            else "matched_in_referring_scope"
                            if targets
                            else "not_found_in_referring_scope"
                        ),
                        source_ids=sorted({f"doc_{target[0]}" for target in targets}),
                    )
                    if len(targets) != 1:
                        # A title collision cannot choose a referenced version,
                        # but does not invalidate independently verified roots.
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
            parent, index = record.get("document_id"), record.get("chunk_index")
            if (
                not isinstance(parent, int)
                or isinstance(parent, bool)
                or parent <= 0
                or not isinstance(index, int)
                or isinstance(index, bool)
                or index < 0
            ):
                raise RuntimeError("Invalid indexed sibling position")
            if record["id"] != f"doc_{parent}_chunk_{index}":
                raise RuntimeError("Invalid indexed sibling identity")
            distance = (
                min(abs(index - selected_index) for _, selected_index in parents[group])
                if group in original_groups
                else 0
            )
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
                extra.update(
                    retrieval_role="source_context",
                    supporting_document_ids=[identity for identity, _ in parents[group]],
                    supporting_source_refs=tuple(
                        dict(
                            document_id=identity,
                            source_id=f"doc_{metadata[identity]['document_id']}",
                            source_title=metadata[identity]["title"],
                            knowledge_base_id=metadata[identity]["knowledge_base_id"],
                            relation="same_source"
                            if metadata[identity]["document_id"] == group[0]
                            else "title_reference",
                            target_title=group[2],
                            text=metadata[identity]["content"],
                        )
                        for identity, _ in parents[group]
                    ),
                )
            added.append(extra)
        results = [*anchors, *added]
        coverage = []
        for group, _supporting in parents.items():
            indexed_ids = [record["id"] for record in groups[group]]
            indexed_set = set(indexed_ids)
            retrieved_ids = [r["id"] for r in results if r["id"] in indexed_set]
            coverage.append(
                {
                    "source_id": f"doc_{group[0]}",
                    "source_title": group[2],
                    "indexed_document_ids": indexed_ids,
                    "retrieved_document_ids": retrieved_ids,
                }
            )
        result = {**bundle, "results": results, "source_context_added": len(added), "source_coverage": tuple(coverage)}
        if scope_excluded:
            from knowledge.rag_helper import get_rag_helper

            # Strong unrelated votes cannot certify the remaining semantic
            # roots. Exact title lookup retains its separate source authority.
            confidence = get_rag_helper().compute_confidence(anchors)
            result.update(
                confidence=min(float(bundle.get("confidence", 0.0)), confidence),
                original_semantic_confidence=bundle.get("confidence", 0.0),
                scope_excluded_semantic_document_ids=scope_excluded,
                citations=[],
            )
        if references:
            result["source_references"] = tuple(references.values())
        if discard_candidates:
            result.update(abstained=not bool(results), citations=[], semantic_candidates_discarded=True)
        if requested_titles:
            result.update(
                requested_source_titles=list(requested_titles),
                unresolved_requested_titles=unresolved_titles,
                ambiguous_requested_titles=ambiguous_titles,
                requested_source_scope="original_filter",
            )
        return result
