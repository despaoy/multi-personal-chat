"""Complete original speech retrieval, separate from current-fact projection.

Sparse scope-first search and selected-claim source expansion; no LLM reviewer,
text cache, recent-N history approximation, or inferred fact ownership.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from html import escape

from db.memory_source_search import literal_project_source_terms, literal_source_fragments, terms


@dataclass(frozen=True)
class SourceRecall:
    context: str = ""
    diagnostics: dict = field(default_factory=dict)
    candidate_context: str = ""


def requested_source_successors(query):
    """A closed request for each anchor's next currently visible raw record.

    This resolves user-requested record order, never semantic project ownership
    or the original physical message sequence before erasure.
    """
    if len(literal_project_source_terms(query)) != 1:
        return False
    compact = "".join(query.split())
    if not re.search(r"每条(?:原始)?(?:交接)?记录", compact):
        return False
    match = re.search(r"(?:紧随其后的|紧接其后的|紧随各条记录的)下一条(?:仍可见)?(?:用户)?原话记录", compact)
    if not match:
        return False
    return not re.search(r"(?:不要|不需要|无需|禁止|别|不含|排除|不包括)(?:列出|读取|查看|包括)?$",
                         compact[max(0, match.start()-12):match.start()])


def select_sources(linked, found, *, contextual=(), limit=4):
    rows, scores = {}, {}
    for lane in (linked, found, contextual):
        for rank, row in enumerate(lane, 1):
            key = row["source_message_id"]
            rows.setdefault(key, row)
            scores[key] = scores.get(key, 0.0) + 1.0 / (60 + rank)
    # Preserve both retrieval routes. Four old accepted facts must not occupy
    # every slot and suppress the top source-only correction/condition.
    required = list(dict.fromkeys(lane[0]["source_message_id"] for lane in (linked, found, contextual) if lane))
    ranking = sorted(rows, key=lambda key: (scores[key], rows[key]["observed_at"], key), reverse=True)
    selected = list(dict.fromkeys([*required, *ranking]))
    if limit is not None:
        selected = selected[:limit]
    return [rows[key] for key in selected], len(rows)


def covered_by_fact(row, memories):
    """Do not add an identical quote merely to disable an already complete read.

    Both complete text AND receipt must match an admitted, unconditional fact.
    Newer repeated speech, observations and conditional evidence still matter.
    """
    for item in memories:
        if (item.memory_type != "user_fact" or item.status != "active" or item.qualifiers
                or item.temporal_mode == "observation" or row["body"] not in item.evidence):
            continue
        try:
            if datetime.fromisoformat(row["observed_at"]) == datetime.fromisoformat(item.observed_at):
                return True
        except (TypeError, ValueError):
            continue
    return False


def compile_sources(rows, *, max_chars=2400, max_items=4):
    if max_chars is not None and (type(max_chars) is not int or max_chars <= 0):
        raise ValueError("Source character budget must be positive or deferred")
    records = []
    omitted = []
    for row in rows:
        if max_items is not None and len(records) == max_items:
            omitted.append(dict(source_id=row["source_message_id"], reason="item_budget"))
            continue
        record = dict(source_id=row["source_message_id"], observed_at=row["observed_at"], text=row["body"])
        candidate = json.dumps(dict(source_kind="historical_user_utterances", speaker_role="user",
            described_subject="not_resolved", current_validity="not_resolved", records=[*records, record]),
            ensure_ascii=False, separators=(",", ":"))
        if max_chars is not None and len(escape(candidate, quote=False)) > max_chars:
            # Do not substitute older/lower-ranked excerpts when the best
            # complete source (possibly a correction) cannot fit.
            omitted.append(dict(source_id=row["source_message_id"], reason="whole_source_budget"))
            return SourceRecall("", dict(status="budget_omitted", omitted=omitted, selected_ids=[]))
        records.append(record)
    packet = (json.dumps(dict(source_kind="historical_user_utterances", speaker_role="user",
                 described_subject="not_resolved", current_validity="not_resolved", records=records),
                 ensure_ascii=False, separators=(",", ":")) if records else "")
    return SourceRecall(packet, dict(status="available" if records else "no_match", omitted=omitted,
                                    selected_ids=[row["source_id"] for row in records]))


class SourceMemoryService:
    def __init__(self, repository, *, max_chars=2400, window_radius=0, defer_budget=False):
        if type(window_radius) is not int or not 0 <= window_radius <= 2:
            raise ValueError("Source window radius must be 0..2")
        self._repo = repository
        self._max_chars = max_chars
        self._window_radius = window_radius
        self._defer_budget = defer_budget

    async def recall(self, character_id, scope, query, *, memories=(), retrieval_context=""):
        """Keep current-query and loaded user-history search as separate scoped lanes."""
        started = time.monotonic()
        if scope.adapter == "narrative":
            return SourceRecall(diagnostics=dict(status="unsupported_branch"))
        reader = getattr(self._repo, "list_sources", None)
        search = getattr(self._repo, "search_sources", None)
        linker = getattr(self._repo, "linked_sources", None)
        if not callable(reader) or not callable(search) or not callable(linker):
            return SourceRecall(diagnostics=dict(status="unsupported_adapter"))
        ids = tuple(dict.fromkeys(int(memory.memory_id) for memory in memories
                                   if str(memory.memory_id).isdigit() and int(memory.memory_id) > 0))[:200]
        try:
            # Complete source packets are admitted by the actual serving budget;
            # a count-only SQL candidate cut cannot prove an exhaustive read.
            search_limit = None if self._defer_budget else 32
            source_fragments = literal_source_fragments(query)
            project_tags = () if source_fragments else literal_project_source_terms(query)
            explicit_source_scope = bool(project_tags or source_fragments)
            requested_order = requested_source_successors(query)
            effective_radius = 1 if requested_order else self._window_radius
            contextual_deferred = bool(explicit_source_scope and retrieval_context.strip())
            lanes = [linker(character_id, scope, memory_ids=ids),
                     search(character_id, scope, query=query, limit=search_limit)]
            if retrieval_context.strip() and not explicit_source_scope:
                lanes.append(search(character_id, scope, query=retrieval_context, limit=search_limit))
            linked, found, *contextual_results = await asyncio.gather(*lanes)
            contextual = contextual_results[0] if contextual_results else []
            original_linked_read_count = len(linked)
            linked_project_omitted = 0
            if project_tags:
                scoped_linked = [row for row in linked if set(project_tags).intersection(terms(row['body']))]
                linked_project_omitted = len(linked) - len(scoped_linked)
                linked = scoped_linked

            linked_fragment_omitted = 0
            if source_fragments:
                scoped_linked = [row for row in linked if any(fragment in row["body"] for fragment in source_fragments)]
                linked_fragment_omitted = len(linked) - len(scoped_linked)
                linked = scoped_linked

            linked_read_count, indexed_read_count = original_linked_read_count, len(found)
            contextual_read_count = len(contextual)
            covered = {row["source_message_id"] for row in [*linked, *found, *contextual]
                       if covered_by_fact(row, memories)}
            if not effective_radius:
                linked = [row for row in linked if row["source_message_id"] not in covered]
                found = [row for row in found if row["source_message_id"] not in covered]
                contextual = [row for row in contextual if row["source_message_id"] not in covered]
            # Linked sources were selected by fact recall; lexical candidates
            # cover utterances whose model produced no usable claim at all.
            # Cloud serving budgets admit complete sources by actual size.
            # Legacy compact callers retain their existing four-source cap.
            selected, candidate_count = select_sources(linked, found, contextual=contextual,
                limit=None if self._defer_budget else 4)
            window_ids = []
            requested_followers = {}
            anchor_ids = [row["source_message_id"] for row in selected]
            if effective_radius and selected:
                windows = []
                # The repository keeps its bounded four-anchor read contract.
                for offset in range(0, len(anchor_ids), 4):
                    windows.extend(await self._repo.source_windows(character_id, scope,
                        source_message_ids=tuple(anchor_ids[offset:offset + 4]), radius=effective_radius))
                rows_by_id = {}
                for window in windows:
                    rows = window["rows"]
                    if requested_order:
                        anchors = [row for row in rows if row['source_message_id'] == window['anchor_id']]
                        if len(anchors) != 1:
                            raise ValueError("Requested source-order anchor unavailable")
                        rows = [anchors[0], *window['following_rows'][:1]]
                        requested_followers[window['anchor_id']] = tuple(row['source_message_id'] for row in rows[1:])

                    # Exact fact dedup must not suppress the anchor needed to
                    # retrieve a source-only later correction.
                    if len(rows) == 1 and window["anchor_id"] in covered and not requested_order:
                        continue
                    window_ids.append(dict(anchor_id=window["anchor_id"],
                                           source_ids=[row["source_message_id"] for row in rows]))
                    rows_by_id.update((row["source_message_id"], row) for row in rows)
                selected = sorted(rows_by_id.values(), key=lambda row: (row["observed_at"], json.dumps(row["source_message_id"]) if requested_order else row["source_message_id"]))
            # Re-read only chosen identities after ranking. Erased sources must
            # not be restored from cached candidate text; use current SQL grant.
            fresh = []
            chosen_ids = tuple(row["source_message_id"] for row in selected)
            # Keep exact scoped fresh reads below the repository's row limit;
            # a larger union/window must not silently lose rows in SQL LIMIT.
            for offset in range(0, len(chosen_ids), 100):
                fresh.extend(await reader(character_id, scope,
                    source_message_ids=chosen_ids[offset:offset + 100]))
            fresh_by_id = {row["source_message_id"]: row for row in fresh}
            selected = [fresh_by_id[row["source_message_id"]] for row in selected
                        if row["source_message_id"] in fresh_by_id
                        and (not source_fragments or any(fragment in fresh_by_id[row["source_message_id"]]["body"]
                                                    for fragment in source_fragments))]
            dependency_omitted = 0
            if requested_order:
                valid_ids = set(anchor_ids) | {child for parent, children in requested_followers.items()
                                                if parent in fresh_by_id for child in children}
                with_authorized_anchor = [row for row in selected if row['source_message_id'] in valid_ids]
                dependency_omitted = len(selected) - len(with_authorized_anchor)
                selected = with_authorized_anchor
            result = compile_sources(selected, max_chars=self._max_chars,
                                     max_items=None if self._defer_budget else 4 * (1 + 2 * self._window_radius))
            trace = dict(result.diagnostics, linked_count=len(linked), indexed_count=len(found),
                         linked_read_count=linked_read_count, indexed_read_count=indexed_read_count,
                         contextual_count=len(contextual), contextual_read_count=contextual_read_count,
                         contextual_search_enabled=bool(contextual_results),
                         literal_project_terms=list(project_tags),
                         contextual_scope_deferred_to_explicit_task=contextual_deferred,
                         linked_project_scope_omitted=linked_project_omitted,
                         literal_fragment_scope=bool(source_fragments),
                         literal_fragment_count=len(source_fragments),
                         literal_fragment_match_counts=[sum(fragment in row["body"] for row in selected)
                                                        for fragment in source_fragments],
                         linked_fragment_scope_omitted=linked_fragment_omitted,
                         covered_by_fact_count=len(covered),
                         window_radius=self._window_radius, effective_window_radius=effective_radius,
                         requested_source_order="next_visible_after_each_anchor" if requested_order else "not_resolved",
                         requested_following=[dict(anchor_id=key, source_ids=list(value)) for key,value in requested_followers.items()],
                         following_missing_for_anchors=[key for key in anchor_ids if requested_order and not requested_followers.get(key)],
                         following_anchor_dependency_omitted=dependency_omitted,
                         windows=window_ids,
                         window_semantic_relation="not_inferred", anchor_ids=anchor_ids,
                         selection_omitted=max(0, candidate_count - len(anchor_ids)),
                         fresh_recheck_omitted=max(0, len(chosen_ids) - len(selected)),
                         source_search_limit=search_limit,
                         candidate_limit_reached=search_limit is not None and (
                             indexed_read_count == search_limit or contextual_read_count == search_limit),
                         ranking=("covered_rrf_linked_sparse_history" if contextual_results
                                  else "covered_rrf_linked_sparse"),
                         query_terms_semantic=False, elapsed_ms=(time.monotonic() - started) * 1000)
            candidate_context = ""
            if self._defer_budget and result.diagnostics["status"] == "budget_omitted":
                candidate_context = compile_sources(selected, max_chars=None,
                    max_items=None if self._defer_budget else 4 * (1 + 2 * self._window_radius)).context
            trace["candidate_budget_pending"] = bool(candidate_context)
            return SourceRecall(result.context, trace, candidate_context)
        except Exception as exc:
            return SourceRecall(diagnostics=dict(status="retrieval_error", error_type=type(exc).__name__,
                                                 elapsed_ms=(time.monotonic() - started) * 1000))


def _share_observation_sources(context, source_context, *, preferred_address, complete_evidence):
    from character.context_builder import compile_reference_context

    try:
        data = json.loads(source_context)
        if (data.get('source_kind') != 'historical_user_utterances' or data.get('speaker_role') != 'user'
                or not isinstance(data.get('records'), list)):
            return context
        records = {row['source_id']: row for row in data['records']}
        if len(records) != len(data['records']):
            return context
        covered = []
        for item in context.memory_packets:
            if (not item.source_observation or item.temporal_mode != 'observation'
                    or item.status != 'active' or item.relation_type != 'ADD'
                    or item.complete_original_source is False
                    or item.historical or item.qualifiers or len(item.evidence) != 1
                    or len(item.source_message_ids) != 1):
                continue
            row = records.get(item.source_message_ids[0])
            if (row and row['text'] == item.evidence[0] and row['observed_at'] == item.observed_at
                    and item.memory_id in context.used_memory_ids):
                covered.append(item.memory_id)
    except (ValueError, TypeError, KeyError, AttributeError):
        return context
    if not covered:
        return context
    original, ids = compile_reference_context(context.memory_packets, preferred_address=preferred_address,
                                               complete_evidence=complete_evidence)
    # Never overwrite additional reference content, a caller-specific packet
    # format, or an address whose compilation parameters were not supplied.
    if original != context.reference_context or ids != context.used_memory_ids:
        return context
    remaining = tuple(item for item in context.memory_packets
                      if item.memory_id in context.used_memory_ids and item.memory_id not in covered)
    reference, _ = compile_reference_context(remaining, preferred_address=preferred_address,
                                             complete_evidence=complete_evidence)
    return replace(context, reference_context=reference, source_shared_memory_ids=tuple(covered),
                   source_reference_backup=context.reference_context)


def attach_sources(context, result, *, preferred_address="", complete_evidence=False, share_observations=False):
    if context.source_shared_memory_ids:
        context = replace(context, reference_context=context.source_reference_backup,
                          source_shared_memory_ids=(), source_reference_backup="", episodic_reference_context="")
    status = str(result.diagnostics.get('status') or 'not_checked')
    context = replace(context, memory_source_status=status, episodic_reference_context='',
                      source_candidate_context=result.candidate_context)
    if not result.context:
        if status in {'budget_omitted', 'retrieval_error'}:
            return replace(context, memory_field_presence=tuple(
                (key, None if value is False else value) for key, value in context.memory_field_presence))
        return context
    # Unclassified but relevant source speech defeats a proof of absence, not
    # an independently established fact. No raw quote becomes a MemoryItem.
    # Offline candidate only: shorter input did not reliably improve live
    # answers. Keep the established reference transport by default.
    if share_observations:
        context = _share_observation_sources(context, result.context, preferred_address=preferred_address,
                                             complete_evidence=complete_evidence)
    return replace(context, episodic_reference_context=result.context,
                   memory_field_presence=tuple((key, None if value is False else value)
                                               for key, value in context.memory_field_presence))
