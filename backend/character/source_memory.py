"""Complete original speech retrieval, separate from current-fact projection.

Sparse scope-first search and selected-claim source expansion; no LLM reviewer,
text cache, recent-N history approximation, or inferred fact ownership.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from html import escape


@dataclass(frozen=True)
class SourceRecall:
    context: str = ""
    diagnostics: dict = field(default_factory=dict)


def select_sources(linked, found, *, limit=4):
    rows, scores = {}, {}
    for lane in (linked, found):
        for rank, row in enumerate(lane, 1):
            key = row["source_message_id"]
            rows.setdefault(key, row)
            scores[key] = scores.get(key, 0.0) + 1.0 / (60 + rank)
    # Preserve both retrieval routes. Four old accepted facts must not occupy
    # every slot and suppress the top source-only correction/condition.
    required = list(dict.fromkeys(lane[0]["source_message_id"] for lane in (linked, found) if lane))
    ranking = sorted(rows, key=lambda key: (scores[key], rows[key]["observed_at"], key), reverse=True)
    selected = list(dict.fromkeys([*required, *ranking]))[:limit]
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
    records = []
    omitted = []
    for row in rows:
        if len(records) == max_items:
            omitted.append(dict(source_id=row["source_message_id"], reason="item_budget"))
            continue
        record = dict(source_id=row["source_message_id"], observed_at=row["observed_at"], text=row["body"])
        candidate = json.dumps(dict(source_kind="historical_user_utterances", speaker_role="user",
            described_subject="not_resolved", current_validity="not_resolved", records=[*records, record]),
            ensure_ascii=False, separators=(",", ":"))
        if len(escape(candidate, quote=False)) > max_chars:
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
    def __init__(self, repository, *, max_chars=2400, window_radius=0):
        if type(window_radius) is not int or not 0 <= window_radius <= 2:
            raise ValueError("Source window radius must be 0..2")
        self._repo = repository
        self._max_chars = max_chars
        self._window_radius = window_radius

    async def recall(self, character_id, scope, query, *, memories=()):
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
            linked, found = await asyncio.gather(
                linker(character_id, scope, memory_ids=ids),
                search(character_id, scope, query=query, limit=32))
            linked_read_count, indexed_read_count = len(linked), len(found)
            covered = {row["source_message_id"] for row in [*linked, *found] if covered_by_fact(row, memories)}
            if not self._window_radius:
                linked = [row for row in linked if row["source_message_id"] not in covered]
                found = [row for row in found if row["source_message_id"] not in covered]
            # Linked sources were selected by fact recall; lexical candidates
            # cover utterances whose model produced no usable claim at all.
            selected, candidate_count = select_sources(linked, found)
            window_ids = []
            anchor_ids = [row["source_message_id"] for row in selected]
            if self._window_radius and selected:
                windows = await self._repo.source_windows(character_id, scope,
                    source_message_ids=tuple(anchor_ids), radius=self._window_radius)
                rows_by_id = {}
                for window in windows:
                    rows = window["rows"]
                    # Exact fact dedup must not suppress the anchor needed to
                    # retrieve a source-only later correction.
                    if len(rows) == 1 and window["anchor_id"] in covered:
                        continue
                    window_ids.append(dict(anchor_id=window["anchor_id"],
                                           source_ids=[row["source_message_id"] for row in rows]))
                    rows_by_id.update((row["source_message_id"], row) for row in rows)
                selected = sorted(rows_by_id.values(), key=lambda row: (row["observed_at"], row["source_message_id"]))
            # Re-read only chosen identities after ranking. Erased sources must
            # not be restored from cached candidate text; use current SQL grant.
            fresh = await reader(character_id, scope, source_message_ids=tuple(
                row["source_message_id"] for row in selected))
            fresh_by_id = {row["source_message_id"]: row for row in fresh}
            selected = [fresh_by_id[row["source_message_id"]] for row in selected
                        if row["source_message_id"] in fresh_by_id]
            result = compile_sources(selected, max_chars=self._max_chars,
                                     max_items=4 * (1 + 2 * self._window_radius))
            trace = dict(result.diagnostics, linked_count=len(linked), indexed_count=len(found),
                         linked_read_count=linked_read_count, indexed_read_count=indexed_read_count,
                         covered_by_fact_count=len(covered),
                         window_radius=self._window_radius, windows=window_ids,
                         window_semantic_relation="not_inferred", anchor_ids=anchor_ids,
                         selection_omitted=max(0, candidate_count - 4),
                         candidate_limit_reached=indexed_read_count == 32, ranking="covered_rrf_linked_sparse",
                         query_terms_semantic=False, elapsed_ms=(time.monotonic() - started) * 1000)
            return SourceRecall(result.context, trace)
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
    context = replace(context, memory_source_status=status, episodic_reference_context='')
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
