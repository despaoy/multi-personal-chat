"""Read views for model-proposed lifetimes; persisted source rows stay intact."""

from __future__ import annotations

import json
from datetime import datetime

from character.memory_extractor import extract_memories
from character.memory_request import memory_statement_body
from character.temporal_expression import source_temporal_spans


def project_temporal_record(row: dict) -> dict:
    """Separate a rule-supported present assertion from an observed statement.

    Only the new semantic provenance schema is interpreted. Legacy and rule
    records retain their existing contract; no implicit data migration occurs.
    A matched independent assertion gets the normal until-revised state view,
    anchored to its observation, never to proposed model endpoints. Other
    statements remain quotes with unknown applicability, not pending claims.
    Lifecycle/authorization/confidence gates remain the caller's responsibility.
    """
    metadata = row.get("metadata")
    provenance = metadata.get("temporal_provenance") if isinstance(metadata, dict) else None
    if not isinstance(provenance, dict) or provenance.get("version") != 1:
        return row
    # Neither guessed endpoints nor their omission establishes applicability.
    # Classify the proposition from its source identically in both cases.
    if (provenance.get("producer") != "semantic_memory"
            or provenance.get("validity_authority") not in ("unverified", "unspecified")):
        return row
    try:
        observed = datetime.fromisoformat(str(row.get("observed_at") or ""))
        if observed.utcoffset() is None:
            return row
    except ValueError:
        return row
    evidence = row.get("evidence")
    sources = row.get("source_message_ids")
    if (not isinstance(evidence, (list, tuple)) or not evidence
            or not all(isinstance(value, str) and value.strip() for value in evidence)
            or not isinstance(sources, (list, tuple)) or not sources):
        return row
    # A partial first clause must not erase an unparsed temporal restriction in
    # the remainder. Do not use model confidence as proof of temporal meaning.
    supported_contents: set[str] = set()
    if (not metadata.get("qualifiers") and row.get("memory_type") == "user_fact"
            and row.get("status", "active") in {"active", "current"}):
        for quote in evidence:
            statement = memory_statement_body(quote)
            # The existing extractor searches predicate spans. A leading time
            # frame or reported speaker must not become a present assertion.
            if not statement.startswith("我") or source_temporal_spans(statement):
                continue
            for item in extract_memories(statement, reference_time=observed):
                if (item.memory_key == row.get("memory_key")
                        and not item.qualifiers and item.evidence.strip().rstrip("。") == statement.strip().rstrip("。")):
                    supported_contents.add(item.content)
    supported = len(supported_contents) == 1
    view = dict(row)
    view.update(valid_from=observed.isoformat(), valid_to="", valid_at="", invalid_at="",
                temporal_mode="asserted_state" if supported else "observation",
                temporal_observed_at=observed.isoformat())
    if supported:
        # A display paraphrase is not the fact's identity. The independent
        # source parser supplies the canonical read view; never use the model
        # summary (or its proposed time bounds) to establish a current value.
        # Conflicting source values remain unresolved, and storage is untouched.
        view['content'] = next(iter(supported_contents))
    else:
        quotes = json.dumps(list(evidence), ensure_ascii=False)
        # Search source semantics, not the display label, clock or uncertainty
        # boilerplate. These fields are a read view and never persisted.
        view["retrieval_content"] = "\n".join(evidence)
        view["content"] = f"用户原话记录（时效未核实，不代表当前状态；记录于{observed.isoformat()}）：{quotes}"
    return view
