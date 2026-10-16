"""Keep temporal proposals separate from source-grounded time expressions.

This is provenance, not permission to execute a proposed interval. In
particular, finding the same timestamp in a statement proves neither its
semantic role (start/end/event) nor continuous validity between two dates.
"""

from __future__ import annotations

from datetime import datetime

from character.memory_clock import observation_clock
from character.memory_read_authority import MemoryRecordError
from character.temporal_expression import resolve_temporal_expression


def semantic_observation_clock(metadata: object, observed_at: object) -> datetime | None:
    """Validate semantic provenance; None denotes the legacy/rule contract."""
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        raise MemoryRecordError("metadata_json", "Memory metadata must be an object")
    if "temporal_provenance" not in metadata:
        return None
    provenance = metadata["temporal_provenance"]
    field = "metadata_json.temporal_provenance"
    if not isinstance(provenance, dict):
        raise MemoryRecordError(field, f"Invalid memory {field}")
    if type(provenance.get("version")) is not int or provenance["version"] != 1:
        raise MemoryRecordError(field + ".version", f"Unsupported memory {field}.version")
    producer = provenance.get("producer")
    if producer == "rule":
        return None
    if producer not in ("semantic_memory", "source_erasure_projection"):
        raise MemoryRecordError(field + ".producer", f"Unsupported memory {field}.producer")
    if provenance.get("validity_authority") not in ("unverified", "unspecified"):
        raise MemoryRecordError(field + ".validity_authority", f"Invalid memory {field}.validity_authority")
    try:
        observed = observed_at if isinstance(observed_at, datetime) else datetime.fromisoformat(observed_at)
    except (ValueError, TypeError):
        raise MemoryRecordError("observed_at", "Semantic memory observed_at must be an ISO timestamp with timezone") from None
    if observed.utcoffset() is None:
        raise MemoryRecordError("observed_at", "Semantic memory observed_at must include a timezone")
    return observed


def model_temporal_provenance(
    *, evidence: str, observed_at: datetime, proposed_from: str = "",
    proposed_to: str = "", time_expression: str = "",
) -> dict:
    """Record bounded inputs already validated by the memory proposal parser.

    Missing endpoints do not prove timelessness. Calendar periods are not
    executable lifetimes. The source clock is trusted; all model endpoints
    remain explicitly unverified, including ones that literally occur in text.
    """
    anchor = observation_clock(observed_at)
    proposals = {}
    instants: dict[str, datetime] = {}
    for role, raw in (("start", proposed_from), ("end", proposed_to)):
        if not raw:
            continue
        resolved = resolve_temporal_expression(raw, observed_at=anchor)
        entry = {"text": raw, "authority": "model_proposal", "role_verified": False,
                 "literal_in_evidence": raw in evidence}
        if resolved is not None:
            entry.update(kind=resolved.kind, precision=resolved.precision,
                         lower=resolved.lower.isoformat(),
                         upper=resolved.upper.isoformat() if resolved.upper else None)
            if resolved.kind == "instant":
                instants[role] = resolved.lower
        else:
            entry.update(kind="unresolved", precision=None)
        proposals[role] = entry

    shape = "unspecified" if not proposals else "open" if len(proposals) == 1 else "coarse_or_unresolved"
    if len(instants) == 2:
        start, end = instants["start"], instants["end"]
        shape = "ordered" if start < end else "zero_width" if start == end else "reversed"

    expression = None
    if time_expression and time_expression in evidence:
        resolved = resolve_temporal_expression(time_expression, observed_at=anchor)
        expression = {"text": time_expression, "role": "unspecified", "authority": "source_span"}
        if resolved is not None:
            expression.update(kind=resolved.kind, precision=resolved.precision,
                              lower=resolved.lower.isoformat(),
                              upper=resolved.upper.isoformat() if resolved.upper else None)
        else:
            expression.update(kind="unresolved", precision=None)
    return {"version": 1, "producer": "semantic_memory", "observed_at": anchor.isoformat(),
            "validity_authority": "unverified" if proposals else "unspecified",
            "proposed_bounds": proposals, "interval_shape": shape, "expression": expression}
