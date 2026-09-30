"""Keep temporal proposals separate from source-grounded time expressions.

This is provenance, not permission to execute a proposed interval. In
particular, finding the same timestamp in a statement proves neither its
semantic role (start/end/event) nor continuous validity between two dates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from character.memory_clock import observation_clock
from character.temporal_expression import resolve_temporal_expression

if TYPE_CHECKING:
    from datetime import datetime


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
