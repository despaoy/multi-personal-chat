"""Admit a complete deferred source packet against the actual generation contract."""

import json
from dataclasses import replace
from datetime import datetime

from character.source_memory import SourceRecall, attach_sources


def _complete_packet(text):
    try:
        packet = json.loads(text)
        if (
            set(packet) != {"source_kind", "speaker_role", "described_subject", "current_validity", "records"}
            or packet["source_kind"] != "historical_user_utterances"
            or packet["speaker_role"] != "user"
            or packet["described_subject"] != "not_resolved"
            or packet["current_validity"] != "not_resolved"
            or not isinstance(packet["records"], list)
            or not packet["records"]
        ):
            return False
        ids = set()
        for row in packet["records"]:
            if (
                set(row) != {"source_id", "observed_at", "text"}
                or not isinstance(row["source_id"], str)
                or not row["source_id"]
                or row["source_id"] in ids
                or not isinstance(row["text"], str)
                or not row["text"].strip()
                or not isinstance(row["observed_at"], str)
                or datetime.fromisoformat(row["observed_at"]).utcoffset() is None
            ):
                return False
            ids.add(row["source_id"])
        return True
    except (ValueError, TypeError, KeyError, AttributeError):
        return False


def admit_deferred_sources(request, build):
    """Source priority is settled once before adding bounded public RAG packets.

    The producer supplies whole, freshly granted speech. This does not infer
    current facts, re-read stored text, split a record or invoke any model.
    """
    context = request.character_context
    candidate = getattr(context, "source_candidate_context", "")
    if not candidate:
        return request

    if not _complete_packet(candidate):
        raise ValueError("Invalid deferred source packet; restore complete attributed source records before retrying")
    admitted = attach_sources(context, SourceRecall(candidate, {"status": "available"}))
    proposed = replace(request, character_context=admitted)
    # Public evidence may consume the remaining allowance, but cannot cause
    # separate packet attempts to silently change this private-source decision.
    base = replace(proposed, retrieval=type(request.retrieval)())
    build(base)
    return proposed
