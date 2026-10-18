"""Shared complete preparation records for isolated context evaluations."""

import json
from dataclasses import asdict


def observe_preparation(prepare_turn, proof, output):
    """Persist the observed context; preparation and file errors remain visible."""
    async def observed_prepare(service, turn, character_id):
        result = await prepare_turn(service, turn, character_id)
        proof["prepared"].append({
            "semantic_status": result.semantic_review_status,
            "policy_status": result.contextual_policy_status,
            "history": list(result.history),
            "recall": result.memory_recall,
            "reference_context": result.compiled.reference_context,
            "used_memory_ids": list(result.compiled.used_memory_ids),
            "memory_packets": [asdict(packet) for packet in result.compiled.memory_packets],
            "selection_status": result.memory_selection_status,
            "received_at": result.received_at.isoformat(),
            "memory_budget": result.memory_budget,
            "episodic_context": result.compiled.episodic_reference_context,
            "source_candidate_context": result.compiled.source_candidate_context,
            "selection_candidate_count": result.memory_selection_candidate_count,
        })
        output.write_text(json.dumps(proof, ensure_ascii=False, indent=2, default=str) + "\n")
        return result

    return observed_prepare
