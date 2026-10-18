"""One observation contract for isolated generation evaluations."""

from dataclasses import asdict


def observe_generation(generate_character_response, records):
    """Record the completed result while preserving return and failure identity."""
    async def observed_generation(request, generate):
        result = await generate_character_response(request, generate)
        records.append({
            "messages": list(result.plan.messages),
            "retrieval": asdict(result.plan.retrieval),
            "response_mode": result.response_mode,
            "model_invoked": result.model_invoked,
            "citation_repair_status": result.citation_repair_status,
        })
        return result

    return observed_generation
