"""Resolve history references through the existing source-blind semantic reviews.

Only proven object names supplement the complete current query. Historical
tasks, model claims and guessed facts never become retrieval query text.
"""

from inference.provider_context import get_provider_context_budget
from knowledge.public_question_binding import bound_retrieval_query, resolve_question_binding
from knowledge.retrieval_query_plan import plan_retrieval


async def contextual_retrieval_query(message: str, history) -> str | None:
    if not history:
        return message
    budget = get_provider_context_budget()
    plan = await plan_retrieval(message, history=history, context_budget=budget.review)
    if not plan.needs_public_retrieval:
        return None
    if not dict(plan.dependencies.groups)["public_knowledge"]:
        return message
    binding = await resolve_question_binding(
        plan.dependencies, message, window_tokens=budget.window_tokens,
        public_obligations=plan.public_obligations, history=history,
        context_budget=budget.review,
    )
    return bound_retrieval_query(binding, message)
