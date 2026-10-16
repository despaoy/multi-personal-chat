"""Resolve history references through the existing source-blind semantic reviews.

Only proven object names supplement the complete current query. Historical
tasks, model claims and guessed facts never become retrieval query text.
"""

from dataclasses import dataclass

from inference.provider_context import get_provider_context_budget
from knowledge.public_question_binding import bound_retrieval_query, resolve_question_binding
from knowledge.retrieval_query_plan import plan_retrieval


@dataclass(frozen=True)
class RetrievalQuery:
    query: str
    question_binding: dict | None = None


async def resolve_retrieval_query(message: str, history, *, domain_id: str | None = None) -> RetrievalQuery | None:
    budget = get_provider_context_budget()
    plan = await plan_retrieval(message, history=history, context_budget=budget.review)
    if not plan.needs_public_retrieval:
        return None
    if not dict(plan.dependencies.groups)["public_knowledge"]:
        return RetrievalQuery(message)
    binding = await resolve_question_binding(
        plan.dependencies, message, window_tokens=budget.window_tokens,
        public_obligations=plan.public_obligations, history=history, with_retrieval_scope=True,
        context_budget=budget.review,
    )
    return RetrievalQuery(bound_retrieval_query(binding, message), binding)
