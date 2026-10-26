"""Compile a grounded chat answer independently from HTTP and platform delivery."""

import asyncio
import os
from dataclasses import dataclass
from typing import Any

from db.schemas import MessageRequest
from inference.generation_request import GenerationRequest as CharacterGenerationRequest
from inference.generation_request import RetrievalResult, generate_character_response
from inference.generation_options import read_generation_options
from inference.history import HistoryFormatError
from inference.lora_registry import get_lora_system_prompt
from inference.openai_protocol import with_provider_error
from inference.provider_context import get_provider_context_budget
from infra.environment import read_bool, read_positive_float
from infra.observability import increment, log_event
from knowledge.chat_retrieval import retrieve_chat_knowledge
from knowledge.public_question_binding import QuestionBindingCapacityError
from knowledge.retrieval_core.errors import RetrievalIndexError

_RAG_TIMEOUT = read_positive_float(os.environ, "RAG_TIMEOUT", 8.0)
_RAG_COLD_START_TIMEOUT = read_positive_float(os.environ, "RAG_COLD_START_TIMEOUT", 60.0)


@dataclass(frozen=True)
class ChatAnswer:
    reply: str
    used_rag: bool
    model_invoked: bool
    metadata: dict[str, Any]


class ChatRetrievalError(RuntimeError):
    http_status = 503

    def __init__(self, detail):
        self.detail = detail
        super().__init__("Chat retrieval or evidence review failed")

    def public_detail(self):
        return self.detail


def _rag_retrieval_timeout() -> float:
    """Allow character-index cold start without weakening steady-state limits."""

    from knowledge.multiscale_rag.runtime import get_multiscale_rag_service

    if not get_multiscale_rag_service().is_warm():
        return max(_RAG_TIMEOUT, _RAG_COLD_START_TIMEOUT)
    return _RAG_TIMEOUT


async def generate_chat_answer(
    request: MessageRequest,
    lora_name: str | None,
    prompt_lora_name: str | None = None,
    *,
    runtime_config: dict[str, Any],
    model_generate,
    enable_rag: bool = True,
    prepared_character_turn=None,
    message_db=None,
    memory_write_enabled: bool | None = None,
) -> ChatAnswer:
    """Shared retrieval, abstention and character policy for either transport."""
    # 使用请求开始时读取的配置快照，避免同一次生成多次同步访问数据库。
    _cfg = runtime_config
    options = read_generation_options(_cfg)
    _temperature, _max_tokens, _top_p = options["temperature"], options["maxTokens"], options["topP"]
    _use_kb = read_bool(_cfg, "useKnowledgeBase", True) and enable_rag

    # RAG 检索（受设置页 useKnowledgeBase 开关控制）
    rag_context = ""
    rag_meta: dict[str, Any] = {}
    citations_enabled = read_bool(os.environ, "RAG_CITATIONS_ENABLED", True)
    independent_tasks_enabled = read_bool(os.environ, "CHARACTER_INDEPENDENT_TASKS_ENABLED", False)
    retrieval = RetrievalResult()
    filters = None
    # Nonempty caller history takes precedence over the prepared server history.
    effective_history = tuple(request.history or []) or (
        prepared_character_turn.history if prepared_character_turn else ()
    )
    if _use_kb:
        try:
            from knowledge.retrieval_query_plan import plan_retrieval

            rag_message = request.message
            budget = get_provider_context_budget()
            query_plan = await plan_retrieval(
                rag_message,
                history=effective_history,
                context_budget=budget.review,
            )
            dependency_groups = dict(query_plan.dependencies.groups)
            if not query_plan.needs_public_retrieval:
                retrieval = RetrievalResult(reason="local_context_dependencies")
                rag_meta = {
                    "citations": [],
                    "confidence": None,
                    "abstained": False,
                    "answerMode": "personal_context" if query_plan.private_context_only else "direct_answer",
                }
            else:
                retrieval_query = rag_message
                from knowledge.public_question_binding import bound_retrieval_query, resolve_question_binding

                question_binding = None
                if dependency_groups["public_knowledge"]:
                    question_binding = await resolve_question_binding(
                        query_plan.dependencies,
                        rag_message,
                        window_tokens=budget.window_tokens,
                        public_obligations=query_plan.public_obligations,
                        with_retrieval_scope=True,
                        **(
                            {"history": effective_history, "context_budget": budget.review} if effective_history else {}
                        ),
                    )
                retrieval_query = bound_retrieval_query(question_binding, rag_message)
                binding_options = {"question_binding": question_binding} if question_binding is not None else {}
                retrieval_timeout = _rag_retrieval_timeout()
                bundle = await asyncio.wait_for(
                    retrieve_chat_knowledge(retrieval_query, 3, filters, database=message_db, **binding_options),
                    timeout=retrieval_timeout,
                )
                from knowledge.public_task_evidence import review_public_candidates

                bundle = await review_public_candidates(
                    bundle,
                    query_plan.dependencies,
                    rag_message,
                    window_tokens=budget.window_tokens,
                    public_obligations=query_plan.public_obligations,
                    question_binding=question_binding,
                )
                bundle["query_plan_status"] = query_plan.status
                rag_meta = {
                    "citations": bundle.get("citations", []) if citations_enabled else [],
                    "confidence": bundle.get("confidence"),
                    "abstained": bundle.get("abstained", False),
                    "answerMode": ("abstention" if bundle.get("abstained", False) else "grounded_answer"),
                    "domainId": next(iter(bundle.get("domains") or []), None),
                    "warnings": bundle.get("warnings") or None,
                }
                from inference.evidence_coverage import requested_source_lookups
                from knowledge.query_tasks import requests_only_source_excerpt

                source_requests = requested_source_lookups(bundle, rag_message)

                source_lookup = bundle.get(
                    "retrieval_strategy"
                ) == "multi_scale_character" and requests_only_source_excerpt(rag_message)
                retrieval_fields = dict(
                    confidence=bundle.get("confidence"),
                    requested_sources=source_requests,
                    source_references=tuple(bundle.get("source_references") or ()),
                    source_lookup=source_lookup,
                    task_coverage=tuple(bundle.get("task_candidate_coverage") or ()),
                    task_query=retrieval_query,
                    public_curated_review=bundle.get("public_curated_review") or {},
                    public_task_review=bundle.get("public_task_review") or {},
                    public_task_query=rag_message
                    if bundle.get("public_task_review")
                    or bundle.get("public_curated_review")
                    or bundle.get("public_domain_branches")
                    else "",
                    public_dependency_indices=tuple(dependency_groups["public_knowledge"]),
                )
                if bundle.get("abstained", False):
                    # Generate only the character's expression of uncertainty.
                    # Unreliable candidates and their citations never reach it.
                    rag_meta["citations"] = []
                    retrieval = RetrievalResult(
                        **retrieval_fields,
                        status="character_abstention",
                        reason="insufficient_retrieval_evidence",
                    )

                # 角色知识检索结果自带按粒度组装的 context_text；
                # 通用知识库结果保留整份检索片段，交给最终请求预算裁决。
                else:
                    character_knowledge_context = bundle.get("context_text") or ""
                    evidence_packets = tuple(bundle.get("evidence_packets") or ())
                    if character_knowledge_context or bundle.get("retrieval_strategy") == "multi_scale_character":
                        rag_context = character_knowledge_context
                    else:
                        from knowledge.evidence_packets import document_evidence_packets

                        # Citation display cannot decide which source text reaches the model.
                        evidence_packets = document_evidence_packets(bundle.get("results", [])) + tuple(
                            bundle.get("original_source_packets") or ()
                        )
                        rag_context = "\n\n".join(packet["text"] for packet in evidence_packets)
                    retrieval = RetrievalResult(
                        **retrieval_fields,
                        status="ok" if rag_context else "character_abstention",
                        reason="" if rag_context else "evidence_budget_exhausted",
                        evidence=rag_context,
                        evidence_packets=evidence_packets,
                        public_domain_branches=bundle.get("public_domain_branches") or {},
                        source_coverage=tuple(bundle.get("source_coverage") or ()),
                        identity_task=bundle.get("identity_task") or {},
                        identity_subtask=bundle.get("identity_subtask") or {},
                        documents=tuple(bundle.get("results", [])),
                        citations=tuple(rag_meta.get("citations", [])),
                        source_excerpts=(
                            (bundle["raw_excerpt"],) if isinstance(bundle.get("raw_excerpt"), dict) else ()
                        ),
                    )
        except HistoryFormatError:
            raise
        except Exception as e:
            increment("rag_failures")
            log_event(
                "rag_failed",
                level="warning",
                traceId=request.traceId,
                platform=request.platform,
                conversationId=request.conversationId or request.sessionId,
                senderId=request.senderId or request.userId,
                model="rag",
                costTime=0,
                errorType=type(e).__name__,
            )
            raise ChatRetrievalError(
                detail=(
                    e.public_detail()
                    if isinstance(e, (RetrievalIndexError, QuestionBindingCapacityError))
                    else with_provider_error("检索或证据审核失败，请检查检索服务及审核模型状态后重试", e)
                ),
            ) from e
    else:
        budget = get_provider_context_budget()

    from inference.private_context_authority import make_private_context_revalidator
    from inference.public_context_authority import make_public_context_revalidator

    generation = await generate_character_response(
        CharacterGenerationRequest(
            message=request.message,
            history=effective_history,
            persona_prompt=get_lora_system_prompt(prompt_lora_name or lora_name),
            interlocutor=request.senderName or request.userName or "普通用户",
            retrieval=retrieval,
            independent_tasks_enabled=independent_tasks_enabled,
            # 编译后的角色上下文：画像/关系/情景/决策进系统提示词，
            # 长期记忆只进用户消息的不可信参考区。
            character_context=(prepared_character_turn.compiled if prepared_character_turn else None),
            memory_write_enabled=memory_write_enabled,
            reply_guard=getattr(prepared_character_turn, "reply_guard", None),
            private_context_revalidator=make_private_context_revalidator(
                prepared_character_turn,
                message_db,
            ),
            public_context_revalidator=make_public_context_revalidator(
                lambda identity: message_db.get_knowledge_document(identity)
            ),
            lora_name=lora_name if lora_name != "default" else None,
            temperature=_temperature,
            max_tokens=_max_tokens,
            top_p=_top_p,
            context_window_tokens=budget.window_tokens,
            evidence_max_chars=budget.evidence_max_chars,
        ),
        model_generate,
    )

    if retrieval.evidence_packets or generation.plan.retrieval.answer_citations_bound:
        rag_meta["citations"] = list(generation.plan.retrieval.citations) if citations_enabled else []
        rag_meta["abstained"] = generation.plan.retrieval.status == "character_abstention"
        rag_meta["answerMode"] = "abstention" if rag_meta["abstained"] else "grounded_answer"
    from inference.evidence_coverage import is_partial_coverage

    if is_partial_coverage(generation.plan.retrieval):
        rag_meta["warnings"] = list(dict.fromkeys([*(rag_meta.get("warnings") or []), "partial_source_context"]))
    if generation.plan.retrieval.answer_citations_bound:
        rag_meta["citations"] = list(generation.response_citations) if citations_enabled else []
    if generation.authority_refreshed:
        rag_meta["warnings"] = list(dict.fromkeys([*(rag_meta.get("warnings") or []), "context_authority_refreshed"]))
    if not generation.model_invoked:
        rag_meta["answerMode"] = generation.response_mode
        rag_meta["abstained"] = generation.response_mode == "source_unavailable"
        rag_meta["citations"] = list(generation.response_citations) if citations_enabled else []
    elif generation.response_mode == "task_composite":
        rag_meta["answerMode"] = "task_composite"
    from knowledge.public_task_evidence import render_public_tasks

    public_tasks = render_public_tasks(generation.plan.retrieval)
    if public_tasks and any(task["status"] != "related_candidate_admitted" for task in public_tasks):
        rag_meta["warnings"] = list(dict.fromkeys([*(rag_meta.get("warnings") or []), "partial_public_evidence"]))
        rag_meta["abstained"] = True
        context = generation.plan.character_context
        from html import unescape

        reference = getattr(context, "reference_context", "")
        visible_message = unescape(generation.plan.messages[-1]["content"])
        speech = getattr(context, "episodic_reference_context", "")
        private_visible = bool(reference and getattr(context, "used_memory_ids", ()) and reference in visible_message)
        private_visible = private_visible or bool(
            speech
            and getattr(context, "memory_source_status", "not_checked") == "available"
            and speech in visible_message
        )
        public_visible = any(
            task["status"] == "related_candidate_admitted"
            or any(
                fact["status"] == "fact_evidence_admitted" and fact["admitted_evidence"]
                for fact in task.get("fact_coverage", ())
            )
            for task in public_tasks
        )
        private_visible = bool(dependency_groups["private_memory"] and private_visible)
        if generation.model_invoked and (public_visible or private_visible):
            rag_meta["answerMode"] = "partial_answer"
        elif rag_meta.get("answerMode") == "grounded_answer":
            rag_meta["answerMode"] = "abstention"
    return ChatAnswer(
        reply=generation.reply,
        used_rag=generation.plan.retrieval.has_evidence or bool(rag_meta.get("abstained")),
        model_invoked=generation.model_invoked,
        metadata=rag_meta,
    )
