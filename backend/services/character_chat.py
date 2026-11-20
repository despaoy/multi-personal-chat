"""Character-library chat using the shared prompt and provider contracts."""

import asyncio
import copy
import json
import os
import time
from contextlib import nullcontext
from dataclasses import replace

from character.chat_relationship import relationship_dimension_guidance
from character.decision_policy import relationship_stage_guidance
from character.models import CompiledCharacterContext
from db.errors import RevisionConflictError
from inference.generation_options import read_generation_options
from inference.generation_request import GenerationRequest, generate_character_response
from infra.environment import read_positive_float
from repositories.character_chat import CharacterChatRepository
from repositories.character_chat_generation import CharacterChatGenerationRepository
from services.character_chat_evidence import CharacterChatCompletion, evidence_snapshot
from services.character_chat_policy import is_local_only, require_local_endpoint, require_local_provider
from services.character_chat_summary import RECENT_TURNS, summaries_for_prompt, valid_summaries


def prepare_request(session, turns, message, retrieved_memories=None, *, proactive_messages=None):
    retained = [*session["baseline"]["history"], *turns]
    summaries = valid_summaries(session["state"], retained)
    covered = {source["turn_id"] for segment in summaries for source in segment["sources"]}
    recent = {turn["id"] for turn in retained[-RECENT_TURNS:]}
    settings = session["character"]["settings"]
    persona = f"扮演以下角色，以人物的立场自然回应。角色名：{settings['name']}\n{settings['persona']}"
    if settings["scenario"]:
        persona += "\n初始场景：" + settings["scenario"]
    if settings["relationships"]:
        persona += "\n人物关系设定：" + settings["relationships"]
    if settings["examples"]:
        persona += "\n以下是表达风格示例，不是与当前用户发生过的经历：\n" + json.dumps(
            settings["examples"], ensure_ascii=False
        )
    relationship = session["state"]["relationship"]
    persona += "\n按当前已生效的关系调整表达：" + relationship_stage_guidance(relationship.get("stage", "stranger"))
    persona += "。尊重用户此刻明确的边界，并保留人物的独立立场。"
    persona += relationship_dimension_guidance(relationship)
    persona += (
        "\n用户明确说先放下某个话题时，只回应当前话题，连举例也不要转回已暂停的话题。"
        "用户说只想听你的看法时，给出自己的判断，不追加对方没要的建议、安排或追问。"
    )
    if session["mode"] == "branch":
        persona += "\n本轮处于假想分支，按用户明确给出的剧情假设推演，并保持未被修改的人物立场。"
    else:
        persona += (
            "\n用户要求回忆前文或确认已经发生的事情时，只复述已有对话、设定和核实资料明确支持的细节。"
            "如果只知道地点的大致范围，就只说到这个范围。用角色自己的语气说明不知道的部分，"
            "不要把新想象的地点细节、物件特征或过去行动说成已经记得的事实。"
        )
    context = CompiledCharacterContext(
        profile_context=persona,
        dynamic_context="",
        reference_context="",
        conversation_reference_context=json.dumps(
            {
                "relationship": relationship,
                "memories": [
                    {"content": item["content"], "origin": "user"} if item.get("origin") == "user" else item
                    for item in (session["state"]["memories"] if retrieved_memories is None else retrieved_memories)
                    if item.get("status") == "confirmed"
                ],
                "creator_reference": settings["reference_text"],
                "branch_hypothesis": session["hypothesis"],
                "conversation_summaries": summaries_for_prompt(summaries, message),
                **({"previous_assistant_outreach": proactive_messages} if proactive_messages else {}),
            },
            ensure_ascii=False,
        ),
    )
    history = [{"role": "assistant", "content": settings["greeting"]}]
    for turn in retained:
        if turn["id"] in covered and turn["id"] not in recent:
            continue
        history.extend([{"role": "user", "content": turn["message"]}, {"role": "assistant", "content": turn["reply"]}])
    return GenerationRequest(
        message=message, persona_prompt=persona, history=history, character_context=context, memory_write_enabled=False
    )


class CharacterChatService:
    def __init__(self, database, complete, enrich=None, relationship=None, *, stream_events=False, background=False,
                 retrieve=None, sources=None, proactive_history=None):
        self.repository = CharacterChatRepository(database)
        self.generations = CharacterChatGenerationRepository(database)
        self.complete = complete
        self.enrich = enrich
        self.relationship = relationship
        self.stream_events = stream_events
        self.background = background
        self.retrieve = retrieve
        self.sources = sources
        self.proactive_history = proactive_history

    async def generate(self, owner, session_id, *, revision, request_id, message, replace_turn_id=None):
        repo = self.repository
        prior = await asyncio.to_thread(repo.prior, owner, session_id, request_id)
        if prior:
            if (
                not prior["active"]
                or prior["message"] != message
                or prior["request_revision"] != revision
                or prior["replaces_turn_id"] != replace_turn_id
            ):
                raise RevisionConflictError()
            return prior
        if replace_turn_id:
            session, turns, state, _ = await asyncio.to_thread(repo.before_turn, owner, session_id, replace_turn_id)
            session = {**session, "state": state}
        else:
            session, turns = await asyncio.to_thread(repo.snapshot, owner, session_id)
        if session["revision"] != revision:
            raise RevisionConflictError()
        token = await asyncio.to_thread(self.generations.claim, session, request_id, message, replace_turn_id)
        try:
            async with asyncio.timeout(600):
                return await self.generate_claimed(session, turns, request_id, message, replace_turn_id, token)
        except BaseException as exc:
            code = "interrupted" if isinstance(exc, asyncio.CancelledError) else "generation_failed"
            await asyncio.shield(asyncio.to_thread(self.generations.fail, session_id, request_id, token, code))
            raise

    async def generate_claimed(self, session, turns, request_id, message, replace_turn_id, token):
        # Edits use historical state; current-session vector indexes cannot represent that state.
        memories = await asyncio.to_thread(self.retrieve, session, message) if self.retrieve and not replace_turn_id else None
        outreach = None
        if self.proactive_history and not replace_turn_id and session["mode"] != "branch":
            outreach = await asyncio.to_thread(self.proactive_history, session["owner_user_id"], session["id"])
        request = prepare_request(session, turns, message, memories, proactive_messages=outreach)
        if self.sources:
            request = await self.sources(request, session)
        async def event_sink(event):
            await asyncio.to_thread(self.generations.publish, session, request_id, token, event)

        completion = await self.complete(
            request, session, request_id, **({"event_sink": event_sink} if self.stream_events else {})
        )
        if request.public_context_revalidator:
            await request.public_context_revalidator(request)
        if isinstance(completion, CharacterChatCompletion):
            reply, model, evidence = completion.reply, completion.model, completion.evidence
        else:
            # Existing completion adapters may return the original two-field contract.
            reply, model = completion
            evidence = {}
        if self.background:
            state = copy.deepcopy(session["state"])
            state.update(memory_processing="pending", relationship_processing="pending", summary_processing="pending")
        else:
            state = await self.enrich(message, session, request_id) if self.enrich else session["state"]
            if self.relationship:
                state = await self.relationship(message, reply, {**session, "state": state}, request_id)
        return await asyncio.to_thread(
            self.repository.commit_turn,
            session,
            request_id,
            message,
            reply,
            model,
            state=state,
            replace_turn_id=replace_turn_id,
            evidence=evidence,
            generation_token=token,
            enqueue_enrichment=self.background,
            source_corpus_id=session.get('source_corpus_id') if self.sources and request.retrieval.status == 'ok' else None,
        )


async def complete_with_provider(request, session, request_id, *, database, http_client_pool, event_sink=None):
    from app.config import get_llm_semaphore
    from inference.local_model_access import get_local_model_lock
    from inference.model_manager import get_model_manager
    from inference.openai_protocol import is_deepseek_api
    from inference.provider_context import get_provider_context_budget
    from inference.request_context import resolve_request_context, use_request_context
    from services.model_usage import account_usage

    manager = get_model_manager(database=database)
    context = await resolve_request_context(manager, http_client_pool=http_client_pool)
    require_local_provider(session, context)
    context = replace(context, local_only=is_local_only(session))
    if context.provider_name == "mock":
        raise RuntimeError("当前模型为 mock，请配置有效的模型提供商")
    deepseek = context.provider_name == "openai_compat" and is_deepseek_api(context.provider.base_url)
    options = read_generation_options(await asyncio.to_thread(lambda: dict(database.config)), default_top_p=1.0 if deepseek else 0.9)
    timeout = read_positive_float(os.environ, "MODEL_INFERENCE_TIMEOUT", 180.0)
    metadata = {
        "traceId": request_id,
        "purpose": "chat",
        "platform": "web",
        "sessionId": session["id"],
        "conversationId": session["id"],
        "modelName": context.provider.get_status()["modelName"],
        "loraName": "default",
        "usedRag": request.retrieval.has_evidence,
        "usedLora": False,
    }
    if not isinstance(metadata["modelName"], str) or not metadata["modelName"].strip():
        raise ValueError("模型提供商未报告有效模型名称")
    with use_request_context(context):
        budget = get_provider_context_budget(database=database)
        request = replace(
            request,
            temperature=options["temperature"],
            max_tokens=options["maxTokens"],
            top_p=options["topP"],
            context_window_tokens=budget.window_tokens,
        )
        async with account_usage(database, metadata), asyncio.timeout(timeout):
            if context.provider_name == "vllm":
                from inference.vllm_client import get_vllm_client

                client = await get_vllm_client()
                for endpoint in client._base_urls:
                    require_local_endpoint(session, endpoint)

                async def generate_local(**kwargs):
                    if event_sink is None:
                        return await client.generate(**kwargs)
                    stream = await client.generate(**kwargs, stream=True)
                    return await collect_stream(stream, event_sink)

                result = await generate_character_response(request, generate_local)
            else:
                semaphore = get_llm_semaphore()
                await asyncio.wait_for(semaphore.acquire(), timeout=30)
                try:
                    async with nullcontext() if context.provider_name == "openai_compat" else get_local_model_lock():
                        if context.provider_name != "openai_compat":
                            context.provider.set_lora_adapter(None)

                        async def generate(**kwargs):
                            if event_sink is not None and context.provider_name == "openai_compat":
                                return await collect_stream(context.provider.async_stream(**kwargs), event_sink)
                            text, _ = await context.provider.async_complete(**kwargs)
                            return text

                        result = await generate_character_response(request, generate)
                finally:
                    semaphore.release()
    return CharacterChatCompletion(result.reply, metadata["modelName"], evidence_snapshot(result))


async def collect_stream(stream, event_sink, *, clock=time.monotonic):
    """Publish provisional deltas; callers must separately commit the validated reply.

    A guard-triggered regeneration invokes this again, replacing the previous draft.
    No completion event is emitted here: normal stream termination is not a DB receipt.
    """
    parts = []
    size = 0
    pending = []
    pending_size = 0
    last_publish = None
    try:
        await event_sink({"type": "draft_reset"})
        async for delta in stream:
            size += len(delta)
            if size > 1_000_000:
                raise ValueError("Completion draft exceeds limit")
            parts.append(delta)
            pending.append(delta)
            pending_size += len(delta)
            now = clock()
            if last_publish is None or now - last_publish >= 0.1 or pending_size >= 1024:
                await event_sink({"type": "draft_delta", "text": "".join(pending)})
                pending, pending_size, last_publish = [], 0, now
        if pending:
            await event_sink({"type": "draft_delta", "text": "".join(pending)})
        return "".join(parts)
    finally:
        await stream.aclose()
