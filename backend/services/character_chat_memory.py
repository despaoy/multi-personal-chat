"""Session-local adapter for the existing semantic memory writer and validators."""

import asyncio
import copy
import logging
import re
import uuid

from character.memory_extractor import extract_memories, memory_write_allowed
from character.memory_llm import (
    MemoryLlmConfig,
    MemoryScopeAuthorization,
    OpenAICompatibleMemoryCompletion,
    build_memory_llm_messages,
    parse_llm_proposals,
)
from repositories.character_chat import timestamp
from services.character_chat_policy import is_local_only, local_endpoint_allowed

logger = logging.getLogger(__name__)

_EXPLICIT_HABIT = re.compile(
    r"(?:(?:请)?记住[，,：:\s]*)?(?P<claim>我(?P<frequency>平时|通常|一向)(?:不)?"
    r"(?P<action>喝|吃|用|买|选)(?P<subject>[\w]{1,20}))[。.!！]?"
)


async def enrich_state(message, session, request_id, *, database=None, config=None, completion=None):
    state = copy.deepcopy(session["state"])
    try:
        config = config or MemoryLlmConfig.from_env()
    except ValueError:
        state["memory_processing"] = "failed"
        logger.warning("Character chat memory configuration is invalid")
        return state
    if not config.enabled or not memory_write_allowed(message):
        state["memory_processing"] = "disabled" if not config.enabled else "skipped"
        return state
    if is_local_only(session) and not local_endpoint_allowed(config.base_url):
        state["memory_processing"] = "skipped"
        state["memory_policy_reason"] = "local_only"
        return state
    writer = completion or OpenAICompatibleMemoryCompletion(config, local_only=is_local_only(session))
    try:
        existing = tuple(
            {
                **item,
                "memory_id": item["id"],
                "status": "active" if item["status"] == "confirmed" else "pending",
                "evidence": [item["evidence"]] if item.get("evidence") else [],
                "source_message_ids": [item["source_turn_id"]] if item.get("source_turn_id") else [],
                "metadata": {"qualifiers": item.get("qualifiers", {}), "fact_kind": item.get("fact_kind")},
            }
            for item in state["memories"]
        )
        authorization = MemoryScopeAuthorization(message)
        messages = build_memory_llm_messages(
            message,
            (),
            (),
            existing,
            config.max_input_chars,
            config.confidence_threshold,
            context_window_tokens=config.context_window_tokens,
            scope_authorization=authorization,
            unkeyed_as_read_only=True,
        )
        async with asyncio.timeout(config.timeout_seconds):
            if database is None:
                output = await writer.complete(messages)
            else:
                from services.model_usage import account_usage

                async with account_usage(
                    database,
                    {
                        "traceId": request_id + ":memory",
                        "purpose": "memory",
                        "platform": "web",
                        "sessionId": session["id"],
                        "conversationId": session["id"],
                        "modelName": config.model,
                        "loraName": "default",
                        "usedRag": False,
                        "usedLora": False,
                    },
                ):
                    output = await writer.complete(messages)
        proposals = parse_llm_proposals(
            output,
            source_message=message,
            existing_memories=existing,
            confidence_threshold=config.confidence_threshold,
            scope_authorization=authorization,
        )
        assertions = extract_memories(message)
        habit = _EXPLICIT_HABIT.fullmatch(message.strip())
        for proposal in proposals:
            memory = proposal.memory
            # Erasure is handled by the explicit user-facing memory controls.
            if memory is None or proposal.operation in {"NOOP", "ERASE", "RETRACT"}:
                continue
            key, content = memory.memory_key, memory.content
            unqualified = not proposal.qualifiers and not proposal.valid_from and not proposal.valid_to
            preference = next(
                (assertion for assertion in assertions
                 if assertion.memory_key == key and assertion.polarity == proposal.fact_kind
                 and not assertion.qualifiers and assertion.evidence in proposal.evidence),
                None,
            )
            verified_habit = (
                habit is not None and proposal.fact_kind == "other_user_fact"
                and not proposal.valid_from and not proposal.valid_to
                # A source-unit reference resolves to the whole clause containing the frequency.
                and proposal.qualifiers in (
                    (), (("frequency", habit["frequency"]),), (("frequency", habit["claim"]),),
                )
                and key in {"fact_" + habit["subject"], f"fact_habit_{habit['action']}_{habit['subject']}"}
                and habit["claim"] in proposal.evidence
            )
            if verified_habit:
                key = f"fact_habit_{habit['action']}_{habit['subject']}"
                content = "用户" + habit["claim"].removeprefix("我")
            elif preference and unqualified and proposal.operation == "ADD":
                # Store the corroborated fact, without model-added wording.
                content = preference.content
            same_key = [item for item in state["memories"] if item.get("memory_key") == key]
            if any(item["content"] == content for item in same_key):
                continue
            confirmed = (
                proposal.operation == "ADD" and ((preference is not None and unqualified) or verified_habit)
                and not same_key
                and not any(not item.get("memory_key") for item in state["memories"])
            )
            now = timestamp()
            state["memories"].append(
                {
                    "id": uuid.uuid4().hex,
                    "content": content,
                    "status": "confirmed" if confirmed else "pending",
                    "origin": "conversation",
                    "memory_key": key,
                    "memory_type": memory.memory_type,
                    "evidence": proposal.evidence,
                    "confidence": proposal.confidence,
                    "qualifiers": dict(proposal.qualifiers),
                    "fact_kind": proposal.fact_kind,
                    "source_session_id": session["id"],
                    "source_request_id": request_id,
                    "source_turn_id": None,
                    "created_at": now,
                    "updated_at": now,
                }
            )
        state["memory_processing"] = "completed"
    except Exception as exc:
        # A failed extraction must not lose a successful chat reply or fabricate facts.
        state = copy.deepcopy(session["state"])
        state["memory_processing"] = "failed"
        logger.warning("Character chat memory extraction failed type=%s", type(exc).__name__)
    finally:
        if completion is None:
            try:
                await writer.close()
            except Exception as exc:
                logger.warning("Character chat memory client cleanup failed type=%s", type(exc).__name__)
    return state
