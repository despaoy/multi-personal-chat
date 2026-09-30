"""Isolated ASGI HTTP -> actual chat/prepare/model/save/complete/scheduler replay.

Authentication is a fixed test identity and admission runs inline. No login,
network listener, production database, or production service lifecycle is
tested. Actual serving-model inference and backend write gates are exercised.
RAG is optional and uses the actual configured index, never fixture answers.
Placement experiments log actual model messages separately from runtime plans.
"""
import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch


async def run(args):
    import httpx
    from fastapi import FastAPI

    import api.generate as gen
    from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig
    from character.profile_registry import CharacterProfileRegistry
    from db.database import SQLiteDB
    from evaluation.deepseek_live_adapter import replay_answer_budget
    from evaluation.location_memory_replay import RecordedCompletion
    from evaluation.temporal_memory_live_replay import load_cases, trace_json_default
    from repositories.character_memory import DatabaseCharacterMemoryRepository
    from repositories.messages import DatabaseMessageRepository
    from services.character_context import CharacterContextService

    answer_budget = replay_answer_budget(cloud=bool(getattr(args, 'deepseek', False)),
                                         requested=getattr(args, 'answer_max_tokens', None))
    database = SQLiteDB(args.output / "dialogue.sqlite")
    database.update_config(dict(useKnowledgeBase=args.rag, temperature=.2, maxTokens=answer_budget, topP=.9))
    repo = DatabaseCharacterMemoryRepository(database)
    registry = CharacterProfileRegistry()
    registry.load_profiles()
    current = {}

    class ObservedContext(CharacterContextService):
        async def prepare_interactive_turn(self, *args, **kwargs):
            result = await super().prepare_interactive_turn(*args, **kwargs)
            current['prepared'] = asdict(result)
            current['scope'] = result.user_scope
            return result

        async def prepare_turn(self, *args, **kwargs):
            result = await super().prepare_turn(*args, **kwargs)
            current["prepared"] = asdict(result)
            current["scope"] = result.user_scope
            return result

        async def complete_turn(self, *args, **kwargs):
            current["completion_source_id"] = kwargs.get("source_message_id")
            result = await super().complete_turn(*args, **kwargs)
            current["completion"] = asdict(result)
            return result

    class ObservedHistory(DatabaseMessageRepository):
        async def list_recent_conversation_history(self, *history_args, **kwargs):
            if args.cold_questions and current.get("phase") == "question":
                return []
            return await super().list_recent_conversation_history(*history_args, **kwargs)

    class InlineAdmission:
        def priority_for(self, *args):
            return 0

        async def check_rate_limits(self, *args):
            pass

        async def submit(self, job, **kwargs):
            return await job()

    cloud = None
    if getattr(args, 'deepseek', False):
        from evaluation.deepseek_live_adapter import DeepSeekEvaluationClient
        cloud = DeepSeekEvaluationClient(args.api_key, args.cloud_model)
    components = {}
    if getattr(args, 'cloud_reviewers', False):
        if cloud is None:
            raise ValueError('Cloud reviewers require --deepseek')
        from evaluation.deepseek_live_adapter import cloud_context_components
        components = cloud_context_components(cloud,
            context_window_tokens=getattr(args, 'cloud_context_tokens', 65536))
    cloud_window = getattr(args, 'cloud_context_tokens', 65536)
    context_budgets = {}
    if cloud is not None:
        from evaluation.deepseek_live_adapter import cloud_context_budgets
        context_budgets = cloud_context_budgets(cloud_window)
    source_recall = not getattr(args, "no_source_recall", False)
    context = ObservedContext(registry, repo, ObservedHistory(database), source_recall_enabled=source_recall,
                              **context_budgets, **components)
    config = replace(MemoryLlmConfig.from_env(), enabled=True,
        base_url="http://127.0.0.1:8001", model="qwen3-8b-instruct-awq")
    if cloud is not None:
        config = replace(config, context_window_tokens=cloud_window, model=args.cloud_model)
    if args.recorded_writer:
        from evaluation.frozen_memory_writer import FrozenMemoryWriter
        writer = FrozenMemoryWriter([json.loads(line) for line in
            args.recorded_writer.read_text(encoding='utf-8').splitlines()], current)
    elif cloud is not None:
        from evaluation.deepseek_live_adapter import DeepSeekRecordedWriter
        writer = DeepSeekRecordedWriter(cloud)
    else:
        writer = RecordedCompletion(config)
    scheduler = MemoryEnrichmentScheduler(config=config, completion=writer)
    service = gen._build_chat_generation_service(InlineAdmission(), context, database)
    app = FastAPI()
    app.include_router(gen.router)
    login = {"id": "evaluation"}
    app.dependency_overrides[gen.get_current_user] = lambda: login
    app.dependency_overrides[gen.get_request_chat_generation_service] = lambda: service
    original = gen.generate_character_response
    all_calls = []

    async def traced_generation(request, generate):
        if cloud is not None:
            request = replace(request, context_window_tokens=cloud_window,
                              evidence_max_chars=cloud_window // 4)
        async def traced(**kwargs):
            if kwargs.get("lora_name"):
                raise RuntimeError("No-LoRA evaluation requested an adapter")
            if args.separate_source_packet:
                from evaluation.source_packet_placement import separate_source_packet
                from inference.generation_request import CONTEXT_SAFETY_MARGIN_TOKENS, _estimated_tokens

                source = getattr(request.character_context, 'episodic_reference_context', '')
                messages = separate_source_packet(kwargs['messages'], source)
                cost = sum(_estimated_tokens(m['content']) + 4 for m in messages)
                if cost + kwargs['max_tokens'] + CONTEXT_SAFETY_MARGIN_TOKENS > request.context_window_tokens:
                    raise ValueError('Experimental packet placement exceeds the actual request budget')
                current['source_placement_applied'] = bool(source)
                kwargs = {**kwargs, 'messages': messages}
            at = time.monotonic()
            reply = await (cloud.generate(**kwargs) if cloud is not None else generate(**kwargs))
            call = dict(request=kwargs, reply=reply, seconds=time.monotonic() - at)
            current.setdefault("model_calls", []).append(call)
            all_calls.append(call)
            return reply
        result = await original(request, traced)
        current["generation"] = asdict(result)
        return result

    manifest = dict(asgi_http=True, authentication="fixed_test_identity", admission="inline_test_adapter",
                    actual_chat_service=True, application_completion_tested=True, source_ids="omitted_by_client",
                    real_model=True, lora=False, rag=args.rag, database="isolated_sqlite", source_recall=source_recall,
                    source_window_radius=0, optional_model_reviewers=bool(components), production_modified=False)
    manifest["cold_questions"] = args.cold_questions
    manifest['echo_client_history'] = bool(getattr(args, 'echo_client_history', False))
    manifest['answer_provider'] = 'deepseek' if cloud is not None else 'local_vllm'
    manifest['answer_model'] = args.cloud_model if cloud is not None else 'qwen3-8b-instruct-awq'
    manifest['writer_provider'] = 'recorded' if args.recorded_writer else manifest['answer_provider']
    manifest['writer_context_window_tokens'] = config.context_window_tokens
    manifest['answer_context_window_tokens'] = cloud_window if cloud is not None else None
    manifest['answer_max_tokens'] = answer_budget
    manifest['context_loading_budgets'] = context_budgets
    manifest['writer_legacy_max_input_chars'] = config.max_input_chars
    manifest['response_model_label_is_legacy'] = cloud is not None
    manifest['reviewer_provider'] = 'deepseek' if components else 'disabled'
    manifest['reviewer_model'] = args.cloud_model if components else None
    manifest['reviewer_triggers_and_timeouts'] = 'native_defaults'
    manifest["source_placement"] = "model_adapter_experiment" if args.separate_source_packet else "runtime_default"
    manifest["canonical_plan_remains_inline"] = args.separate_source_packet
    unexpected_local_calls = []
    from evaluation.retrieval_trace import RecordedRetrieval
    retrieval_trace = RecordedRetrieval(gen._retrieve_rag_bundle, current,
                                        top_k=getattr(args, 'rag_top_k', None))
    manifest['experimental_rag_top_k'] = retrieval_trace.top_k

    async def reject_local_generation(*unused_args, **unused_kwargs):
        unexpected_local_calls.append(current.get('case', 'startup'))
        raise RuntimeError('Unexpected local model call in cloud-only evaluation')

    try:
        with ExitStack() as routes, \
             patch("character.memory_llm.get_memory_enrichment_scheduler", return_value=scheduler), \
             patch.object(gen, "generate_character_response", traced_generation), patch.object(gen, "response_cache", None):
            routes.enter_context(patch.object(gen, '_retrieve_rag_bundle', retrieval_trace))
            if cloud is not None:
                routes.enter_context(patch('inference.vllm_client.VLLMClient.generate', reject_local_generation))
                if args.rag:
                    from knowledge.multiscale_rag.runtime import MultiScaleRagRuntime
                    cloud_rag = MultiScaleRagRuntime(context_max_chars=cloud_window // 4)
                    routes.enter_context(patch('knowledge.multiscale_rag.runtime.get_multiscale_rag_service',
                                               return_value=cloud_rag))
                    manifest['rag_context_max_chars'] = cloud_rag.context_max_chars
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://evaluation") as client:
                cases = load_cases(args.fixtures)
                if args.case and set(args.case) - {case["id"] for case in cases}:
                    raise ValueError("Unknown evaluation case selection")
                for case in cases:
                    if args.case and case["id"] not in args.case:
                        continue
                    login["id"] = "r111-" + case["id"]
                    client_history = []
                    for index, message in enumerate([*case["turns"], *case["questions"]]):
                        current.clear()
                        current.update(case=case["id"], index=index, message=message,
                                       phase="source" if index < len(case["turns"]) else "question")
                        count = len(writer.calls)
                        cloud_count = len(cloud.calls) if cloud is not None else 0
                        response = await client.post("/api/generate", json=dict(message=message,
                            characterId="tsukiyashiro_kisaki", loraName="default", sessionId=case["id"],
                            history=client_history if getattr(args, 'echo_client_history', False) else [],
                            senderId="not-authoritative", userId="not-authoritative"))
                        current['client_history_count'] = len(client_history) if getattr(args, 'echo_client_history', False) else 0
                        if response.status_code == 200:
                            client_history.extend([dict(role='user', content=message),
                                                   dict(role='assistant', content=response.json().get('reply', ''))])
                        current.update(http_status=response.status_code, response=response.json())
                        if not await scheduler.flush_memory(timeout=120):
                            raise RuntimeError("Scheduled memory writer did not drain")
                        current["writer_calls"] = writer.calls[count:]
                        if cloud is not None:
                            current['provider_calls'] = cloud.calls[cloud_count:]
                        if scope := current.pop("scope", None):
                            current["sources"] = await repo.list_sources("tsukiyashiro_kisaki", scope)
                            current["claims"] = await repo.list_memory_records("tsukiyashiro_kisaki", scope,
                                limit=None, include_inactive=True)
                        current["scheduler"] = asdict(scheduler.status)
                        current["write_results"] = [item for item in scheduler.status.recent_results
                            if item.get("source_message_id") == current.get("completion_source_id")]
                        with (args.output / "traces.jsonl").open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps(current, ensure_ascii=False, default=trace_json_default) + "\n")
                        print(json.dumps(dict(case=case["id"], index=index, status=response.status_code,
                            reply=current["response"].get("reply"), source_id=current.get("completion_source_id"),
                            outcome=current.get("completion")), ensure_ascii=False), flush=True)
                        response.raise_for_status()
    finally:
        await scheduler.shutdown(timeout=3)
        await gen.close_vllm_client()
        if cloud is not None:
            await cloud.close()
        manifest.update(generation_calls=len(all_calls),
                        writer_calls=0 if args.recorded_writer else len(writer.calls),
                        recorded_writer_calls=len(writer.calls) if args.recorded_writer else 0)
        manifest['provider_calls_by_purpose'] = dict(Counter(call['purpose'] for call in cloud.calls)) if cloud else {}
        manifest['unexpected_local_generation_calls'] = unexpected_local_calls
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if unexpected_local_calls:
        raise RuntimeError('Cloud-only evaluation attempted local generation; inspect manifest')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--case", action="append")
    parser.add_argument("--recorded-writer", type=Path, help="Replay independent ADD/NOOP proposals; no fresh writer")
    parser.add_argument("--cold-questions", action="store_true", help="Suppress recent history only during question phase")
    parser.add_argument("--no-source-recall", action="store_true",
                        help="Disable independent utterance recall to audit the stored claim/observation lane")
    parser.add_argument('--echo-client-history', action='store_true',
                        help='Resend browser transcript including deleted turns to audit server history authority')
    parser.add_argument("--rag", action="store_true", help="Use the actual configured knowledge retrieval service")
    parser.add_argument('--rag-top-k', type=int, choices=range(1, 33), default=None,
                        help='Isolated retrieval coverage experiment; omitted preserves runtime count')
    parser.add_argument('--answer-max-tokens', type=int, default=None,
                        help='Output budget (default: cloud 1024, local 256); not an input window')
    parser.add_argument("--separate-source-packet", action="store_true", help="Experiment at model adapter, preserve runtime plan")
    parser.add_argument('--deepseek', action='store_true', help='Cloud answer AND writer via evaluation adapter')
    parser.add_argument('--cloud-model', default='deepseek-v4-pro')
    parser.add_argument('--cloud-context-tokens', type=int, default=65536,
                        help='API working window; independent of local vLLM (default: 64K)')
    parser.add_argument('--cloud-reviewers', action='store_true',
                        help='Explicit cloud semantic review, memory selection and decision policy')
    args = parser.parse_args()
    if args.cloud_reviewers and not args.deepseek:
        parser.error('--cloud-reviewers requires --deepseek')
    args.output = args.output.resolve()
    if (not args.output.is_relative_to(Path("/home/boot/lhm/multipersonal-runtime/evaluations"))
            or not args.output.name.startswith(("r111-", "r112-", "r113-", "r114-", "r115-", "r117-", "r118-", "r121-", "r122-", "r123-", "r124-", "r125-", "r128-", "r129-", "r130-", "r131-", "r132-", "r136-", "r137-", "r138-", "r139-", "r140-", "r141-", "r142-", "r144-", "r145-", "r146-", "r147-"))):
        raise ValueError("Only an allowlisted isolated evaluation directory is accepted")
    args.output.mkdir(exist_ok=False)
    if args.deepseek:
        args.api_key = sys.stdin.readline().strip()
        if not args.api_key:
            raise ValueError('Provide cloud credential through stdin, not command-line arguments')
    os.environ.update(USE_POSTGRESQL="false", DATABASE_PATH=str(args.output / "bootstrap.sqlite"),
        JWT_SECRET="isolated-r111-web-http-source-receipt", MODEL_PROVIDER="vllm",
        VLLM_BASE_URL="http://127.0.0.1:8001", VLLM_BASE_URLS="http://127.0.0.1:8001",
        VLLM_SERVED_MODEL_NAME="qwen3-8b-instruct-awq", VLLM_MODEL="qwen3-8b-instruct-awq",
        MEMORY_LLM_ENABLED="true", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="",
        REDIS_URL="redis://127.0.0.1:1/0", CONTEXTUAL_MEMORY_SELECTION_ENABLED="false",
        CONTEXTUAL_DECISION_POLICY_ENABLED="false", CHARACTER_INDEPENDENT_TASKS_ENABLED="false")
    from dotenv import load_dotenv
    load_dotenv("/home/boot/lhm/multi-personal-chat/backend/.env", override=False)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
