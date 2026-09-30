"""Fresh no-LoRA temporal memory evaluation using the serving model.

Exercises semantic writes, disposable SQLite, embedding recall, the actual
character reference compiler and shared generation; optionally full dynamic
preparation. Writes call the scheduler directly, not complete_turn's upstream
gates. Not HTTP/RAG E2E. Gold review criteria never enter requests.
"""

import argparse
import asyncio
import json
import os
import re
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from character.context_builder import compile_character_context
from character.memory_extractor import extract_memories
from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig
from character.memory_service import CharacterMemoryService
from character.models import CharacterContext, UserScope
from character.profile_registry import CharacterProfileRegistry
from character.source_memory import SourceMemoryService, attach_sources
from db.database import SQLiteDB
from evaluation.episode_subject_audit import complete
from evaluation.location_memory_replay import RecordedCompletion
from inference.generation_request import GenerationRequest, generate_character_response
from inference.memory_response import read_memory_fields
from repositories.character_memory import DatabaseCharacterMemoryRepository
from repositories.messages import DatabaseMessageRepository
from services.character_context import CharacterContextService, TurnInput

CASES = (
    dict(id="stable_control", turns=["我叫林溪，我现在住在衡水。"],
         questions=["我的姓名和现居地是什么？"], criteria="林溪/衡水应能回答，不能因模型日期笼统拒绝。"),
    dict(id="temporary_present", turns=["我现在住在池州。", "我这周在许昌出差，下周回去。"],
         questions=["我现在住哪里？", "我之前说过的出差安排是什么？"],
         criteria="住址池州；原话可恢复许昌和下周返回，不能编精确日期或自动断言已返回。"),
    dict(id="absolute_ended", turns=["我从2026年7月1日到7月31日在漳州实习，现在已经结束了。"],
         questions=["我现在还在漳州实习吗？"], criteria="实习已结束，不能因来源引用而过度拒答。"),
    dict(id="future_move", turns=["我现在住在眉山。", "下个月我会搬到宝鸡，现在还没有搬。"],
         questions=["我现在住哪里？", "我之后打算搬到哪里？"],
         criteria="目前眉山，未来宝鸡；不能提前覆盖当前地址，也不应完全遗忘未来计划。"),
    dict(id="name_timeword", turns=["我叫三月。"], questions=["我叫什么名字？"],
         criteria="三月是姓名，不应被日历词面信号误解为日期或拒答。"),
    dict(id="conditional_residence", turns=["我工作日住在荆门，周末住在宜昌。"],
         questions=["我工作日和周末分别住哪里？"], criteria="两种条件分别保留，不合并成一个无条件地址。"),
    dict(id="historical_only", turns=["去年我在临汾的图书馆工作，后来已经离职了。"],
         questions=["我现在还在图书馆工作吗？", "我以前在哪里工作？"],
         criteria="现在不再任职；旧工作可作为历史回答，不应以当前过期过滤抹去。"),
    dict(id="future_event", turns=["我下周三要去襄阳参加面试。"],
         questions=["我的面试是什么时候，在哪儿？"],
         criteria="襄阳、下周三是已表达的安排，不保证事件必然发生，也不应因计划pending完全丢失。"),
    dict(id="deadline", turns=["我目前在唐山轮岗，到月底结束，之后回苏州。"],
         questions=["你还记得我的轮岗和返程安排吗？"],
         criteria="保留唐山/月底/苏州；不补具体返程时刻，不把短期轮岗当长期住址。"),
)


def load_cases(path=None):
    cases = json.loads(path.read_text(encoding="utf-8")) if path else list(CASES)
    if not isinstance(cases, list) or not 1 <= len(cases) <= 100:
        raise ValueError("Expected 1..100 evaluation cases")
    ids = set()
    for case in cases:
        if (not isinstance(case, dict) or not re.fullmatch(r"[a-z0-9_]{1,60}", case.get("id", ""))
                or case["id"] in ids):
            raise ValueError("Invalid or duplicate case ID")
        ids.add(case["id"])
        for key in ("turns", "questions"):
            if (not isinstance(case.get(key), list) or not 1 <= len(case[key]) <= 32
                    or any(not isinstance(text, str) or not text.strip() for text in case[key])):
                raise ValueError("Invalid evaluation dialogue")
        if not isinstance(case.get("criteria"), str):
            raise ValueError("Review criteria required (never sent to model)")
    return cases


def trace_json_default(value):
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Unsupported evaluation trace type: {type(value).__name__}")


async def run(args):
    cases = load_cases(args.fixtures)
    if args.case and set(args.case) - {case["id"] for case in cases}:
        raise ValueError("Unknown case selection")
    if args.full_prepare and any(os.getenv(key, "false").lower() in {"1", "true", "yes", "on"}
        for key in ("CONTEXTUAL_MEMORY_SELECTION_ENABLED", "CONTEXTUAL_DECISION_POLICY_ENABLED")):
        raise ValueError("Disable optional model reviewers to keep evaluation call accounting complete")
    root = Path("/home/boot/lhm/multipersonal-runtime/evaluations").resolve()
    output = args.output.resolve()
    if not output.is_relative_to(root) or not output.name.startswith("r104-"):
        raise ValueError("A new isolated r104 evaluation directory is required")
    output.mkdir(exist_ok=False)
    registry = CharacterProfileRegistry()
    registry.load_profiles()
    profile = registry.get_profile("tsukiyashiro_kisaki")
    config = replace(MemoryLlmConfig.from_env(), enabled=True, base_url=args.endpoint, model=args.model)
    recorded_cases = ({case["case"]: case for case in (
        json.loads(line) for line in args.recorded_writes.read_text(encoding="utf-8-sig").splitlines() if line.strip())}
        if args.recorded_writes else {})
    manifest = dict(model=args.model, lora=False, production_modified=False, fresh_write_model_calls=0,
                    write_path="scheduler", application_completion_tested=False,
                    fresh_read_model_calls=0, recorded_write_outputs=0, http_app_e2e=False, rag_tested=False,
                    full_dynamic_preparation=args.full_prepare, optional_model_reviewers=False,
                    source_window_radius=args.source_window_radius,
                    read_history=args.read_history, character_profile_version=profile.version,
                    writes_only=args.writes_only, source_capture_enabled=True, source_retrieval_enabled=args.source_recall)
    try:
        for case in cases:
            if args.case and case["id"] not in args.case:
                continue
            case_id = case["id"]
            scope = UserScope("evaluation", "r104", case_id, case_id, "private")
            database = SQLiteDB(output / f"{case_id}.sqlite")
            repo = DatabaseCharacterMemoryRepository(database)
            frozen = recorded_cases.get(case_id)
            if args.recorded_writes and frozen is None:
                raise ValueError("Missing recorded case; do not silently issue a fresh write request")

            class ReplayCompletion:
                def __init__(self, original):
                    self.original = original
                    self.calls = []

                async def complete(self, messages):
                    call = self.original["write_calls"][len(self.calls)]
                    actual = json.loads(messages[-1]["content"])["current_user_message"]
                    expected = json.loads(call["messages"][-1]["content"])["current_user_message"]
                    if actual != expected:
                        raise ValueError("Source turn does not match recorded output")
                    self.calls.append(dict(messages=messages, output=call["output"], recorded=True))
                    return call["output"]

                async def close(self):
                    pass

            writer = ReplayCompletion(frozen) if frozen else RecordedCompletion(config)
            scheduler = MemoryEnrichmentScheduler(config=config, completion=writer)
            steps, history = [], []
            try:
                for index, message in enumerate(case["turns"]):
                    observed = (datetime.fromisoformat(frozen["steps"][index]["observed_at"])
                                if frozen else datetime.now(timezone.utc))
                    scheduled = scheduler.schedule(repository=repo, character_id=profile.character_id,
                        user_scope=scope, message=message, history=tuple(history[-4:]),
                        rule_hints=extract_memories(message, reference_time=observed),
                        source_message_id=f"{case_id}-{index}", observed_at=observed)
                    if scheduled and not await scheduler.flush_memory(timeout=120):
                        raise RuntimeError("Source write did not complete")
                    rows = await repo.list_memory_records(profile.character_id, scope, limit=None, include_inactive=True)
                    steps.append(dict(message=message, observed_at=observed.isoformat(),
                                      rows=rows, scheduled=scheduled, status=asdict(scheduler.status),
                                      sources=await repo.list_sources(profile.character_id, scope)))
                    history.append({"role": "user", "content": message})
                    print(json.dumps(dict(case=case_id, write_step=index, saved=scheduler.status.saved,
                                          failed=scheduler.status.failed)), flush=True)
            finally:
                await scheduler.shutdown(timeout=3)
                manifest["recorded_write_outputs" if frozen else "fresh_write_model_calls"] += len(writer.calls)
            reads = []
            service = CharacterMemoryService(repo)
            orchestration = CharacterContextService(registry, repo, DatabaseMessageRepository(database),
                memory_service=service, source_recall_enabled=args.source_recall,
                source_window_radius=args.source_window_radius) if args.full_prepare else None
            for query in (() if args.writes_only else case["questions"]):
                prepared = None
                read_history = tuple(history) if args.read_history else ()
                received_at = datetime.now(timezone.utc)
                if orchestration is not None:
                    prepared = await orchestration.prepare_turn(TurnInput(message=query, platform=scope.platform,
                        adapter=scope.adapter, sender_id=scope.sender_id, conversation_id=scope.conversation_id,
                        conversation_type=scope.conversation_type, history=read_history), profile.character_id)
                    context, trace = prepared.compiled, prepared.memory_recall
                else:
                    items, _, trace = await service.recall_with_diagnostics(
                        profile.character_id, scope, query, reference_time=received_at)
                    context = compile_character_context(CharacterContext(profile=profile, user_scope=scope, memories=items),
                                                        complete_memory_evidence=True)
                    context = replace(context, memory_status="retrieval_error" if trace["status"] == "retrieval_error"
                                      else "available" if items else "no_match",
                                      memory_field_presence=tuple(trace.get("field_presence", {}).items()))
                    if args.source_recall:
                        sources = await SourceMemoryService(repo, window_radius=args.source_window_radius).recall(profile.character_id, scope, query,
                                                                         memories=context.memory_packets)
                        context = attach_sources(context, sources, complete_evidence=True)
                        trace["sources"] = sources.diagnostics
                calls = []

                async def generate(_calls=calls, **kwargs):
                    body = dict(model=args.model, messages=kwargs["messages"], temperature=kwargs["temperature"],
                        max_tokens=kwargs["max_tokens"], top_p=kwargs["top_p"],
                        repetition_penalty=kwargs["repetition_penalty"], frequency_penalty=kwargs["frequency_penalty"],
                        chat_template_kwargs={"enable_thinking": kwargs["enable_thinking"]}, seed=0)
                    started = time.monotonic()
                    response = await asyncio.to_thread(complete, args.endpoint.rstrip("/") + "/v1/chat/completions", body)
                    _calls.append(dict(request=body, response=response, seconds=time.monotonic() - started))
                    return response["choices"][0]["message"]["content"]

                started = time.monotonic()
                answer = await generate_character_response(GenerationRequest(
                    message=query, character_context=context, max_tokens=256, temperature=.2,
                    history=prepared.history if prepared else read_history,
                    reply_guard=prepared.reply_guard if prepared else None), generate)
                manifest["fresh_read_model_calls"] += len(calls)
                reads.append(dict(query=query, received_at=received_at.isoformat(), recall=trace,
                                  compiled=asdict(context), fields=[asdict(f) for f in read_memory_fields(query, context)],
                                  prepared=asdict(prepared) if prepared else None,
                                  answer=asdict(answer), calls=calls, seconds=time.monotonic() - started))
                print(json.dumps(dict(case=case_id, query=query, reply=answer.reply,
                                      mode=answer.response_mode, calls=len(calls)),
                                 ensure_ascii=False), flush=True)
            with (output / "results.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(dict(case=case_id, criteria=case["criteria"], steps=steps,
                                            write_calls=writer.calls, reads=reads), ensure_ascii=False,
                                        default=trace_json_default) + "\n")
    finally:
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--case", action="append")
    parser.add_argument("--fixtures", type=Path, help="Independent dialogue cases, review criteria kept out of requests")
    parser.add_argument("--full-prepare", action="store_true", help="Use actual prepare_turn and output guard")
    parser.add_argument("--read-history", action="store_true", help="Also supply original user turns as live short-term history")
    parser.add_argument("--recorded-writes", type=Path)
    parser.add_argument("--writes-only", action="store_true",
                        help="Exercise actual source/claim writes without answer generation")
    parser.add_argument("--source-recall", action="store_true", help="Include independently scoped full speech quotes")
    parser.add_argument("--source-window-radius", type=int, choices=(0, 1, 2), default=0,
                        help="Experimental indexed neighbor expansion, not inferred semantic dependencies")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
