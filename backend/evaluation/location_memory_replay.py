"""Real model + semantic writer + SQLite + recall + shared generation replay.

This intentionally isolates memory from HTTP, persona preparation and RAG.
Raw model outputs and final rows are retained; gold fields never enter prompts.
"""

import argparse
import asyncio
import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

from character.memory_extractor import extract_memories
from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig, OpenAICompatibleMemoryCompletion
from character.memory_service import CharacterMemoryService
from character.models import CompiledCharacterContext, UserScope
from db.database import SQLiteDB
from evaluation.episode_subject_audit import complete
from inference.generation_request import GenerationRequest, generate_character_response
from inference.memory_response import read_memory_fields

CASES = (
    ("separate_fields", ("我来自衡阳。", "我现在住在舟山。"), {"origin": "衡阳", "residence": "舟山"}),
    ("combined_fields", ("我来自通辽，我现在住在嘉兴。",), {"origin": "通辽", "residence": "嘉兴"}),
    ("explicit_move", ("我来自渭南。", "我现在住在湖州。", "我搬家了，我现在住在淄博。"),
     {"origin": "渭南", "residence": "淄博"}),
    ("repeated_fact", ("我现在住在遵义。", "我现在住在遵义。"), {"residence": "遵义"}),
    ("friend_not_user", ("我来自抚顺。", "我的朋友住在梧州。"), {"origin": "抚顺"}),
    ("temporary_visit", ("我现在住在黄山。", "我这周在柳州出差，下周就回去。"), {"residence": "黄山"}),
    ("implicit_withdrawal", ("我现在住在咸阳。", "之前那个住址不对，新的我以后再告诉你。"), {}),
    ("natural_paraphrase", ("老家是常德，现在定居在台州。",), {"origin": "常德", "residence": "台州"}),
)


class RecordedCompletion:
    def __init__(self, config):
        self.delegate = OpenAICompatibleMemoryCompletion(config)
        self.calls = []

    async def complete(self, messages):
        start = time.monotonic()
        raw = await self.delegate.complete(messages)
        self.calls.append(dict(messages=messages, output=raw, seconds=time.monotonic() - start))
        return raw

    async def close(self):
        await self.delegate.close()


async def run(args):
    from repositories.character_memory import DatabaseCharacterMemoryRepository

    root = Path("/home/boot/lhm/multipersonal-runtime/evaluations").resolve()
    output = args.output.resolve()
    if not output.is_relative_to(root) or not output.name.startswith("r97-"):
        raise ValueError("Only a new isolated r97 evaluation directory is accepted")
    output.mkdir(exist_ok=False)
    config = replace(MemoryLlmConfig.from_env(), enabled=True, base_url=args.endpoint, model=args.model)
    results = []
    for case, statements, gold in CASES:
        if args.case and case not in args.case:
            continue
        repo = DatabaseCharacterMemoryRepository(SQLiteDB(output / (case + ".sqlite")))
        scope = UserScope("evaluation", "r97", case, case, "private")
        completion = RecordedCompletion(config)
        scheduler = MemoryEnrichmentScheduler(config=config, completion=completion)
        history = []
        steps = []
        try:
            for index, message in enumerate(statements):
                start = time.monotonic()
                scheduled = scheduler.schedule(repository=repo, character_id="role", user_scope=scope,
                    message=message, history=tuple(history[-4:]), rule_hints=extract_memories(message),
                    source_message_id=f"{case}-{index}")
                drained = await scheduler.flush_memory(timeout=90)
                if not drained:
                    raise TimeoutError("memory worker did not drain; do not treat partial state as final")
                history.append({"role": "user", "content": message})
                rows = await repo.list_memory_records("role", scope, limit=None, include_inactive=True)
                steps.append(dict(message=message, scheduled=scheduled, seconds=time.monotonic()-start,
                                  status=asdict(scheduler.status), rows=rows))
                print(json.dumps(dict(case=case, step=index, scheduled=scheduled,
                                      saved=scheduler.status.saved)), flush=True)
        finally:
            await scheduler.shutdown(timeout=3)
        query = "我来自哪里，目前住哪里？"
        service = CharacterMemoryService(repo)
        items, _, trace = await service.recall_with_diagnostics("role", scope, query)
        context = CompiledCharacterContext("", "", "\n".join(item.content for item in items),
            tuple(item.memory_id for item in items), memory_packets=items,
            memory_status="available" if items else "no_match",
            memory_field_presence=tuple(trace.get("field_presence", {}).items()))
        read_calls = []

        async def generate(_calls=read_calls, **kwargs):
            body = dict(model=args.model, messages=kwargs["messages"], temperature=kwargs["temperature"],
                max_tokens=kwargs["max_tokens"], top_p=kwargs["top_p"],
                repetition_penalty=kwargs["repetition_penalty"], frequency_penalty=kwargs["frequency_penalty"],
                chat_template_kwargs={"enable_thinking": kwargs["enable_thinking"]}, seed=0)
            response = await asyncio.to_thread(complete, args.endpoint.rstrip("/") + "/v1/chat/completions", body)
            _calls.append(dict(request=body, response=response))
            return response["choices"][0]["message"]["content"]

        answer = await generate_character_response(GenerationRequest(
            message=query, character_context=context, max_tokens=192), generate)
        result = dict(case=case, gold_fields=gold, steps=steps, write_calls=completion.calls,
            recall=trace, packets=[asdict(item) for item in items],
            reads=[asdict(item) for item in read_memory_fields(query, context)],
            answer=asdict(answer), read_calls=read_calls)
        results.append(result)
        with (output / "results.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
        print(json.dumps(dict(case=case, write_calls=len(completion.calls), read_calls=len(read_calls),
                              reads=result["reads"]), ensure_ascii=False), flush=True)
    backend = Path(__file__).resolve().parents[1]
    paths = ("character/memory_llm.py", "character/memory_query.py", "character/memory_service.py",
             "db/database.py", "db/memory_claim_guard.py", "evaluation/location_memory_replay.py")
    (output / "manifest.json").write_text(json.dumps(dict(model=args.model,
        model_calls=sum(len(r["write_calls"]) + len(r["read_calls"]) for r in results),
        real_model=True, http_app_e2e=False, rag_tested=False, persona_preparation_tested=False,
        production_modified=False, sources={name: hashlib.sha256((backend/name).read_bytes()).hexdigest()
                                            for name in paths}), indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--case", action="append", choices=[case[0] for case in CASES])
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
