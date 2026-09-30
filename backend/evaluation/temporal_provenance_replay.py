"""Replay recorded write outputs through scheduler/SQLite, without inference.

Input is a location_memory_replay JSONL artifact. Output is always a new
directory, never an existing database. This diagnoses provenance, not model
quality or full persona/RAG E2E behavior.
"""

import argparse
import asyncio
import json
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

from character.memory_extractor import extract_memories
from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig
from character.memory_service import CharacterMemoryService, _is_current_record
from character.models import CompiledCharacterContext, UserScope
from db.database import SQLiteDB
from inference.memory_response import read_memory_fields, render_memory_response
from repositories.character_memory import DatabaseCharacterMemoryRepository


async def run(args):
    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    case = next(record for record in records if record["case"] == args.case)
    if len(case["write_calls"]) != len(case["steps"]):
        raise ValueError("Replay requires one recorded output per source turn")
    args.output.mkdir(parents=True, exist_ok=False)

    class Completion:
        index = 0

        async def complete(self, messages):
            call = case["write_calls"][self.index]
            expected = json.loads(call["messages"][-1]["content"])["current_user_message"]
            actual = json.loads(messages[-1]["content"])["current_user_message"]
            if actual != expected:
                raise ValueError("Recorded completion must match this source message")
            self.index += 1
            return call["output"]

        async def close(self):
            pass

    completion = Completion()
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(args.output / "memory.sqlite"))
    scope = UserScope("evaluation", "temporal-replay", args.case, args.case, "private")
    scheduler = MemoryEnrichmentScheduler(
        config=MemoryLlmConfig(enabled=True, base_url="http://unused", model="recorded-output"),
        completion=completion)
    history = []
    try:
        for index, step in enumerate(case["steps"]):
            source = f"{args.case}-{index}"
            original = next(row for row in step["rows"] if row.get("source_message_id") == source)
            observed = datetime.fromisoformat(original["observed_at"])
            if not scheduler.schedule(repository=repo, character_id="role", user_scope=scope,
                                      message=step["message"], history=tuple(history),
                                      rule_hints=extract_memories(step["message"], reference_time=observed),
                                      source_message_id=source, observed_at=observed):
                raise RuntimeError("Source turn was not scheduled")
            if not await scheduler.flush_memory(timeout=90):
                raise TimeoutError("Recorded-output worker did not drain")
            if scheduler.status.failed:
                raise RuntimeError(scheduler.status.last_error)
            history.append({"role": "user", "content": step["message"]})
    finally:
        await scheduler.shutdown(timeout=3)
    rows = await repo.list_memory_records("role", scope, limit=None, include_inactive=True)
    audit = []
    for row in rows:
        audit.append({"source_message_id": row["source_message_id"], "evidence": row["evidence"],
                      "provenance": row["metadata"]["temporal_provenance"],
                      "legacy_current_after_one_minute": _is_current_record(
                          row, datetime.fromisoformat(row["observed_at"]) + timedelta(minutes=1),
                          include_pending=False)})
    reads = []
    service = CharacterMemoryService(repo)
    reference = max(datetime.fromisoformat(row["observed_at"]) for row in rows) + timedelta(minutes=1)
    for query in args.query:
        items, _, trace = await service.recall_with_diagnostics("role", scope, query, reference_time=reference)
        if trace["status"] == "retrieval_error":
            raise RuntimeError(f"Read failed: {trace}")
        context = CompiledCharacterContext("", "", "", tuple(item.memory_id for item in items),
            memory_status="available" if items else "no_match", memory_packets=items,
            memory_field_presence=tuple(trace.get("field_presence", {}).items()))
        reads.append({"query": query, "trace": trace, "packets": [asdict(item) for item in items],
                      "fields": [asdict(field) for field in read_memory_fields(query, context)],
                      "direct_answer": render_memory_response(query, context)})
    result = {"input": str(args.input.resolve()), "case": args.case, "new_generation_calls": 0,
              "write_retrieval": "normal_scheduler_including_local_embeddings",
              "recorded_outputs_replayed": completion.index, "rows": audit,
              "reads": reads, "projection_policy_changed": True}
    (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query", action="append", default=[])
    asyncio.run(run(parser.parse_args()))
