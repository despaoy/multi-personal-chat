"""Replay recorded real writer outputs through actual scheduler and source store.

No fresh generative-model calls, answer generation or retrieval certification.
The existing writer's embedding-based target lookup remains enabled.
Preserves original receipt times and verifies complete input survives all writer
outcomes without accepting invalid mutations. Only fresh isolated output roots.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
from dataclasses import asdict
from datetime import datetime
from pathlib import Path


async def run(recorded, output):
    from character.memory_extractor import extract_memories
    from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig
    from character.models import UserScope
    from db.database import SQLiteDB
    from repositories.character_memory import DatabaseCharacterMemoryRepository

    results = []
    cases = [json.loads(line) for line in recorded.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    for index, case in enumerate(cases):
        scope = UserScope("evaluation", "capture-replay", str(index), str(index), "private")
        repo = DatabaseCharacterMemoryRepository(SQLiteDB(output / f"case-{index}.sqlite"))

        class Replay:
            def __init__(self, calls):
                self.calls = calls
                self.used = 0

            async def complete(self, messages):
                call = self.calls[self.used]
                expected = json.loads(call["messages"][-1]["content"])["current_user_message"]
                actual = json.loads(messages[-1]["content"])["current_user_message"]
                if expected != actual:
                    raise AssertionError("Recorded output belongs to another input")
                self.used += 1
                return call["output"]

            async def close(self):
                pass

        writer = Replay(case["write_calls"])
        scheduler = MemoryEnrichmentScheduler(
            config=MemoryLlmConfig(enabled=True, base_url="http://unused", model="recorded"), completion=writer)
        history, steps = [], []
        try:
            for step_index, step in enumerate(case["steps"]):
                source, at = step["message"], datetime.fromisoformat(step["observed_at"])
                source_id = f"source-{step_index}"
                if not scheduler.schedule(repository=repo, character_id="role", user_scope=scope,
                    message=source, rule_hints=extract_memories(source, reference_time=at), history=tuple(history),
                    source_message_id=source_id, observed_at=at):
                    raise AssertionError("Source was not admitted")
                if not await scheduler.flush_memory(timeout=120):
                    raise AssertionError("Writer did not finish")
                sources = await repo.list_sources("role", scope, source_message_ids=(source_id,))
                if len(sources) != 1 or sources[0]["body"] != source:
                    raise AssertionError("Complete source was not preserved")
                steps.append(dict(message=source, sources=sources, status=asdict(scheduler.status),
                    claims=await repo.list_memory_records("role", scope, limit=None, include_inactive=True)))
                history.append(dict(role="user", content=source))
        finally:
            await scheduler.shutdown(timeout=3)
        if writer.used != len(writer.calls):
            raise AssertionError("Not all recorded outputs replayed")
        results.append(dict(case=case["case"], recorded_outputs=writer.used, steps=steps))
    payload = dict(results=results, fresh_model_calls=0, generation_tested=False, source_retrieval_tested=False,
                   production_modified=False)
    (output / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(dict(cases=len(results), complete_sources=sum(len(c["steps"]) for c in results),
                          recorded_outputs=sum(c["recorded_outputs"] for c in results), fresh_model_calls=0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recorded-writes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    allowed = (Path(__file__).resolve().parents[2] / ".tmp",
               Path("/home/boot/lhm/multipersonal-runtime/evaluations"))
    if not any(output.is_relative_to(root.resolve()) and output != root.resolve() for root in allowed):
        raise ValueError("Output must be a new isolated evaluation directory")
    output.mkdir(exist_ok=False)
    os.environ["DATABASE_PATH"] = str(output / "bootstrap.sqlite")
    os.environ["USE_POSTGRESQL"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("JWT_SECRET", secrets.token_urlsafe(48))
    asyncio.run(run(args.recorded_writes, output))


if __name__ == "__main__":
    main()
