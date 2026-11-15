"""Explicit live Flash smoke test using synthetic dialogue and in-memory credentials."""

import asyncio
import json
import os
import re
import secrets
import sys
import time
from pathlib import Path


async def main():
    root = Path(__file__).resolve().parents[1]
    runtime = root / ".codex-test-runtime-deepseek"
    runtime.mkdir(exist_ok=True)
    secret_file = Path.home() / "Desktop" / "deepseekapi.txt"
    key = re.search(r"sk-[A-Za-z0-9_-]+", secret_file.read_text(encoding="utf-8-sig"))
    if key is None:
        raise RuntimeError("No API key found in the configured local file")
    os.environ.update(JWT_SECRET=secrets.token_urlsafe(48), OPENAI_COMPAT_API_KEY=key.group(), OPENAI_COMPAT_BASE_URL="https://api.deepseek.com",
                      OPENAI_COMPAT_MODEL="deepseek-flash", MODEL_PROVIDER="openai_compat", USE_POSTGRESQL="false",
                      VLLM_ENABLED="false", MODEL_STORAGE_PATH=str(runtime / "models"),
                      DATABASE_PATH=str(runtime / "bootstrap.sqlite"), VLLM_TOP_P="1")
    sys.path.insert(0, str(root / "backend"))
    import httpx
    from character.library import CharacterSettings
    from db.database import SQLiteDB
    from infra.resource_pool import HttpClientPool
    from repositories.character_chat import CharacterChatRepository
    from services.character_chat import CharacterChatService, complete_with_provider

    async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
        response = await client.get("https://api.deepseek.com/models", headers={"Authorization": "Bearer " + key.group()})
        if response.status_code != 200:
            raise RuntimeError(f"Model listing failed HTTP {response.status_code}")
        models = [item["id"] for item in response.json()["data"]]
        print(json.dumps({"available_models": models}, ensure_ascii=False), flush=True)
    if "deepseek-flash" not in models:
        raise RuntimeError("Configured Flash model unavailable; no fallback attempted")
    db = SQLiteDB(runtime / ("live-" + str(time.time_ns()) + ".sqlite"))
    pool = HttpClientPool()
    pool._client = httpx.AsyncClient(timeout=120, trust_env=False)
    db.update_config({"modelProvider": "openai_compat", "openaiCompatModel": "deepseek-flash",
                      "openaiCompatBaseUrl": "https://api.deepseek.com", "temperature": 0.7,
                      "maxTokens": 384, "topP": 1.0, "useKnowledgeBase": False})
    repo = CharacterChatRepository(db)
    char = repo.characters.create("live-test", CharacterSettings(name="林舟", persona="谨慎而温和的旅行者，有自己的看法。不编造与用户的共同经历。每次回复不超过100字。",
                                                               greeting="你好，你也是来等车的吗？"))
    session = repo.create("live-test", char["id"], "Flash 接线验收")
    events = []

    async def complete(request, session, request_id, *, event_sink):
        async def observe(event):
            events.append((time.monotonic(), event["type"]))
            await event_sink(event)
        return await complete_with_provider(request, session, request_id, database=db, http_client_pool=pool, event_sink=observe)

    service = CharacterChatService(db, complete, stream_events=True)
    results = []
    try:
        for index, message in enumerate(("第一次见面，我叫小远，喜欢热茶，不喝咖啡。你在等哪班车？", "你会替我选什么喝的？如果我坚持雨天走危险山路，你会怎么说？")):
            started = time.monotonic()
            events.clear()
            args = {"revision": session["revision"], "request_id": f"live-{index}", "message": message}
            turn = await service.generate("live-test", session["id"], **args)
            first = next(moment for moment, kind in events if kind == "draft_delta")
            count = len(events)
            assert (await service.generate("live-test", session["id"], **args))["id"] == turn["id"]
            assert len(events) == count  # A replay must not issue another model request.
            results.append({"model": turn["model"], "message": message, "reply": turn["reply"],
                            "first_delta_seconds": round(first - started, 2), "total_seconds": round(time.monotonic() - started, 2),
                            "draft_events": count, "replay_same_turn": True})
            session = repo.get("live-test", session["id"])
        assert len(repo.turns("live-test", session["id"])) == 2
        (runtime / "latest-result.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(results, ensure_ascii=False, indent=2))
    finally:
        await pool.close()
        db.close_connection()


if __name__ == "__main__":
    asyncio.run(main())
