"""Read real pro-written isolated claims with an offline scoped retrieval diagnostic."""

import asyncio
import base64
import hashlib
import importlib.util
import json
import os
import secrets
import sys
from dataclasses import asdict
from pathlib import Path


async def run():
    runtime = Path("/home/boot/lhm/multipersonal-runtime")
    phase = runtime / "backups/backend-chain-20261001/stage78"
    cluster = runtime / "evaluations/r148pg.s3"
    label = "stage3_stage78_native_pg_baseline"
    os.environ.update(
        ENVIRONMENT="production",
        JWT_SECRET=secrets.token_urlsafe(48),
        ENCRYPTION_KEY=base64.urlsafe_b64encode(secrets.token_bytes(32)).decode(),
        USE_POSTGRESQL="true",
        DATABASE_URL="postgresql+asyncpg://boot@/" + label + "?host=" + str(cluster / "socket") + "&port=25433",
        DATABASE_PATH=str(phase / "pg-diagnostic-import-only.sqlite"),
        MODEL_PROVIDER="openai_compat",
        OPENAI_COMPAT_MODEL="deepseek-v4-pro",
        OPENAI_COMPAT_CONTEXT_WINDOW_TOKENS="65536",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        EMBEDDING_MODEL_PATH=str(runtime / "models/paraphrase-multilingual-MiniLM-L12-v2"),
        CUDA_VISIBLE_DEVICES="",
        VECTOR_DB_PATH=str(phase / "native-pg-baseline/vectors"),
    )
    import asyncpg
    import httpx

    cloud = []
    original_send = httpx.AsyncClient.send

    async def no_cloud(client, request, **kwargs):
        if request.url.host == "api.deepseek.com":
            cloud.append(str(request.url))
            raise AssertionError("Offline diagnostic must not call the real model")
        return await original_send(client, request, **kwargs)

    httpx.AsyncClient.send = no_cloud
    f = json.loads((phase / "fixture.json").read_text())
    before = json.loads((phase / "provider-block-storage.json").read_text())
    con = await asyncpg.connect(user="boot", database=label, host=str(cluster / "socket"), port=25433)
    try:
        assert await con.fetchval("SHOW data_directory") == str(cluster / "data")
    finally:
        await con.close()
    from repositories.character_memory import DatabaseCharacterMemoryRepository

    from character.memory_service import CharacterMemoryService
    from character.models import UserScope
    from db.adapter import db, is_pg_mode

    assert is_pg_mode()
    repo = DatabaseCharacterMemoryRepository(db)
    scope = UserScope("web", "web-character", "6", "6", "private")
    records = await repo.list_memory_records("tsukiyashiro_kisaki", scope, limit=None, include_inactive=True)
    assert len(records) == 3
    name = "_stage78_pg_candidate"
    spec = importlib.util.spec_from_file_location(name, phase / "candidate_memory_service.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    outputs = []
    for kind, service_type in [
        ("current_unmodified_backend", CharacterMemoryService),
        ("isolated_candidate", module.CharacterMemoryService),
    ]:
        trace = {}
        items, total = await service_type(repo, semantic_enabled=False).load_relevant_memories(
            "tsukiyashiro_kisaki", scope, f["question"], diagnostics=trace, for_contextual_selection=True
        )
        outputs.append({"kind": kind, "diagnostics": trace, "total": total, "items": [asdict(x) for x in items]})
    baseline, candidate = outputs
    assert (
        baseline["diagnostics"]["records_read"] == 3
        and baseline["diagnostics"]["usable_records"] == 0
        and not baseline["items"]
    )
    assert candidate["diagnostics"]["records_read"] == candidate["diagnostics"]["usable_records"] == 3 and {
        x["memory_id"] for x in candidate["items"]
    } == {str(r["id"]) for r in records}
    assert all(f["source_message"] in x["evidence"] for x in candidate["items"])
    assert any(dict(x["qualifiers"]) == {"condition": "完整表格已提交且身份核验通过"} for x in candidate["items"])
    con = await asyncpg.connect(user="boot", database=label, host=str(cluster / "socket"), port=25433)
    try:
        async with con.transaction(readonly=True):
            assert await con.fetchval("SHOW data_directory") == str(cluster / "data")
            after = await con.fetch(
                "SELECT * FROM character_memories WHERE character_id=$1 AND platform='web' AND adapter='web-character' AND sender_id='6' AND conversation_id='6'",
                "tsukiyashiro_kisaki",
            )
            assert json.loads(json.dumps([dict(x) for x in after], default=str)) == before["claims"]
            assert not await con.fetchval(
                'SELECT EXISTS(SELECT 1 FROM messages WHERE "senderId"=$1 AND message=$2)', "6", f["question"]
            )
    finally:
        await con.close()
    assert not cloud
    result = {
        "mechanism_only": True,
        "real_pro_written_claims_read": True,
        "actual_native_final_task_attempted": False,
        "actual_semantic_reviewer_run": False,
        "cloud_requests": 0,
        "candidate_deployed": False,
        "raw_original_source_bytes_unchanged": True,
        "full_native_qualification": 0,
        "outputs": outputs,
        "records": records,
        "candidate_sha256": hashlib.sha256((phase / "candidate_memory_service.py").read_bytes()).hexdigest(),
    }
    (phase / "offline-pg-diagnostic.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str) + chr(10)
    )
    print(
        json.dumps(
            {
                "real_three_claims": True,
                "current_usable": baseline["diagnostics"]["usable_records"],
                "candidate_usable": candidate["diagnostics"]["usable_records"],
                "same_exact_source_and_conjunction": True,
                "no_cloud_requests": True,
                "full_native_qualification": 0,
            }
        )
    )


if __name__ == "__main__":
    asyncio.run(run())
