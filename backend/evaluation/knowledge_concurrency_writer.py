"""Actual authenticated writes; a gate delays a real revision read unchanged."""

import asyncio
import json
import os
import sys
import time
from pathlib import Path


async def main():
    request = json.loads(sys.stdin.read())
    output = Path(request["output"]).resolve()
    label = request["label"]
    assert output.parent == Path("/home/boot/lhm/multipersonal-runtime/evaluations/r148pg.s3") and label in {
        "A",
        "B",
        "C",
    }
    os.environ["VECTOR_DB_PATH"] = str(output / f"vectors-{label}")
    os.environ["AUDIT_LOG_DIR"] = str(output / f"audit-{label}")
    import httpx

    from api import knowledge
    from app.main import create_app
    from db.adapter import db

    app = create_app()
    proof = dict(
        pid=os.getpid(),
        label=label,
        database_name=db.execute_sql("SELECT current_database() AS name", {})[0]["name"],
        atomic_invalidation=callable(getattr(db, "mark_knowledge_index_dirty", None)),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="https://stage3-evaluation.invalid",
            timeout=180,
        ) as client,
    ):
        login = await client.post(
            "/api/auth/login", json=dict(username="stage45-scope-fixed", password=request["password"])
        )
        me = await client.get("/api/auth/me")
        proof["auth_statuses"] = [login.status_code, me.status_code]
        assert proof["auth_statuses"] == [200, 200]
        proof["role"] = me.json()["user"]["role"]
        assert proof["role"] == "admin"
        proof["before_revision"] = knowledge._get_rebuild_revision()
        gated = False

        def hold(value, location):
            nonlocal gated
            if gated:
                return
            gated = True
            (output / "B-observed.json").write_text(
                json.dumps(dict(pid=os.getpid(), revision=value, location=location))
            )
            deadline = time.monotonic() + 25
            while not (output / "B-release").exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("Real revision gate was not released")
                time.sleep(0.025)

        if label == "B":
            if proof["atomic_invalidation"]:
                from db import knowledge_index_state

                original = knowledge_index_state.parse_revision

                def observed_parse(raw):
                    value = original(raw)
                    hold(value, "locked_database_transaction")
                    return value

                knowledge_index_state.parse_revision = observed_parse
            else:
                original = knowledge._get_rebuild_revision

                def observed_read():
                    value = original()
                    hold(value, "unlocked_original_api_read")
                    return value

                knowledge._get_rebuild_revision = observed_read
        updated = await client.put(f"/api/knowledge/documents/{request['doc_id']}", json=request["payload"])
        proof["update_response"] = dict(http_status=updated.status_code, response=updated.json())
        assert updated.status_code == 200
        proof["after_revision"] = knowledge._get_rebuild_revision()
        proof["after_status"] = knowledge._read_rebuild_status()
        proof["sync_pending_final"] = len(db._pending)
    (output / f"{label}-result.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
