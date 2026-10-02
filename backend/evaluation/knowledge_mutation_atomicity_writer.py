"""An independent authenticated writer against the disposable probe DB."""

import asyncio
import json
import os
import sys
from pathlib import Path


async def main():
    request = json.loads(sys.stdin.read())
    output = Path(request["output"]).resolve()
    label = request["label"]
    assert output.parent == Path("/home/boot/lhm/multipersonal-runtime/evaluations/r148pg.s3") and label in {
        "failure",
        "retry",
    }
    os.environ["VECTOR_DB_PATH"] = str(output / f"vectors-{label}")
    os.environ["AUDIT_LOG_DIR"] = str(output / f"audit-{label}")
    import httpx

    from api import knowledge
    from app.main import create_app
    from db.adapter import db

    app = create_app()
    proof = {"pid": os.getpid(), "database": db.execute_sql("SELECT current_database() AS name", {})[0]["name"]}
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="https://stage3-evaluation.invalid",
            timeout=180,
        ) as client,
    ):
        login = await client.post(
            "/api/auth/login", json={"username": "stage45-scope-fixed", "password": request["password"]}
        )
        me = await client.get("/api/auth/me")
        proof["auth_statuses"] = [login.status_code, me.status_code]
        assert proof["auth_statuses"] == [200, 200] and me.json()["user"]["role"] == "admin"
        response = await client.put(f"/api/knowledge/documents/{request['doc_id']}", json=request["payload"])
        proof["http_status"] = response.status_code
        proof["after_document"] = db.get_knowledge_document(request["doc_id"])
        proof["after_chunks"] = db.get_knowledge_chunks(request["doc_id"])
        proof["after_revision"] = knowledge._get_rebuild_revision()
        proof["after_status"] = knowledge._read_rebuild_status()
        proof["sync_pending_final"] = len(db._pending)
    (output / (label + "-result.json")).write_text(json.dumps(proof, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
