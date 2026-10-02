"""Publish arrangement then confirmation through independent authenticated requests."""

import asyncio
import json
import os
import sys
from pathlib import Path


async def main():
    request = json.loads(sys.stdin.read())
    output = Path(request["output"]).resolve()
    assert output.parent == Path("/home/boot/lhm/multipersonal-runtime/evaluations/r148pg.s3")
    os.environ["VECTOR_DB_PATH"] = str(output / "writer-vectors")
    os.environ["AUDIT_LOG_DIR"] = str(output / "writer-audit")
    import httpx

    from api import knowledge
    from app.main import create_app
    from db.adapter import db

    app = create_app()
    proof = {
        "pid": os.getpid(),
        "database": db.execute_sql("SELECT current_database() AS name", {})[0]["name"],
        "states": [],
    }

    def snapshot():
        return {
            "documents": [db.get_knowledge_document(i) for i in request["ids"]],
            "revision": knowledge._get_rebuild_revision(),
        }

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
        proof["role"] = me.json()["user"]["role"]
        assert proof["auth_statuses"] == [200, 200] and proof["role"] == "admin"
        proof["states"].append(snapshot())
        proof["http_statuses"] = []
        for doc_id, body in zip(request["ids"], request["updates"]):
            response = await client.put(f"/api/knowledge/documents/{doc_id}", json=body)
            proof["http_statuses"].append(response.status_code)
            assert response.status_code == 200
            proof["states"].append(snapshot())
        proof["sync_pending_final"] = len(db._pending)
    (output / "writer-result.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
