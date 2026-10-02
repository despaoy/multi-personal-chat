"""Independent actual admin requests for rename and folder removal."""

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
        "rename",
        "detach",
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
        proof["role"] = me.json()["user"]["role"]
        assert proof["auth_statuses"] == [200, 200] and proof["role"] == "admin"
        if label == "rename":
            response = await client.put(
                f"/api/knowledge/bases/{request['kb_id']}", json={"name": request["new_base_name"]}
            )
        else:
            response = await client.delete(f"/api/knowledge/folders/{request['folder_id']}")
        proof["http_status"] = response.status_code
        assert response.status_code == 200
        proof["after_base"] = db.get_knowledge_base(request["kb_id"])
        proof["after_folder"] = db.get_knowledge_folder(request["folder_id"])
        proof["after_document"] = db.get_knowledge_document(request["doc_id"])
        proof["after_revision"] = knowledge._get_rebuild_revision()
        proof["after_status"] = knowledge._read_rebuild_status()
        proof["sync_pending_final"] = len(db._pending)
    (output / (label + "-result.json")).write_text(json.dumps(proof, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
