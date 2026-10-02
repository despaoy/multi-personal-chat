"""A separate authenticated writer against the disposable reader database."""

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
    proof = dict(
        pid=os.getpid(),
        database_name=db.execute_sql("SELECT current_database() AS name", {})[0]["name"],
        vector_path=os.environ["VECTOR_DB_PATH"],
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
        updated = await client.put(f"/api/knowledge/documents/{request['doc_id']}", json=request["payload"])
        proof["update_response"] = dict(http_status=updated.status_code, response=updated.json())
        assert updated.status_code == 200
        proof["after_update"] = dict(
            revision=knowledge._get_rebuild_revision(),
            status=knowledge._read_rebuild_status(),
            index_built=knowledge._vector_index_built,
        )
        # An actual writer-side search rebuilds its independent local index.
        rebuilt = await client.post(
            "/api/knowledge/search", json=dict(query=request["query"], topK=3, knowledgeBaseName=request["new_base"])
        )
        assert rebuilt.status_code == 200
        proof["search_response"] = rebuilt.json()
        proof["after_rebuild"] = dict(
            revision=knowledge._get_rebuild_revision(),
            status=knowledge._read_rebuild_status(),
            index_built=knowledge._vector_index_built,
        )
        proof["sync_pending_final"] = len(db._pending)
    (output / "writer-result.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
