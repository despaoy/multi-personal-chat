"""Literal teaching examples remain data through validation, model transport and storage."""

from knowledge import index_lifecycle
import httpx
import pytest
from fastapi import FastAPI
from test_compiled_cloud_transport import installed as installed

from api import generate, knowledge
from db.database import SQLiteDB
from db.schemas import KnowledgeDocumentCreate, MessageRequest
from middleware.security import InputValidationMiddleware, SecurityMiddleware

SQL_TEXTS = (
    "数据库课程题设：UNION SELECT 和 UNION ALL SELECT 分别如何处理重复行？只解释语义，不执行语句。",
    "数据库课程题设：DELETE FROM lessons WHERE id = 7 是什么作用？字符串 '; DROP TABLE messages;-- 只作教学引用，不执行。",
)
COURSE_TEXTS = (*SQL_TEXTS,
    "数据库课程：请解释下面的代码，不执行。\n```sql\nSELECT name FROM students UNION SELECT name FROM teachers;\n```",
    "网络课程：请解释 curl https://example.invalid/notes 的用途，不访问网址或执行命令。",
    pytest.param('网页课程：解释原样代码，不渲染或执行：<button onclick="showNote()">课程说明</button>。', id="literal-html"),
    pytest.param('脚本课程：解释字符串 eval("2 + 3") 的求值含义，只说明结果，不执行代码。', id="literal-javascript"),
)


@pytest.mark.asyncio
@pytest.mark.parametrize("message", COURSE_TEXTS)
async def test_literal_course_text_reaches_compiled_provider_unchanged(installed, message):
    app = FastAPI()
    app.add_middleware(InputValidationMiddleware)
    app.add_middleware(SecurityMiddleware, api_keys=["synthetic-text-key"])

    @app.post("/api/generate")
    async def reply(request: MessageRequest):
        return await generate._generate_reply_impl(request, persist_message=False, record_invocation=False, message_db=generate.db)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://synthetic") as client:
        response = await client.post("/api/generate", json={"message": message, "loraName": "default"},
                                     headers={"X-API-Key": "synthetic-text-key"})
    assert response.status_code == 200 and response.json()["modelInvoked"]
    assert len(installed.calls) == 1 and message in installed.calls[0]["body"]["messages"][-1]["content"]
    installed.manager.async_generate.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("content", SQL_TEXTS)
async def test_sql_example_is_preserved_as_document_and_bound_parameter(tmp_path, monkeypatch, content):
    database = SQLiteDB(tmp_path / "literal-text.sqlite")
    monkeypatch.setattr(knowledge, "db", database)
    monkeypatch.setattr(index_lifecycle, "db", database)
    monkeypatch.setattr(index_lifecycle, "_vector_index_revision", None)
    try:
        database.add_message(dict(sessionId="sentinel", message="隔离保留记录", reply="必须保留", modelName="synthetic"))
        request = KnowledgeDocumentCreate(title="SQL课程：UNION SELECT", content=content, category="课程")
        result = await knowledge.create_knowledge_document(request, {"role": "admin"})
        assert result["success"]
        document = database.get_knowledge_document(result["document"]["id"])
        assert document["title"] == request.title and document["content"] == content
        assert "".join(row["content"] for row in database.get_knowledge_chunks(document["id"])) == content
        database.update_config({"course_note": content})
        assert database.get_config_value("course_note") == content
        assert len(database.get_messages(limit=10)) == 1
        assert database.get_messages(limit=10)[0]["message"] == "隔离保留记录"
    finally:
        database.close_connection()
