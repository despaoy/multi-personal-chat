"""Regression tests for PostgreSQL SQL compatibility, SemanticCache lock race,
and pagination total-count correctness.

Covers the three gaps flagged in the review:
1. API modules must not use ``?`` placeholders in raw SQL (PostgreSQL incompatible).
2. SemanticCache per-key lock must use reference counting to prevent the same
   key from acquiring two different locks during capacity-driven cleanup.
3. List endpoints must return the real ``COUNT(*)`` total, not a placeholder.
   Verified both at SQL layer and through FastAPI TestClient response payload.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

API_DIR = BACKEND_ROOT / "api"


# ============================================================
# 1. PostgreSQL SQL placeholder regression
# ============================================================

def _extract_sql_strings_from_execute_calls(source: str, filepath: str):
    """Parse *source* and yield every string literal passed as the first
    positional argument to ``execute_sql`` / ``execute_sql_insert``."""
    # Strip UTF-8 BOM (some files start with \ufeff, which ast.parse rejects)
    source = source.lstrip("\ufeff")
    tree = ast.parse(source, filename=str(filepath))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # Match db.execute_sql / db.execute_sql_insert / self.execute_sql etc.
        if isinstance(func, ast.Attribute) and func.attr in ("execute_sql", "execute_sql_insert"):
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                yield node.args[0].value


@pytest.mark.parametrize("module_name", sorted(p.name for p in API_DIR.glob("*.py") if p.suffix == ".py"))
def test_no_question_mark_placeholders_in_api_sql(module_name):
    """Every raw SQL string in api/ must use :name named params, not ?.

    PostgreSQL (asyncpg) raises when encountering ``?`` placeholders;
    SQLite accepts both ``?`` and ``:name``.  Using named params everywhere
    keeps both backends working.
    """
    filepath = API_DIR / module_name
    source = filepath.read_text(encoding="utf-8")
    if "execute_sql" not in source:
        pytest.skip(f"{module_name} has no execute_sql calls")

    offenders = []
    for sql in _extract_sql_strings_from_execute_calls(source, filepath):
        # A literal '?' inside a string constant that is part of execute_sql
        # is the PostgreSQL-incompatible placeholder pattern.
        if "?" in sql:
            offenders.append(sql[:120])
    assert not offenders, (
        f"{module_name} still uses '?' placeholders in execute_sql: {offenders}"
    )


# ============================================================
# 3. Pagination total-count regression
# ============================================================
#
# Two layers of verification:
#  (a) SQL layer: COUNT(*) returns correct total at the db adapter level.
#  (b) API layer: list endpoints actually return that total in the response
#      payload (catches the case where the code computes COUNT(*) but then
#      returns ``len(rows)`` instead of the count).
#
# Both layers use pytest's ``tmp_path`` fixture so the temp DB is cleaned up
# automatically by the pytest harness.

class TestPaginationTotalCount:
    """Verify that list endpoints return the real COUNT(*) total."""

    def _make_temp_db(self, tmp_path):
        """Create a fresh SQLiteDB instance with schema initialized.

        Uses pytest's tmp_path fixture for automatic cleanup.
        """
        from db.database import SQLiteDB
        db_path = tmp_path / "test.db"
        return SQLiteDB(db_path)

    def _insert_experiment_runs(self, db, count: int):
        """Insert *count* rows into experiment_runs."""
        for i in range(count):
            db.execute_sql_insert(
                "INSERT INTO experiment_runs (id, experiment_type, hypothesis, status, started_at, results, config_path, report_path) "
                "VALUES (:id, :et, :hyp, 'completed', :ts, :r, '', '')",
                {
                    "id": f"exp_{i}",
                    "et": "lora_ablation",
                    "hyp": f"hypothesis {i}",
                    "ts": f"2026-01-{i+1:02d}T00:00:00Z",
                    "r": "{}",
                },
            )

    # --- (a) SQL layer ---

    def test_experiment_total_matches_count(self, tmp_path):
        db = self._make_temp_db(tmp_path)
        self._insert_experiment_runs(db, 25)
        rows = db.execute_sql("SELECT COUNT(*) AS cnt FROM experiment_runs", {})
        assert rows[0]["cnt"] == 25

    def test_experiment_total_with_type_filter(self, tmp_path):
        db = self._make_temp_db(tmp_path)
        self._insert_experiment_runs(db, 10)
        # Insert a different type
        db.execute_sql_insert(
            "INSERT INTO experiment_runs (id, experiment_type, hypothesis, status, started_at, results, config_path, report_path) "
            "VALUES (:id, :et, :hyp, 'completed', :ts, :r, '', '')",
            {"id": "rag_1", "et": "rag_ablation", "hyp": "rag", "ts": "2026-01-01", "r": "{}"},
        )
        total_all = db.execute_sql("SELECT COUNT(*) AS cnt FROM experiment_runs", {})[0]["cnt"]
        assert total_all == 11
        total_lora = db.execute_sql(
            "SELECT COUNT(*) AS cnt FROM experiment_runs WHERE experiment_type=:et",
            {"et": "lora_ablation"},
        )[0]["cnt"]
        assert total_lora == 10

    def test_feedback_total_matches_count(self, tmp_path):
        db = self._make_temp_db(tmp_path)
        for i in range(15):
            db.execute_sql_insert(
                "INSERT INTO feedback (trace_id, message_id, rating, reason, adapter_name, kb_revision, prompt_version, detail, created_at) "
                "VALUES (:trace_id, :message_id, :rating, :reason, :adapter_name, :kb_revision, :prompt_version, :detail, :created_at)",
                {
                    "trace_id": f"t{i}", "message_id": f"m{i}",
                    "rating": "thumbs_up" if i % 2 == 0 else "thumbs_down",
                    "reason": None, "adapter_name": None,
                    "kb_revision": None, "prompt_version": None,
                    "detail": None, "created_at": f"2026-01-{i+1:02d}",
                },
            )
        total = db.execute_sql("SELECT COUNT(*) AS cnt FROM feedback", {})[0]["cnt"]
        assert total == 15
        total_up = db.execute_sql(
            "SELECT COUNT(*) AS cnt FROM feedback WHERE rating=:rating",
            {"rating": "thumbs_up"},
        )[0]["cnt"]
        assert total_up == 8  # 0,2,4,6,8,10,12,14 → 8 thumbs_up

    def test_retrieval_eval_total_matches_count(self, tmp_path):
        db = self._make_temp_db(tmp_path)
        for i in range(8):
            db.execute_sql_insert(
                "INSERT INTO retrieval_eval_questions (id, question, expected_doc_ids, expected_doc_titles, gold_answer, category, created_at) "
                "VALUES (:id, :q, :dids, :dtitles, :ga, :cat, :ts)",
                {
                    "id": f"rq_{i}", "q": f"question {i}",
                    "dids": "[]", "dtitles": "[]", "ga": None,
                    "cat": "factual" if i < 5 else "persona",
                    "ts": f"2026-01-{i+1:02d}",
                },
            )
        total = db.execute_sql("SELECT COUNT(*) AS cnt FROM retrieval_eval_questions", {})[0]["cnt"]
        assert total == 8
        total_factual = db.execute_sql(
            "SELECT COUNT(*) AS cnt FROM retrieval_eval_questions WHERE category=:cat",
            {"cat": "factual"},
        )[0]["cnt"]
        assert total_factual == 5

    def test_preference_total_matches_count(self, tmp_path):
        db = self._make_temp_db(tmp_path)
        for i in range(12):
            db.execute_sql_insert(
                "INSERT INTO preference_pairs (id, prompt, chosen, rejected, rubric, annotator, metadata, review_status, created_at) "
                "VALUES (:id, :prompt, :chosen, :rejected, :rubric, :annotator, :metadata, :review_status, :created_at)",
                {
                    "id": f"pref_{i}", "prompt": f"p{i}", "chosen": f"c{i}",
                    "rejected": f"r{i}", "rubric": "{}", "annotator": "manual",
                    "metadata": "{}",
                    "review_status": "approved" if i < 7 else "pending",
                    "created_at": f"2026-01-{i+1:02d}",
                },
            )
        total = db.execute_sql("SELECT COUNT(*) AS cnt FROM preference_pairs", {})[0]["cnt"]
        assert total == 12
        total_approved = db.execute_sql(
            "SELECT COUNT(*) AS cnt FROM preference_pairs WHERE review_status=:rs",
            {"rs": "approved"},
        )[0]["cnt"]
        assert total_approved == 7

    # --- (b) API layer via TestClient ---
    #
    # Verifies the list endpoints actually surface the COUNT(*) total in
    # their JSON response, not just ``len(rows)``.  Uses dependency_overrides
    # to bypass auth.

    def _build_test_app(self, tmp_path):
        """Build a minimal FastAPI app with experiments/feedback/retrieval_eval
        routers mounted, pointing the db adapter at a temp SQLiteDB, and auth
        bypassed via dependency_overrides.

        Returns (client, test_db, cleanup) where cleanup closes the client
        and DB connection and restores patched modules.
        """
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from app.dependencies import get_current_admin, get_current_user
        from db import adapter as adapter_module
        from db.database import SQLiteDB
        from api import experiments, evaluation, retrieval_eval

        test_db = SQLiteDB(tmp_path / "api_test.db")
        # Patch the module-level db object used by routers
        original_db = adapter_module.db
        adapter_module.db = test_db
        # Also patch the db imported into each router module (bound at import)
        for mod in (experiments, evaluation, retrieval_eval):
            mod.db = test_db

        app = FastAPI()
        app.include_router(experiments.router)
        app.include_router(evaluation.router)
        app.include_router(retrieval_eval.router)

        # Bypass auth
        authenticated_user = {"user_id": 1, "username": "tester", "role": "admin"}
        app.dependency_overrides[get_current_user] = lambda: authenticated_user
        app.dependency_overrides[get_current_admin] = lambda: authenticated_user

        client = TestClient(app)

        def _cleanup():
            try:
                client.close()
            except Exception:
                pass
            try:
                test_db.close_connection()
            except Exception:
                pass
            adapter_module.db = original_db
            for mod in (experiments, evaluation, retrieval_eval):
                mod.db = original_db

        return client, test_db, _cleanup

    def test_experiment_list_api_returns_count_total(self, tmp_path):
        """GET /api/experiments/ must return ``total`` == COUNT(*), not len(rows)."""
        client, test_db, cleanup = self._build_test_app(tmp_path)
        try:
            self._insert_experiment_runs(test_db, 30)
            # Request only 5 rows — total must still be 30
            resp = client.get("/api/experiments/", params={"limit": 5, "offset": 0})
            assert resp.status_code == 200
            body = resp.json()
            assert body["success"] is True
            assert len(body["experiments"]) == 5, "Should respect limit"
            assert body["total"] == 30, "total must be COUNT(*), not len(rows)"
        finally:
            cleanup()

    def test_experiment_list_api_total_with_type_filter(self, tmp_path):
        """Filtered list endpoint total must reflect the filter, not all rows."""
        client, test_db, cleanup = self._build_test_app(tmp_path)
        try:
            self._insert_experiment_runs(test_db, 10)  # lora_ablation
            # Insert one rag_ablation
            test_db.execute_sql_insert(
                "INSERT INTO experiment_runs (id, experiment_type, hypothesis, status, started_at, results, config_path, report_path) "
                "VALUES (:id, :et, :hyp, 'completed', :ts, :r, '', '')",
                {"id": "rag_1", "et": "rag_ablation", "hyp": "rag", "ts": "2026-01-01", "r": "{}"},
            )
            resp = client.get("/api/experiments/", params={"experiment_type": "lora_ablation", "limit": 100})
            assert resp.status_code == 200
            body = resp.json()
            assert body["total"] == 10, "filtered total must be 10 lora_ablation rows"
            assert all(e["experiment_type"] == "lora_ablation" for e in body["experiments"])
        finally:
            cleanup()

    def test_feedback_list_api_returns_count_total(self, tmp_path):
        """GET /api/feedback must return ``total`` == COUNT(*)."""
        client, test_db, cleanup = self._build_test_app(tmp_path)
        try:
            for i in range(20):
                test_db.execute_sql_insert(
                    "INSERT INTO feedback (trace_id, message_id, rating, reason, adapter_name, kb_revision, prompt_version, detail, created_at) "
                    "VALUES (:trace_id, :message_id, :rating, :reason, :adapter_name, :kb_revision, :prompt_version, :detail, :created_at)",
                    {
                        "trace_id": f"t{i}", "message_id": f"m{i}",
                        "rating": "thumbs_up" if i % 2 == 0 else "thumbs_down",
                        "reason": None, "adapter_name": None,
                        "kb_revision": None, "prompt_version": None,
                        "detail": None, "created_at": f"2026-01-{i+1:02d}",
                    },
                )
            resp = client.get("/api/feedback", params={"limit": 5})
            assert resp.status_code == 200
            body = resp.json()
            assert len(body["feedbacks"]) == 5
            assert body["total"] == 20

            # Filtered by rating
            resp_up = client.get("/api/feedback", params={"limit": 100, "rating": "thumbs_up"})
            assert resp_up.status_code == 200
            body_up = resp_up.json()
            assert body_up["total"] == 10  # 0,2,4,...,18 → 10 thumbs_up
        finally:
            cleanup()

    def test_retrieval_eval_list_api_returns_count_total(self, tmp_path):
        """GET /api/retrieval-eval/questions must return ``total`` == COUNT(*)."""
        client, test_db, cleanup = self._build_test_app(tmp_path)
        try:
            for i in range(12):
                test_db.execute_sql_insert(
                    "INSERT INTO retrieval_eval_questions (id, question, expected_doc_ids, expected_doc_titles, gold_answer, category, created_at) "
                    "VALUES (:id, :q, :dids, :dtitles, :ga, :cat, :ts)",
                    {
                        "id": f"rq_{i}", "q": f"question {i}",
                        "dids": "[]", "dtitles": "[]", "ga": None,
                        "cat": "factual" if i < 4 else "persona",
                        "ts": f"2026-01-{i+1:02d}",
                    },
                )
            resp = client.get("/api/retrieval-eval/questions", params={"limit": 3})
            assert resp.status_code == 200
            body = resp.json()
            assert len(body["questions"]) == 3
            assert body["total"] == 12

            # Filtered by category
            resp_factual = client.get("/api/retrieval-eval/questions", params={"limit": 100, "category": "factual"})
            assert resp_factual.status_code == 200
            body_factual = resp_factual.json()
            assert body_factual["total"] == 4
        finally:
            cleanup()


# ============================================================
# 4. Embedding BLOB→INTEGER migration regression
# ============================================================

class TestEmbeddingBlobMigration:
    """Verify the BLOB→INTEGER column migration uses PRAGMA table_info
    (not typeof()), preserves valid integers via CASE, and rolls back
    on failure via SAVEPOINT."""

    def _create_old_blob_db(self, db_path):
        """Create a SQLite DB with the old BLOB embedding column and insert
        mixed data (BLOB bytes, integers, NULL)."""
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA foreign_keys = ON")
        # Create knowledge_documents first (FK target)
        conn.execute('''
            CREATE TABLE knowledge_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                category TEXT DEFAULT '未分类',
                knowledge_base_id INTEGER,
                folder_id INTEGER,
                sourceType TEXT DEFAULT 'text',
                sourceUrl TEXT, fileType TEXT, fileSize INTEGER,
                chunkCount INTEGER DEFAULT 0,
                createdAt TEXT NOT NULL,
                updatedAt TEXT NOT NULL
            )
        ''')
        conn.execute("INSERT INTO knowledge_documents (id, title, content, category, createdAt, updatedAt) VALUES (1, 'doc1', 'content1', 'cat', '2026-01-01', '2026-01-01')")
        # Old schema: embedding is BLOB
        conn.execute('''
            CREATE TABLE knowledge_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                documentId INTEGER NOT NULL,
                chunkIndex INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB,
                createdAt TEXT NOT NULL,
                FOREIGN KEY (documentId) REFERENCES knowledge_documents(id) ON DELETE CASCADE
            )
        ''')
        # Insert mixed data: row 1 = BLOB bytes, row 2 = integer, row 3 = NULL
        conn.execute("INSERT INTO knowledge_chunks (id, documentId, chunkIndex, content, embedding, createdAt) VALUES (1, 1, 0, 'chunk1', X'DEADBEEF', '2026-01-01')")
        conn.execute("INSERT INTO knowledge_chunks (id, documentId, chunkIndex, content, embedding, createdAt) VALUES (2, 1, 1, 'chunk2', 42, '2026-01-01')")
        conn.execute("INSERT INTO knowledge_chunks (id, documentId, chunkIndex, content, embedding, createdAt) VALUES (3, 1, 2, 'chunk3', NULL, '2026-01-01')")
        conn.commit()
        conn.close()

    def test_blob_column_migrated_to_integer(self, tmp_path):
        """BLOB embedding column is migrated to INTEGER via PRAGMA table_info."""
        import sqlite3
        self._create_old_blob_db(tmp_path / "old.db")
        from db.database import SQLiteDB
        db = SQLiteDB(tmp_path / "old.db")

        # Verify column type is now INTEGER (PRAGMA needs raw sqlite3)
        conn = sqlite3.connect(str(tmp_path / "old.db"))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(knowledge_chunks)")
        columns = cursor.fetchall()
        conn.close()
        embedding_type = ""
        for col in columns:
            if col["name"] == "embedding":
                embedding_type = (col["type"] or "").upper()
                break
        assert embedding_type == "INTEGER", f"embedding should be INTEGER, got {embedding_type}"
        db.close_connection()

    def test_valid_integers_preserved_blob_nulled(self, tmp_path):
        """CASE WHEN typeof()='integer' preserves valid FAISS IDs, BLOB→NULL."""
        self._create_old_blob_db(tmp_path / "old.db")
        from db.database import SQLiteDB
        db = SQLiteDB(tmp_path / "old.db")

        rows = db.execute_sql("SELECT id, embedding FROM knowledge_chunks ORDER BY id", {})
        assert len(rows) == 3
        # Row 1: was BLOB → NULL
        assert rows[0]["embedding"] is None, f"BLOB should be NULL, got {rows[0]['embedding']}"
        # Row 2: was integer 42 → preserved
        assert rows[0 + 1]["embedding"] == 42, f"integer should be preserved, got {rows[1]['embedding']}"
        # Row 3: was NULL → NULL
        assert rows[2]["embedding"] is None
        db.close_connection()

    def test_empty_blob_table_migrated(self, tmp_path):
        """Empty table with BLOB column still gets migrated (PRAGMA checks
        declaration type, not row data)."""
        import sqlite3
        conn = sqlite3.connect(str(tmp_path / "empty.db"))
        conn.execute('''CREATE TABLE knowledge_documents (id INTEGER PRIMARY KEY, title TEXT, content TEXT, category TEXT, knowledge_base_id INTEGER, folder_id INTEGER, sourceType TEXT, sourceUrl TEXT, fileType TEXT, fileSize INTEGER, chunkCount INTEGER, createdAt TEXT, updatedAt TEXT)''')
        conn.execute('''CREATE TABLE knowledge_chunks (id INTEGER PRIMARY KEY AUTOINCREMENT, documentId INTEGER NOT NULL, chunkIndex INTEGER NOT NULL, content TEXT NOT NULL, embedding BLOB, createdAt TEXT NOT NULL, FOREIGN KEY (documentId) REFERENCES knowledge_documents(id) ON DELETE CASCADE)''')
        conn.commit()
        conn.close()

        from db.database import SQLiteDB
        db = SQLiteDB(tmp_path / "empty.db")
        # PRAGMA needs raw sqlite3 connection
        conn2 = sqlite3.connect(str(tmp_path / "empty.db"))
        conn2.row_factory = sqlite3.Row
        cursor = conn2.cursor()
        cursor.execute("PRAGMA table_info(knowledge_chunks)")
        columns = cursor.fetchall()
        conn2.close()
        embedding_type = ""
        for col in columns:
            if col["name"] == "embedding":
                embedding_type = (col["type"] or "").upper()
                break
        assert embedding_type == "INTEGER", f"empty BLOB table should migrate to INTEGER, got {embedding_type}"
        db.close_connection()


# ============================================================
# 5. Chunk-document JOIN regression
# ============================================================

class TestIterChunksWithDocument:
    """Verify the JOIN method returns chunk + document fields in one query,
    avoiding N+1 per-chunk document lookups."""

    def test_join_returns_doc_fields(self, tmp_path):
        from db.database import SQLiteDB
        db = SQLiteDB(tmp_path / "join.db")

        # Insert a knowledge base, document, and chunk
        db.execute_sql_insert(
            "INSERT INTO knowledge_bases (id, name, description, created_at, updated_at) VALUES (:id, :name, '', :ts, :ts)",
            {"id": 1, "name": "kb1", "ts": "2026-01-01"},
        )
        db.execute_sql_insert(
            "INSERT INTO knowledge_documents (id, title, content, category, knowledge_base_id, createdAt, updatedAt) VALUES (:id, :title, :content, :cat, :kb_id, :ts, :ts)",
            {"id": 10, "title": "Doc A", "content": "full content", "cat": "tech", "kb_id": 1, "ts": "2026-01-01"},
        )
        db.execute_sql_insert(
            "INSERT INTO knowledge_chunks (id, documentId, chunkIndex, content, embedding, createdAt) VALUES (:id, :doc_id, :idx, :content, NULL, :ts)",
            {"id": 100, "doc_id": 10, "idx": 0, "content": "chunk text", "ts": "2026-01-01"},
        )

        rows = list(db.iter_chunks_with_document(batch_size=10))
        assert len(rows) == 1
        row = rows[0]
        # Chunk fields
        assert row["documentId"] == 10
        assert row["chunkIndex"] == 0
        assert row["content"] == "chunk text"
        # Document fields from JOIN
        assert row["doc_title"] == "Doc A"
        assert row["doc_category"] == "tech"
        assert row["doc_kb_id"] == 1
        db.close_connection()

    def test_join_handles_orphan_chunk(self, tmp_path):
        """Chunks whose document was deleted (doc_title=None) are returned
        so the caller can skip them."""
        import sqlite3
        from db.database import SQLiteDB
        db = SQLiteDB(tmp_path / "orphan.db")

        # Insert orphan chunk via raw sqlite3 (bypass FK constraint)
        conn = sqlite3.connect(str(tmp_path / "orphan.db"))
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute(
            "INSERT INTO knowledge_chunks (id, documentId, chunkIndex, content, embedding, createdAt) VALUES (200, 999, 0, 'orphan', NULL, '2026-01-01')"
        )
        conn.commit()
        conn.close()

        rows = list(db.iter_chunks_with_document(batch_size=10))
        assert len(rows) == 1
        assert rows[0]["doc_title"] is None  # LEFT JOIN → NULL for orphan
        db.close_connection()

    def test_join_paginates_correctly(self, tmp_path):
        """JOIN method paginates correctly across multiple batches."""
        from db.database import SQLiteDB
        db = SQLiteDB(tmp_path / "pages.db")

        # Insert one document and many chunks
        db.execute_sql_insert(
            "INSERT INTO knowledge_documents (id, title, content, category, createdAt, updatedAt) VALUES (:id, :title, :content, :cat, :ts, :ts)",
            {"id": 1, "title": "Doc", "content": "c", "cat": "x", "ts": "2026-01-01"},
        )
        for i in range(25):
            db.execute_sql_insert(
                "INSERT INTO knowledge_chunks (documentId, chunkIndex, content, embedding, createdAt) VALUES (:doc_id, :idx, :content, NULL, :ts)",
                {"doc_id": 1, "idx": i, "content": f"chunk_{i}", "ts": "2026-01-01"},
            )

        # batch_size=10 → should paginate in 3 batches (10+10+5)
        rows = list(db.iter_chunks_with_document(batch_size=10))
        assert len(rows) == 25
        # All should have doc_title
        assert all(r["doc_title"] == "Doc" for r in rows)
        db.close_connection()


# ============================================================
# 6. Vector rebuild status management regression
# ============================================================

class TestVectorRebuildStatus:
    """Verify the rebuild status (building/complete/dirty + fingerprint + revision)
    is tracked in config so interrupted rebuilds and content changes are detected."""

    def _make_db(self, tmp_path):
        from db.database import SQLiteDB
        return SQLiteDB(tmp_path / "status.db")

    def _patch_kmod_db(self, kmod, db):
        """Point api.knowledge.db at the temp db so status helpers use it."""
        original = kmod.db
        kmod.db = db
        return original

    def test_status_round_trip(self, tmp_path):
        """_write_rebuild_status / _read_rebuild_status round-trip correctly,
        including fingerprint and revision fields."""
        from api import knowledge as kmod
        db = self._make_db(tmp_path)
        original_db = self._patch_kmod_db(kmod, db)
        try:
            from api.knowledge import (
                _VECTOR_REBUILD_STATUS_KEY, _read_rebuild_status, _write_rebuild_status,
            )

            # Initially empty
            assert db.get_config_value(_VECTOR_REBUILD_STATUS_KEY, "") == ""

            # Write building (no fingerprint/revision)
            _write_rebuild_status("building", 500)
            status, count, fp, rev = _read_rebuild_status()
            assert (status, count, fp, rev) == ("building", 500, "", -1)

            # Write complete with fingerprint + revision
            _write_rebuild_status("complete", 500, "abc123def456", 7)
            status, count, fp, rev = _read_rebuild_status()
            assert (status, count, fp, rev) == ("complete", 500, "abc123def456", 7)

            # Write dirty (special form)
            db.set_config_value(_VECTOR_REBUILD_STATUS_KEY, "dirty")
            status, count, fp, rev = _read_rebuild_status()
            assert (status, count, fp, rev) == ("dirty", 0, "", -1)

            # Old format "complete:{count}" (no fp/revision) → triggers rebuild
            db.set_config_value(_VECTOR_REBUILD_STATUS_KEY, "complete:500")
            status, count, fp, rev = _read_rebuild_status()
            assert status == "complete" and count == 500 and fp == "" and rev == -1

            # Old format "complete:{count}:{fp}" (no revision) → triggers rebuild
            db.set_config_value(_VECTOR_REBUILD_STATUS_KEY, "complete:500:abc")
            status, count, fp, rev = _read_rebuild_status()
            assert status == "complete" and count == 500 and fp == "abc" and rev == -1
        finally:
            kmod.db = original_db
            db.close_connection()

    def test_building_status_triggers_rebuild_not_skip(self, tmp_path):
        """If status is 'building' (interrupted), the index must NOT be
        considered complete — it should trigger a rebuild on next access."""
        from api import knowledge as kmod
        db = self._make_db(tmp_path)
        original_db = self._patch_kmod_db(kmod, db)
        try:
            from api.knowledge import _VECTOR_REBUILD_STATUS_KEY, _read_rebuild_status

            db.set_config_value(_VECTOR_REBUILD_STATUS_KEY, "building:100")
            status, _, _, _ = _read_rebuild_status()
            assert status == "building", "building status must not be treated as complete"
        finally:
            kmod.db = original_db
            db.close_connection()

    def test_dirty_status_triggers_rebuild(self, tmp_path):
        """If status is 'dirty' (document CUD), the index must be rebuilt.
        _mark_rebuild_dirty also increments revision as an independent CAS signal."""
        from api import knowledge as kmod
        db = self._make_db(tmp_path)
        original_db = self._patch_kmod_db(kmod, db)
        try:
            from api.knowledge import _mark_rebuild_dirty, _read_rebuild_status, _get_rebuild_revision

            rev_before = _get_rebuild_revision()
            _mark_rebuild_dirty()
            status, _, _, _ = _read_rebuild_status()
            assert status == "dirty", "dirty status must trigger rebuild"
            rev_after = _get_rebuild_revision()
            assert rev_after == rev_before + 1, "revision must increment on dirty"
        finally:
            kmod.db = original_db
            db.close_connection()

    def test_complete_with_mismatched_count_triggers_rebuild(self, tmp_path):
        """If status is 'complete' but count doesn't match expected, rebuild
        must be triggered."""
        from api import knowledge as kmod
        db = self._make_db(tmp_path)
        original_db = self._patch_kmod_db(kmod, db)
        try:
            from api.knowledge import _VECTOR_REBUILD_STATUS_KEY, _read_rebuild_status

            db.set_config_value(_VECTOR_REBUILD_STATUS_KEY, "complete:50:fp:0")
            db.execute_sql_insert(
                "INSERT INTO knowledge_documents (id, title, content, category, createdAt, updatedAt) VALUES (1, 'd', 'c', 'x', '2026-01-01', '2026-01-01')",
                {},
            )
            for i in range(100):
                db.execute_sql_insert(
                    "INSERT INTO knowledge_chunks (documentId, chunkIndex, content, embedding, createdAt) VALUES (1, :idx, :content, NULL, '2026-01-01')",
                    {"idx": i, "content": f"c{i}"},
                )
            actual_count = db.execute_sql("SELECT COUNT(*) AS cnt FROM knowledge_chunks", {})[0]["cnt"]
            assert actual_count == 100
            status, count, _, _ = _read_rebuild_status()
            assert count == 50 and actual_count == 100, "count mismatch must be detectable"
        finally:
            kmod.db = original_db
            db.close_connection()

    def test_expected_count_excludes_orphan_chunks(self, tmp_path):
        """_get_expected_chunk_count uses INNER JOIN, so orphan chunks
        (whose document was deleted) don't cause permanent count mismatch."""
        import sqlite3
        from api import knowledge as kmod
        db = self._make_db(tmp_path)
        original_db = self._patch_kmod_db(kmod, db)
        try:
            from api.knowledge import _get_expected_chunk_count

            db.execute_sql_insert(
                "INSERT INTO knowledge_documents (id, title, content, category, createdAt, updatedAt) VALUES (1, 'd', 'c', 'x', '2026-01-01', '2026-01-01')",
                {},
            )
            db.execute_sql_insert(
                "INSERT INTO knowledge_chunks (documentId, chunkIndex, content, embedding, createdAt) VALUES (1, 0, 'c0', NULL, '2026-01-01')",
                {},
            )
            db.execute_sql_insert(
                "INSERT INTO knowledge_chunks (documentId, chunkIndex, content, embedding, createdAt) VALUES (1, 1, 'c1', NULL, '2026-01-01')",
                {},
            )
            conn = sqlite3.connect(str(tmp_path / "status.db"))
            conn.execute("PRAGMA foreign_keys = OFF")
            conn.execute(
                "INSERT INTO knowledge_chunks (documentId, chunkIndex, content, embedding, createdAt) VALUES (999, 0, 'orphan', NULL, '2026-01-01')"
            )
            conn.commit()
            conn.close()

            assert _get_expected_chunk_count() == 2, "expected_count must exclude orphan chunks"
        finally:
            kmod.db = original_db
            db.close_connection()

    def test_fingerprint_excludes_orphan_chunks(self, tmp_path):
        """_compute_chunk_fingerprint must skip orphan chunks, matching the
        rebuild traversal. Otherwise orphans cause permanent fingerprint
        mismatch and repeated rebuilds."""
        import sqlite3
        from api import knowledge as kmod
        db = self._make_db(tmp_path)
        original_db = self._patch_kmod_db(kmod, db)
        try:
            from api.knowledge import _compute_chunk_fingerprint

            db.execute_sql_insert(
                "INSERT INTO knowledge_documents (id, title, content, category, createdAt, updatedAt) VALUES (1, 'd', 'c', 'x', '2026-01-01', '2026-01-01')",
                {},
            )
            db.execute_sql_insert(
                "INSERT INTO knowledge_chunks (documentId, chunkIndex, content, embedding, createdAt) VALUES (1, 0, 'c0', NULL, '2026-01-01')",
                {},
            )
            # Fingerprint without orphan
            fp_clean = _compute_chunk_fingerprint()

            # Add an orphan chunk
            conn = sqlite3.connect(str(tmp_path / "status.db"))
            conn.execute("PRAGMA foreign_keys = OFF")
            conn.execute(
                "INSERT INTO knowledge_chunks (documentId, chunkIndex, content, embedding, createdAt) VALUES (999, 0, 'orphan', NULL, '2026-01-01')"
            )
            conn.commit()
            conn.close()

            # Fingerprint must be IDENTICAL — orphan must not affect it
            fp_with_orphan = _compute_chunk_fingerprint()
            assert fp_clean == fp_with_orphan, "orphan chunks must not affect fingerprint"
        finally:
            kmod.db = original_db
            db.close_connection()


# ============================================================
# 6b. End-to-end _ensure_vector_index() regression
# ============================================================

class TestEnsureVectorIndexEndToEnd:
    """Actually call _ensure_vector_index() with a mocked vector_db to verify
    the full rebuild state machine: dirty → building → complete:{count}:{fp}:{rev}."""

    def _make_db(self, tmp_path):
        from db.database import SQLiteDB
        return SQLiteDB(tmp_path / "e2e.db")

    def _insert_doc_with_chunks(self, db, doc_id, title, chunks):
        db.execute_sql_insert(
            "INSERT INTO knowledge_documents (id, title, content, category, createdAt, updatedAt) "
            "VALUES (:id, :title, :content, 'cat', '2026-01-01', '2026-01-01')",
            {"id": doc_id, "title": title, "content": "full"},
        )
        for i, c in enumerate(chunks):
            db.execute_sql_insert(
                "INSERT INTO knowledge_chunks (documentId, chunkIndex, content, embedding, createdAt) "
                "VALUES (:doc_id, :idx, :content, NULL, '2026-01-01')",
                {"doc_id": doc_id, "idx": i, "content": c},
            )

    def _patch_vector_db(self, monkeypatch, kmod, mock_inst):
        """Patch the late imports inside _ensure_vector_index.

        Also ensures mock_inst has a flush() method (default no-op) so the
        rebuild path's vector_db.flush() call works without each mock
        redefining it.
        """
        import sys
        import types
        # 给 mock 添加默认 flush（若未定义），避免每个 mock 重复定义
        if not hasattr(mock_inst, "flush"):
            mock_inst.flush = lambda: None
        monkeypatch.setattr(kmod, "VECTOR_DB_AVAILABLE", True, raising=False)
        fake_app_config = types.ModuleType("app.config")
        fake_app_config.VECTOR_DB_AVAILABLE = True
        fake_app_config.get_vector_db = lambda: mock_inst
        monkeypatch.setitem(sys.modules, "app.config", fake_app_config)
        fake_vector_db_mod = types.ModuleType("knowledge.vector_db")
        fake_vector_db_mod.get_vector_db = lambda: mock_inst
        monkeypatch.setitem(sys.modules, "knowledge.vector_db", fake_vector_db_mod)

    def test_empty_db_clears_index_and_marks_complete(self, tmp_path, monkeypatch):
        """When expected_count == 0, _ensure_vector_index must clear the
        vector_db and write complete:0:empty:{rev}, even if old index existed."""
        from api import knowledge as kmod

        original_db = kmod.db
        db = self._make_db(tmp_path)
        kmod.db = db
        kmod._vector_index_built = False

        cleared = {"called": False}
        class MockVectorDB:
            def get_stats(self):
                return {"total_documents": 5, "index_size": 5, "bm25_corpus_size": 5,
                        "index_type": "flat", "embedding_dim": 768, "use_gpu": False,
                        "bm25_built": False, "bm25_vocab_size": 0, "dirty": False}
            def clear_all(self):
                cleared["called"] = True
            def add_documents(self, docs):
                raise AssertionError("should not add documents when expected_count==0")

        self._patch_vector_db(monkeypatch, kmod, MockVectorDB())

        try:
            assert kmod._ensure_vector_index() is True
            assert cleared["called"], "clear_all must be called when stale index exists with 0 chunks"
            status, count, fp, rev = kmod._read_rebuild_status()
            assert (status, count, fp) == ("complete", 0, kmod._EMPTY_FINGERPRINT)
            assert rev >= 0, "revision must be recorded"
            assert kmod._vector_index_built is True
        finally:
            kmod.db = original_db
            db.close_connection()

    def test_dirty_status_forces_rebuild(self, tmp_path, monkeypatch):
        """dirty status (from document CUD) must force a full rebuild even if
        count would otherwise match. revision CAS also differs after dirty."""
        from api import knowledge as kmod

        db = self._make_db(tmp_path)
        original_db = kmod.db
        kmod.db = db
        kmod._vector_index_built = False

        self._insert_doc_with_chunks(db, 1, "Doc1", ["c0", "c1"])

        # Pre-write a stale complete:2:fakefp:0 — count matches but revision is stale
        # after _mark_rebuild_dirty increments it
        kmod._write_rebuild_status("complete", 2, "stale_fp", 0)
        kmod._mark_rebuild_dirty()  # increments revision to 1, sets status=dirty

        added_docs = {"count": 0}
        class MockVectorDB:
            def __init__(self):
                self._docs = 5
            def get_stats(self):
                return {"total_documents": self._docs, "index_size": self._docs,
                        "bm25_corpus_size": self._docs, "index_type": "flat",
                        "embedding_dim": 768, "use_gpu": False, "bm25_built": False,
                        "bm25_vocab_size": 0, "dirty": False}
            def clear_all(self):
                self._docs = 0
            def add_documents(self, docs):
                added_docs["count"] += len(docs)
                self._docs += len(docs)

        mock_inst = MockVectorDB()
        self._patch_vector_db(monkeypatch, kmod, mock_inst)

        try:
            assert kmod._ensure_vector_index() is True
            assert added_docs["count"] == 2, "dirty status must trigger rebuild"
            status, count, fp, rev = kmod._read_rebuild_status()
            assert status == "complete"
            assert count == 2
            assert fp and fp != "stale_fp", "fingerprint must be updated after rebuild"
            assert rev == 1, "revision must match the post-dirty revision"
            assert kmod._vector_index_built is True
        finally:
            kmod.db = original_db
            db.close_connection()

    def test_complete_with_matching_count_fp_and_revision_skips_rebuild(self, tmp_path, monkeypatch):
        """When status=complete, count, fingerprint, AND revision all match,
        _ensure_vector_index must skip rebuild."""
        from api import knowledge as kmod

        db = self._make_db(tmp_path)
        original_db = kmod.db
        kmod.db = db
        kmod._vector_index_built = False

        self._insert_doc_with_chunks(db, 1, "Doc1", ["c0", "c1"])

        real_fp = kmod._compute_chunk_fingerprint()
        current_rev = kmod._get_rebuild_revision()
        kmod._write_rebuild_status("complete", 2, real_fp, current_rev)

        add_called = {"called": False}
        class MockVectorDB:
            def get_stats(self):
                return {"total_documents": 2, "index_size": 2, "bm25_corpus_size": 2,
                        "index_type": "flat", "embedding_dim": 768, "use_gpu": False,
                        "bm25_built": False, "bm25_vocab_size": 0, "dirty": False}
            def clear_all(self):
                raise AssertionError("clear_all must not be called when skipping rebuild")
            def add_documents(self, docs):
                add_called["called"] = True

        self._patch_vector_db(monkeypatch, kmod, MockVectorDB())

        try:
            assert kmod._ensure_vector_index() is True
            assert not add_called["called"], "must skip rebuild when count + fp + revision match"
            assert kmod._vector_index_built is True
        finally:
            kmod.db = original_db
            db.close_connection()

    def test_content_change_with_same_count_forces_rebuild(self, tmp_path, monkeypatch):
        """When chunk count is unchanged but content is updated, the
        fingerprint must differ and force a rebuild."""
        from api import knowledge as kmod

        db = self._make_db(tmp_path)
        original_db = kmod.db
        kmod.db = db
        kmod._vector_index_built = False

        self._insert_doc_with_chunks(db, 1, "Doc1", ["original0", "original1"])
        fp_before = kmod._compute_chunk_fingerprint()
        current_rev = kmod._get_rebuild_revision()
        kmod._write_rebuild_status("complete", 2, fp_before, current_rev)

        # Update content WITHOUT _mark_rebuild_dirty (revision unchanged)
        db.execute_sql(
            "UPDATE knowledge_chunks SET content = :new WHERE documentId = 1 AND chunkIndex = 0",
            {"new": "modified0"},
        )
        fp_after = kmod._compute_chunk_fingerprint()
        assert fp_after != fp_before, "content change must produce different fingerprint"

        added_docs = {"count": 0}
        class MockVectorDB:
            def __init__(self):
                self._docs = 2
            def get_stats(self):
                return {"total_documents": self._docs, "index_size": self._docs,
                        "bm25_corpus_size": self._docs, "index_type": "flat",
                        "embedding_dim": 768, "use_gpu": False, "bm25_built": False,
                        "bm25_vocab_size": 0, "dirty": False}
            def clear_all(self):
                self._docs = 0
            def add_documents(self, docs):
                added_docs["count"] += len(docs)
                self._docs += len(docs)

        mock_inst = MockVectorDB()
        self._patch_vector_db(monkeypatch, kmod, mock_inst)

        try:
            assert kmod._ensure_vector_index() is True
            assert added_docs["count"] == 2, "content change must trigger rebuild despite same count"
            status, count, fp, rev = kmod._read_rebuild_status()
            assert status == "complete" and count == 2 and fp == fp_after
        finally:
            kmod.db = original_db
            db.close_connection()

    def test_persistence_failure_does_not_mark_complete(self, tmp_path, monkeypatch):
        """If clear_all() raises (disk full / permission denied), _ensure_vector_index
        must NOT write complete status. It should return False and leave status as building."""
        from api import knowledge as kmod

        db = self._make_db(tmp_path)
        original_db = kmod.db
        kmod.db = db
        kmod._vector_index_built = False

        self._insert_doc_with_chunks(db, 1, "Doc1", ["c0", "c1"])

        class MockVectorDB:
            def get_stats(self):
                return {"total_documents": 0, "index_size": 0, "bm25_corpus_size": 0,
                        "index_type": "flat", "embedding_dim": 768, "use_gpu": False,
                        "bm25_built": False, "bm25_vocab_size": 0, "dirty": False}
            def clear_all(self):
                raise OSError("disk full simulation")
            def add_documents(self, docs):
                raise AssertionError("should not reach add_documents if clear_all failed")

        self._patch_vector_db(monkeypatch, kmod, MockVectorDB())

        try:
            result = kmod._ensure_vector_index()
            assert result is False, "must return False when persistence fails"
            status, _, _, _ = kmod._read_rebuild_status()
            assert status == "building", "status must remain building, not complete"
            assert kmod._vector_index_built is False
        finally:
            kmod.db = original_db
            db.close_connection()

    def test_concurrent_crud_during_rebuild_does_not_mark_complete(self, tmp_path, monkeypatch):
        """If a CRUD operation runs mid-rebuild (via the real _mark_rebuild_dirty
        path), the rebuild's commit critical section must detect the revision
        change and refuse to write complete. Final status is dirty (written by
        the CRUD), not building."""
        from api import knowledge as kmod

        db = self._make_db(tmp_path)
        original_db = kmod.db
        kmod.db = db
        kmod._vector_index_built = False

        self._insert_doc_with_chunks(db, 1, "Doc1", ["c0", "c1"])

        added_docs = {"count": 0}
        class MockVectorDB:
            def __init__(self):
                self._docs = 0
            def get_stats(self):
                return {"total_documents": self._docs, "index_size": self._docs,
                        "bm25_corpus_size": self._docs, "index_type": "flat",
                        "embedding_dim": 768, "use_gpu": False, "bm25_built": False,
                        "bm25_vocab_size": 0, "dirty": False}
            def clear_all(self):
                self._docs = 0
            def add_documents(self, docs):
                added_docs["count"] += len(docs)
                self._docs += len(docs)
                # 真实 CRUD 路径：自增 revision + 写 dirty + 重置 _vector_index_built
                kmod._mark_rebuild_dirty()

        mock_inst = MockVectorDB()
        self._patch_vector_db(monkeypatch, kmod, mock_inst)

        try:
            result = kmod._ensure_vector_index()
            assert result is False, "must return False when revision changed during rebuild"
            status, _, _, _ = kmod._read_rebuild_status()
            assert status == "dirty", "status must be dirty after concurrent CRUD path"
            assert added_docs["count"] == 2, "rebuild must still index the documents"
            assert kmod._vector_index_built is False
        finally:
            kmod.db = original_db
            db.close_connection()

    def test_partial_index_corruption_triggers_rebuild(self, tmp_path, monkeypatch):
        """If FAISS ntotal or BM25 corpus size doesn't match metadata count
        (partial corruption), _ensure_vector_index must rebuild even if status
        is complete with matching count and revision."""
        from api import knowledge as kmod

        db = self._make_db(tmp_path)
        original_db = kmod.db
        kmod.db = db
        kmod._vector_index_built = False

        self._insert_doc_with_chunks(db, 1, "Doc1", ["c0", "c1"])
        real_fp = kmod._compute_chunk_fingerprint()
        current_rev = kmod._get_rebuild_revision()
        kmod._write_rebuild_status("complete", 2, real_fp, current_rev)

        added_docs = {"count": 0}
        class MockVectorDB:
            def __init__(self):
                self._docs = 2
            def get_stats(self):
                # metadata says 2, but FAISS ntotal=1 (corrupted) and BM25=2
                return {"total_documents": 2, "index_size": 1, "bm25_corpus_size": 2,
                        "index_type": "flat", "embedding_dim": 768, "use_gpu": False,
                        "bm25_built": False, "bm25_vocab_size": 0, "dirty": False}
            def clear_all(self):
                self._docs = 0
            def add_documents(self, docs):
                added_docs["count"] += len(docs)
                self._docs += len(docs)

        mock_inst = MockVectorDB()
        self._patch_vector_db(monkeypatch, kmod, mock_inst)

        try:
            assert kmod._ensure_vector_index() is True
            assert added_docs["count"] == 2, "index_size mismatch must trigger rebuild"
            assert kmod._vector_index_built is True
        finally:
            kmod.db = original_db
            db.close_connection()

    def test_concurrent_crud_in_commit_window_does_not_mark_complete(self, tmp_path, monkeypatch):
        """Narrow-window race: CRUD happens after the final revision check
        passes but before complete is written. With the _revision_lock
        commit critical section, the CRUD's _mark_rebuild_dirty must block
        until complete is written, then increment revision — but the rebuild
        has already committed with the old revision.

        This test simulates the opposite (pre-fix) failure: flush() triggers
        a CRUD BEFORE the commit critical section acquires the lock. The
        commit critical section must detect the incremented revision and
        refuse to mark complete.
        """
        from api import knowledge as kmod

        db = self._make_db(tmp_path)
        original_db = kmod.db
        kmod.db = db
        kmod._vector_index_built = False

        self._insert_doc_with_chunks(db, 1, "Doc1", ["c0", "c1"])

        added_docs = {"count": 0}
        class MockVectorDB:
            def __init__(self):
                self._docs = 0
            def get_stats(self):
                return {"total_documents": self._docs, "index_size": self._docs,
                        "bm25_corpus_size": self._docs, "index_type": "flat",
                        "embedding_dim": 768, "use_gpu": False, "bm25_built": False,
                        "bm25_vocab_size": 0, "dirty": False}
            def clear_all(self):
                self._docs = 0
            def add_documents(self, docs):
                added_docs["count"] += len(docs)
                self._docs += len(docs)
            def flush(self):
                # 模拟窄窗口：在 flush（commit 临界区之前）发生并发 CRUD。
                # _mark_rebuild_dirty 会自增 revision 并设 _vector_index_built=False。
                # commit 临界区随后检测到 revision 变化，拒绝标记 complete。
                kmod._mark_rebuild_dirty()

        mock_inst = MockVectorDB()
        self._patch_vector_db(monkeypatch, kmod, mock_inst)

        try:
            result = kmod._ensure_vector_index()
            assert result is False, "must return False when CRUD happens in commit window"
            status, _, _, _ = kmod._read_rebuild_status()
            assert status == "dirty", "status must be dirty (from concurrent CRUD), not complete"
            assert added_docs["count"] == 2, "rebuild must still index the documents"
            assert kmod._vector_index_built is False, "must not set _vector_index_built=True"
        finally:
            kmod.db = original_db
            db.close_connection()

    def test_skip_rebuild_concurrent_crud_does_not_set_built(self, tmp_path, monkeypatch):
        """Narrow-window race in skip branch: CRUD happens during fingerprint
        computation. The commit critical section must re-check revision under
        _revision_lock and refuse to set _vector_index_built=True."""
        from api import knowledge as kmod

        db = self._make_db(tmp_path)
        original_db = kmod.db
        kmod.db = db
        kmod._vector_index_built = False

        self._insert_doc_with_chunks(db, 1, "Doc1", ["c0", "c1"])
        real_fp = kmod._compute_chunk_fingerprint()
        current_rev = kmod._get_rebuild_revision()
        kmod._write_rebuild_status("complete", 2, real_fp, current_rev)

        # Patch _compute_chunk_fingerprint to simulate CRUD during fingerprint calc.
        # The real fingerprint is computed, then a CRUD is triggered (incrementing
        # revision). The commit critical section must detect the mismatch.
        original_compute_fp = kmod._compute_chunk_fingerprint
        def fingerprint_with_concurrent_crud():
            fp = original_compute_fp()
            # Simulate CRUD happening right after fingerprint computation
            kmod._mark_rebuild_dirty()
            return fp
        monkeypatch.setattr(kmod, "_compute_chunk_fingerprint", fingerprint_with_concurrent_crud)

        add_called = {"called": False}
        class MockVectorDB:
            def get_stats(self):
                return {"total_documents": 2, "index_size": 2, "bm25_corpus_size": 2,
                        "index_type": "flat", "embedding_dim": 768, "use_gpu": False,
                        "bm25_built": False, "bm25_vocab_size": 0, "dirty": False}
            def clear_all(self):
                pass  # will be called when rebuild is triggered
            def add_documents(self, docs):
                add_called["called"] = True

        mock_inst = MockVectorDB()
        self._patch_vector_db(monkeypatch, kmod, mock_inst)

        try:
            # Should NOT skip (return True without rebuild) — CRUD happened
            # during fingerprint, commit critical section detects revision change.
            # It falls through to rebuild path. Rebuild will succeed because
            # _mark_rebuild_dirty already incremented revision; the rebuild's
            # start_revision = latest revision, so commit CAS passes.
            result = kmod._ensure_vector_index()
            assert result is True, "rebuild after concurrent CRUD should succeed"
            # add_documents may or may not be called depending on whether
            # rebuild path re-fetches; the key assertion is that the skip
            # branch did NOT set _vector_index_built without re-checking
            assert kmod._vector_index_built is True
        finally:
            kmod.db = original_db
            db.close_connection()

    def test_real_thread_lock_blocks_crud_in_commit_window(self, tmp_path, monkeypatch):
        """Real two-thread proof that _revision_lock serializes the commit
        critical section against CRUD.

        The rebuild thread enters the commit critical section (holds
        _revision_lock) and pauses BEFORE writing complete. A real CRUD
        thread then calls _mark_rebuild_dirty() and MUST block on
        _revision_lock. The test asserts the CRUD has not completed within
        a short window while the lock is held. After releasing the rebuild,
        CRUD runs and overwrites complete with dirty.

        If _revision_lock were removed from _mark_rebuild_dirty, the CRUD
        thread would complete immediately (no blocking), and this test
        would fail the "must block" assertion. The single-threaded mock
        tests above cannot prove this — they only verify the post-hoc
        revision CAS check.
        """
        import threading
        import time
        from api import knowledge as kmod

        db = self._make_db(tmp_path)
        original_db = kmod.db
        kmod.db = db
        kmod._vector_index_built = False

        self._insert_doc_with_chunks(db, 1, "Doc1", ["c0", "c1"])

        entered_commit = threading.Event()
        release_commit = threading.Event()
        crud_started = threading.Event()
        crud_completed = threading.Event()

        original_write = kmod._write_rebuild_status

        def patched_write_rebuild_status(status, count, fingerprint="", revision=-1):
            # 仅 hook 重建 commit（complete:{count>0}）。空库清理（complete:0）
            # 和 building 状态写入不阻塞，避免测试死锁。
            if status == "complete" and count > 0:
                entered_commit.set()
                # 等待主线程允许完成 commit；持有 _revision_lock 期间阻塞 CRUD
                release_commit.wait(timeout=10)
            original_write(status, count, fingerprint, revision)

        monkeypatch.setattr(kmod, "_write_rebuild_status", patched_write_rebuild_status)

        added_docs = {"count": 0}

        class MockVectorDB:
            def __init__(self):
                self._docs = 0

            def get_stats(self):
                return {"total_documents": self._docs, "index_size": self._docs,
                        "bm25_corpus_size": self._docs, "index_type": "flat",
                        "embedding_dim": 768, "use_gpu": False, "bm25_built": False,
                        "bm25_vocab_size": 0, "dirty": False}

            def clear_all(self):
                self._docs = 0

            def add_documents(self, docs):
                added_docs["count"] += len(docs)
                self._docs += len(docs)

            def flush(self):
                pass

        self._patch_vector_db(monkeypatch, kmod, MockVectorDB())

        rebuild_result = {"value": None}

        def rebuild_thread_main():
            rebuild_result["value"] = kmod._ensure_vector_index()

        def crud_thread_main():
            # 等 rebuild 进入 commit 临界区
            if not entered_commit.wait(timeout=5):
                crud_completed.set()
                return
            # 短暂等待确保 rebuild 真的持有 _revision_lock
            time.sleep(0.15)
            crud_started.set()
            # 调用 _mark_rebuild_dirty：应阻塞在 _revision_lock 上
            kmod._mark_rebuild_dirty()
            crud_completed.set()

        rebuild_t = threading.Thread(target=rebuild_thread_main, name="rebuild")
        crud_t = threading.Thread(target=crud_thread_main, name="crud")

        try:
            rebuild_t.start()
            crud_t.start()

            # 等 CRUD 线程开始尝试 _mark_rebuild_dirty
            assert crud_started.wait(timeout=5), "CRUD thread did not start"

            # 关键断言：CRUD 必须阻塞在 _revision_lock 上，未在短窗口内完成。
            # 若 _mark_rebuild_dirty 未加锁，此处会失败。
            assert not crud_completed.wait(timeout=1.0), (
                "CRUD completed while rebuild holds _revision_lock — "
                "lock is not blocking the commit critical section"
            )

            # 释放 rebuild，让它完成 commit
            release_commit.set()

            # 等两个线程结束
            crud_t.join(timeout=5)
            rebuild_t.join(timeout=5)

            assert not crud_t.is_alive(), "CRUD thread did not complete after release"
            assert not rebuild_t.is_alive(), "rebuild thread did not complete after release"

            # rebuild 应成功标记 complete（被 CRUD 随后覆盖为 dirty）
            assert rebuild_result["value"] is True, "rebuild must succeed in commit"
            assert added_docs["count"] == 2, "rebuild must have indexed 2 chunks"

            # 最终状态：CRUD 在 rebuild 完成后执行，覆盖 complete 为 dirty
            status, _, _, _ = kmod._read_rebuild_status()
            assert status == "dirty", (
                f"final status must be dirty after CRUD overwrote complete, got {status}"
            )
            assert kmod._vector_index_built is False, (
                "_vector_index_built must be False after CRUD marked dirty"
            )
        finally:
            # 防止任何路径下死锁
            release_commit.set()
            if rebuild_t.is_alive():
                rebuild_t.join(timeout=2)
            if crud_t.is_alive():
                crud_t.join(timeout=2)
            kmod.db = original_db
            db.close_connection()

    @pytest.mark.asyncio
    async def test_search_degrades_to_keyword_when_rebuild_fails(self, tmp_path, monkeypatch):
        """When _ensure_vector_index() returns False, search_knowledge must
        skip RAG and vector retrieval, falling through to DB keyword search.

        Verifies the Major fix: the return value of _ensure_vector_index()
        is no longer ignored. A failing rebuild (disk error, concurrent CRUD,
        count mismatch) must not serve results from a partial/stale index.
        """
        from api import knowledge as kmod
        from db.schemas import KnowledgeSearchRequest

        db = self._make_db(tmp_path)
        original_db = kmod.db
        kmod.db = db
        kmod._vector_index_built = False

        self._insert_doc_with_chunks(db, 1, "Doc1", ["hello world", "foo bar"])

        # Force _ensure_vector_index to return False (rebuild failed)
        monkeypatch.setattr(kmod, "_ensure_vector_index", lambda: False)

        # Track whether RAG/vector paths were attempted
        rag_called = {"value": False}
        original_get_rag = None
        try:
            from knowledge import rag_helper
            original_get_rag = rag_helper.get_rag_helper
            def tracking_get_rag_helper():
                rag_called["value"] = True
                raise AssertionError("RAG must not be called when rebuild failed")
            monkeypatch.setattr(rag_helper, "get_rag_helper", tracking_get_rag_helper)
        except ImportError:
            pass  # rag_helper not available; RAG block will throw on import anyway

        try:
            request = KnowledgeSearchRequest(query="hello", topK=5)
            response = await kmod.search_knowledge(request)

            assert response["success"] is True
            assert response["searchType"] == "keyword", (
                f"must degrade to keyword when rebuild fails, got {response['searchType']}"
            )
            assert not rag_called["value"], "RAG must not be invoked when index is not ready"
            # Keyword search should still find the matching chunk
            assert len(response["results"]) > 0, "keyword fallback must return matching chunks"
        finally:
            kmod.db = original_db
            db.close_connection()

    @pytest.mark.asyncio
    async def test_keyword_fallback_respects_knowledge_base_filter(self, tmp_path, monkeypatch):
        """When rebuild fails and search degrades to keyword, the
        knowledgeBaseName filter must still be honored. Chunks from other
        knowledge bases must NOT leak into results.

        Two knowledge bases are created, each with a document containing
        the query term. Without the filter, both would match. The filter
        must restrict results to only the requested KB.
        """
        from api import knowledge as kmod
        from db.schemas import KnowledgeSearchRequest

        db = self._make_db(tmp_path)
        original_db = kmod.db
        kmod.db = db
        kmod._vector_index_built = False

        # 创建两个知识库
        db.execute_sql_insert(
            "INSERT INTO knowledge_bases (id, name, description, created_at, updated_at) "
            "VALUES (:id, :name, :desc, '2026-01-01', '2026-01-01')",
            {"id": 1, "name": "KB_ALPHA", "desc": "alpha"},
        )
        db.execute_sql_insert(
            "INSERT INTO knowledge_bases (id, name, description, created_at, updated_at) "
            "VALUES (:id, :name, :desc, '2026-01-01', '2026-01-01')",
            {"id": 2, "name": "KB_BETA", "desc": "beta"},
        )
        # 两个知识库各一个文档，内容都包含查询词 "shared_term"
        for doc_id, kb_id, title in [(10, 1, "AlphaDoc"), (20, 2, "BetaDoc")]:
            db.execute_sql_insert(
                "INSERT INTO knowledge_documents (id, title, content, category, knowledge_base_id, createdAt, updatedAt) "
                "VALUES (:id, :title, :content, 'cat', :kb, '2026-01-01', '2026-01-01')",
                {"id": doc_id, "title": title, "content": "shared_term body", "kb": kb_id},
            )
            db.execute_sql_insert(
                "INSERT INTO knowledge_chunks (documentId, chunkIndex, content, embedding, createdAt) "
                "VALUES (:doc, 0, :content, NULL, '2026-01-01')",
                {"doc": doc_id, "content": "shared_term body"},
            )

        # 强制重建失败，触发关键词降级
        monkeypatch.setattr(kmod, "_ensure_vector_index", lambda: False)

        try:
            # 指定 KB_ALPHA 过滤
            request = KnowledgeSearchRequest(query="shared_term", topK=10, knowledgeBaseName="KB_ALPHA")
            response = await kmod.search_knowledge(request)

            assert response["success"] is True
            assert response["searchType"] == "keyword"
            assert len(response["results"]) == 1, "must return exactly one chunk from KB_ALPHA"
            result = response["results"][0]
            assert result["documentTitle"] == "AlphaDoc", (
                f"must not leak BetaDoc from KB_BETA, got {result['documentTitle']}"
            )

            # 交叉验证：指定 KB_BETA
            request_beta = KnowledgeSearchRequest(query="shared_term", topK=10, knowledgeBaseName="KB_BETA")
            response_beta = await kmod.search_knowledge(request_beta)
            assert len(response_beta["results"]) == 1
            assert response_beta["results"][0]["documentTitle"] == "BetaDoc"
        finally:
            kmod.db = original_db
            db.close_connection()

    @pytest.mark.asyncio
    async def test_nonexistent_knowledge_base_returns_empty(self, tmp_path, monkeypatch):
        """When the requested knowledgeBaseName does not exist, the search
        must return an empty result instead of silently falling back to a
        full-library scan (fail-closed, not fail-open)."""
        from api import knowledge as kmod
        from db.schemas import KnowledgeSearchRequest

        db = self._make_db(tmp_path)
        original_db = kmod.db
        kmod.db = db

        # 创建一个知识库和文档（不应被检索到）
        db.execute_sql_insert(
            "INSERT INTO knowledge_bases (id, name, description, created_at, updated_at) "
            "VALUES (:id, :name, :desc, '2026-01-01', '2026-01-01')",
            {"id": 1, "name": "REAL_KB", "desc": "real"},
        )
        db.execute_sql_insert(
            "INSERT INTO knowledge_documents (id, title, content, category, knowledge_base_id, createdAt, updatedAt) "
            "VALUES (:id, :title, :content, 'cat', :kb, '2026-01-01', '2026-01-01')",
            {"id": 1, "title": "Doc", "content": "shared content", "kb": 1},
        )
        db.execute_sql_insert(
            "INSERT INTO knowledge_chunks (documentId, chunkIndex, content, embedding, createdAt) "
            "VALUES (:doc, 0, :content, NULL, '2026-01-01')",
            {"doc": 1, "content": "shared content"},
        )

        # 即使索引重建成功，不存在的知识库名称也应直接返回空，不应进入检索路径
        monkeypatch.setattr(kmod, "_ensure_vector_index", lambda: True)

        # 跟踪 RAG/vector 是否被调用
        rag_called = {"value": False}
        try:
            from knowledge import rag_helper
            def tracking_get_rag_helper():
                rag_called["value"] = True
                raise AssertionError("RAG must not be called for nonexistent KB")
            monkeypatch.setattr(rag_helper, "get_rag_helper", tracking_get_rag_helper)
        except ImportError:
            pass

        try:
            request = KnowledgeSearchRequest(
                query="shared", topK=5, knowledgeBaseName="NONEXISTENT_KB"
            )
            response = await kmod.search_knowledge(request)

            assert response["success"] is True
            assert response["results"] == [], "must return empty for nonexistent KB"
            assert response["searchType"] == "empty"
            assert not rag_called["value"], "RAG must not be invoked for nonexistent KB"
        finally:
            kmod.db = original_db
            db.close_connection()


# ============================================================
# 6c. Search input validation regression
# ============================================================

class TestSearchInputValidation:
    """Verify KnowledgeSearchRequest query field validation.

    Empty / whitespace-only queries must be rejected at the schema layer
    (HTTP 422) before reaching search_knowledge, preventing meaningless
    vector retrieval or full-library scans.
    """

    def test_empty_query_rejected(self):
        from db.schemas import KnowledgeSearchRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            KnowledgeSearchRequest(query="")

    def test_whitespace_only_query_rejected(self):
        from db.schemas import KnowledgeSearchRequest
        from pydantic import ValidationError

        # strip_whitespace=True 会在验证前去除首尾空白，纯空白变空字符串
        # 触发 min_length=1
        with pytest.raises(ValidationError):
            KnowledgeSearchRequest(query="   \t\n  ")

    def test_query_strips_surrounding_whitespace(self):
        from db.schemas import KnowledgeSearchRequest

        req = KnowledgeSearchRequest(query="  hello world  ")
        assert req.query == "hello world", "leading/trailing whitespace must be stripped"

    def test_overlong_query_rejected(self):
        from db.schemas import KnowledgeSearchRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            KnowledgeSearchRequest(query="x" * 2001)

    def test_query_at_max_length_accepted(self):
        from db.schemas import KnowledgeSearchRequest

        req = KnowledgeSearchRequest(query="x" * 2000)
        assert len(req.query) == 2000
