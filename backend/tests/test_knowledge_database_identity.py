"""A matching revision is meaningful only inside its database/index owner."""

from knowledge import index_lifecycle

import pytest

from api import knowledge
from db.database import SQLiteDB
from knowledge import vector_db
from test_knowledge_cross_process_freshness import RecordingVector


@pytest.mark.parametrize("replace_database,replace_vector", [(True, False), (True, True), (False, True)])
def test_rebound_database_with_equal_revision_rebuilds_its_own_evidence(tmp_path, monkeypatch, replace_database, replace_vector):
    first = SQLiteDB(tmp_path / "first.db")
    second = SQLiteDB(tmp_path / "second.db")
    original = "甲库完整规则：普通雨照常，红色暴雨停课。必须由甲库负责人书面确认。"
    expected = "乙库完整规则：普通雨与红色暴雨均停课。只接受乙库负责人书面确认。"
    try:
        for database, text in ((first, original), (second, expected)):
            database.save_knowledge_document(dict(title="课程规则", content=text), chunks=[text])
        assert first.get_config_value("vector_index_rebuild_revision") == second.get_config_value("vector_index_rebuild_revision")
        vector = RecordingVector()
        monkeypatch.setattr(knowledge, "db", first)
        monkeypatch.setattr(index_lifecycle, "db", first)
        monkeypatch.setattr(index_lifecycle, "_vector_index_revision", None)
        monkeypatch.setattr(vector_db, "get_vector_db", lambda: vector)
        assert index_lifecycle._ensure_vector_index()
        assert original in vector.metadata[0]["content"]
        selected = second if replace_database else first
        monkeypatch.setattr(knowledge, "db", selected)
        monkeypatch.setattr(index_lifecycle, "db", selected)
        if replace_vector:
            vector = RecordingVector()
        assert index_lifecycle._ensure_vector_index()
        assert len(vector.metadata) == 1
        wanted, excluded = (expected, original) if replace_database else (original, expected)
        assert wanted in vector.metadata[0]["content"]
        assert excluded not in vector.metadata[0]["content"]
        assert selected.get_config_value("vector_index_rebuild_status").startswith("complete:")
    finally:
        first.close_connection()
        second.close_connection()
