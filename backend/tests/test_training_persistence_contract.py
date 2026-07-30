from __future__ import annotations

from datetime import datetime

from db.database import SQLiteDB


def test_sqlite_training_task_timestamps_are_storage_managed(tmp_path):
    database = SQLiteDB(tmp_path / "training.db")
    try:
        created = database.add_training_task({
            "task_id": "task-1",
            "lora_name": "kisaki",
        })
        assert created["created_at"]
        assert created["updated_at"]

        previous_updated_at = created["updated_at"]
        updated = database.update_training_task(
            "task-1",
            {
                "status": "running",
                "updated_at": "caller-controlled-value",
            },
        )

        assert updated["status"] == "running"
        assert updated["updated_at"] != "caller-controlled-value"
        assert datetime.fromisoformat(updated["updated_at"])
        assert updated["updated_at"] >= previous_updated_at
    finally:
        database.close_connection()
