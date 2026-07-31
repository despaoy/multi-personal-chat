from __future__ import annotations

import json


def test_scan_registers_multiple_adapters_with_unique_ids(tmp_path, monkeypatch) -> None:
    from api import loras
    from db import database as database_module

    for name in ("kisaki", "minamo"):
        adapter_dir = tmp_path / name
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps({"r": 16, "lora_alpha": 32}),
            encoding="utf-8",
        )

    class Database:
        def __init__(self) -> None:
            self.records = [{"id": "legacy-name", "name": "old", "status": "inactive"}]

        def get_loras(self):
            return list(self.records)

        def add_lora(self, record):
            assert all(item["id"] != record["id"] for item in self.records)
            self.records.append(dict(record))
            return record

        def execute_sql(self, _query, _params):
            return []

    database = Database()
    monkeypatch.setattr(loras, "LORA_ROOT", tmp_path)
    monkeypatch.setattr(loras, "db", database)

    result = loras._scan_loras_sync()

    added = [item for item in database.records if item["name"] in {"kisaki", "minamo"}]
    assert result["success"] is True
    assert result["new_count"] == 2
    assert [item["id"] for item in added] == ["1", "2"]
    database_module.refresh_lora_dir_map(database_module.LORA_ROOT)