import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.mark.asyncio
async def test_services_omit_platform_connection_checks(monkeypatch, tmp_path):
    from services import stats as statistics
    from stats_fixtures import StatsRig

    rig = StatsRig(tmp_path / "service-names.sqlite")
    monkeypatch.setattr(statistics, "check_service", lambda port: False)
    async with rig.client() as client:
        response = await client.get("/api/stats/services")
    assert response.status_code == 200
    names = {item["name"] for item in response.json()["services"]}
    assert "QQ Adapter (NapCat)" not in names
    assert "NoneBot Bot" not in names
    assert not any(name.startswith("Platform ") for name in names)
