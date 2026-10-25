"""Complete, isolated dashboard applications; no external model or host probes."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI

from api import stats
from app.dependencies import get_current_admin
from app.runtime import RuntimeContainer
from db.database import SQLiteDB


@pytest.fixture
def fresh_observability(monkeypatch):
    from collections import defaultdict
    from infra import observability

    for name in ("_COUNTERS", "_RECENT", "_CONSECUTIVE"):
        current = getattr(observability, name)
        monkeypatch.setattr(observability, name, defaultdict(current.default_factory))
    return observability


class CountingStatsDB:
    def __init__(self, database):
        self.database = database
        self.failure = ""
        self.queries = []
        self.returned_rows = 0
        self.config_reads = 0

    @property
    def config(self):
        self.config_reads += 1
        if self.failure == "config":
            raise OSError("synthetic-private-statistics-location")
        if self.failure == "config_shape":
            return []
        if self.failure == "config_boolean":
            return {"astrbotEnabled": "synthetic-invalid-boolean"}
        return self.database.config

    def execute_sql(self, query, params):
        self.queries.append(query)
        if "ROW_NUMBER()" in query:
            stage = "quantiles"
        elif "GROUP BY SUBSTR" in query:
            stage = "activity"
        elif "GROUP BY platform" in query:
            stage = "events"
        elif "FROM model_invocations" in query:
            stage = "invocations"
        else:
            stage = "messages"
        if self.failure == stage:
            raise OSError("synthetic-private-statistics-location")
        if self.failure == "query_shape":
            return None
        if self.failure == "aggregate_shape":
            return []
        rows = self.database.execute_sql(query, params)
        self.returned_rows += len(rows)
        return rows


class StatsRig:
    def __init__(self, path, *, env=None):
        self.database = SQLiteDB(path)
        self.db = CountingStatsDB(self.database)
        self.queue = SimpleNamespace(
            stats=Mock(return_value={"queue_size": 0, "max_queue_size": 100, "workers": 2, "active": 1})
        )
        self.probe = SimpleNamespace(
            get=AsyncMock(
                return_value={
                    "ready": True,
                    "deps": {"model": True, "database": True},
                    "details": {"rag": "not_probed"},
                }
            )
        )
        self.app = FastAPI()
        self.app.state.runtime_container = RuntimeContainer(
            db=self.db,
            is_pg_mode=lambda: False,
            inference_runtime=self.queue,
            startup_env={"STATS_CACHE_TTL_SECONDS": "60", "ASTRBOT_ENABLED": "false"} if env is None else env,
        )
        self.app.state.readiness_probe = self.probe
        self.app.include_router(stats.router)
        self.app.dependency_overrides[get_current_admin] = lambda: {"role": "admin"}

    def client(self):
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://isolated.test")

    def seed(self, count=1, *, hour=11, cost=1.0):
        observed = datetime.now().replace(hour=hour, minute=1, second=0, microsecond=0).isoformat()
        for index in range(count):
            self.database.add_message(
                {
                    "sessionType": "private",
                    "sessionId": f"synthetic:{index}",
                    "platform": "web",
                    "conversationId": f"synthetic:{index}",
                    "message": "synthetic-private-dialogue-body",
                    "reply": "complete synthetic reply",
                    "costTime": cost,
                    "createdAt": observed,
                }
            )
