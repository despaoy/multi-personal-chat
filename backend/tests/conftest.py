"""Pytest configuration for the QQ assistant backend test suite."""

from __future__ import annotations

import base64
import os
import secrets
import sys
import tempfile
from pathlib import Path

import pytest

# Never let unit tests migrate or write the developer's local database. PostgreSQL
# integration runs must opt in explicitly with USE_POSTGRESQL=true.
_TEST_RUNTIME_ROOT = Path(tempfile.mkdtemp(prefix="qqchat-pytest-"))
# Local synthetic dataset tests must not write into the developer's model cache.
os.environ.setdefault("HF_HOME", str(_TEST_RUNTIME_ROOT / "huggingface"))
os.environ.setdefault("USE_POSTGRESQL", "false")
os.environ.setdefault("DATABASE_PATH", str(_TEST_RUNTIME_ROOT / "qq_assistant.db"))
# Test configuration supplies an explicit process-local signing key before
# importing the application. Replace only invalid test credentials; never write
# the workspace configuration or change an already valid caller-supplied key.
_test_jwt_secret = os.environ.get("JWT_SECRET", "")
if len(_test_jwt_secret) < 32 or _test_jwt_secret == "multipersonal-jwt-secret-change-in-production":
    os.environ["JWT_SECRET"] = secrets.token_urlsafe(48)
# Test-only explicit encryption configuration; never provision workspace keys.
os.environ.setdefault("ENCRYPTION_KEY", base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii"))
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
for path in (PROJECT_ROOT, BACKEND_ROOT):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

collect_ignore_glob = [
    "fault_injection_test.py",
]



@pytest.fixture
def no_live_context_review(monkeypatch):
    """These unit tests must supply a reviewer rather than contact a provider."""
    from inference import review_client

    async def forbidden():
        pytest.fail("Unit test attempted live review; supply complete semantic controls")

    monkeypatch.setattr(review_client, "get_context_review_client", forbidden)


@pytest.fixture
def semantic_plan(monkeypatch):
    """Validate an explicitly supplied unit response through the actual planner."""
    import json

    from knowledge import retrieval_query_plan
    from knowledge.turn_dependencies import KINDS, parse_dependencies

    actual = retrieval_query_plan.plan_retrieval

    def install(query, groups):
        complete = {kind: groups.get(kind, []) for kind in KINDS}
        parse_dependencies(complete, query)
        calls = []

        async def reviewer(messages):
            payload = json.loads(messages[-1]["content"])
            assert payload["query"] == query
            calls.append(payload)
            return json.dumps(dict(dependencies=complete, public_partitions=[
                dict(segment_id=i, cuts=[]) for i in complete["public_knowledge"]
            ]))

        async def planner(message, **kwargs):
            assert message == query
            return await actual(message, reviewer=reviewer, **kwargs)

        monkeypatch.setattr(retrieval_query_plan, "plan_retrieval", planner)
        return calls

    return install


@pytest.fixture
def literal_grounded_query(monkeypatch):
    """Answer contract tests control preparation; this is not semantic evidence."""
    from knowledge import dialogue_query

    async def resolve(query, history, **kwargs):
        return dialogue_query.RetrievalQuery(query)

    monkeypatch.setattr("knowledge.grounded_answer.service.resolve_retrieval_query", resolve)


@pytest.fixture
def controlled_public_bundle(monkeypatch):
    """Pre-reviewed synthetic bundles for transport tests, not semantic evidence."""
    async def binding(*args, **kwargs):
        return None

    async def reviewed(bundle, *args, **kwargs):
        return bundle

    monkeypatch.setattr("knowledge.public_question_binding.resolve_question_binding", binding)
    monkeypatch.setattr("knowledge.public_task_evidence.review_public_candidates", reviewed)
