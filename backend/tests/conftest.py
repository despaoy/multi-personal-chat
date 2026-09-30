"""Pytest configuration for the QQ assistant backend test suite."""

from __future__ import annotations

import os
import secrets
import sys
import tempfile
from pathlib import Path

# Never let unit tests migrate or write the developer's local database. PostgreSQL
# integration runs must opt in explicitly with USE_POSTGRESQL=true.
_TEST_RUNTIME_ROOT = Path(tempfile.mkdtemp(prefix="qqchat-pytest-"))
os.environ.setdefault("USE_POSTGRESQL", "false")
os.environ.setdefault("DATABASE_PATH", str(_TEST_RUNTIME_ROOT / "qq_assistant.db"))
# Importing app.config otherwise rotates backend/.env's development secret.
# Ephemeral process-local test credentials must never write application config.
# A caller-supplied short/deprecated test value also triggers auto-rotation.
# Replace only invalid test credentials before any app configuration imports.
_test_jwt_secret = os.environ.get("JWT_SECRET", "")
if len(_test_jwt_secret) < 32 or _test_jwt_secret == "multipersonal-jwt-secret-change-in-production":
    os.environ["JWT_SECRET"] = secrets.token_urlsafe(48)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
for path in (PROJECT_ROOT, BACKEND_ROOT):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

collect_ignore_glob = [
    "security_test.py",
    "fault_injection_test.py",
]
