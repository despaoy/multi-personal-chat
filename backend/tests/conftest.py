"""Pytest configuration for the QQ assistant backend test suite."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Never let unit tests migrate or write the developer's local database. PostgreSQL
# integration runs must opt in explicitly with USE_POSTGRESQL=true.
_TEST_RUNTIME_ROOT = Path(tempfile.mkdtemp(prefix="qqchat-pytest-"))
os.environ.setdefault("USE_POSTGRESQL", "false")
os.environ.setdefault("DATABASE_PATH", str(_TEST_RUNTIME_ROOT / "qq_assistant.db"))
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