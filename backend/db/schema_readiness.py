"""Check that a production database has run the application's migrations."""

from pathlib import Path

from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory


def require_current_migrations(connection):
    scripts = ScriptDirectory(str(Path(__file__).resolve().parents[1] / "alembic"))
    expected = set(scripts.get_heads())
    actual = set(MigrationContext.configure(connection).get_current_heads())
    if actual != expected:
        raise RuntimeError(
            "PostgreSQL schema does not match this application version. "
            "From backend/, run python -m alembic current against the configured database "
            "and apply the required migrations before restarting."
        )
