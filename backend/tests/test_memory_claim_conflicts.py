"""Storage invariants for updates prepared against an obsolete memory version."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from db.database import SQLiteDB

SCOPE = dict(character_id="character", platform="test", adapter="test", sender_id="user",
             conversation_type="private", conversation_id="user")


def write(db, **kwargs):
    return db.append_character_memory_claim(
        **SCOPE, memory_type="user_fact", memory_key=kwargs.pop("memory_key", "field"),
        content=kwargs.pop("content", "fact"), **kwargs,
    )


@pytest.mark.parametrize("relation", ["SUPERSEDE", "MERGE", "RETRACT"])
@pytest.mark.parametrize("same_key", [True, False])
def test_obsolete_target_cannot_be_mutated_again(tmp_path, relation, same_key):
    db = SQLiteDB(tmp_path / "conflict.db")
    first = write(db)
    current = write(db, relation_type="SUPERSEDE", supersedes_memory_id=first["id"])
    with pytest.raises(ValueError, match="memory target changed"):
        write(db, relation_type=relation, supersedes_memory_id=first["id"],
              memory_key="field" if same_key else "other-field")
    rows = db.list_character_memory_claims(**SCOPE, include_inactive=True)
    assert len(rows) == 2
    assert [row["id"] for row in rows if row["status"] == "active"] == [current["id"]]


@pytest.mark.parametrize("relation", ["SUPERSEDE", "MERGE", "RETRACT"])
def test_older_observation_cannot_close_newer_active_fact(tmp_path, relation):
    db = SQLiteDB(tmp_path / "order.db")
    first = write(db, observed_at="2026-09-20T12:00:00+00:00")
    with pytest.raises(ValueError, match="memory observation precedes target"):
        write(db, relation_type=relation, supersedes_memory_id=first["id"],
              observed_at="2026-09-20T19:59:59+08:00")
    rows = db.list_character_memory_claims(**SCOPE, include_inactive=True)
    assert len(rows) == 1
    assert rows[0]["status"] == "active"
    assert rows[0]["valid_to"] is None


def test_concurrent_successors_have_one_winner(tmp_path):
    db = SQLiteDB(tmp_path / "parallel.db")
    first = write(db)

    def change(index):
        try:
            return write(db, relation_type="SUPERSEDE", supersedes_memory_id=first["id"],
                         memory_key=f"new-key-{index}", content=f"candidate {index}")
        except ValueError as exc:
            assert "memory target changed" in str(exc)
            return None

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(change, range(4)))
    assert sum(result is not None for result in results) == 1
    rows = db.list_character_memory_claims(**SCOPE, include_inactive=True)
    assert len(rows) == 2


def test_coexisting_fact_may_reference_historical_parent(tmp_path):
    db = SQLiteDB(tmp_path / "coexist.db")
    first = write(db)
    write(db, relation_type="SUPERSEDE", supersedes_memory_id=first["id"])
    coexist = write(db, relation_type="COEXIST", parent_memory_id=first["id"])
    assert coexist["status"] == "active"


@pytest.mark.parametrize("old_time,new_time", [
    ("2026-09-20T12:00:00+00:00", "2026-09-20T20:00:00+08:00"),
    ("2026-09-20T12:00:00+00:00", "2026-09-20T12:00:01+00:00"),
    ("2026-09-20T12:00:00", "2026-09-20T10:00:00+00:00"),
    (None, "2026-09-20T10:00:00+00:00"),
])
def test_equal_later_and_unknown_legacy_times_are_not_guessed_stale(tmp_path, old_time, new_time):
    db = SQLiteDB(tmp_path / "known-time.db")
    first = write(db, observed_at=old_time)
    current = write(db, relation_type="SUPERSEDE", supersedes_memory_id=first["id"],
                    observed_at=new_time)
    rows = db.list_character_memory_claims(**SCOPE)
    assert [row["id"] for row in rows] == [current["id"]]


@pytest.mark.parametrize("status", ["pending", "retracted", "archived"])
def test_nonactive_target_is_not_reopened_as_current_fact(tmp_path, status):
    db = SQLiteDB(tmp_path / "nonactive.db")
    first = write(db, status=status)
    with pytest.raises(ValueError, match="memory target changed"):
        write(db, relation_type="SUPERSEDE", supersedes_memory_id=first["id"])
    rows = db.list_character_memory_claims(**SCOPE, include_inactive=True)
    assert len(rows) == 1
    assert rows[0]["status"] == status
