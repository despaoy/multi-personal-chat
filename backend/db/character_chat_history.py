"""Store inherited conversation content once and keep state local to each path."""

import hashlib
import json

from .errors import RevisionConflictError

HISTORY_NODES = "character_chat_history_nodes"
HISTORY_PATH = "character_chat_history_path"
HISTORY_OWNER_INDEX = "ix_character_chat_session_owner_scope"
CREATE_HISTORY_OWNER_INDEX = (
    f"CREATE UNIQUE INDEX IF NOT EXISTS {HISTORY_OWNER_INDEX} ON character_chat_sessions (id,owner_user_id)"
)

READ_PATH = (
    f"SELECT p.position,n.payload_json FROM {HISTORY_PATH} p JOIN {HISTORY_NODES} n "
    "ON n.owner_user_id=p.owner_user_id AND n.node_key=p.node_key "
    "WHERE p.session_id=:session AND p.owner_user_id=:owner ORDER BY p.position"
)
ORPHANED_HISTORY_NODE = (
    f"NOT EXISTS (SELECT 1 FROM {HISTORY_PATH} p WHERE p.owner_user_id={HISTORY_NODES}.owner_user_id "
    f"AND p.node_key={HISTORY_NODES}.node_key)"
)


def stored_baseline(baseline):
    """Keep only the path-local state in the session snapshot."""
    result = {key: value for key, value in baseline.items() if key != "history"}
    result["history_states"] = [
        {"state_after": turn["state_after"]} if "state_after" in turn else {}
        for turn in baseline["history"]
    ]
    return json.dumps(result, ensure_ascii=False)


def materialize_baseline(baseline, rows):
    states = baseline.pop("history_states")
    if len(states) != len(rows) or any(row["position"] != position for position, row in enumerate(rows)):
        raise RevisionConflictError()
    baseline["history"] = [
        {**json.loads(row["payload_json"]), **state} for row, state in zip(rows, states, strict=True)
    ]
    return baseline


def _insert_batches(table, columns, rows, conflict=""):
    # Keep the bound-parameter count comfortably below SQLite's older 999-variable limit.
    for start in range(0, len(rows), 100):
        batch = rows[start:start + 100]
        params = {}
        values = []
        for index, row in enumerate(batch):
            names = [f"{column}_{index}" for column in columns]
            params.update(zip(names, row, strict=True))
            values.append("(" + ",".join(f":{name}" for name in names) + ")")
        yield f"INSERT INTO {table} ({','.join(columns)}) VALUES {','.join(values)}{conflict}", params, len(batch)


def create_path(owner, session_id, history):
    nodes = {}
    path = []
    for position, turn in enumerate(history):
        content = {key: value for key, value in turn.items() if key != "state_after"}
        payload = json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        node_key = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        nodes[node_key] = (owner, node_key, content.get("session_id"), payload)
        path.append((session_id, position, owner, node_key))

    # Lock reused nodes in stable order before inserting references. The payload is immutable.
    yield from _insert_batches(
        HISTORY_NODES, ("owner_user_id", "node_key", "origin_session_id", "payload_json"),
        sorted(nodes.values()),
        f" ON CONFLICT (owner_user_id,node_key) DO UPDATE SET node_key=excluded.node_key "
        f"WHERE {HISTORY_NODES}.payload_json=excluded.payload_json",
    )
    yield from _insert_batches(HISTORY_PATH, ("session_id", "position", "owner_user_id", "node_key"), path)


def migrate_baselines(execute, *, downgrade=False):
    """Translate stored paths in the caller's schema transaction, in bounded batches."""
    after = ""
    while rows := execute(
        "SELECT id,owner_user_id,baseline_json FROM character_chat_sessions WHERE id>:after ORDER BY id LIMIT 25",
        {"after": after},
    ).fetchall():
        for row in rows:
            session_id, owner, encoded = row
            baseline = json.loads(encoded)
            params = {"session": session_id, "owner": owner}
            if downgrade:
                if "history_states" not in baseline:
                    continue
                path = [
                    {"position": position, "payload_json": payload}
                    for position, payload in execute(READ_PATH, params).fetchall()
                ]
                baseline = materialize_baseline(baseline, path)
                encoded = json.dumps(baseline, ensure_ascii=False)
            else:
                if "history_states" in baseline:
                    continue
                for sql, values, expected in create_path(owner, session_id, baseline["history"]):
                    if execute(sql, values).rowcount != expected:
                        raise RevisionConflictError()
                encoded = stored_baseline(baseline)
            execute("UPDATE character_chat_sessions SET baseline_json=:baseline WHERE id=:session AND owner_user_id=:owner",
                    {**params, "baseline": encoded})
        after = rows[-1][0]
