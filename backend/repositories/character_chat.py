"""Persist version-bound sessions, inherited snapshots and revision-checked turns."""

import copy
import json
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone

from db.character_chat_erasure import ERASURE_TABLE, SESSION_ROWS
from db.character_chat_history import (
    HISTORY_NODES,
    HISTORY_PATH,
    ORPHANED_HISTORY_NODE,
    READ_PATH,
    create_path,
    materialize_baseline,
    stored_baseline,
)
from db.errors import RevisionConflictError
from repositories.character_library import CharacterLibraryRepository, CharacterNotFoundError


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def initial_state():
    return {"relationship": {"stage": "stranger", "summary": ""}, "memories": []}


class CharacterChatRepository:
    def __init__(self, database):
        self.database = database
        self.characters = CharacterLibraryRepository(database)

    def get(self, owner, session_id):
        rows = self.database.execute_sql(
            "SELECT * FROM character_chat_sessions WHERE id=:id AND owner_user_id=:owner",
            {"id": session_id, "owner": owner},
        )
        if not rows:
            raise CharacterNotFoundError("会话不存在")
        row = dict(rows[0])
        baseline = json.loads(row.pop("baseline_json"))
        path = (
            self.database.execute_sql(READ_PATH, {"session": session_id, "owner": owner})
            if baseline["history_states"] else []
        )
        row["baseline"] = materialize_baseline(baseline, path)
        row["network_policy"] = row["baseline"].get("network_policy", "cloud_allowed")
        row["source_corpus_id"] = row["baseline"].get("source_corpus_id")
        row["state"] = json.loads(row.pop("state_json"))
        row["character"] = self.characters.get(owner, row["character_id"], row["character_revision"])
        return row

    def list(self, owner, character_id=None, limit=50, offset=0):
        character_filter = " AND character_id=:character" if character_id is not None else ""
        return self.database.execute_sql(
            "SELECT id,character_id,character_revision,title,revision,mode,source_session_id,created_at,updated_at "
            "FROM character_chat_sessions WHERE owner_user_id=:owner" + character_filter
            + " ORDER BY updated_at DESC,id LIMIT :limit OFFSET :offset",
            {"owner": owner, "character": character_id, "limit": limit, "offset": offset},
        )

    def turns(self, owner, session_id):
        self.get(owner, session_id)
        return self._active_turns(owner, session_id)

    def _active_turns(self, owner, session_id):
        rows = self.database.execute_sql(
            "SELECT t.*,e.payload_json FROM character_chat_turns t "
            "JOIN character_chat_sessions s ON s.id=t.session_id "
            "LEFT JOIN character_chat_evidence e ON e.turn_id=t.id "
            "WHERE t.session_id=:id AND s.owner_user_id=:owner AND t.active=1 ORDER BY t.position",
            {"id": session_id, "owner": owner},
        )
        return [self.decode_turn(row) for row in rows]

    @staticmethod
    def decode_turn(row):
        result = dict(row)
        result["evidence"] = json.loads(result.pop("payload_json", None) or "{}")
        return result

    @staticmethod
    def _deletion_ids(selections):
        if not selections or len(selections) > 100:
            raise ValueError("请选择 1 至 100 个会话")
        ids = [item["id"] for item in selections]
        if len(set(ids)) != len(ids):
            raise ValueError("会话不能重复选择")
        return ids

    def preview_deletion(self, owner, selections):
        """Return account-owned session metadata and links, without transcript contents."""
        ids = self._deletion_ids(selections)
        rows = self.database.execute_sql(
            "SELECT id,character_id,title,revision,mode,source_session_id,baseline_json,state_json "
            "FROM character_chat_sessions WHERE owner_user_id=:owner ORDER BY updated_at DESC,id",
            {"owner": owner},
        )
        by_id = {row["id"]: row for row in rows}
        for selection in selections:
            row = by_id.get(selection["id"])
            if row is None:
                raise CharacterNotFoundError("会话不存在")
            if row["revision"] != selection["revision"]:
                raise RevisionConflictError()

        origins = defaultdict(set)
        for row in self.database.execute_sql(
            f"SELECT DISTINCT p.session_id,n.origin_session_id FROM {HISTORY_PATH} p JOIN {HISTORY_NODES} n "
            "ON n.owner_user_id=p.owner_user_id AND n.node_key=p.node_key "
            "WHERE p.owner_user_id=:owner AND n.origin_session_id IS NOT NULL",
            {"owner": owner},
        ):
            origins[row["session_id"]].add(row["origin_session_id"])
        children = defaultdict(set)
        for row in rows:
            baseline = json.loads(row["baseline_json"])
            parents = {row["source_session_id"], *origins[row["id"]]}
            for ancestry in baseline.get("shared_memory_lineage", {}).values():
                parents.update(ancestry)
            states = [json.loads(row["state_json"]), baseline.get("state", {})]
            states.extend(step.get("state_after", {}) for step in baseline["history_states"])
            for state in states:
                for memory in state.get("memories", []):
                    if memory.get("shared_memory_id"):
                        parents.update((memory.get("shared_from_session_id"), memory.get("source_session_id")))
            for parent in parents - {None, row["id"]}:
                children[parent].add(row["id"])

        related = set(ids)
        queue = deque(ids)
        while queue:
            for child in children[queue.popleft()] - related:
                related.add(child)
                queue.append(child)

        def summary(row):
            return {key: row[key] for key in ("id", "character_id", "title", "revision", "mode")}

        return {
            "selected": [summary(by_id[session_id]) for session_id in ids],
            "related": [summary(row) for row in rows if row["id"] in related and row["id"] not in ids],
        }

    def delete_sessions(self, owner, selections):
        """Erase exactly the selected snapshots, including inactive replacement history."""
        ids = self._deletion_ids(selections)
        statements = []
        # Lock all selected rows in stable order before deleting their dependent data.
        # Creation/branching locks the same source row; stale in-flight writers cannot revive it.
        for item in sorted(selections, key=lambda item: item["id"]):
            self.get(owner, item["id"])
            statements.append(
                (
                    "UPDATE character_chat_sessions SET revision=revision+1 "
                    "WHERE id=:id AND owner_user_id=:owner AND revision=:revision",
                    {"id": item["id"], "owner": owner, "revision": item["revision"]},
                    1,
                )
            )
        path_params = {f"session_{index}": session_id for index, session_id in enumerate(ids)}
        placeholders = ",".join(f":{name}" for name in path_params)
        nodes = self.database.execute_sql(
            f"SELECT DISTINCT node_key FROM {HISTORY_PATH} "
            f"WHERE owner_user_id=:owner AND session_id IN ({placeholders}) ORDER BY node_key",
            {**path_params, "owner": owner},
        )
        node_params = [{"owner": owner, "node": row["node_key"]} for row in nodes]
        # A path creation holds the same node locks until its references have committed.
        statements.extend(
            (f"UPDATE {HISTORY_NODES} SET node_key=node_key WHERE owner_user_id=:owner AND node_key=:node", params, 1)
            for params in node_params
        )
        for session_id in sorted(ids):
            params = {"id": session_id}
            statements.extend((
                (f"INSERT INTO {ERASURE_TABLE} (session_id) VALUES (:id) ON CONFLICT (session_id) DO NOTHING", params, None),
                ("DELETE FROM character_chat_evidence WHERE turn_id IN "
                 "(SELECT id FROM character_chat_turns WHERE session_id=:id)", params, None),
            ))
            statements.extend(
                (f"DELETE FROM {table} WHERE {column}=:id", params, 1 if table == "character_chat_sessions" else None)
                for table, column in SESSION_ROWS
            )
        statements.extend(
            (f"DELETE FROM {HISTORY_NODES} WHERE owner_user_id=:owner AND node_key=:node AND {ORPHANED_HISTORY_NODE}",
             params, None)
            for params in node_params
        )
        self.database.revision_transaction(statements)
        return {"deleted_session_ids": ids}

    def snapshot(self, owner, session_id):
        session = self.get(owner, session_id)
        turns = self._active_turns(owner, session_id)
        rows = self.database.execute_sql(
            "SELECT revision FROM character_chat_sessions WHERE id=:id AND owner_user_id=:owner",
            {"id": session_id, "owner": owner},
        )
        if not rows:
            raise CharacterNotFoundError("会话不存在")
        if rows[0]["revision"] != session["revision"]:
            raise RevisionConflictError()
        return session, turns

    def continuation_preview(self, owner, session_id):
        from repositories.character_sources import CharacterSourceRepository

        source = self.get(owner, session_id)
        latest = self.characters.get(owner, source["character_id"])
        sources = CharacterSourceRepository(self.database)
        bound_corpus = source["source_corpus_id"]
        editions = sources.list(
            owner, source["character_id"], latest["settings_revision"], 1,
        )
        return {
            "source_session_id": session_id,
            "source_revision": source["revision"],
            "character_revision": latest["settings_revision"],
            "previous_character_revision": source["character_revision"],
            "source_corpus_id": bound_corpus,
            "source_corpus_available": bool(bound_corpus) and sources.is_available(
                owner, source["character_id"], source["character_revision"], bound_corpus,
            ),
            "latest_source_corpus": editions[0] if editions else None,
            "changes": {
                key: {"previous": source["character"]["settings"][key], "next": value}
                for key, value in latest["settings"].items()
                if value != source["character"]["settings"][key]
            },
            "relationship": source["state"]["relationship"],
            "memories": source["state"]["memories"],
        }

    def create(
        self,
        owner,
        character_id,
        title,
        *,
        source_session_id=None,
        source_revision=None,
        source_turn_id=None,
        hypothesis="",
        accept_settings_change=False,
        expected_character_revision=None,
        excluded_memory_ids=None,
        reset_relationship=False,
        inherit_history=True,
        shared_memory_ids=None,
        network_policy=None,
        source_corpus_id=None,
        without_original_sources=False,
    ):
        if network_policy not in {None, "cloud_allowed", "local_only"}:
            raise ValueError("无效的模型网络策略")
        character = self.characters.get(owner, character_id)
        state = initial_state()
        baseline = {"history": [], "state": state}
        mode = "initial"
        statements = []
        if source_session_id:
            source, turns = self.snapshot(owner, source_session_id)
            if source["network_policy"] == "local_only":
                if network_policy == "cloud_allowed":
                    raise ValueError("仅本地会话的继承与分支必须保留仅本地策略")
                network_policy = "local_only"
            if source["character_id"] != character_id:
                raise ValueError("只能继承同一个角色的会话")
            if source_revision != source["revision"]:
                raise RevisionConflictError()
            mode = "branch" if source_turn_id else "continuation"
            if source["mode"] == "branch" and mode == "continuation":
                raise ValueError("假想分支只能继续原分支或创建子分支，不能作为普通会话的关系起点")
            history = [
                *source["baseline"]["history"],
                *[
                    {
                        "id": turn["id"],
                        "session_id": source_session_id,
                        "message": turn["message"],
                        "reply": turn["reply"],
                        "evidence": turn["evidence"],
                        "state_after": json.loads(turn["state_after_json"]),
                        "character_revision": source["character_revision"],
                        "hypothesis": source["hypothesis"],
                        "source_corpus_id": source["source_corpus_id"],
                    }
                    for turn in turns
                ],
            ]
            if source_turn_id:
                selected = next((turn for turn in history if turn["id"] == source_turn_id), None)
                if selected is None:
                    raise CharacterNotFoundError("分支起点不存在或已撤回")
                if "state_after" not in selected or "character_revision" not in selected:
                    raise ValueError("这段旧版继承记录未保存逐轮状态，无法准确创建分支")
                history = history[: history.index(selected) + 1]
                state = selected["state_after"]
                character = self.characters.get(owner, character_id, selected["character_revision"])
                # Preserve the premise at the selected point when branching again.
                previous_hypothesis = selected.get("hypothesis", "")
                if previous_hypothesis:
                    hypothesis = previous_hypothesis + ("\n后续调整：" + hypothesis if hypothesis else "")
            else:
                state = source["state"]
                if source["character_revision"] != character["settings_revision"] and not accept_settings_change:
                    raise ValueError("角色设定已有新版本，请查看差异并确认继承")
                if (
                    source["character_revision"] != character["settings_revision"]
                    and expected_character_revision is None
                ):
                    raise ValueError("请先查看并指定确认的角色设定版本")
            baseline = {"history": history, "state": state}
            # Lock/check the source revision in the same transaction as the new snapshot.
            statements.append(
                (
                    "UPDATE character_chat_sessions SET revision=revision WHERE id=:id AND owner_user_id=:owner AND revision=:revision",
                    {"id": source_session_id, "owner": owner, "revision": source_revision},
                    1,
                )
            )
        elif source_turn_id or source_revision is not None:
            raise ValueError("继承起点需要指定来源会话")
        if hypothesis and mode != "branch":
            raise ValueError("剧情假设仅能用于假想分支")
        excluded = set(excluded_memory_ids or [])
        if excluded or reset_relationship or not inherit_history:
            if mode != "continuation":
                raise ValueError("继承调整仅适用于延续关系")
            known = {item.get("id") for item in state["memories"] if isinstance(item, dict)}
            if not excluded.issubset(known):
                raise ValueError("所选记忆已变化，请刷新继承预览")
            baseline = copy.deepcopy(baseline)

            def adjusted(value):
                result = copy.deepcopy(value)
                # Summaries may contain excluded facts or relationship context.
                result.pop("summaries", None)
                result.pop("summary_processing", None)
                result["memories"] = [item for item in result["memories"] if item.get("id") not in excluded]
                if reset_relationship:
                    result["relationship"] = initial_state()["relationship"]
                    result.pop("relationship_proposal", None)
                return result

            state = adjusted(state)
            baseline["state"] = state
            if not inherit_history:
                baseline["history"] = []
            else:
                for turn in baseline["history"]:
                    if "state_after" in turn:
                        turn["state_after"] = adjusted(turn["state_after"])
            baseline["inheritance_choices"] = {
                "excluded_memory_ids": sorted(excluded),
                "reset_relationship": reset_relationship,
                "inherit_history": inherit_history,
            }
        if mode != "branch":
            if (
                expected_character_revision is not None
                and expected_character_revision != character["settings_revision"]
            ):
                raise RevisionConflictError()
            statements.append(
                (
                    "UPDATE character_library SET revision=revision WHERE id=:id AND owner_user_id=:owner AND revision=:revision",
                    {"id": character_id, "owner": owner, "revision": character["settings_revision"]},
                    1,
                )
            )
        session_id, now = uuid.uuid4().hex, timestamp()
        shared_lineage = {}
        if source_session_id:
            retained_ids = {memory.get("id") for memory in state["memories"] if isinstance(memory, dict)}
            shared_lineage = {
                identity: ancestry for identity, ancestry in source["baseline"].get("shared_memory_lineage", {}).items()
                if baseline["history"] or identity in retained_ids
            }
        if shared_memory_ids:
            if mode != "initial":
                raise ValueError("共享记忆仅在初识新会话中显式带入")
            state = copy.deepcopy(state)
            for shared_id in dict.fromkeys(shared_memory_ids):
                rows = self.database.execute_sql(
                    "SELECT * FROM character_shared_memories WHERE id=:id AND owner_user_id=:owner",
                    {"id": shared_id, "owner": owner},
                )
                if not rows:
                    raise CharacterNotFoundError("共享记忆不存在或已取消共享")
                shared_source = self.get(owner, rows[0]["source_session_id"])
                if shared_source["network_policy"] == "local_only":
                    if network_policy == "cloud_allowed":
                        raise ValueError("仅本地会话的共享记忆只能带入仅本地会话")
                    network_policy = "local_only"
                memory = json.loads(rows[0]["memory_json"])
                # Share the chosen fact, not an entire source utterance containing other facts.
                memory.pop("evidence", None)
                memory.pop("source_request_id", None)
                memory_id = uuid.uuid4().hex
                ancestry = set(shared_source["baseline"].get("shared_memory_lineage", {}).get(rows[0]["source_memory_id"], ()))
                ancestry.update((memory.get("source_session_id"), memory.get("shared_from_session_id"), shared_source["id"]))
                shared_lineage[memory_id] = sorted(ancestry - {None})
                collision = any(item.get("memory_key") == memory.get("memory_key") for item in state["memories"])
                state["memories"].append(
                    {
                        **memory,
                        "id": memory_id,
                        "origin": "shared",
                        "shared_memory_id": shared_id,
                        "shared_from_session_id": shared_source["id"],
                        "status": "pending" if collision else "confirmed",
                    }
                )
                statements.append(
                    (
                        "UPDATE character_shared_memories SET id=id WHERE id=:id AND owner_user_id=:owner",
                        {"id": shared_id, "owner": owner},
                        1,
                    )
                )
            baseline["state"] = state
        if shared_lineage:
            # Provenance belongs to the snapshot, so scrubbing the copied memory
            # does not conceal a conversation that may have used it.
            baseline["shared_memory_lineage"] = shared_lineage
        baseline["network_policy"] = network_policy or "cloud_allowed"
        from repositories.character_sources import CharacterSourceRepository

        source_repo = CharacterSourceRepository(self.database)
        if without_original_sources:
            if mode != 'continuation' or source_corpus_id is not None:
                raise ValueError('仅可为延续会话选择不使用原作资料，且不能同时指定资料版本')
        elif mode == 'branch':
            inherited_corpus = selected.get('source_corpus_id')
            if source_corpus_id is not None and source_corpus_id != inherited_corpus:
                raise ValueError('分支必须使用起点的资料版本')
            source_corpus_id = inherited_corpus
        elif mode == 'continuation' and source['character_revision'] == character['settings_revision']:
            source_corpus_id = source_corpus_id or source['source_corpus_id']
        elif source_corpus_id is None:
            editions = source_repo.list(owner, character_id, character['settings_revision'], 1)
            source_corpus_id = editions[0]['id'] if editions else None
        if source_corpus_id:
            corpus = source_repo.get(owner, source_corpus_id)
            if corpus['character_id'] != character_id or corpus['character_revision'] != character['settings_revision']:
                raise ValueError('资料版本不属于当前角色设定版本')
            statements.append((
                'UPDATE character_source_corpora SET revoked=revoked WHERE id=:corpus AND owner_user_id=:owner AND revoked=0',
                {'corpus': source_corpus_id, 'owner': owner}, 1,
            ))
        baseline['source_corpus_id'] = source_corpus_id
        params = {
            "id": session_id,
            "owner": owner,
            "character": character_id,
            "version": character["settings_revision"],
            "title": title,
            "mode": mode,
            "source": source_session_id,
            "turn": source_turn_id,
            "hypothesis": hypothesis,
            "baseline": stored_baseline(baseline),
            "state": json.dumps(state, ensure_ascii=False),
            "now": now,
        }
        statements.append(
            (
                "INSERT INTO character_chat_sessions (id,owner_user_id,character_id,character_revision,title,revision,mode,"
                "source_session_id,source_turn_id,hypothesis,baseline_json,state_json,created_at,updated_at) "
                "VALUES (:id,:owner,:character,:version,:title,1,:mode,:source,:turn,:hypothesis,:baseline,:state,:now,:now)",
                params,
                1,
            )
        )
        statements.extend(create_path(owner, session_id, baseline["history"]))
        self.database.revision_transaction(statements)
        return self.get(owner, session_id)

    @staticmethod
    def cas(session, state):
        return (
            "UPDATE character_chat_sessions SET revision=revision+1,state_json=:state,updated_at=:now "
            "WHERE id=:id AND owner_user_id=:owner AND revision=:revision",
            {
                "id": session["id"],
                "owner": session["owner_user_id"],
                "revision": session["revision"],
                "state": json.dumps(state, ensure_ascii=False),
                "now": timestamp(),
            },
            1,
        )

    def prior(self, owner, session_id, request_id):
        self.get(owner, session_id)
        rows = self.database.execute_sql(
            "SELECT t.*,e.payload_json FROM character_chat_turns t LEFT JOIN character_chat_evidence e ON e.turn_id=t.id "
            "WHERE t.session_id=:id AND t.request_id=:request",
            {"id": session_id, "request": request_id},
        )
        return self.decode_turn(rows[0]) if rows else None

    def commit_turn(
        self, session, request_id, message, reply, model, *, state=None, replace_turn_id=None, evidence=None,
        generation_token=None,
        enqueue_enrichment=False,
        source_corpus_id=None,
    ):
        owner, session_id = session["owner_user_id"], session["id"]
        if not reply.strip():
            raise ValueError("模型回复为空")
        state = copy.deepcopy(session["state"] if state is None else state)
        turn_id = uuid.uuid4().hex
        relationship = state["relationship"]
        if relationship.get("source_request_id") == request_id and relationship.get("source_session_id") == session_id:
            relationship["source_turn_id"] = turn_id
        proposal = state.get("relationship_proposal", {})
        if proposal.get("source_request_id") == request_id and proposal.get("source_session_id") == session_id:
            proposal["source_turn_id"] = turn_id
        for memory in state.get("memories", []):
            if (
                isinstance(memory, dict)
                and memory.get("source_request_id") == request_id
                and memory.get("source_session_id") == session_id
            ):
                memory["source_turn_id"] = turn_id
        statements = [self.cas(session, state), *self.revoke_changed_shares(session_id, state)]
        if source_corpus_id:
            if source_corpus_id != session.get('source_corpus_id'):
                raise ValueError('回答资料版本与会话绑定不一致')
            statements.append((
                'UPDATE character_source_corpora SET revoked=revoked WHERE id=:corpus AND owner_user_id=:owner '
                'AND character_id=:character AND character_revision=:revision AND revoked=0',
                {'corpus': source_corpus_id, 'owner': owner, 'character': session['character_id'], 'revision': session['character_revision']}, 1,
            ))
        if generation_token is not None:
            from repositories.character_chat_generation import CharacterChatGenerationRepository

            statements.append(CharacterChatGenerationRepository.completion_statement(
                session_id, request_id, generation_token, turn_id
            ))
        if replace_turn_id:
            selected = self.selected_turn(owner, session_id, replace_turn_id)
            statements.extend(self.deactivate_after(session_id, selected["position"]))
        position = self.database.execute_sql(
            "SELECT COALESCE(MAX(position),0)+1 AS next_position FROM character_chat_turns WHERE session_id=:id",
            {"id": session_id},
        )[0]["next_position"]
        params = {
            "id": turn_id,
            "session": session_id,
            "position": position,
            "request": request_id,
            "message": message,
            "reply": reply,
            "model": model,
            "revision": session["revision"],
            "replaces": replace_turn_id,
            "state": json.dumps(state, ensure_ascii=False),
            "now": timestamp(),
        }
        statements.append(
            (
                "INSERT INTO character_chat_turns (id,session_id,position,request_id,request_revision,replaces_turn_id,message,reply,model,active,state_after_json,created_at) "
                "VALUES (:id,:session,:position,:request,:revision,:replaces,:message,:reply,:model,1,:state,:now)",
                params,
                1,
            )
        )
        if evidence:
            statements.append(
                (
                    "INSERT INTO character_chat_evidence (turn_id,payload_json) VALUES (:turn,:payload)",
                    {"turn": turn_id, "payload": json.dumps(evidence, ensure_ascii=False)},
                    1,
                )
            )
        if enqueue_enrichment:
            statements.append((
                "INSERT INTO character_chat_enrichment_jobs (turn_id,session_id,expected_revision,status,created_at) "
                "VALUES (:turn,:session,:revision,'pending',:now)",
                {"turn": turn_id, "session": session_id, "revision": session["revision"] + 1, "now": timestamp()}, 1,
            ))
        statements.append((
            "UPDATE character_chat_proactive_policies SET inactivity_turn_id=:turn,"
            "inactivity_due_at=CAST(:now_epoch AS DOUBLE PRECISION)+inactivity_hours*3600 "
            "WHERE session_id=:session AND enabled=1 AND inactivity_hours IS NOT NULL",
            {"turn": turn_id, "now_epoch": datetime.fromisoformat(params["now"]).timestamp(), "session": session_id}, None,
        ))
        self.database.revision_transaction(statements)
        return self.prior(owner, session_id, request_id)

    def selected_turn(self, owner, session_id, turn_id):
        turn = next((row for row in self.turns(owner, session_id) if row["id"] == turn_id), None)
        if turn is None:
            raise CharacterNotFoundError("消息不存在或已撤回")
        return turn

    @staticmethod
    def deactivate_after(session_id, position):
        params = {"id": session_id, "position": position}
        return (
            ("UPDATE character_chat_proactive_jobs SET status='cancelled',topic='',content=NULL "
             "WHERE session_id=:id AND status IN ('pending','sent') AND source_turn_id IN ("
             "SELECT id FROM character_chat_turns WHERE session_id=:id AND position>=:position AND active=1)",
             params, None),
            ("UPDATE character_chat_turns SET active=0 WHERE session_id=:id AND position>=:position AND active=1",
             params, None),
        )

    def before_turn(self, owner, session_id, turn_id):
        session, turns = self.snapshot(owner, session_id)
        selected = next((turn for turn in turns if turn["id"] == turn_id), None)
        if selected is None:
            raise CharacterNotFoundError("消息不存在或已撤回")
        earlier = [turn for turn in turns if turn["position"] < selected["position"]]
        state = json.loads(earlier[-1]["state_after_json"]) if earlier else session["baseline"]["state"]
        return session, earlier, state, selected

    def retract(self, owner, session_id, turn_id, revision):
        session, _, state, selected = self.before_turn(owner, session_id, turn_id)
        if session["revision"] != revision:
            raise RevisionConflictError()
        self.database.revision_transaction(
            [
                self.cas(session, state),
                *self.deactivate_after(session_id, selected["position"]),
                ("UPDATE character_chat_proactive_policies SET inactivity_due_at=NULL,inactivity_turn_id=NULL "
                 "WHERE session_id=:session AND inactivity_due_at IS NOT NULL", {"session": session_id}, None),
                *self.revoke_changed_shares(session_id, state),
            ]
        )
        return self.get(owner, session_id)

    def update_relationship(self, owner, session_id, revision, stage, summary, dimensions=None):
        """Anchor an explicit relationship adjustment at the current conversation point."""
        if stage not in {"stranger", "acquaintance", "familiar", "close"} or len(summary) > 2000:
            raise ValueError("关系阶段或摘要无效")
        session, turns = self.snapshot(owner, session_id)
        if session["revision"] != revision:
            raise RevisionConflictError()
        state = copy.deepcopy(session["state"])
        choices = state["relationship"].get("dimensions", {})
        if dimensions is not None:
            choices = dimensions.apply_to(choices)
        state["relationship"] = {
            "stage": stage,
            "summary": summary.strip(),
            "origin": "user",
            "source_turn_id": turns[-1]["id"] if turns else None,
            **({"dimensions": choices} if choices else {}),
        }
        state.pop("relationship_proposal", None)
        return self.save_relationship_state(session, turns, state)

    def resolve_relationship_proposal(self, owner, session_id, revision, proposal_id, accept):
        session, turns = self.snapshot(owner, session_id)
        if session["revision"] != revision:
            raise RevisionConflictError()
        state = copy.deepcopy(session["state"])
        proposal = state.pop("relationship_proposal", None)
        if not proposal or proposal["id"] != proposal_id:
            raise RevisionConflictError()
        if accept:
            proposal.pop("id")
            state["relationship"] = {**proposal, "origin": "user"}
        return self.save_relationship_state(session, turns, state)

    def save_relationship_state(self, session, turns, state):
        owner, session_id = session["owner_user_id"], session["id"]
        statements = [self.cas(session, state)]
        if turns:
            statements.append(
                (
                    "UPDATE character_chat_turns SET state_after_json=:state WHERE id=:id AND session_id=:session AND active=1",
                    {"id": turns[-1]["id"], "session": session_id, "state": json.dumps(state, ensure_ascii=False)},
                    1,
                )
            )
        else:
            baseline = copy.deepcopy(session["baseline"])
            baseline["state"] = state
            statements.append(
                (
                    "UPDATE character_chat_sessions SET baseline_json=:baseline WHERE id=:id",
                    {"id": session_id, "baseline": stored_baseline(baseline)},
                    1,
                )
            )
        self.database.revision_transaction(statements)
        return self.get(owner, session_id)

    def manage_memory(self, owner, session_id, revision, *, content=None, memory_id=None):
        """User corrections apply to every snapshot in this session, never other branches.

        New manual facts are anchored to the latest active turn (or the baseline).
        Rewinding past that turn removes them. Corrections/deletions rewrite older
        snapshots too, so rewinding cannot resurrect the superseded fact.
        """
        session, turns = self.snapshot(owner, session_id)
        if session["revision"] != revision:
            raise RevisionConflictError()
        if content is not None:
            content = content.strip()
            if not content or len(content) > 2000:
                raise ValueError("记忆内容需要为 1 至 2000 个字符")
        memories = session["state"].get("memories", [])
        if memory_id and not any(item.get("id") == memory_id for item in memories):
            raise CharacterNotFoundError("记忆不存在")
        if memory_id is None and content is None:
            raise ValueError("新增记忆需要提供内容")
        adding = memory_id is None
        memory_id = memory_id or uuid.uuid4().hex
        now = timestamp()
        latest_id = turns[-1]["id"] if turns else None

        excluded_summary_sources = set(session["state"].get("summary_excluded_turn_ids", []))
        if not adding:
            # A corrected/deleted fact must not survive in generated summaries, including on rewind.
            excluded_summary_sources.update(
                source["turn_id"] for segment in session["state"].get("summaries", [])
                for source in segment.get("sources", [])
            )
            excluded_summary_sources.update(
                item["source_turn_id"] for item in memories
                if item.get("id") == memory_id and item.get("source_turn_id")
            )

        def changed(state, *, anchor=False):
            result = copy.deepcopy(state)
            if not adding:
                result.pop("summaries", None)
                result["summary_excluded_turn_ids"] = sorted(excluded_summary_sources)
            target = next((item for item in result.get("memories", []) if item.get("id") == memory_id), None)
            confirmed_key = (
                target.get("memory_key")
                if target and target.get("status") == "pending" and content is not None
                else None
            )
            items = []
            for item in result.get("memories", []):
                if confirmed_key and item.get("memory_key") == confirmed_key and item.get("id") != memory_id:
                    continue
                if item.get("id") != memory_id:
                    items.append(item)
                elif content is not None:
                    items.append({**item, "content": content, "status": "confirmed", "origin": "user", "updated_at": now})
            if adding and anchor:
                items.append(
                    {
                        "id": memory_id,
                        "content": content,
                        "status": "confirmed",
                        "origin": "user",
                        "source_session_id": session_id,
                        "source_turn_id": latest_id,
                        "created_at": now,
                        "updated_at": now,
                    }
                )
            result["memories"] = items
            return result

        current_state = changed(session["state"], anchor=True)
        statements = [self.cas(session, current_state), *self.revoke_changed_shares(session_id, current_state)]
        baseline = copy.deepcopy(session["baseline"])
        baseline["state"] = changed(baseline["state"], anchor=latest_id is None)
        for inherited in baseline["history"]:
            if "state_after" in inherited:
                inherited["state_after"] = changed(inherited["state_after"])
        statements.append(
            (
                "UPDATE character_chat_sessions SET baseline_json=:baseline WHERE id=:id",
                {"id": session_id, "baseline": stored_baseline(baseline)},
                1,
            )
        )
        # Include inactive turns: superseded snapshots must not retain corrected facts.
        snapshots = self.database.execute_sql(
            "SELECT id,state_after_json FROM character_chat_turns WHERE session_id=:id", {"id": session_id}
        )
        for row in snapshots:
            state = changed(json.loads(row["state_after_json"]), anchor=row["id"] == latest_id)
            statements.append(
                (
                    "UPDATE character_chat_turns SET state_after_json=:state WHERE id=:id AND session_id=:session",
                    {"id": row["id"], "session": session_id, "state": json.dumps(state, ensure_ascii=False)},
                    1,
                )
            )
        self.database.revision_transaction(statements)
        return self.get(owner, session_id)

    def shared_memories(self, owner, limit=100, offset=0):
        rows = self.database.execute_sql(
            "SELECT * FROM character_shared_memories WHERE owner_user_id=:owner ORDER BY created_at DESC,id LIMIT :limit OFFSET :offset",
            {"owner": owner, "limit": limit, "offset": offset},
        )
        return [
            {"id": row["id"], "source_session_id": row["source_session_id"], "memory": json.loads(row["memory_json"])}
            for row in rows
        ]

    def share_memory(self, owner, session_id, memory_id, revision):
        session = self.get(owner, session_id)
        if session["revision"] != revision:
            raise RevisionConflictError()
        if session["mode"] == "branch":
            raise ValueError("假想分支记忆不能作为普通会话的共享记忆")
        memory = next((item for item in session["state"]["memories"] if item.get("id") == memory_id), None)
        if memory is None:
            raise CharacterNotFoundError("记忆不存在")
        if memory["status"] != "confirmed":
            raise ValueError("请先确认记忆，再选择共享")
        params = {
            "id": uuid.uuid4().hex,
            "owner": owner,
            "session": session_id,
            "memory": memory_id,
            "payload": json.dumps(memory, ensure_ascii=False),
            "now": timestamp(),
        }
        self.database.revision_transaction(
            [
                self.cas(session, session["state"]),
                (
                    "DELETE FROM character_shared_memories WHERE source_session_id=:session AND source_memory_id=:memory",
                    params,
                    None,
                ),
                (
                    "INSERT INTO character_shared_memories (id,owner_user_id,source_session_id,source_memory_id,memory_json,created_at) "
                    "VALUES (:id,:owner,:session,:memory,:payload,:now)",
                    params,
                    1,
                ),
            ]
        )
        return self.get(owner, session_id)

    def unshare_memory(self, owner, shared_id):
        self.database.revision_transaction(
            [
                (
                    "DELETE FROM character_shared_memories WHERE id=:id AND owner_user_id=:owner",
                    {"id": shared_id, "owner": owner},
                    1,
                )
            ]
        )

    def revoke_changed_shares(self, session_id, state):
        current = {item.get("id"): item for item in state["memories"] if isinstance(item, dict)}
        rows = self.database.execute_sql(
            "SELECT id,source_memory_id,memory_json FROM character_shared_memories WHERE source_session_id=:id",
            {"id": session_id},
        )
        return [
            (
                "DELETE FROM character_shared_memories WHERE id=:id",
                {"id": row["id"]},
                None,
            )
            for row in rows
            if current.get(row["source_memory_id"]) != json.loads(row["memory_json"])
        ]
