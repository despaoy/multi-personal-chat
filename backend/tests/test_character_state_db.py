"""角色关系与长期记忆数据库层（SQLite）原子性回归测试。

覆盖两类并发回归：
1. upsert_character_relationship 不得用旧计数覆盖
   increment_character_interaction 刚自增的交互计数
   （管理端更新关系与新消息并发时计数回退）；
2. add_or_update_character_memory 相同 memory_key 并发写入
   不得触发唯一约束错误（单条 UPSERT 原子完成）。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

SCOPE = {
    "character_id": "kisaki",
    "platform": "qq",
    "adapter": "nonebot",
    "sender_id": "user_1",
    "conversation_type": "private",
    "conversation_id": "user_1",
}


def _make_db(tmp_path):
    from db.database import SQLiteDB

    return SQLiteDB(tmp_path / "character_state.db")


def _increment(db, times: int) -> int:
    count = 0
    for _ in range(times):
        count = db.increment_character_interaction(**SCOPE)
    return count


def _upsert_relationship(db, stage: str = "familiar", interaction_count=None):
    return db.upsert_character_relationship(
        SCOPE["character_id"],
        SCOPE["platform"],
        SCOPE["adapter"],
        SCOPE["sender_id"],
        SCOPE["conversation_type"],
        SCOPE["conversation_id"],
        stage,
        "小明",
        "一起聊过天",
        interaction_count,
    )


# ============================================
# 关系 UPSERT：计数不被覆盖
# ============================================

def test_upsert_without_count_preserves_incremented_value(tmp_path):
    db = _make_db(tmp_path)
    assert _increment(db, 5) == 5
    record = _upsert_relationship(db, stage="close")
    assert record["interaction_count"] == 5
    assert record["relationship_stage"] == "close"
    row = db.get_character_relationship(**SCOPE)
    assert row["interaction_count"] == 5
    assert row["relationship_stage"] == "close"


def test_upsert_explicit_count_overrides(tmp_path):
    db = _make_db(tmp_path)
    _increment(db, 3)
    record = _upsert_relationship(db, interaction_count=42)
    assert record["interaction_count"] == 42
    assert db.get_character_relationship(**SCOPE)["interaction_count"] == 42


def test_upsert_on_missing_row_starts_from_zero(tmp_path):
    db = _make_db(tmp_path)
    record = _upsert_relationship(db)
    assert record["interaction_count"] == 0
    assert record["relationship_stage"] == "familiar"


def test_upsert_preserves_created_at(tmp_path):
    db = _make_db(tmp_path)
    first = _upsert_relationship(db, stage="stranger")
    second = _upsert_relationship(db, stage="close")
    assert second["created_at"] == first["created_at"]


def test_relationship_upsert_never_loses_concurrent_increments(tmp_path):
    """并发混合自增与关系覆盖写：自增结果不得丢失。

    旧实现（先 SELECT 计数再 UPSERT 全列）在并发下会用旧计数覆盖
    刚自增的值；单条 UPSERT 且 interaction_count=None 时不触碰计数列，
    任何交错下最终计数都等于自增次数。
    """
    db = _make_db(tmp_path)
    increments = 12

    def increment(_):
        db.increment_character_interaction(**SCOPE)

    def upsert(_):
        _upsert_relationship(db, stage="familiar")

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(increment, i) for i in range(increments)]
        futures += [pool.submit(upsert, i) for i in range(6)]
        for future in futures:
            future.result()

    row = db.get_character_relationship(**SCOPE)
    assert row["interaction_count"] == increments
    assert row["relationship_stage"] == "familiar"


# ============================================
# 记忆 UPSERT：同 key 原子写入
# ============================================

def test_memory_upsert_same_key_updates_in_place(tmp_path):
    db = _make_db(tmp_path)
    first = db.add_or_update_character_memory(
        **SCOPE,
        memory_type="user_fact",
        memory_key="user_name",
        content="用户说自己叫小明",
        importance=0.8,
    )
    second = db.add_or_update_character_memory(
        **SCOPE,
        memory_type="user_fact",
        memory_key="user_name",
        content="用户说自己叫大明",
        importance=0.9,
    )
    assert second["id"] == first["id"]
    assert second["content"] == "用户说自己叫大明"
    assert second["created_at"] == first["created_at"]
    rows = db.list_character_memories(**SCOPE)
    assert len(rows) == 1
    assert rows[0]["content"] == "用户说自己叫大明"
    assert rows[0]["importance"] == 0.9


def test_memory_upsert_distinct_keys_create_rows(tmp_path):
    db = _make_db(tmp_path)
    db.add_or_update_character_memory(
        **SCOPE,
        memory_type="user_fact",
        memory_key="user_name",
        content="用户说自己叫小明",
    )
    db.add_or_update_character_memory(
        **SCOPE,
        memory_type="promise",
        memory_key="promise_friday",
        content="约定周五见面",
    )
    assert len(db.list_character_memories(**SCOPE)) == 2


def test_memory_upsert_concurrent_same_key_is_atomic(tmp_path):
    """相同 memory_key 并发写入：单条 UPSERT 原子完成，不撞唯一约束。"""
    db = _make_db(tmp_path)

    def write(index: int):
        return db.add_or_update_character_memory(
            **SCOPE,
            memory_type="user_fact",
            memory_key="user_name",
            content=f"用户说自己叫小明{index}",
            importance=0.5,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(write, range(8)))

    assert len({record["id"] for record in records}) == 1
    rows = db.list_character_memories(**SCOPE)
    assert len(rows) == 1
