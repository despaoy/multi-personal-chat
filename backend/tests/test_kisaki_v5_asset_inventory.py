"""阶段 1 资产清单脚本 build_kisaki_v5_asset_inventory.py 的单元测试。

覆盖审查要求的六项：
- assistant_supervision=all/last 的监督口径
- 未知来源分类
- 相邻同角色消息（user-user 与 assistant-assistant）
- train/validation ID 重叠
- 926 条真实清单完整性
- V4 文件在运行前后不被修改
"""

import hashlib
import importlib.util
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V4_DIR = PROJECT_ROOT / "backend/data/character_dialogues/experiments/v4"


def _module():
    spec = importlib.util.spec_from_file_location(
        "build_kisaki_v5_asset_inventory",
        PROJECT_ROOT / "scripts/build_kisaki_v5_asset_inventory.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )


def _record(rid: str, messages: list[dict], metadata: dict) -> dict:
    return {"id": rid, "messages": messages, "metadata": metadata}


# ============================================
# 监督口径：assistant_supervision = all / last
# ============================================


def test_supervision_all_counts_every_assistant_message():
    mod = _module()
    messages = [
        {"role": "user", "content": "问题一"},
        {"role": "assistant", "content": "回答一"},
        {"role": "user", "content": "问题二"},
        {"role": "assistant", "content": "回答二"},
    ]
    supervised = mod.supervised_assistant_messages(messages, "all")
    assert [m["content"] for m in supervised] == ["回答一", "回答二"]


def test_supervision_last_only_supervises_final_assistant_message():
    mod = _module()
    messages = [
        {"role": "user", "content": "问题一"},
        {"role": "assistant", "content": "回答一"},
        {"role": "user", "content": "问题二"},
        {"role": "assistant", "content": "回答二"},
    ]
    supervised = mod.supervised_assistant_messages(messages, "last")
    # 与训练契约一致：last 只监督最后一条 assistant 消息
    assert [m["content"] for m in supervised] == ["回答二"]


def test_build_inventory_separates_raw_and_supervised_counts(tmp_path):
    mod = _module()
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    _write_jsonl(
        train,
        [
            _record(
                "m1",
                [
                    {"role": "user", "content": "问题一"},
                    {"role": "assistant", "content": "第一轮回答"},
                    {"role": "user", "content": "问题二"},
                    {"role": "assistant", "content": "最后一轮回答"},
                ],
                {
                    "data_source": "game_extraction",
                    "assistant_supervision": "last",
                    "final_review": {"approved": True},
                },
            ),
            _record(
                "m2",
                [
                    {"role": "user", "content": "问题"},
                    {"role": "assistant", "content": "回答"},
                ],
                {
                    "data_source": "game_extraction",
                    "final_review": {"approved": True},
                },
            ),
        ],
    )
    _write_jsonl(validation, [])

    result = mod.build_inventory(train, validation)
    rec = result["inventory"][0]
    # 原始 assistant 消息 2 条，last 契约下监督目标只有 1 条
    assert rec["turns"]["assistant"] == 2
    assert rec["supervised_assistant_targets"] == 1
    assert rec["chars"]["raw_assistant"] == len("第一轮回答") + len("最后一轮回答")
    assert rec["chars"]["supervised_assistant"] == len("最后一轮回答")

    rec2 = result["inventory"][1]
    # 缺省 all：监督目标等于原始消息数
    assert rec2["assistant_supervision"] == "all"
    assert rec2["supervised_assistant_targets"] == 1
    assert rec2["chars"]["raw_assistant"] == rec2["chars"]["supervised_assistant"]


# ============================================
# 未知来源
# ============================================


def test_unknown_source_is_flagged_not_silently_grouped(tmp_path):
    mod = _module()
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    _write_jsonl(
        train,
        [
            _record(
                "u1",
                [
                    {"role": "user", "content": "问题"},
                    {"role": "assistant", "content": "回答"},
                ],
                {"data_source": "mystery_source"},
            )
        ],
    )
    _write_jsonl(validation, [])
    result = mod.build_inventory(train, validation)
    rec = result["inventory"][0]
    assert rec["source_group"] == "unknown"
    assert any("未知 data_source" in p for p in rec["issues"])


# ============================================
# 相邻同角色消息（user 与 assistant 一律拒绝）
# ============================================


def test_adjacent_user_messages_are_flagged():
    mod = _module()
    messages = [
        {"role": "user", "content": "问题一"},
        {"role": "user", "content": "补充问题"},
        {"role": "assistant", "content": "回答"},
    ]
    problems = mod.check_role_order(messages)
    assert any("相邻两条 user 消息" in p for p in problems)


def test_adjacent_assistant_messages_are_flagged():
    mod = _module()
    messages = [
        {"role": "user", "content": "问题"},
        {"role": "assistant", "content": "回答一"},
        {"role": "assistant", "content": "回答二"},
    ]
    problems = mod.check_role_order(messages)
    assert any("相邻两条 assistant 消息" in p for p in problems)


def test_alternating_roles_pass():
    mod = _module()
    messages = [
        {"role": "user", "content": "问题一"},
        {"role": "assistant", "content": "回答一"},
        {"role": "user", "content": "问题二"},
        {"role": "assistant", "content": "回答二"},
    ]
    assert mod.check_role_order(messages) == []


# ============================================
# train/validation ID 重叠
# ============================================


def test_validation_id_overlap_is_flagged(tmp_path):
    mod = _module()
    meta = {"data_source": "game_extraction", "final_review": {"approved": True}}
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    _write_jsonl(
        train,
        [
            _record(
                "shared_id",
                [
                    {"role": "user", "content": "问题"},
                    {"role": "assistant", "content": "回答"},
                ],
                meta,
            )
        ],
    )
    _write_jsonl(
        validation,
        [
            _record(
                "shared_id",
                [
                    {"role": "user", "content": "验证问题"},
                    {"role": "assistant", "content": "验证回答"},
                ],
                {"data_source": "game_extraction"},
            )
        ],
    )
    result = mod.build_inventory(train, validation)
    assert any("id 与 validation 重叠" in p for p in result["inventory"][0]["issues"])


def test_duplicate_train_ids_are_flagged(tmp_path):
    mod = _module()
    meta = {"data_source": "game_extraction", "final_review": {"approved": True}}
    train = tmp_path / "train.jsonl"
    _write_jsonl(
        train,
        [
            _record(
                "dup",
                [
                    {"role": "user", "content": "问题"},
                    {"role": "assistant", "content": "回答"},
                ],
                meta,
            ),
            _record(
                "dup",
                [
                    {"role": "user", "content": "问题2"},
                    {"role": "assistant", "content": "回答2"},
                ],
                meta,
            ),
        ],
    )
    validation = tmp_path / "validation.jsonl"
    _write_jsonl(validation, [])
    result = mod.build_inventory(train, validation)
    assert any("重复 id" in p for p in result["inventory"][1]["issues"])


# ============================================
# 真实 V4 数据：926 条完整性 + V4 不被修改
# ============================================


def test_real_v4_inventory_completeness_and_supervision_totals():
    mod = _module()
    train = V4_DIR / "train.jsonl"
    validation = V4_DIR / "validation.jsonl"
    manifest = json.loads((V4_DIR / "canonical_dataset_manifest.json").read_text(encoding="utf-8"))

    train_sha = hashlib.sha256(train.read_bytes()).hexdigest()
    assert train_sha == manifest["train"]["sha256"], "V4 train 与冻结 manifest 不一致"

    result = mod.build_inventory(train, validation)
    inventory = result["inventory"]

    # 926 条记录全部入清单
    assert len(inventory) == 926
    assert result["train_count"] == 926
    assert result["val_count"] == 70

    # 状态只有三种且未预写人工批准
    statuses = {i["status"] for i in inventory}
    assert statuses <= {"pending_review", "keep_core", "excluded_candidate"}
    assert statuses == {"keep_core", "pending_review"}

    # 监督口径权威数字：2,068 原始 / 1,961 监督目标 / 117,345 监督字符
    assert sum(i["turns"]["assistant"] for i in inventory) == 2068
    assert sum(i["supervised_assistant_targets"] for i in inventory) == 1961
    assert sum(i["chars"]["supervised_assistant"] for i in inventory) == 117345

    # 107 条 last 契约全部来自原作多轮记录
    last_records = [i for i in inventory if i["assistant_supervision"] == "last"]
    assert len(last_records) == 107
    assert all(i["source_group"] == "game_extraction_current_sft" for i in last_records)
    assert all(i["multiturn"] for i in last_records)


def test_real_v4_files_unchanged_after_inventory(tmp_path):
    """build_inventory 只读：运行前后 V4 文件哈希不变。"""
    mod = _module()
    train = V4_DIR / "train.jsonl"
    validation = V4_DIR / "validation.jsonl"

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    before = {"train": digest(train), "validation": digest(validation)}
    mod.build_inventory(train, validation)
    after = {"train": digest(train), "validation": digest(validation)}
    assert before == after
