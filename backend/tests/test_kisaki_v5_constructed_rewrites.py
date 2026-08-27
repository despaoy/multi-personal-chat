"""阶段 4 改写脚本 build_kisaki_v5_constructed_rewrites.py 的单元测试。

覆盖：
- 决定文件门禁：仅接受 approved，draft 拒绝
- 内置改写定义与真实 needs_revision 集合一致（13 条全覆盖）
- 消息结构校验：user 开头/交替/assistant 结尾/空文本
- 事实风险词：频率词、assistant 未引入人物名、用户引入不罚
- validation 重叠：完全匹配与 ≥15 连续字符公共子串
- 真实数据端到端：12 候选 + 1 drop、pending_review、ID 规则、
  长度 ≤100、与 130 条保留样本无完全重复、审核材料完整、
  V4 冻结数据不被修改（sha256 前后一致）
"""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V5_DIR = PROJECT_ROOT / "backend/data/character_dialogues/experiments/v5_candidate"
V4_DIR = PROJECT_ROOT / "backend/data/character_dialogues/experiments/v4"
OUT_DIR = V5_DIR / "constructed_rewrite_v1"


def _module():
    spec = importlib.util.spec_from_file_location(
        "build_kisaki_v5_constructed_rewrites",
        PROJECT_ROOT / "scripts/build_kisaki_v5_constructed_rewrites.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _qa(user: str, assistant: str) -> list[dict]:
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


# ============================================
# 门禁与定义完整性
# ============================================


def test_decisions_must_be_approved(tmp_path, monkeypatch):
    """draft 决定文件必须被拒绝，不产出任何结果"""
    mod = _module()
    doc = {
        "review_status": "draft",
        "decisions": {},
        "needs_revision": [],
    }
    fake = tmp_path / "decisions.json"
    fake.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(mod, "DECISIONS_PATH", fake)
    with pytest.raises(SystemExit, match="approved"):
        mod.main()


def test_builtin_definitions_match_needs_revision():
    """内置 REVIEW_ISSUES / REWRITES 与真实 approved 决定的 needs_revision 一致"""
    mod = _module()
    doc = json.loads((V5_DIR / "constructed_review_decisions.json").read_text(encoding="utf-8"))
    assert doc["review_status"] == "approved"
    needs = set(doc["needs_revision"])
    assert needs, "needs_revision 不应为空"
    assert set(mod.REVIEW_ISSUES) == needs
    assert set(mod.REWRITES) == needs
    assert len(needs) == doc["stats"]["needs_revision"] == 13


# ============================================
# 消息结构校验
# ============================================


def test_validate_messages_ok():
    mod = _module()
    msgs = [
        {"role": "user", "content": "你有朋友吗"},
        {"role": "assistant", "content": "……不多。"},
        {"role": "user", "content": "有几个"},
        {"role": "assistant", "content": "随你怎么理解。"},
    ]
    assert mod.validate_messages(msgs) == []


@pytest.mark.parametrize(
    "msgs,fragment",
    [
        # assistant 开头
        ([{"role": "assistant", "content": "在"}], "首轮不是 user"),
        # 末轮 user
        (
            [
                {"role": "user", "content": "在吗"},
                {"role": "assistant", "content": "在"},
                {"role": "user", "content": "没事"},
            ],
            "末轮不是 assistant",
        ),
        # 角色不交替（连续 user）
        (
            [
                {"role": "user", "content": "在吗"},
                {"role": "user", "content": "还在吗"},
            ],
            "角色应为 assistant",
        ),
        # 空文本
        (
            [
                {"role": "user", "content": "在吗"},
                {"role": "assistant", "content": "  "},
            ],
            "文本为空",
        ),
        # 空消息列表
        ([], "消息为空"),
    ],
)
def test_validate_messages_errors(msgs, fragment):
    mod = _module()
    errors = mod.validate_messages(msgs)
    assert errors and any(fragment in e for e in errors)


# ============================================
# 事实风险词
# ============================================


def test_fact_risks_frequency_word():
    """频率词（一直/总是/很多次/别以为/以前/曾/已经）在 assistant 侧被标记"""
    mod = _module()
    risks = mod.check_fact_risks(_qa("你害怕失去什么", "……已经失去了。很多次。"))
    words = {r["word"] for r in risks}
    assert "已经" in words and "很多次" in words


def test_fact_risks_character_not_introduced():
    """assistant 提及用户未引入的人物名 → 标记"""
    mod = _module()
    risks = mod.check_fact_risks(_qa("你有朋友吗", "夜子算一个。"))
    assert {"type": "character_name_not_introduced", "word": "夜子"} in risks


def test_fact_risks_character_introduced_by_user():
    """用户引入的人物名不罚"""
    mod = _module()
    risks = mod.check_fact_risks(_qa("夜子没来图书馆", "她大概想一个人待着。"))
    assert risks == []


# ============================================
# validation / Gold Set 重叠
# ============================================


def test_validation_overlap_exact_match():
    mod = _module()
    flags = mod.check_validation_overlap(["书会一直停在原处。"], ["书会一直停在原处。"])
    assert flags and flags[0]["kind"] == "exact_match"


def test_validation_overlap_lcs():
    """≥15 连续字符公共子串被标记"""
    mod = _module()
    flags = mod.check_validation_overlap(
        ["雨一停，那些吵闹的家伙就会回到图书馆的每个角落。"],
        ["雨一停，那些吵闹的家伙就会回到图书馆的每个角落了吧。"],
    )
    assert flags and flags[0]["kind"] == "lcs>=15"


def test_validation_overlap_none():
    mod = _module()
    assert mod.check_validation_overlap(["在。"], ["晚安。别熬到明天。"]) == []


def test_normalize_text():
    mod = _module()
    assert mod.normalize_text("在。 有事？") == mod.normalize_text("在。有事？")


# ============================================
# 真实数据端到端（重跑脚本 = 幂等重生成产物）
# ============================================


@pytest.fixture(scope="module")
def run_real():
    mod = _module()
    train_path = V4_DIR / "train.jsonl"
    sha_before = hashlib.sha256(train_path.read_bytes()).hexdigest()
    rc = mod.main()
    sha_after = hashlib.sha256(train_path.read_bytes()).hexdigest()
    return rc, sha_before, sha_after


def test_real_run_success_and_v4_untouched(run_real):
    """脚本退出码 0，且 V4 train.jsonl 前后 sha256 一致（冻结数据只读）"""
    rc, sha_before, sha_after = run_real
    assert rc == 0
    assert sha_before == sha_after


def test_real_products_structure(run_real):
    """12 候选 + 1 drop；候选状态 pending_review；ID/父 ID 规则"""
    _module()
    report = json.loads((OUT_DIR / "validation_report.json").read_text(encoding="utf-8"))
    assert report["counts"] == {"tasks": 13, "candidates": 12, "drops": 1}

    tasks = json.loads((OUT_DIR / "rewrite_tasks.json").read_text(encoding="utf-8"))["tasks"]
    drops = [t for t in tasks if t["action"] == "drop"]
    assert len(drops) == 1
    assert drops[0]["parent_record_id"] == "kisaki_llm_v4_blindfix_0048"
    assert drops[0]["drop_reason"]

    candidates = [
        json.loads(line)
        for line in (OUT_DIR / "candidates.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(candidates) == 12
    doc = json.loads((V5_DIR / "constructed_review_decisions.json").read_text(encoding="utf-8"))
    needs = set(doc["needs_revision"])
    for c in candidates:
        parent = c["metadata"]["parent_record_id"]
        assert parent in needs
        assert c["id"] == f"{parent}__rewrite_v1"
        assert c["metadata"]["status"] == "pending_review"
        assert c["metadata"]["data_source"] == "constructed_rewrite_v1"


def test_real_candidates_quality(run_real):
    """结构合法、长度 ≤100、与 130 条保留样本无完全重复、开头唯一"""
    mod = _module()
    report = json.loads((OUT_DIR / "validation_report.json").read_text(encoding="utf-8"))
    checks = report["checks"]
    assert checks["tasks_complete"]
    assert checks["new_ids_unique"]
    assert checks["parent_ids_in_needs_revision"]
    assert checks["structure_ok"]
    assert checks["assistant_length_ok"]
    assert checks["no_exact_duplicate_with_kept"]
    assert checks["openings_unique"]
    assert checks["validation_overlap_flags"] == []

    candidates = [
        json.loads(line)
        for line in (OUT_DIR / "candidates.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for c in candidates:
        assert mod.validate_messages(c["messages"]) == []


def test_real_review_batch_material(run_real):
    """审核材料：13 条对照 + 勾选行（keep/revise/drop）+ 人工备注栏"""
    _module()
    md = (OUT_DIR / "review_batch.md").read_text(encoding="utf-8")
    assert md.count("- **人工选择**: [ ] keep  [ ] revise  [ ] drop") == 13
    assert md.count("- 人工备注: ______") == 13
    assert md.count("## [") == 13
    # 每条展示原始对话与（rewrite 时）改写后对话
    assert md.count("- 原始对话：") == 13
    assert md.count("- 改写后对话（") == 12
    assert md.count("- **建议: drop**") == 1
