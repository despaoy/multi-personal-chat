"""阶段 3 复查脚本 build_kisaki_v5_constructed_review.py 的单元测试。

覆盖：
- 人物研究场景标签保底（scenario_value floor 3）；factual 类不保底
- 单轮一致性记 5（维度不适用）；多轮标记需人工检查
- 事实根基扩展人物名单（琉璃/夜子/理央）：
  用户/场景引入不罚；assistant 未引入提及 → needs_human；
  经历声称 → auto_fail；世界观事实声称 → auto_fail
- needs_human 阻断 prefer_keep 门禁
- 提问重复簇：相似问句聚簇、无关问句不合并
- 复查排序：事实可疑优先
- 历史审核信息（改写前后文本）进入批次材料
- 决定校验与阶段 3 后预算重算（原作/短构造/模拟按 approved 决定）
- 阶段 2 approved 模拟决定文件缺失时拒绝
- 真实 V4 数据：150 条短构造全部进入复查材料，分批 Markdown 生成
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V4_DIR = PROJECT_ROOT / "backend/data/character_dialogues/experiments/v4"


def _module():
    spec = importlib.util.spec_from_file_location(
        "build_kisaki_v5_constructed_review",
        PROJECT_ROOT / "scripts/build_kisaki_v5_constructed_review.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _base_module():
    """阶段 2 模块（apply_decisions 依赖其 V4_DIR）。"""
    return sys.modules["build_kisaki_v5_candidate"]


def _con_record(
    rid: str,
    scene: str,
    messages: list[dict],
    data_source: str = "llm_v4_manual",
    prior: dict | None = None,
) -> dict:
    return {
        "id": rid,
        "messages": messages,
        "metadata": {
            "data_source": data_source,
            "scene": scene,
            "interlocutor_kind": "generic_user",
            "human_review": prior or {},
        },
    }


def _qa(user: str, assistant: str) -> list[dict]:
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


# ============================================
# 场景价值：人物研究标签保底 / factual 不保底
# ============================================


def test_persona_research_scene_floor():
    """人物研究主线标签（人物关系等）场景价值保底 3——短构造的设计前提"""
    mod = _module()
    record = _con_record("c1", "人物关系", _qa("在吗", "在。……有事？"))
    score, hits = mod.score_scenario_value_constructed(record)
    assert score >= 3
    assert "persona_research_scene_floor" in {h["signal"] for h in hits}


def test_factual_scene_not_floored():
    """factual/事实与安全类不保底（世界观声称走事实根基重点检查）"""
    mod = _module()
    record = _con_record("c2", "factual", _qa("魔法之书是啥", "……书。"))
    score, hits = mod.score_scenario_value_constructed(record)
    assert score < 3
    assert not any(h["signal"] == "persona_research_scene_floor" for h in hits)


# ============================================
# 多轮一致性：单轮不适用 / 多轮需人工
# ============================================


def test_single_turn_coherence_not_applicable():
    mod = _module()
    record = _con_record("c3", "问候闲聊", _qa("在吗", "在。"))
    score, notes = mod.score_multiturn_coherence_constructed(record)
    assert score == 5
    assert "single_turn_not_applicable" in {n["signal"] for n in notes}


def test_multi_turn_marked_for_human_check():
    mod = _module()
    record = _con_record(
        "c4",
        "multiturn",
        [
            {"role": "user", "content": "你害怕什么"},
            {"role": "assistant", "content": "失去。"},
            {"role": "user", "content": "比如呢"},
            {"role": "assistant", "content": "……不说了。"},
        ],
    )
    score, notes = mod.score_multiturn_coherence_constructed(record)
    assert score < 5
    assert "multi_turn_needs_human_check" in {n["signal"] for n in notes}


# ============================================
# 事实根基：扩展人物名单（琉璃/夜子/理央）
# ============================================


def test_user_introduced_name_not_flagged():
    """用户先提到夜子 → assistant 再提不罚"""
    mod = _module()
    record = _con_record(
        "c5",
        "夜子日常",
        _qa("夜子今天很安静", "是吗。她大概又在想复杂的事。"),
    )
    verdict, notes = mod.check_factual_grounding_constructed(record)
    assert verdict == "no_auto_flag"
    assert not any(n["signal"] == "character_name_uninvited" for n in notes)


def test_scene_label_introduces_name():
    """scene 标签（琉璃斗嘴）已引入琉璃 → assistant 提及不罚"""
    mod = _module()
    record = _con_record(
        "c6",
        "琉璃斗嘴",
        _qa("又在和你哥吵架？", "哼。啰嗦的哥哥而已。"),
    )
    # assistant 未直接说"琉璃"，改用直接提及的例子
    record2 = _con_record(
        "c6b",
        "琉璃斗嘴",
        _qa("又在和你哥吵架？", "……琉璃他就是啰嗦。"),
    )
    for r in (record, record2):
        verdict, _ = mod.check_factual_grounding_constructed(r)
        assert verdict == "no_auto_flag"


def test_uninvited_name_needs_human():
    """assistant 未引入提及理央 → needs_human（角色问答需人工确认关系设定）"""
    mod = _module()
    record = _con_record("c7", "兴趣偏好", _qa("教我做菜", "我不太会。找理央。"))
    verdict, notes = mod.check_factual_grounding_constructed(record)
    assert verdict == "needs_human"
    assert "character_name_uninvited" in {n["signal"] for n in notes}


def test_experience_claim_auto_fail():
    """人物名 + 经历语境（夜子以前生病时）→ auto_fail"""
    mod = _module()
    record = _con_record(
        "c8",
        "温柔关心",
        _qa("她怎么了", "夜子以前生病时也这样。让她休息吧。"),
    )
    verdict, notes = mod.check_factual_grounding_constructed(record)
    assert verdict == "auto_fail"
    assert "character_experience_claim" in {n["signal"] for n in notes}


def test_world_fact_claim_auto_fail():
    mod = _module()
    record = _con_record(
        "c9",
        "factual",
        _qa("这个世界是怎样的", "在我们的世界，魔法浓度决定一切。"),
    )
    verdict, _ = mod.check_factual_grounding_constructed(record)
    assert verdict == "auto_fail"


def test_gate_blocks_needs_human():
    """needs_human 阻断 prefer_keep——即使其他维度全优"""
    mod = _module()
    record = _con_record("c10", "人物关系", _qa("你有朋友吗", "……夜子算一个。"))
    result = mod.score_constructed_record(record)
    assert result["dimensions"]["factual_grounding"] == "needs_human"
    assert result["gate_pass"] is False
    assert result["machine_suggestion"] != "prefer_keep"


# ============================================
# 提问重复簇
# ============================================


def test_question_clusters_similar():
    """ "你觉得什么是X"系列聚同簇（实测 7 个相似对的主体）"""
    mod = _module()
    entries = [
        {"id": "q1", "user_text": "你觉得什么是幸福"},
        {"id": "q2", "user_text": "你觉得什么是勇敢"},
        {"id": "q3", "user_text": "你觉得什么是爱"},
    ]
    cluster_map = mod.cluster_user_questions(entries)
    assert len({cluster_map[e["id"]] for e in entries}) == 1


def test_question_clusters_dont_merge_unrelated():
    mod = _module()
    entries = [
        {"id": "q1", "user_text": "你觉得什么是幸福"},
        {"id": "q4", "user_text": "晚饭吃什么"},
        {"id": "q5", "user_text": "在吗"},
    ]
    cluster_map = mod.cluster_user_questions(entries)
    assert cluster_map["q1"] != cluster_map["q4"]
    assert cluster_map["q4"] != cluster_map["q5"]


# ============================================
# 复查排序与历史审核信息
# ============================================


def test_sorting_needs_human_first():
    """事实可疑（needs_human）排在 no_auto_flag 之前"""
    mod = _module()
    records = [
        _con_record("clean1", "问候闲聊", _qa("在吗", "在。")),
        _con_record("suspect", "兴趣偏好", _qa("教我做菜", "找理央。")),
        _con_record("clean2", "问候闲聊", _qa("早", "……嗯。")),
    ]
    entries = mod.build_constructed_packet(records)
    assert entries[0]["id"] == "suspect"


def test_prior_review_info_in_batches(tmp_path, monkeypatch):
    """曾改写记录（blindfix 改写前后文本）必须进入批次材料"""
    mod = _module()
    monkeypatch.setattr(mod, "CONSTRUCTED_BATCH_DIR", tmp_path / "batches")
    prior = {
        "status": "approved_after_revision",
        "reviewed_by": "project_owner",
        "reviewed_at": "2026-08-09",
        "reason": "修复指代。",
        "original_assistant_messages": ["旧的回复。"],
    }
    record = _con_record("bf1", "multiturn", _qa("问题", "新回复。"), data_source="llm_v4_blindfix", prior=prior)
    entries = mod.build_constructed_packet([record])
    paths = mod.write_constructed_batches(entries, {})
    text = paths[0].read_text(encoding="utf-8")
    assert "改写理由: 修复指代。" in text
    assert "旧的回复。" in text
    assert "approved_after_revision" in text


# ============================================
# --decisions：预算重算与门禁
# ============================================


def _write_train(tmp_path: Path, records: list[dict]) -> None:
    train = tmp_path / "train.jsonl"
    train.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "dataset_id": "TEST",
        "train": {"path": "train.jsonl", "sha256": hashlib.sha256(train.read_bytes()).hexdigest()},
        "validation": {"sha256": "v"},
    }
    (tmp_path / "canonical_dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _minimal_v4_records() -> list[dict]:
    game = {
        "id": "g1",
        "messages": _qa("你好", "……嗯。你好。"),
        "metadata": {"data_source": "game_extraction", "scene": "日常"},
    }
    con1 = _con_record("con1", "问候闲聊", _qa("在吗", "在。……有事？"))
    con2 = _con_record("con2", "兴趣偏好", _qa("晚饭吃什么", "随便。"))
    sim1 = {
        "id": "sim1",
        "messages": _qa("早上好", "……早。有事快说。"),
        "metadata": {
            "data_source": "codex_user_simulation_v41_reviewed",
            "scene": "问候",
            "task_type": "casual_chat",
        },
    }
    return [game, con1, con2, sim1]


def test_apply_decisions_budget_recalc(tmp_path, monkeypatch):
    """阶段 3 后预算重算：原作/短构造/模拟全部按 approved 决定计入"""
    mod = _module()
    _write_train(tmp_path, _minimal_v4_records())
    monkeypatch.setattr(_base_module(), "V4_DIR", tmp_path)
    monkeypatch.setattr(mod, "SIMULATION_DECISIONS_PATH", tmp_path / "sim_decisions.json")

    (tmp_path / "sim_decisions.json").write_text(
        json.dumps({"review_status": "approved", "reviewed_by": "o", "decisions": {"sim1": "keep"}}),
        encoding="utf-8",
    )
    con_decisions = tmp_path / "con_decisions.json"
    con_decisions.write_text(
        json.dumps(
            {
                "review_status": "approved",
                "reviewed_by": "o",
                "decisions": {"con1": "keep", "con2": "exclude"},
            }
        ),
        encoding="utf-8",
    )

    result = mod.apply_decisions(con_decisions)
    budget = result["budget_final_after_phase3"]
    assert result["constructed_kept_records"] == ["con1"]
    assert budget["game_extraction"]["records"] == 1
    assert budget["constructed_kept"]["records"] == 1
    assert budget["simulation_kept"]["records"] == 1
    # 总字符 = 原作 + 保留短构造 + 保留模拟
    assert budget["total_sup_chars"] == (
        budget["game_extraction"]["sup_chars"]
        + budget["constructed_kept"]["sup_chars"]
        + budget["simulation_kept"]["sup_chars"]
    )
    # 占比合计约 100（消除"假定全留"初步口径）
    assert round(sum(budget["share_pct"].values())) == 100
    assert "不再有假定全留" in budget["note"]


def test_apply_decisions_requires_simulation_decisions(tmp_path, monkeypatch):
    """阶段 2 approved 模拟决定文件缺失 → 拒绝（预算重算依赖）"""
    mod = _module()
    _write_train(tmp_path, _minimal_v4_records())
    monkeypatch.setattr(_base_module(), "V4_DIR", tmp_path)
    monkeypatch.setattr(mod, "SIMULATION_DECISIONS_PATH", tmp_path / "missing.json")
    con_decisions = tmp_path / "con_decisions.json"
    con_decisions.write_text(
        json.dumps(
            {
                "review_status": "approved",
                "reviewed_by": "o",
                "decisions": {"con1": "keep", "con2": "exclude"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="阶段 2"):
        mod.apply_decisions(con_decisions)


def test_apply_decisions_requires_simulation_approved(tmp_path, monkeypatch):
    """阶段 2 决定文件存在但未 approved → 拒绝"""
    mod = _module()
    _write_train(tmp_path, _minimal_v4_records())
    monkeypatch.setattr(_base_module(), "V4_DIR", tmp_path)
    monkeypatch.setattr(mod, "SIMULATION_DECISIONS_PATH", tmp_path / "sim_decisions.json")
    (tmp_path / "sim_decisions.json").write_text(
        json.dumps({"review_status": "draft", "reviewed_by": None, "decisions": {"sim1": "keep"}}),
        encoding="utf-8",
    )
    con_decisions = tmp_path / "con_decisions.json"
    con_decisions.write_text(
        json.dumps(
            {
                "review_status": "approved",
                "reviewed_by": "o",
                "decisions": {"con1": "keep", "con2": "exclude"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="review_status"):
        mod.apply_decisions(con_decisions)


def test_apply_decisions_rejects_invalid_value(tmp_path, monkeypatch):
    """非法决定值（revise 未转换）必须报错，绝不静默当 exclude"""
    mod = _module()
    _write_train(tmp_path, _minimal_v4_records())
    monkeypatch.setattr(_base_module(), "V4_DIR", tmp_path)
    monkeypatch.setattr(mod, "SIMULATION_DECISIONS_PATH", tmp_path / "sim_decisions.json")
    (tmp_path / "sim_decisions.json").write_text(
        json.dumps({"review_status": "approved", "reviewed_by": "o", "decisions": {"sim1": "keep"}}),
        encoding="utf-8",
    )
    con_decisions = tmp_path / "con_decisions.json"
    con_decisions.write_text(
        json.dumps(
            {
                "review_status": "approved",
                "reviewed_by": "o",
                "decisions": {"con1": "keep", "con2": "revise"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="非法决定值"):
        mod.apply_decisions(con_decisions)


# ============================================
# 真实 V4 数据
# ============================================


def test_real_v4_constructed_packet_covers_150(tmp_path, monkeypatch):
    mod = _module()
    monkeypatch.setattr(mod, "V5_DIR", tmp_path)
    monkeypatch.setattr(mod, "CONSTRUCTED_BATCH_DIR", tmp_path / "constructed_review_batches")

    records, manifest = mod.load_canonical_records()
    constructed = [r for r in records if mod.classify_record_source(r) == "llm_v4_reviewed_constructed"]
    assert len(constructed) == 150

    entries = mod.build_constructed_packet(constructed)
    cluster_map = mod.cluster_user_questions(entries)
    paths = mod.write_constructed_batches(entries, cluster_map)

    # 150 条 / 每批 ≤25 → 6 批
    assert len(paths) == 6
    total_lines = 0
    ids = []
    for p in paths:
        text = p.read_text(encoding="utf-8")
        rec_ids = [line[7:-1] for line in text.splitlines() if line.startswith("- ID: `")]
        ids.extend(rec_ids)
        total_lines += len(rec_ids)
    assert total_lines == 150
    assert len(set(ids)) == 150

    # 排序：needs_human 条目必须出现在第一批
    first_batch = paths[0].read_text(encoding="utf-8")
    assert "需人工确认（未引入原作人物/元素）" in first_batch

    # packet 结构
    packet = {
        "entries": entries,
    }
    for e in packet["entries"]:
        dims = e["scoring"]["dimensions"]
        assert set(dims) == {
            "scenario_value",
            "persona_fidelity",
            "factual_grounding",
            "multiturn_coherence",
            "generic_assistant_risk",
        }
        assert dims["factual_grounding"] in ("no_auto_flag", "needs_human", "auto_fail")
        assert e["human_decision"] is None
        # 历史审核信息结构完整（阶段 3 复查的关键上下文）
        assert "prior_review" in e
        assert "status" in e["prior_review"]


def test_real_v4_factual_states(tmp_path):
    """真实数据三态分布已知：needs_human=8（未引入人物名），其余 no_auto_flag"""
    mod = _module()
    records, _ = mod.load_canonical_records()
    constructed = [r for r in records if mod.classify_record_source(r) == "llm_v4_reviewed_constructed"]
    entries = mod.build_constructed_packet(constructed)
    states = [e["scoring"]["dimensions"]["factual_grounding"] for e in entries]
    assert states.count("needs_human") == 8
    assert states.count("auto_fail") == 0
    assert states.count("no_auto_flag") == 142
