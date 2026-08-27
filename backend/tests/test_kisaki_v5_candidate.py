"""阶段 2 筛选脚本 build_kisaki_v5_candidate.py 的单元测试（v3 五维/三态事实）。

覆盖：
- 五个维度独立评分（场景/人物/事实/一致/通用助手风险）
- 事实根基三态：no_auto_flag / needs_human / auto_fail
- prefer_keep 门禁互不抵消：场景价值不能补偿人物还原差
- 原作元素分级：否定/比喻不罚，事实声称/人物经历判 auto_fail
- 技术排除队列：task_type 优先，场景价值封顶 2
- 场景概念聚类：同义改写归一（屡次/频繁→反复），重复簇完整
- 预算基线动态计算（不硬编码）
- 决定文件严格校验：ID 全等、值域、review_status=approved
- 真实 V4 数据：254 条模拟全部进入审核材料，分批 Markdown 生成
"""

import importlib.util
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V4_DIR = PROJECT_ROOT / "backend/data/character_dialogues/experiments/v4"


def _module():
    spec = importlib.util.spec_from_file_location(
        "build_kisaki_v5_candidate",
        PROJECT_ROOT / "scripts/build_kisaki_v5_candidate.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sim_record(rid: str, scene: str, messages: list[dict], task_type: str = "") -> dict:
    return {
        "id": rid,
        "messages": messages,
        "metadata": {
            "data_source": "codex_user_simulation_v41_reviewed",
            "scene": scene,
            "task_type": task_type,
            "interlocutor_kind": "generic_user",
            "human_review": {"status": "approved_after_revision"},
        },
    }


# ============================================
# 维度 1：场景价值
# ============================================


def test_scenario_value_counts_positive_signals():
    mod = _module()
    record = _sim_record(
        "s1",
        "朋友反复取消约定的边界沟通",
        [
            {"role": "user", "content": "我们约好了周五看电影，他又说没空，我该生气吗"},
            {"role": "assistant", "content": "……知道了。"},
            {"role": "user", "content": "明天要不要再约一次"},
            {"role": "assistant", "content": "随便你。"},
        ],
    )
    score, hits = mod.score_scenario_value(record)
    signal_names = {h["signal"] for h in hits}
    assert "promise_plan" in signal_names
    assert "relationship_boundary" in signal_names
    # 4 用户轮的加分也应触发
    assert score >= 2


def test_scenario_value_tech_task_type_capped_at_2():
    mod = _module()
    record = _sim_record(
        "s2",
        "迭代中途处理需求变更",
        [
            {"role": "user", "content": "答应好的功能要改，怎么办"},
            {"role": "assistant", "content": "……哼。"},
        ],
        task_type="project_collaboration",
    )
    score, hits = mod.score_scenario_value(record)
    # 技术队列：场景价值封顶 2，即使正向信号再多
    assert score <= 2
    assert "tech_task_type_capped" in {h["signal"] for h in hits}


def test_tech_scenario_routed_to_prefer_exclude():
    mod = _module()
    record = _sim_record(
        "s3",
        "初学者理解 Git merge 与 rebase",
        [
            {"role": "user", "content": "git rebase 和 merge 有什么区别"},
            {"role": "assistant", "content": "简单说……"},
        ],
        task_type="tool_usage",
    )
    result = mod.score_simulation_value(record)
    assert result["tech_queue"] is True
    assert result["machine_suggestion"] == "prefer_exclude"


# ============================================
# 维度 2：人物风格（机器代理）
# ============================================


def test_persona_fidelity_short_sarcasm_reply_scores_well():
    mod = _module()
    record = _sim_record(
        "s4",
        "晚饭吃什么",
        [
            {"role": "user", "content": "晚饭吃什么"},
            {"role": "assistant", "content": "随便。哼。"},
        ],
    )
    score, notes = mod.score_persona_fidelity(record)
    assert score >= 4
    assert "sarcasm_style_markers" in {n["signal"] for n in notes}


def test_persona_fidelity_service_language_penalized():
    mod = _module()
    record = _sim_record(
        "s5",
        "咨询建议",
        [
            {"role": "user", "content": "我该怎么办"},
            {"role": "assistant", "content": "您好，很高兴为您服务。"},
        ],
    )
    score, notes = mod.score_persona_fidelity(record)
    assert score < 4
    assert "service_language" in {n["signal"] for n in notes}


# ============================================
# 维度 3：事实根基三态（原作元素分级）
# ============================================


def test_factual_grounding_negation_is_no_auto_flag():
    """否定语境（"这不是魔法"）→ no_auto_flag，不升级——用户指出的误判"""
    mod = _module()
    record = _sim_record(
        "s6",
        "减缓香蕉成熟",
        [
            {"role": "user", "content": "怎么让香蕉成熟慢一点？"},
            {"role": "assistant", "content": "可以把香蕉分开存放并让空气流通，这不是魔法。"},
        ],
    )
    verdict, notes = mod.check_factual_grounding(record)
    assert verdict == "no_auto_flag"
    assert any(n["signal"] == "lore_negated" for n in notes)


def test_factual_grounding_metaphor_is_no_auto_flag():
    """比喻语境（"仿佛魔法"）→ no_auto_flag"""
    mod = _module()
    record = _sim_record(
        "s7",
        "学习计划",
        [
            {"role": "user", "content": "帮我做个学习计划"},
            {"role": "assistant", "content": "临时抱佛脚仿佛魔法，其实没用。"},
        ],
    )
    verdict, notes = mod.check_factual_grounding(record)
    assert verdict == "no_auto_flag"
    assert any(n["signal"] == "lore_metaphor" for n in notes)


def test_factual_grounding_negated_metaphor_is_no_auto_flag():
    """否定+比喻（"不是能解决一切的魔法"）同样不升级（任一语境命中即可）"""
    mod = _module()
    record = _sim_record(
        "s7b",
        "学习计划",
        [
            {"role": "user", "content": "帮我做个学习计划"},
            {"role": "assistant", "content": "计划不是能解决一切的魔法，先自己做。"},
        ],
    )
    verdict, notes = mod.check_factual_grounding(record)
    assert verdict == "no_auto_flag"
    assert any(n["signal"] in ("lore_negated", "lore_metaphor") for n in notes)


def test_factual_grounding_world_fact_claim_is_auto_fail():
    """世界观事实声称（"我们的世界由魔法决定"）→ auto_fail"""
    mod = _module()
    record = _sim_record(
        "s8",
        "聊世界观",
        [
            {"role": "user", "content": "你相信什么"},
            {"role": "assistant", "content": "在我们的世界，很多事情由魔法决定。"},
        ],
    )
    verdict, notes = mod.check_factual_grounding(record)
    assert verdict == "auto_fail"
    assert any(n["signal"] == "world_fact_claim" for n in notes)


def test_factual_grounding_character_experience_is_auto_fail():
    """人物经历声称（"琉璃以前生病时"）→ auto_fail"""
    mod = _module()
    record = _sim_record(
        "s9",
        "周末在家整理房间",
        [
            {"role": "user", "content": "周末在家整理房间好累"},
            {"role": "assistant", "content": "我以前和琉璃在社刊室也整理过，真是的。"},
        ],
    )
    verdict, notes = mod.check_factual_grounding(record)
    assert verdict == "auto_fail"
    assert any(n["signal"] == "character_experience_claim" for n in notes)


def test_factual_grounding_uninvited_name_is_needs_human():
    """未由用户引入的人物名、无虚构语境 → needs_human（非 fail，需人工判断）"""
    mod = _module()
    record = _sim_record(
        "s9b",
        "周末在家整理房间",
        [
            {"role": "user", "content": "周末在家整理房间好累"},
            {"role": "assistant", "content": "琉璃大概又会说这是偷懒吧，真是的。"},
        ],
    )
    verdict, notes = mod.check_factual_grounding(record)
    assert verdict == "needs_human"
    assert any(n["signal"] == "character_name_uninvited" for n in notes)
    # needs_human 不应触发 prefer_exclude（留给人工）
    result = mod.score_simulation_value(record)
    assert result["machine_suggestion"] != "prefer_exclude" or result["tech_queue"]


def test_factual_grounding_needs_human_blocks_prefer_keep():
    """needs_human 不能进入 prefer_keep（必须先经人工确认）"""
    mod = _module()
    record = _sim_record(
        "s9c",
        "朋友反复取消约定的边界沟通",
        [
            {"role": "user", "content": "我们约好了周五看电影，他又说没空"},
            {"role": "assistant", "content": "琉璃大概也会这样吧。"},
            {"role": "user", "content": "明天要不要再约一次，我不确定"},
            {"role": "assistant", "content": "随便你。哼。"},
            {"role": "user", "content": "他上次也是这样"},
            {"role": "assistant", "content": "真是的。"},
        ],
    )
    result = mod.score_simulation_value(record)
    assert result["dimensions"]["factual_grounding"] == "needs_human"
    assert result["gate_pass"] is False
    assert result["machine_suggestion"] != "prefer_keep"


def test_factual_grounding_user_introduced_is_no_auto_flag():
    """用户引入的人物名不应升级"""
    mod = _module()
    record = _sim_record(
        "s10",
        "聊聊社团活动",
        [
            {"role": "user", "content": "你和琉璃平时在社刊室做什么"},
            {"role": "assistant", "content": "……和你没关系。"},
        ],
    )
    verdict, notes = mod.check_factual_grounding(record)
    assert verdict == "no_auto_flag"
    assert not any(n["signal"] in ("character_experience_claim", "world_fact_claim", "experience_claim") for n in notes)


# ============================================
# 维度 5：通用助手风险
# ============================================


def test_generic_assistant_risk_structured_reply():
    mod = _module()
    record = _sim_record(
        "s11",
        "选择困难",
        [
            {"role": "user", "content": "两个都想买，怎么选"},
            {"role": "assistant", "content": "首先看预算，其次看需求，最后综合建议如下：\n1. 列清单\n2. 打分"},
        ],
    )
    score, notes = mod.score_generic_assistant_risk(record)
    assert score >= 2
    assert "structured_reply" in {n["signal"] for n in notes}


def test_generic_assistant_risk_short_terse_reply():
    mod = _module()
    record = _sim_record(
        "s12",
        "晚饭吃什么",
        [
            {"role": "user", "content": "晚饭吃什么"},
            {"role": "assistant", "content": "随便。"},
        ],
    )
    score, _ = mod.score_generic_assistant_risk(record)
    assert score == 0


# ============================================
# 门禁：互不抵消
# ============================================


def test_gate_high_scenario_value_cannot_offset_low_persona():
    """核心要求：场景价值很高不能补偿人物还原很差。

    通用心理咨询式回答（长、服务语、建议密度高）即使场景
    涉及承诺/边界，也必须被挡在 prefer_keep 之外。
    """
    mod = _module()
    record = _sim_record(
        "s13",
        "朋友反复取消约定",
        [
            {"role": "user", "content": "我们约好了周五看电影，他又说没空，我该生气吗，该拒绝他吗"},
            {
                "role": "assistant",
                "content": "首先理解你的感受，其次建议直接沟通。建议如下：\n1. 表达感受\n2. 倾听对方\n3. 建议约定下次时间。希望这些建议对你有帮助。",
            },
        ],
    )
    result = mod.score_simulation_value(record)
    d = result["dimensions"]
    # 场景价值达标
    assert d["scenario_value"] >= mod.GATE["scenario_value_min"]
    # 但人物还原差 + 通用助手风险高
    assert d["persona_fidelity"] < mod.GATE["persona_fidelity_min"]
    assert d["generic_assistant_risk"] > mod.GATE["generic_assistant_risk_max"]
    assert result["gate_pass"] is False
    assert result["machine_suggestion"] != "prefer_keep"


def test_gate_all_dimensions_pass_yields_prefer_keep():
    mod = _module()
    record = _sim_record(
        "s14",
        "朋友反复取消约定的边界沟通",
        [
            {"role": "user", "content": "我们约好了周五看电影，他又说没空"},
            {"role": "assistant", "content": "……知道了。"},
            {"role": "user", "content": "明天要不要再约一次，我不确定"},
            {"role": "assistant", "content": "随便你。哼。"},
            {"role": "user", "content": "他上次也是这样"},
            {"role": "assistant", "content": "真是的。"},
            {"role": "user", "content": "那我拒绝他？"},
            {"role": "assistant", "content": "……啰嗦。"},
        ],
    )
    result = mod.score_simulation_value(record)
    assert result["gate_pass"] is True, result
    assert result["machine_suggestion"] == "prefer_keep"


def test_gate_factual_auto_fail_forces_prefer_exclude():
    mod = _module()
    record = _sim_record(
        "s15",
        "朋友反复取消约定的边界沟通",
        [
            {"role": "user", "content": "我们约好了周五看电影，他又说没空"},
            {"role": "assistant", "content": "琉璃以前生病时也放过我鸽子，在我们的世界这由魔法决定。哼。"},
            {"role": "user", "content": "明天要不要再约一次"},
            {"role": "assistant", "content": "随便你。"},
        ],
    )
    result = mod.score_simulation_value(record)
    assert result["dimensions"]["factual_grounding"] == "auto_fail"
    assert result["machine_suggestion"] == "prefer_exclude"


# ============================================
# 决定门禁：validate_decision_document
# ============================================


def test_decision_doc_requires_approved_status():
    mod = _module()
    # 只有 reviewed_by，没有 review_status=approved → 拒绝
    with pytest.raises(SystemExit, match="review_status"):
        mod.validate_decision_document({"reviewed_by": "owner", "decisions": {"a": "keep"}}, {"a"})


def test_decision_doc_requires_reviewed_by():
    mod = _module()
    with pytest.raises(SystemExit, match="reviewed_by"):
        mod.validate_decision_document({"review_status": "approved", "decisions": {"a": "keep"}}, {"a"})


def test_decision_doc_rejects_unknown_ids():
    mod = _module()
    with pytest.raises(SystemExit, match="未知 ID"):
        mod.validate_decision_document(
            {"review_status": "approved", "reviewed_by": "o", "decisions": {"a": "keep", "ghost": "keep"}},
            {"a"},
        )


def test_decision_doc_rejects_missing_ids():
    mod = _module()
    with pytest.raises(SystemExit, match="未出现在决定文件"):
        mod.validate_decision_document(
            {"review_status": "approved", "reviewed_by": "o", "decisions": {"a": "keep"}},
            {"a", "b"},
        )


def test_decision_doc_rejects_invalid_values_not_silent_exclude():
    """非法决定值（kepe/空串/pending）必须报错，绝不静默当成 exclude"""
    mod = _module()
    for bad in ("kepe", "", "pending", "KEEP", "revise"):
        with pytest.raises(SystemExit, match="非法决定值"):
            mod.validate_decision_document(
                {"review_status": "approved", "reviewed_by": "o", "decisions": {"a": bad}},
                {"a"},
            )


def test_decision_doc_accepts_valid_keep_exclude():
    mod = _module()
    mod.validate_decision_document(
        {"review_status": "approved", "reviewed_by": "o", "decisions": {"a": "keep", "b": "exclude"}},
        {"a", "b"},
    )


# ============================================
# build_candidate_dataset：只认人工批准决定
# ============================================


def _write_train(tmp_path: Path, records: list[dict]) -> None:
    import hashlib

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


def _two_sims() -> list[dict]:
    return [
        _sim_record(
            f"sim{i}",
            f"场景{i}",
            [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "……嗯。"},
            ],
        )
        for i in range(2)
    ]


def test_candidate_dataset_requires_approved_status(tmp_path, monkeypatch):
    mod = _module()
    _write_train(tmp_path, _two_sims())
    monkeypatch.setattr(mod, "V4_DIR", tmp_path)

    decisions = tmp_path / "decisions.json"
    decisions.write_text(
        json.dumps({"decisions": {"sim0": "keep", "sim1": "keep"}, "reviewed_by": "owner"}),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="review_status"):
        mod.build_candidate_dataset(decisions)


def test_candidate_dataset_rejects_missing_records(tmp_path, monkeypatch):
    mod = _module()
    _write_train(tmp_path, _two_sims())
    monkeypatch.setattr(mod, "V4_DIR", tmp_path)

    # 只决定了 sim0，sim1 未出现在决定文件中 → 拒绝
    decisions = tmp_path / "decisions.json"
    decisions.write_text(
        json.dumps({"decisions": {"sim0": "keep"}, "reviewed_by": "owner", "review_status": "approved"}),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="未出现在决定文件"):
        mod.build_candidate_dataset(decisions)


def test_candidate_dataset_rejects_invalid_decision_value(tmp_path, monkeypatch):
    mod = _module()
    _write_train(tmp_path, _two_sims())
    monkeypatch.setattr(mod, "V4_DIR", tmp_path)

    decisions = tmp_path / "decisions.json"
    decisions.write_text(
        json.dumps(
            {
                "decisions": {"sim0": "kepe", "sim1": "keep"},
                "reviewed_by": "owner",
                "review_status": "approved",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="非法决定值"):
        mod.build_candidate_dataset(decisions)


def test_candidate_dataset_applies_approved_decisions(tmp_path, monkeypatch):
    mod = _module()
    _write_train(tmp_path, _two_sims())
    monkeypatch.setattr(mod, "V4_DIR", tmp_path)

    decisions = tmp_path / "decisions.json"
    decisions.write_text(
        json.dumps(
            {
                "decisions": {"sim0": "keep", "sim1": "exclude"},
                "reviewed_by": "owner",
                "review_status": "approved",
            }
        ),
        encoding="utf-8",
    )
    result = mod.build_candidate_dataset(decisions)
    assert result["kept_records"] == 1
    assert result["excluded_records"] == 1
    assert [r["id"] for r in result["records"]] == ["sim0"]
    # 预算占比字段名带 preliminary 标记（阶段 3 后重算）
    assert isinstance(result["sim_sup_char_share_pct_preliminary"], float)


# ============================================
# 场景重复簇（概念标签 + 同义归一）
# ============================================


def test_scene_concepts_synonym_normalization():
    """同义改写归一：屡次/频繁/总是 → 反复"""
    mod = _module()
    assert mod.scene_concepts("朋友屡次临时取消约定") == mod.scene_concepts("朋友频繁临时取消约定")
    assert "freq:反复" in mod.scene_concepts("朋友屡次爽约")


def test_cluster_scenes_four_cancellation_variants_merge():
    """用户指出的漏检：4 条"朋友反复临时取消约定"变体必须聚到同簇"""
    mod = _module()
    entries = [
        {"id": "b006", "scene": "朋友反复临时取消约定", "task_type": "emotional_relationship"},
        {"id": "b026", "scene": "面对朋友反复临时取消约定的失落与边界", "task_type": "emotional_relationship"},
        {"id": "b035", "scene": "朋友屡次临时取消约定时表达失望并调整投入", "task_type": "emotional_relationship"},
        {"id": "b048", "scene": "面对朋友频繁临时取消约定时调整期待", "task_type": "emotional_relationship"},
    ]
    cluster_map = mod.cluster_scenes(entries)
    clusters = {cluster_map[e["id"]] for e in entries}
    assert len(clusters) == 1  # 4 条全进同一簇


def test_cluster_scenes_different_behavior_not_merged():
    """同对象不同行为不合并：借钱 vs 取消约定"""
    mod = _module()
    entries = [
        {"id": "a", "scene": "朋友反复借钱不还的边界", "task_type": "emotional_relationship"},
        {"id": "b", "scene": "朋友反复临时取消约定", "task_type": "emotional_relationship"},
    ]
    cluster_map = mod.cluster_scenes(entries)
    assert cluster_map["a"] != cluster_map["b"]


def test_cluster_scenes_requires_same_task_type():
    """不同 task_type 不比较（用户要求先按 task_type 分组）"""
    mod = _module()
    entries = [
        {"id": "a", "scene": "朋友反复临时取消约定", "task_type": "emotional_relationship"},
        {"id": "b", "scene": "面对朋友屡次临时取消约定的边界", "task_type": "casual_chat"},
    ]
    cluster_map = mod.cluster_scenes(entries)
    assert cluster_map["a"] != cluster_map["b"]


def test_cluster_scenes_ignores_unrelated_scenes():
    """无关场景不并入取消约定簇"""
    mod = _module()
    entries = [
        {"id": "a", "scene": "朋友反复临时取消约定", "task_type": "emotional_relationship"},
        {"id": "b", "scene": "晚饭吃什么", "task_type": "emotional_relationship"},
    ]
    cluster_map = mod.cluster_scenes(entries)
    assert cluster_map["a"] != cluster_map["b"]


# ============================================
# 预算基线动态计算
# ============================================


def test_budget_baseline_computed_from_records_matches_phase1(tmp_path):
    """动态基线必须与阶段 1 资产清单实测一致（不硬编码也不漂移）"""
    mod = _module()
    records, _ = mod.load_canonical_records()
    baseline = mod.compute_budget_baseline(records)
    stats = json.loads(
        (PROJECT_ROOT / "backend/data/character_dialogues/experiments/v5_candidate/asset_stats.json").read_text(
            encoding="utf-8"
        )
    )
    assert baseline["game_records"] == 522
    assert baseline["game_sup_chars"] == 16433
    assert baseline["constructed_records"] == 150
    assert baseline["constructed_sup_chars"] == 2883
    assert "阶段 3" in baseline["note"]


def test_budget_baseline_rejects_unknown_source():
    mod = _module()
    bad = {
        "id": "x",
        "messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}],
        "metadata": {"data_source": "mystery_source"},
    }
    with pytest.raises(ValueError, match="未知 data_source"):
        mod.compute_budget_baseline([bad])


# ============================================
# 真实 V4 数据
# ============================================


def test_real_v4_packet_covers_all_254_sims(tmp_path, monkeypatch):
    mod = _module()
    monkeypatch.setattr(mod, "V5_DIR", tmp_path)
    monkeypatch.setattr(mod, "REVIEW_BATCH_DIR", tmp_path / "review_batches")

    records, manifest = mod.load_canonical_records()
    sim_records = [r for r in records if mod.classify_record_source(r) in mod.SIMULATION_SOURCES]
    assert len(sim_records) == 254

    baseline = mod.compute_budget_baseline(records)
    entries = mod.build_review_packet(sim_records)
    mod.write_review_packet(entries, manifest, baseline)

    packet = json.loads((tmp_path / "simulation_review_packet.json").read_text(encoding="utf-8"))
    assert len(packet["entries"]) == 254
    # 机器建议与人工决定严格分离：全部人工决定留空待批准
    assert all(e["human_decision"] is None for e in packet["entries"])
    # 建议带只有三种合法值
    assert {e["scoring"]["machine_suggestion"] for e in packet["entries"]} <= {
        "prefer_keep",
        "review_priority",
        "prefer_exclude",
    }
    # 五维结构完整、事实三态合法
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
        # 门禁通过当且仅当全维度达标且不在技术队列
        expect_pass = (
            dims["scenario_value"] >= mod.GATE["scenario_value_min"]
            and dims["persona_fidelity"] >= mod.GATE["persona_fidelity_min"]
            and dims["factual_grounding"] == "no_auto_flag"
            and dims["multiturn_coherence"] >= mod.GATE["multiturn_coherence_min"]
            and dims["generic_assistant_risk"] <= mod.GATE["generic_assistant_risk_max"]
            and not e["scoring"]["tech_queue"]
        )
        assert e["scoring"]["gate_pass"] == expect_pass
        if e["scoring"]["gate_pass"]:
            assert e["scoring"]["machine_suggestion"] == "prefer_keep"
        if dims["factual_grounding"] == "auto_fail":
            assert e["scoring"]["machine_suggestion"] == "prefer_exclude"
    # 按建议带排序（prefer_keep > review_priority > prefer_exclude）
    band_order = {"prefer_keep": 0, "review_priority": 1, "prefer_exclude": 2}
    bands = [band_order[e["scoring"]["machine_suggestion"]] for e in packet["entries"]]
    assert bands == sorted(bands)
    # 预算：基线动态写入 packet、模拟表随 k 单调递增、全部标记 preliminary
    assert packet["budget_baseline_computed"]["game_records"] == 522
    assert packet["budget_baseline_computed"]["constructed_records"] == 150
    budget = packet["budget_simulation_preliminary"]
    assert len(budget) == 254
    shares = [row["sim_sup_char_share_pct"] for row in budget]
    assert shares == sorted(shares)
    assert all(row["preliminary"] is True for row in budget)
    # 4 条取消约定样本必须落在同一重复簇
    cancel_ids = {e["id"] for e in packet["entries"] if "cancel" in e["id"].lower()}
    assert len(cancel_ids) == 4
    cluster_ids = [set(v) for v in packet["scene_clusters"].values()]
    assert any(cancel_ids <= c for c in cluster_ids), "4 条取消约定未聚到同簇"
    # 数据缺口说明已记录（差异化反应维度缺失）
    assert "interlocutor_kind" in packet["policy"]["data_gap"]
    # 三态说明与簇说明已写入 policy
    assert "no_auto_flag" in packet["policy"]["factual_grounding_note"]
    assert "不自动排除" in packet["policy"]["cluster_note"]
    # 分批 Markdown 与摘要同时生成
    batches = sorted((tmp_path / "review_batches").glob("batch_*.md"))
    assert len(batches) == 11  # 254 / 25 → 11 批
    assert (tmp_path / "simulation_review_summary.md").exists()
    # 批次含人工选择栏、三态事实展示与完整对话
    first_batch = batches[0].read_text(encoding="utf-8")
    assert "keep" in first_batch and "exclude" in first_batch and "revise" in first_batch
    assert "自动未发现问题" in first_batch
    assert "pass" not in first_batch.split("五维")[1].split("\n")[0]  # 不显示"事实 pass"
    assert "用户" in first_batch
