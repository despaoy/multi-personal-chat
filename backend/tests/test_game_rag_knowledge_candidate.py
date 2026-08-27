"""P5 知识卡候选（事实卡/关系卡/事件卡）基础设施测试。

覆盖：enriched 输入门禁、prompt 构建、三类严格 JSON 解析（围栏/重复键/额外字段/
evidence 越界/别名归一/非人物拒绝）、运行状态绑定、断点续跑与成功结果保护、
失败重试收敛、稳定 ID、确定性排序、去重与冲突分组、数量上限、draft/needs_review
门禁（候选不可 approved）、原子保存与 JSONL roundtrip。
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import ValidationError

from knowledge.game_rag.knowledge_candidate import (
    CONFLICT_PRONE_FACT_PREDICATES,
    MAX_EVENTS_PER_SCENE,
    MAX_FACTS_PER_SCENE,
    MAX_RELATIONS_PER_SCENE,
    AliasConfig,
    KnowledgeGenerationStatus,
    KnowledgeParseError,
    KnowledgeReviewDocument,
    build_knowledge_prompt,
    build_knowledge_quality_report,
    build_knowledge_run_manifest,
    chunk_limits,
    create_knowledge_review,
    create_knowledge_run,
    finalize_knowledge_candidates,
    generate_knowledge_candidates,
    load_enriched_scene_bundle,
    load_knowledge_review,
    load_knowledge_run,
    parse_knowledge_candidates,
    save_documents_jsonl,
    save_knowledge_review,
    save_knowledge_run,
    save_knowledge_run_with_manifest,
    select_knowledge_scenes,
    validate_knowledge_run,
)
from knowledge.game_rag.models import (
    ContentScope,
    RealityStatus,
    ReviewStatus,
    SceneDocument,
    SourceSpan,
    StoryContext,
    TemporalScope,
)

if TYPE_CHECKING:
    from pathlib import Path

UNIT_A = "vol99_9合成知识A"
PATH_A = "gametext/纸上魔法使/synth_knowledge.txt"

# 通用合成别名配置（与真实 character_aliases.json 同构的小型配置）。
ALIASES = AliasConfig(
    canonical_names=["妃", "琉璃", "夜子", "汀"],
    aliases={"月社妃": "妃", "四条琉璃": "琉璃", "遊行寺夜子": "夜子"},
    non_person_terms=["魔法之书", "翡翠"],
)


def _enriched_scene(
    scene_id: str,
    line_start: int,
    line_end: int,
    *,
    text: str | None = None,
    viewpoint: str = "琉璃第一人称",
    temporal: TemporalScope = TemporalScope.current,
    reality: RealityStatus = RealityStatus.objective,
    content_scope: ContentScope = ContentScope.main_story,
    present: list[str] | None = None,
    mentioned: list[str] | None = None,
) -> SceneDocument:
    span_lines = line_end - line_start + 1
    body = text if text is not None else "\n".join(f"合成原文第{i}行" for i in range(1, span_lines + 1))
    return SceneDocument(
        id=scene_id,
        title=f"合成 {scene_id} L{line_start}-{line_end}",
        text=body,
        story=StoryContext(
            volume_number=99,
            story_unit_id=UNIT_A,
            story_title=UNIT_A,
            content_scope=content_scope,
            viewpoint=viewpoint,
            temporal_scope=temporal,
        ),
        source=SourceSpan(source_path=PATH_A, line_start=line_start, line_end=line_end),
        speakers=[],
        mentioned_characters=mentioned or [],
        present_characters=present or [],
        reality_status=reality,
        review_status=ReviewStatus.approved,
    )


def _enriched_manifest(total: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generator": "knowledge.game_rag.scene_metadata_review",
        "source_boundary_manifest": {
            "schema_version": 1,
            "boundary_review_status": "approved",
            "reviewer": "project_owner_01",
            "total_scenes": total,
            "manifest_sha256": "a" * 64,
            "scenes_sha256": "b" * 64,
            "bundle_sha256": "c" * 64,
        },
        "total_scenes": total,
        "scene_review_status": "approved",
        "note": "synthetic enriched manifest",
    }


def _write_enriched(
    tmp_path: Path,
    scenes: list[SceneDocument],
    *,
    manifest: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    scenes_path = tmp_path / "enriched_scenes.jsonl"
    manifest_path = tmp_path / "enriched_manifest.json"
    scenes_path.write_text("".join(s.model_dump_json() + "\n" for s in scenes), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(manifest or _enriched_manifest(len(scenes)), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return scenes_path, manifest_path


def _load_bundle(tmp_path: Path, scenes: list[SceneDocument], **kwargs: Any):
    scenes_path, manifest_path = _write_enriched(tmp_path, scenes, **kwargs)
    return load_enriched_scene_bundle(scenes_path, manifest_path)


class ScriptedClient:
    """替身模型客户端：按序返回预置输出，统计调用次数。"""

    def __init__(self, responses: list[str | Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        item = self.responses.pop(0) if self.responses else "{}"
        if isinstance(item, Exception):
            raise item
        return item


def _payload_json(
    scene_id: str,
    *,
    facts: list[dict] | None = None,
    relations: list[dict] | None = None,
    events: list[dict] | None = None,
) -> str:
    return json.dumps(
        {"scene_id": scene_id, "facts": facts or [], "relations": relations or [], "events": events or []},
        ensure_ascii=False,
    )


def _fact_item(**overrides: Any) -> dict:
    base = {
        "subject": "琉璃",
        "predicate": "身份",
        "value": "从本岛转学回到小岛的少年",
        "title": "琉璃的身份",
        "summary": "琉璃从本岛转学回到小岛。",
        "reality_status": "objective",
        "line_start": 1,
        "line_end": 2,
    }
    base.update(overrides)
    return base


def _relation_item(**overrides: Any) -> dict:
    base = {
        "subject": "琉璃",
        "relation": "妹妹",
        "target": "妃",
        "title": "兄妹关系",
        "summary": "琉璃的妹妹是妃。",
        "reality_status": "objective",
        "line_start": 1,
        "line_end": 2,
    }
    base.update(overrides)
    return base


def _event_item(**overrides: Any) -> dict:
    base = {
        "title": "琉璃与妃重逢",
        "summary": "琉璃回到小岛与妃重逢。",
        "participants": ["琉璃", "妃"],
        "causes": ["琉璃归岛"],
        "outcomes": ["兄妹再会"],
        "reality_status": "objective",
        "line_start": 1,
        "line_end": 2,
    }
    base.update(overrides)
    return base


# ---------- enriched 输入门禁 ----------


def test_load_enriched_bundle_ok_and_digest_stable(tmp_path: Path) -> None:
    scenes = [_enriched_scene("scene_k1", 1, 10), _enriched_scene("scene_k2", 11, 20)]
    bundle = _load_bundle(tmp_path, scenes)
    assert [s.id for s in bundle.scenes] == ["scene_k1", "scene_k2"]
    bundle2 = _load_bundle(tmp_path / "again", scenes)
    assert bundle.bundle_digest == bundle2.bundle_digest
    assert not validate_knowledge_run(create_knowledge_run(bundle, model_id="m"), bundle)


def test_load_rejects_missing_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="不存在"):
        load_enriched_scene_bundle(tmp_path / "nope.jsonl", tmp_path / "nope.json")


def test_load_rejects_not_approved_scene(tmp_path: Path) -> None:
    scene = _enriched_scene("scene_k1", 1, 10)
    scene.review_status = ReviewStatus.draft
    with pytest.raises(ValueError, match="非 approved"):
        _load_bundle(tmp_path, [scene])


def test_load_rejects_unfilled_metadata(tmp_path: Path) -> None:
    scene = _enriched_scene("scene_k1", 1, 10)
    scene.story.viewpoint = None
    with pytest.raises(ValueError, match="viewpoint 未填写"):
        _load_bundle(tmp_path, [scene])


def test_load_rejects_line_count_mismatch(tmp_path: Path) -> None:
    scene = _enriched_scene("scene_k1", 1, 10, text="只有一行")
    with pytest.raises(ValueError, match="行数"):
        _load_bundle(tmp_path, [scene])


def test_load_rejects_count_mismatch(tmp_path: Path) -> None:
    scenes = [_enriched_scene("scene_k1", 1, 10)]
    with pytest.raises(ValueError, match="不一致"):
        _load_bundle(tmp_path, scenes, manifest=_enriched_manifest(2))


def test_load_rejects_duplicate_scene_id(tmp_path: Path) -> None:
    scenes = [_enriched_scene("scene_k1", 1, 10), _enriched_scene("scene_k1", 11, 20)]
    with pytest.raises(ValueError, match="重复"):
        _load_bundle(tmp_path, scenes)


# ---------- prompt ----------


def test_prompt_contains_contract_elements(tmp_path: Path) -> None:
    scene = _enriched_scene("scene_k1", 1, 10, present=["琉璃"], mentioned=["妃"])
    prompt = build_knowledge_prompt(scene, alias_config=ALIASES)
    assert "scene_k1" in prompt
    assert "L1: 合成原文第1行" in prompt
    assert "琉璃" in prompt and "妃" in prompt
    assert "身份" in prompt and "妹妹" in prompt  # 规范表
    assert "月社妃=妃" in prompt  # 别名表
    assert prompt.index("facts ≤ 8") < prompt.index("relations ≤ 5") < prompt.index("events ≤ 3")


def test_prompt_chunk_note(tmp_path: Path) -> None:
    scene = _enriched_scene("scene_k1", 1, 4, text="行一\n行二\n行三\n行四")
    prompt = build_knowledge_prompt(
        scene,
        SourceSpan(source_path=PATH_A, line_start=3, line_end=4),
        chunk_index=2,
        total_chunks=2,
        alias_config=ALIASES,
    )
    assert "第 2/2 片" in prompt
    assert "L3: 行三" in prompt and "L4: 行四" in prompt
    assert "L1:" not in prompt


def test_chunk_limits_scale_with_chunks() -> None:
    assert chunk_limits(1) == (MAX_FACTS_PER_SCENE, MAX_RELATIONS_PER_SCENE, MAX_EVENTS_PER_SCENE)
    facts, relations, events = chunk_limits(3)
    assert facts < MAX_FACTS_PER_SCENE and relations >= 1 and events >= 1


# ---------- 严格 JSON 解析 ----------


def _parse(tmp_path: Path, raw: str, scene: SceneDocument | None = None, **kwargs: Any):
    scene = scene or _enriched_scene("scene_k1", 1, 10)
    return parse_knowledge_candidates(raw, scene, alias_config=ALIASES, **kwargs)


def test_parse_fact_ok_with_alias_normalization(tmp_path: Path) -> None:
    parsed = _parse(tmp_path, _payload_json("scene_k1", facts=[_fact_item(subject="四条琉璃")]))
    assert parsed.facts[0].subject == "琉璃"  # 别名归一
    assert parsed.facts[0].payload_key == "琉璃|身份|从本岛转学回到小岛的少年"


def test_parse_relation_ok_with_alias_and_symmetric_direction(tmp_path: Path) -> None:
    # 夜子-恋人-琉璃（对称关系）按名字排序 canonical 化：夜子(U+591C) < 琉璃(U+7409)
    parsed = _parse(
        tmp_path,
        _payload_json("scene_k1", relations=[_relation_item(subject="夜子", relation="恋人", target="琉璃")]),
    )
    assert (parsed.relations[0].subject, parsed.relations[0].target) == ("夜子", "琉璃")
    # 非对称关系保持方向
    parsed2 = _parse(
        tmp_path,
        _payload_json("scene_k1", relations=[_relation_item(subject="妃", relation="哥哥", target="琉璃")]),
    )
    assert (parsed2.relations[0].subject, parsed2.relations[0].target) == ("妃", "琉璃")


def test_parse_event_ok_with_participant_dedupe_and_alias(tmp_path: Path) -> None:
    parsed = _parse(
        tmp_path,
        _payload_json(
            "scene_k1",
            events=[_event_item(participants=["琉璃", "月社妃", "琉璃"])],
        ),
    )
    assert parsed.events[0].participants == ["琉璃", "妃"]


def test_parse_empty_output_valid(tmp_path: Path) -> None:
    parsed = _parse(tmp_path, _payload_json("scene_k1"))
    assert parsed.facts == [] and parsed.relations == [] and parsed.events == []


def test_parse_rejects_markdown_fence(tmp_path: Path) -> None:
    with pytest.raises(KnowledgeParseError) as exc_info:
        _parse(tmp_path, "```json\n{}\n```")
    assert exc_info.value.error_kind == "markdown_fence"


def test_parse_rejects_invalid_json(tmp_path: Path) -> None:
    with pytest.raises(KnowledgeParseError) as exc_info:
        _parse(tmp_path, "{not json")
    assert exc_info.value.error_kind == "invalid_json"


def test_parse_rejects_non_object_json(tmp_path: Path) -> None:
    with pytest.raises(KnowledgeParseError) as exc_info:
        _parse(tmp_path, "[1, 2]")
    assert exc_info.value.error_kind == "invalid_json"


def test_parse_rejects_non_string_output(tmp_path: Path) -> None:
    with pytest.raises(KnowledgeParseError) as exc_info:
        _parse(tmp_path, 123)  # type: ignore[arg-type]
    assert exc_info.value.error_kind == "invalid_output"


def test_parse_rejects_duplicate_keys(tmp_path: Path) -> None:
    raw = '{"scene_id": "scene_k1", "scene_id": "scene_k1", "facts": [], "relations": [], "events": []}'
    with pytest.raises(KnowledgeParseError) as exc_info:
        _parse(tmp_path, raw)
    assert exc_info.value.error_kind == "duplicate_json_key"


def test_parse_rejects_extra_top_level_fields(tmp_path: Path) -> None:
    raw = json.dumps(
        {
            "scene_id": "scene_k1",
            "facts": [],
            "relations": [],
            "events": [],
            "extra": 1,
        }
    )
    with pytest.raises(KnowledgeParseError) as exc_info:
        _parse(tmp_path, raw)
    assert exc_info.value.error_kind == "schema_violation"


def test_parse_drops_card_with_extra_fields(tmp_path: Path) -> None:
    # 卡片级额外字段：丢弃该卡（不污染其余合法卡片），计入 dropped_invalid
    raw = _payload_json(
        "scene_k1",
        facts=[_fact_item(extra_field="x"), _fact_item(value="合法事实")],
    )
    parsed = _parse(tmp_path, raw)
    assert len(parsed.facts) == 1
    assert parsed.facts[0].value == "合法事实"
    assert parsed.dropped_invalid == 1


def test_parse_rejects_missing_top_level_fields(tmp_path: Path) -> None:
    with pytest.raises(KnowledgeParseError) as exc_info:
        _parse(tmp_path, '{"scene_id": "scene_k1", "facts": []}')
    assert exc_info.value.error_kind == "schema_violation"


def test_parse_drops_invalid_predicate_card(tmp_path: Path) -> None:
    raw = _payload_json(
        "scene_k1",
        facts=[_fact_item(predicate="职业"), _fact_item(value="合法事实")],
    )
    parsed = _parse(tmp_path, raw)
    assert [f.value for f in parsed.facts] == ["合法事实"]
    assert parsed.dropped_invalid == 1


def test_parse_drops_invalid_relation_label_card(tmp_path: Path) -> None:
    raw = _payload_json(
        "scene_k1",
        relations=[_relation_item(relation="亲妹妹"), _relation_item(relation="哥哥", subject="妃", target="琉璃")],
    )
    parsed = _parse(tmp_path, raw)
    assert len(parsed.relations) == 1
    assert parsed.dropped_invalid == 1


def test_parse_drops_invalid_reality_status_card(tmp_path: Path) -> None:
    raw = _payload_json("scene_k1", facts=[_fact_item(reality_status="real")])
    parsed = _parse(tmp_path, raw)
    assert parsed.facts == []
    assert parsed.dropped_invalid == 1


def test_parse_rejects_scene_id_mismatch(tmp_path: Path) -> None:
    with pytest.raises(KnowledgeParseError) as exc_info:
        _parse(tmp_path, _payload_json("scene_other"))
    assert exc_info.value.error_kind == "scene_id_mismatch"


def test_parse_drops_out_of_range_evidence_card(tmp_path: Path) -> None:
    # 卡片级越界：丢弃该卡（evidence 越界拒绝）；合法卡保留
    raw = _payload_json(
        "scene_k1",
        facts=[_fact_item(line_start=1, line_end=99), _fact_item(value="合法事实")],
    )
    parsed = _parse(tmp_path, raw)
    assert [f.value for f in parsed.facts] == ["合法事实"]
    assert parsed.dropped_invalid == 1


def test_parse_drops_line_order_card(tmp_path: Path) -> None:
    raw = _payload_json("scene_k1", facts=[_fact_item(line_start=5, line_end=2)])
    parsed = _parse(tmp_path, raw)
    assert parsed.facts == []
    assert parsed.dropped_invalid == 1


def test_parse_drops_overbroad_evidence(tmp_path: Path) -> None:
    scene = _enriched_scene("scene_k1", 1, 100)
    parsed = _parse(
        tmp_path,
        _payload_json(
            "scene_k1",
            facts=[_fact_item(line_start=1, line_end=25)],
            relations=[_relation_item(line_start=1, line_end=25)],
            events=[_event_item(line_start=1, line_end=61)],
        ),
        scene=scene,
    )
    assert parsed.facts == []
    assert parsed.relations == []
    assert parsed.events == []
    assert parsed.dropped_invalid == 3


def test_parse_evidence_checked_against_chunk_span(tmp_path: Path) -> None:
    scene = _enriched_scene("scene_k1", 1, 10)
    span = SourceSpan(source_path=PATH_A, line_start=5, line_end=8)
    # 在场景内但在分片外 → 卡片被丢弃
    raw = _payload_json("scene_k1", facts=[_fact_item(line_start=1, line_end=2)])
    parsed = parse_knowledge_candidates(raw, scene, span=span, alias_config=ALIASES)
    assert parsed.facts == []
    assert parsed.dropped_invalid == 1
    # 恰在分片边界 → 合法
    ok = parse_knowledge_candidates(
        _payload_json("scene_k1", facts=[_fact_item(line_start=5, line_end=8)]),
        scene,
        span=span,
        alias_config=ALIASES,
    )
    assert ok.facts[0].line_start == 5


def test_parse_drops_non_person_relation_target(tmp_path: Path) -> None:
    raw = _payload_json("scene_k1", relations=[_relation_item(target="魔法之书"), _relation_item()])
    parsed = _parse(tmp_path, raw)
    assert len(parsed.relations) == 1
    assert parsed.dropped_invalid == 1


def test_parse_drops_unknown_person_in_relation(tmp_path: Path) -> None:
    raw = _payload_json("scene_k1", relations=[_relation_item(subject="神秘人")])
    parsed = _parse(tmp_path, raw)
    assert parsed.relations == []
    assert parsed.dropped_invalid == 1


def test_parse_drops_self_relation_after_alias_normalization(tmp_path: Path) -> None:
    raw = _payload_json("scene_k1", relations=[_relation_item(subject="月社妃", target="妃")])
    parsed = _parse(tmp_path, raw)
    assert parsed.relations == []
    assert parsed.dropped_invalid == 1


def test_parse_drops_non_person_event_participant(tmp_path: Path) -> None:
    raw = _payload_json(
        "scene_k1",
        events=[_event_item(participants=["《潘多拉的狂乱剧场》的后续书页"])],
    )
    parsed = _parse(tmp_path, raw)
    assert parsed.events == []
    assert parsed.dropped_invalid == 1


def test_parse_accepts_behavior_predicate_and_expanded_role_nouns(tmp_path: Path) -> None:
    # smoke 修正：谓词「行为」与语料核对扩展的通用角色称谓（校方/加害者/班主任等）
    parsed = _parse(
        tmp_path,
        _payload_json(
            "scene_k1",
            facts=[_fact_item(predicate="行为", value="掩盖了少女轻生的事实", subject="校方")],
            events=[_event_item(participants=["少女", "班主任", "加害者"])],
        ),
    )
    assert parsed.facts[0].predicate == "行为"
    assert parsed.facts[0].subject == "校方"
    assert parsed.events[0].participants == ["少女", "班主任", "加害者"]


def test_parse_allows_role_noun_and_entity_fact_subject(tmp_path: Path) -> None:
    # fact.subject 允许实体名（设定类事实）；relation 人物位允许称谓
    # （敌对为对称关系：母亲/夜子按名字排序固定方向为 夜子-敌对-母亲）。
    parsed = _parse(
        tmp_path,
        _payload_json(
            "scene_k1",
            facts=[_fact_item(subject="魔法之书", predicate="设定", value="把书写内容引进现实")],
            relations=[_relation_item(subject="母亲", relation="敌对", target="夜子")],
        ),
    )
    assert parsed.facts[0].subject == "魔法之书"
    assert {parsed.relations[0].subject, parsed.relations[0].target} == {"母亲", "夜子"}


# ---------- 运行状态 ----------


def test_create_run_state_initial_pending(tmp_path: Path) -> None:
    bundle = _load_bundle(tmp_path, [_enriched_scene("scene_k1", 1, 10)])
    state = create_knowledge_run(bundle, model_id="ollama:qwen2.5:7b")
    assert len(state.scene_states) == 1
    assert state.scene_states[0].status is KnowledgeGenerationStatus.pending
    assert state.scene_states[0].candidates is None


def test_create_run_rejects_blank_model_id(tmp_path: Path) -> None:
    bundle = _load_bundle(tmp_path, [_enriched_scene("scene_k1", 1, 10)])
    with pytest.raises(ValueError, match="model_id"):
        create_knowledge_run(bundle, model_id="  ")


def test_validate_run_rejects_cross_bundle(tmp_path: Path) -> None:
    bundle = _load_bundle(tmp_path, [_enriched_scene("scene_k1", 1, 10)])
    state = create_knowledge_run(bundle, model_id="m")
    other = _load_bundle(tmp_path / "other", [_enriched_scene("scene_k1", 1, 10, text="不同的原文\n" * 10)])
    errors = validate_knowledge_run(state, other)
    assert errors and "不一致" in errors[0]


def test_validate_run_rejects_scene_id_replacement(tmp_path: Path) -> None:
    bundle = _load_bundle(tmp_path, [_enriched_scene("scene_k1", 1, 10)])
    state = create_knowledge_run(bundle, model_id="m")
    data = json.loads(state.model_dump_json())
    data["scene_states"][0]["scene_id"] = "scene_tampered"
    errors = validate_knowledge_run(data, bundle)
    assert errors and "scene_id" in errors[0]


def test_select_scenes_rejects_unknown_id(tmp_path: Path) -> None:
    bundle = _load_bundle(tmp_path, [_enriched_scene("scene_k1", 1, 10)])
    state = create_knowledge_run(bundle, model_id="m")
    with pytest.raises(ValueError, match="未知 scene id"):
        select_knowledge_scenes(state, scene_ids=["scene_nope"])


# ---------- 生成（替身客户端） ----------


def test_generate_success_persists_state(tmp_path: Path) -> None:
    bundle = _load_bundle(tmp_path, [_enriched_scene("scene_k1", 1, 10)])
    state = create_knowledge_run(bundle, model_id="m")
    client = ScriptedClient([_payload_json("scene_k1", facts=[_fact_item()])])
    state_path = tmp_path / "run.json"
    result = generate_knowledge_candidates(bundle, state, client, state_path=state_path, alias_config=ALIASES)
    assert result.succeeded_scene_ids == ["scene_k1"]
    assert result.new_state.scene_states[0].status is KnowledgeGenerationStatus.success
    assert len(result.new_state.scene_states[0].candidates.facts) == 1
    reloaded = load_knowledge_run(state_path)
    assert reloaded == result.new_state  # JSON roundtrip 一致


def test_generate_skips_successful_scenes(tmp_path: Path) -> None:
    bundle = _load_bundle(tmp_path, [_enriched_scene("scene_k1", 1, 10)])
    state = create_knowledge_run(bundle, model_id="m")
    client = ScriptedClient([_payload_json("scene_k1", facts=[_fact_item()])])
    result = generate_knowledge_candidates(bundle, state, client, alias_config=ALIASES)
    # 第二次默认运行：成功场景不入选（不选中即不触碰），零模型调用
    client2 = ScriptedClient([])
    result2 = generate_knowledge_candidates(bundle, result.new_state, client2, alias_config=ALIASES)
    assert result2.attempted_scene_ids == []
    assert result2.skipped_scene_ids == []
    assert client2.calls == []
    # 显式选中已成功场景 → 记为 skipped，成功结果不被覆盖
    client3 = ScriptedClient([])
    result3 = generate_knowledge_candidates(
        bundle, result.new_state, client3, scene_ids=["scene_k1"], alias_config=ALIASES
    )
    assert result3.skipped_scene_ids == ["scene_k1"]
    assert client3.calls == []
    assert result3.new_state.scene_states[0].candidates == result.new_state.scene_states[0].candidates


def test_generate_failure_retries_then_marks_failed(tmp_path: Path) -> None:
    bundle = _load_bundle(tmp_path, [_enriched_scene("scene_k1", 1, 10)])
    state = create_knowledge_run(bundle, model_id="m")
    client = ScriptedClient([KnowledgeParseError("bad", error_kind="invalid_json")] * 3)
    result = generate_knowledge_candidates(bundle, state, client, max_attempts=3, alias_config=ALIASES)
    item = result.new_state.scene_states[0]
    assert item.status is KnowledgeGenerationStatus.failed
    assert item.candidates is None
    assert item.attempts == 3
    assert item.last_failure is not None
    assert item.last_failure.error_kind == "invalid_json"
    assert item.last_failure.detail == "KnowledgeParseError"  # 只落盘类型名


def test_generate_retries_when_all_cards_invalid(tmp_path: Path) -> None:
    # 整片卡片全部非法（结构合法但全被丢弃）→ 视为失败并重试；耗尽则场景失败
    bundle = _load_bundle(tmp_path, [_enriched_scene("scene_k1", 1, 10)])
    state = create_knowledge_run(bundle, model_id="m")
    all_invalid = _payload_json("scene_k1", facts=[_fact_item(predicate="职业")])
    client = ScriptedClient([all_invalid, all_invalid, all_invalid])
    result = generate_knowledge_candidates(bundle, state, client, max_attempts=3, alias_config=ALIASES)
    item = result.new_state.scene_states[0]
    assert item.status is KnowledgeGenerationStatus.failed
    assert item.last_failure is not None
    assert item.last_failure.error_kind == "schema_violation"
    assert len(client.calls) == 3


def test_generate_partial_drop_success_keeps_valid_cards(tmp_path: Path) -> None:
    # 部分卡片非法：丢弃计数记录，合法卡片保留，场景成功
    bundle = _load_bundle(tmp_path, [_enriched_scene("scene_k1", 1, 10)])
    state = create_knowledge_run(bundle, model_id="m")
    raw = _payload_json(
        "scene_k1",
        facts=[_fact_item(predicate="职业"), _fact_item(value="合法事实")],
        relations=[_relation_item(target="魔法之书"), _relation_item(relation="哥哥", subject="妃", target="琉璃")],
    )
    client = ScriptedClient([raw])
    result = generate_knowledge_candidates(bundle, state, client, alias_config=ALIASES)
    item = result.new_state.scene_states[0]
    assert item.status is KnowledgeGenerationStatus.success
    assert item.candidates is not None
    assert len(item.candidates.facts) == 1
    assert len(item.candidates.relations) == 1
    assert item.candidates.dropped_invalid == 2


def test_generate_timeout_and_model_error_converge(tmp_path: Path) -> None:
    bundle = _load_bundle(tmp_path / "t1", [_enriched_scene("scene_k1", 1, 10)])
    state = create_knowledge_run(bundle, model_id="m")
    client = ScriptedClient([TimeoutError("http://localhost secret")] * 3)
    result = generate_knowledge_candidates(bundle, state, client, max_attempts=3, alias_config=ALIASES)
    assert result.new_state.scene_states[0].last_failure.error_kind == "timeout"
    assert "localhost" not in result.new_state.scene_states[0].last_failure.detail

    bundle2 = _load_bundle(tmp_path / "t2", [_enriched_scene("scene_k1", 1, 10)])
    state2 = create_knowledge_run(bundle2, model_id="m")
    client2 = ScriptedClient([RuntimeError("secret-key leaked")] * 3)
    result2 = generate_knowledge_candidates(bundle2, state2, client2, max_attempts=3, alias_config=ALIASES)
    failure = result2.new_state.scene_states[0].last_failure
    assert failure.error_kind == "model_error"
    assert failure.detail == "RuntimeError"
    assert "secret" not in failure.detail


def test_generate_single_chunk_failure_fails_whole_scene(tmp_path: Path) -> None:
    # 长场景分两片：第一片成功、第二片耗尽重试 → 整场景失败，不产出部分候选。
    long_text = "\n".join(f"第{i}行原文内容" for i in range(1, 121))
    scene = _enriched_scene("scene_k1", 1, 120, text=long_text)
    bundle = _load_bundle(tmp_path, [scene])
    state = create_knowledge_run(bundle, model_id="m", generation_params={"chunk_max_chars": 400})
    client = ScriptedClient(
        [
            _payload_json("scene_k1"),  # 第 1 片成功（空产出合法）
            KnowledgeParseError("bad", error_kind="invalid_json"),
            KnowledgeParseError("bad", error_kind="invalid_json"),
            KnowledgeParseError("bad", error_kind="invalid_json"),
        ]
    )
    result = generate_knowledge_candidates(bundle, state, client, max_attempts=3, alias_config=ALIASES)
    item = result.new_state.scene_states[0]
    assert item.status is KnowledgeGenerationStatus.failed
    assert item.candidates is None
    assert item.chunk_count >= 2
    assert len(client.calls) == 4


def test_generate_chunk_merge_union(tmp_path: Path) -> None:
    long_text = "\n".join(f"第{i}行原文内容" for i in range(1, 121))
    scene = _enriched_scene("scene_k1", 1, 120, text=long_text)
    bundle = _load_bundle(tmp_path, [scene])
    state = create_knowledge_run(bundle, model_id="m", generation_params={"chunk_max_chars": 400})
    # 按实际分片数生成响应：每片各产出一条事实（行号取该片首行附近）。
    from knowledge.game_rag.knowledge_candidate import _chunk_spans

    spans = _chunk_spans(scene, chunk_max_chars=400)
    assert len(spans) >= 2
    responses = [
        _payload_json(
            "scene_k1", facts=[_fact_item(value=f"片段{i}事实", line_start=span.line_start, line_end=span.line_start)]
        )
        for i, span in enumerate(spans, start=1)
    ]
    client = ScriptedClient(responses)
    result = generate_knowledge_candidates(bundle, state, client, alias_config=ALIASES)
    candidates = result.new_state.scene_states[0].candidates
    assert candidates is not None
    assert len(candidates.facts) == len(spans)


def test_generate_zero_calls_on_invalid_state(tmp_path: Path) -> None:
    bundle = _load_bundle(tmp_path, [_enriched_scene("scene_k1", 1, 10)])
    other = _load_bundle(tmp_path / "other", [_enriched_scene("scene_k1", 1, 10, text="别的原文\n" * 10)])
    state = create_knowledge_run(other, model_id="m")
    client = ScriptedClient([])
    with pytest.raises(ValueError):
        generate_knowledge_candidates(bundle, state, client, alias_config=ALIASES)
    assert client.calls == []


def test_generate_rejects_bad_max_attempts(tmp_path: Path) -> None:
    bundle = _load_bundle(tmp_path, [_enriched_scene("scene_k1", 1, 10)])
    state = create_knowledge_run(bundle, model_id="m")
    with pytest.raises(ValueError, match="max_attempts"):
        generate_knowledge_candidates(bundle, state, ScriptedClient([]), max_attempts=0, alias_config=ALIASES)


# ---------- finalize：稳定 ID / 排序 / 去重 / 上限 / 冲突 ----------


def _completed_state(
    tmp_path: Path,
    scenes: list[SceneDocument],
    responses: list[str],
    *,
    alias_config: AliasConfig = ALIASES,
):
    bundle = _load_bundle(tmp_path, scenes)
    state = create_knowledge_run(bundle, model_id="m")
    client = ScriptedClient(responses)
    result = generate_knowledge_candidates(bundle, state, client, alias_config=alias_config)
    assert not result.failed_scene_ids, "测试夹具不应失败"
    return bundle, result.new_state


def test_finalize_stable_ids_and_deterministic_order(tmp_path: Path) -> None:
    scenes = [_enriched_scene("scene_k1", 1, 10), _enriched_scene("scene_k2", 11, 20)]
    responses = [
        _payload_json(
            "scene_k1",
            facts=[_fact_item(line_start=2, line_end=3), _fact_item(value="乙事实", line_start=1, line_end=1)],
            relations=[_relation_item()],
            events=[_event_item()],
        ),
        _payload_json("scene_k2", facts=[_fact_item(value="丙事实", line_start=12, line_end=13)]),
    ]
    bundle, state = _completed_state(tmp_path / "a", scenes, responses)
    final1 = finalize_knowledge_candidates(bundle, state, alias_config=ALIASES)
    final2 = finalize_knowledge_candidates(bundle, state, alias_config=ALIASES)
    assert [d.id for d in final1.fact_documents] == [d.id for d in final2.fact_documents]
    assert all(d.id.startswith("fact_") for d in final1.fact_documents)
    assert all(d.id.startswith("rel_") for d in final1.relation_documents)
    assert all(d.id.startswith("event_") for d in final1.event_documents)
    # 排序：同场景内按 line_start 升序（1 行的乙事实在前）
    assert [d.source.line_start for d in final1.fact_documents] == [1, 2, 12]


def test_finalize_id_depends_on_span_and_payload(tmp_path: Path) -> None:
    scene = _enriched_scene("scene_k1", 1, 10)
    bundle, state = _completed_state(
        tmp_path,
        [scene],
        [_payload_json("scene_k1", facts=[_fact_item(line_start=1, line_end=2)])],
    )
    final_a = finalize_knowledge_candidates(bundle, state, alias_config=ALIASES)
    bundle_b, state_b = _completed_state(
        tmp_path / "b",
        [scene],
        [_payload_json("scene_k1", facts=[_fact_item(line_start=1, line_end=3)])],
    )
    final_b = finalize_knowledge_candidates(bundle_b, state_b, alias_config=ALIASES)
    assert final_a.fact_documents[0].id != final_b.fact_documents[0].id


def test_finalize_evidence_text_verbatim(tmp_path: Path) -> None:
    scene = _enriched_scene(
        "scene_k1", 1, 10, text="第一行\n第二行\n第三行\n第四行\n第五行\n第六行\n第七行\n第八行\n第九行\n第十行"
    )
    bundle, state = _completed_state(
        tmp_path, [scene], [_payload_json("scene_k1", facts=[_fact_item(line_start=2, line_end=4)])]
    )
    final = finalize_knowledge_candidates(bundle, state, alias_config=ALIASES)
    assert final.fact_documents[0].evidence_text == "第二行\n第三行\n第四行"
    assert final.fact_documents[0].source == SourceSpan(source_path=PATH_A, line_start=2, line_end=4)


def test_finalize_story_context_inherited(tmp_path: Path) -> None:
    scene = _enriched_scene(
        "scene_k1", 1, 10, viewpoint="夜子第一人称", temporal=TemporalScope.flashback, reality=RealityStatus.objective
    )
    bundle, state = _completed_state(
        tmp_path, [scene], [_payload_json("scene_k1", facts=[_fact_item(reality_status="objective")])]
    )
    final = finalize_knowledge_candidates(bundle, state, alias_config=ALIASES)
    doc = final.fact_documents[0]
    assert doc.story.viewpoint == "夜子第一人称"
    assert doc.story.temporal_scope is TemporalScope.flashback
    assert doc.reality_status is RealityStatus.objective


def test_finalize_dedup_within_scene(tmp_path: Path) -> None:
    scene = _enriched_scene("scene_k1", 1, 10)
    bundle, state = _completed_state(
        tmp_path,
        [scene],
        [
            _payload_json(
                "scene_k1",
                facts=[
                    _fact_item(line_start=4, line_end=5),
                    _fact_item(line_start=1, line_end=2),  # 相同 payload，行号更靠前
                ],
            )
        ],
    )
    final = finalize_knowledge_candidates(bundle, state, alias_config=ALIASES)
    assert len(final.fact_documents) == 1
    assert final.fact_documents[0].source.line_start == 1  # 保留行号最靠前的证据
    assert final.scene_stats[0].deduped_in_scene == 1


def test_finalize_caps_with_deterministic_truncation(tmp_path: Path) -> None:
    scene = _enriched_scene("scene_k1", 1, 30)
    facts = [_fact_item(value=f"事实{i}", line_start=i, line_end=i) for i in range(1, 12)]  # 11 条
    bundle, state = _completed_state(tmp_path, [scene], [_payload_json("scene_k1", facts=facts)])
    final = finalize_knowledge_candidates(bundle, state, alias_config=ALIASES)
    assert len(final.fact_documents) == MAX_FACTS_PER_SCENE
    assert [d.source.line_start for d in final.fact_documents] == list(range(1, 9))  # 行号最小的保留
    assert final.scene_stats[0].dropped_by_cap == 3


def test_finalize_empty_scene_valid(tmp_path: Path) -> None:
    bundle, state = _completed_state(tmp_path, [_enriched_scene("scene_k1", 1, 10)], [_payload_json("scene_k1")])
    final = finalize_knowledge_candidates(bundle, state, alias_config=ALIASES)
    assert final.total_documents == 0
    assert final.scene_stats[0].facts == 0


def test_finalize_duplicate_group_cross_scene(tmp_path: Path) -> None:
    scenes = [_enriched_scene("scene_k1", 1, 10), _enriched_scene("scene_k2", 11, 20)]
    responses = [
        _payload_json("scene_k1", relations=[_relation_item(line_start=1, line_end=2)]),
        _payload_json("scene_k2", relations=[_relation_item(line_start=11, line_end=12)]),
    ]
    bundle, state = _completed_state(tmp_path, scenes, responses)
    final = finalize_knowledge_candidates(bundle, state, alias_config=ALIASES)
    assert len(final.relation_documents) == 1
    assert final.relation_documents[0].source.line_start == 1
    assert final.deduped_cross_scene == 1
    assert final.scene_stats[1].deduped_cross_scene == 1
    assert final.duplicate_groups == []


def test_finalize_keeps_dynamic_duplicate_across_scenes(tmp_path: Path) -> None:
    scenes = [_enriched_scene("scene_k1", 1, 10), _enriched_scene("scene_k2", 11, 20)]
    responses = [
        _payload_json("scene_k1", facts=[_fact_item(predicate="状态", value="闭门不出")]),
        _payload_json("scene_k2", facts=[_fact_item(predicate="状态", value="闭门不出", line_start=11, line_end=12)]),
    ]
    bundle, state = _completed_state(tmp_path, scenes, responses)
    final = finalize_knowledge_candidates(bundle, state, alias_config=ALIASES)
    assert len(final.fact_documents) == 2
    assert final.deduped_cross_scene == 0
    assert len(final.duplicate_groups) == 1


@pytest.mark.parametrize(
    ("second_scene_kwargs", "second_reality"),
    [
        ({"temporal": TemporalScope.flashback}, "objective"),
        ({"content_scope": ContentScope.bonus_story}, "objective"),
        ({}, "fictional"),
    ],
)
def test_finalize_keeps_stable_duplicate_across_narrative_domains(
    tmp_path: Path, second_scene_kwargs: dict[str, Any], second_reality: str
) -> None:
    scenes = [
        _enriched_scene("scene_k1", 1, 10),
        _enriched_scene("scene_k2", 11, 20, **second_scene_kwargs),
    ]
    responses = [
        _payload_json("scene_k1", facts=[_fact_item()]),
        _payload_json("scene_k2", facts=[_fact_item(line_start=11, line_end=12, reality_status=second_reality)]),
    ]
    bundle, state = _completed_state(tmp_path, scenes, responses)
    final = finalize_knowledge_candidates(bundle, state, alias_config=ALIASES)
    assert len(final.fact_documents) == 2
    assert final.deduped_cross_scene == 0


def test_finalize_normalizes_verified_sibling_direction_and_summary(tmp_path: Path) -> None:
    scene = _enriched_scene("scene_k1", 1, 2, text="妃是琉璃的妹妹。\n琉璃是妃的哥哥。")
    bundle, state = _completed_state(
        tmp_path,
        [scene],
        [
            _payload_json(
                "scene_k1",
                relations=[
                    _relation_item(
                        subject="妃",
                        relation="妹妹",
                        target="琉璃",
                        summary="妃是琉璃的姐姐",
                        line_start=1,
                        line_end=1,
                    )
                ],
            )
        ],
    )
    final = finalize_knowledge_candidates(bundle, state, alias_config=ALIASES)
    relation = final.relation_documents[0]
    assert (relation.subject, relation.relation, relation.target) == ("琉璃", "妹妹", "妃")
    assert relation.summary == "妃是琉璃的妹妹"


def test_finalize_drops_unverified_named_sibling_relation(tmp_path: Path) -> None:
    aliases = AliasConfig(canonical_names=["妃", "琉璃", "夜子", "汀"])
    scene = _enriched_scene("scene_k1", 1, 1, text="妃和汀讨论往事。")
    bundle = _load_bundle(tmp_path, [scene])
    state = create_knowledge_run(bundle, model_id="m")
    result = generate_knowledge_candidates(
        bundle,
        state,
        ScriptedClient(
            [
                _payload_json(
                    "scene_k1",
                    relations=[
                        _relation_item(
                            subject="妃",
                            relation="哥哥",
                            target="汀",
                            summary="汀是妃的哥哥",
                        )
                    ],
                )
            ]
        ),
        alias_config=aliases,
    )
    final = finalize_knowledge_candidates(bundle, result.new_state, alias_config=aliases)
    assert final.relation_documents == []


def test_finalize_drops_relationship_identity_fact_and_repairs_name_summary(tmp_path: Path) -> None:
    scene = _enriched_scene(tmp_path.stem, 1, 3, text="妃是琉璃的妹妹。\n琉璃是四条琉璃。\n妃是月社妃。")
    bundle, state = _completed_state(
        tmp_path,
        [scene],
        [
            _payload_json(
                scene.id,
                facts=[
                    _fact_item(value="妃的妹妹", summary="琉璃是妃的妹妹", line_start=1, line_end=1),
                    _fact_item(value="四条琉璃", summary="琉璃是四条家的女儿", line_start=2, line_end=2),
                ],
            )
        ],
    )
    final = finalize_knowledge_candidates(bundle, state, alias_config=ALIASES)
    assert len(final.fact_documents) == 1
    assert final.fact_documents[0].summary == "琉璃的姓名是四条琉璃"


def test_finalize_drops_known_sibling_pair_without_direct_family_evidence(tmp_path: Path) -> None:
    aliases = AliasConfig(canonical_names=["妃", "琉璃", "夜子", "汀"])
    scene = _enriched_scene(tmp_path.stem, 1, 3, text="遊行寺夜子闭门不出。\n汀在回忆妃。\n他把妃当做自己的妹妹看待。")
    bundle, state = _completed_state(
        tmp_path,
        [scene],
        [
            _payload_json(
                scene.id,
                relations=[
                    _relation_item(
                        subject="汀",
                        relation="妹妹",
                        target="夜子",
                        summary="夜子是汀的妹妹",
                        line_start=1,
                        line_end=3,
                    )
                ],
            )
        ],
    )
    final = finalize_knowledge_candidates(bundle, state, alias_config=aliases)
    assert final.relation_documents == []


def test_finalize_rewrites_known_event_overclaims(tmp_path: Path) -> None:
    scene = _enriched_scene(
        tmp_path.stem,
        2827,
        2886,
        text="\n".join(f"第{i}行" for i in range(1, 61)),
    )
    scene.story.story_unit_id = "vol03_3蓝宝石的存在证明"
    bundle, state = _completed_state(
        tmp_path,
        [scene],
        [
            _payload_json(
                scene.id,
                events=[
                    _event_item(
                        title="蓝宝石影响下的恋爱",
                        summary="妃试图与父亲建立恋爱关系",
                        line_start=2827,
                        line_end=2886,
                    )
                ],
            )
        ],
    )
    final = finalize_knowledge_candidates(bundle, state, alias_config=ALIASES)
    event = final.event_documents[0]
    assert event.title == "蓝宝石影响下妃与琉璃的亲密互动"
    assert "父亲" not in event.summary


def test_finalize_repairs_or_drops_known_false_fact_cards(tmp_path: Path) -> None:
    aliases = AliasConfig(
        canonical_names=["妃", "琉璃", "夜子", "彼方", "克丽索贝莉露"],
    )
    scene = _enriched_scene(tmp_path.stem, 271, 294, text="\n".join(f"第{i}行" for i in range(1, 25)))
    scene.story.story_unit_id = "vol13_13璀璨的紫翠玉"
    bundle, state = _completed_state(
        tmp_path,
        [scene],
        [
            _payload_json(
                scene.id,
                facts=[
                    _fact_item(
                        subject="克丽索贝莉露",
                        predicate="性格",
                        value="被父亲欺骗",
                        title="性格特征",
                        summary="克丽索贝莉露被父亲欺骗",
                        line_start=271,
                        line_end=294,
                    ),
                    _fact_item(
                        subject="琉璃",
                        value="与彼方有复杂关系",
                        summary="琉璃与彼方有复杂关系",
                        line_start=271,
                        line_end=272,
                    ),
                    _fact_item(
                        subject="夜子",
                        predicate="性格",
                        value="闭门不出，说话恶毒，爱板着脸的妹妹",
                        summary="夜子的性格特征",
                        line_start=271,
                        line_end=272,
                    ),
                    _fact_item(
                        subject="妃",
                        predicate="设定",
                        value="兄妹关系",
                        summary="妃和琉璃设定为兄妹",
                        line_start=271,
                        line_end=272,
                    ),
                ],
            )
        ],
        alias_config=aliases,
    )
    final = finalize_knowledge_candidates(bundle, state, alias_config=aliases)
    facts = {(item.subject, item.predicate, item.value): item for item in final.fact_documents}
    assert ("琉璃", "身份", "与彼方有复杂关系") not in facts
    assert ("妃", "设定", "兄妹关系") not in facts
    assert ("夜子", "性格", "闭门不出，说话恶毒，爱板着脸") in facts
    assert ("克丽索贝莉露", "经历", "被父亲欺骗") in facts
    assert facts[("克丽索贝莉露", "经历", "被父亲欺骗")].title == "克丽索贝莉露被父亲欺骗"


def test_finalize_drops_known_false_relation_card(tmp_path: Path) -> None:
    scene = _enriched_scene(tmp_path.stem, 1000, 1000, text="彼方调查琉璃和游行寺家的关系。")
    scene.story.story_unit_id = "vol01_1翡翠的排挤原理"
    aliases = AliasConfig(canonical_names=["妃", "琉璃", "夜子", "汀", "彼方"])
    bundle, state = _completed_state(
        tmp_path,
        [scene],
        [
            _payload_json(
                scene.id,
                relations=[
                    _relation_item(
                        subject="彼方",
                        relation="敌对",
                        target="琉璃",
                        summary="彼方敌对琉璃",
                        line_start=1000,
                        line_end=1000,
                    )
                ],
            )
        ],
        alias_config=aliases,
    )
    final = finalize_knowledge_candidates(bundle, state, alias_config=aliases)
    assert final.relation_documents == []


def test_finalize_status_values_are_not_automatically_conflicts(tmp_path: Path) -> None:
    scenes = [_enriched_scene("scene_k1", 1, 10), _enriched_scene("scene_k2", 11, 20)]
    responses = [
        _payload_json("scene_k1", facts=[_fact_item(predicate="状态", value="已死亡", line_start=1, line_end=2)]),
        _payload_json("scene_k2", facts=[_fact_item(predicate="状态", value="在世", line_start=11, line_end=12)]),
    ]
    bundle, state = _completed_state(tmp_path, scenes, responses)
    final = finalize_knowledge_candidates(bundle, state, alias_config=ALIASES)
    assert final.conflict_groups == []
    assert all(doc.review_status is ReviewStatus.draft for doc in final.fact_documents)


def test_finalize_conflict_group_relations(tmp_path: Path) -> None:
    scenes = [_enriched_scene("scene_k1", 1, 10), _enriched_scene("scene_k2", 11, 20)]
    responses = [
        _payload_json(
            "scene_k1",
            relations=[_relation_item(relation="母亲", target="妃", summary="琉璃的母亲是妃")],
        ),
        _payload_json(
            "scene_k2",
            relations=[
                _relation_item(relation="母亲", target="夜子", summary="琉璃的母亲是夜子", line_start=11, line_end=12)
            ],
        ),
    ]
    bundle, state = _completed_state(tmp_path, scenes, responses)
    final = finalize_knowledge_candidates(bundle, state, alias_config=ALIASES)
    assert any(g.card_type == "relation" for g in final.conflict_groups)
    assert all(doc.review_status is ReviewStatus.needs_review for doc in final.relation_documents)


def test_finalize_no_conflict_for_non_unique_predicates(tmp_path: Path) -> None:
    # 偏好等多值谓词不构成冲突
    scenes = [_enriched_scene("scene_k1", 1, 10), _enriched_scene("scene_k2", 11, 20)]
    responses = [
        _payload_json("scene_k1", facts=[_fact_item(predicate="偏好", value="苦味食物")]),
        _payload_json("scene_k2", facts=[_fact_item(predicate="偏好", value="钢琴", line_start=11, line_end=12)]),
    ]
    bundle, state = _completed_state(tmp_path, scenes, responses)
    final = finalize_knowledge_candidates(bundle, state, alias_config=ALIASES)
    assert final.conflict_groups == []
    assert all(doc.review_status is ReviewStatus.draft for doc in final.fact_documents)


def test_finalize_conflict_prone_predicates_defined() -> None:
    assert set(CONFLICT_PRONE_FACT_PREDICATES) == {"死因"}


def test_finalize_excludes_failed_scenes(tmp_path: Path) -> None:
    scenes = [_enriched_scene("scene_k1", 1, 10)]
    bundle = _load_bundle(tmp_path, scenes)
    state = create_knowledge_run(bundle, model_id="m")
    client = ScriptedClient([KnowledgeParseError("bad", error_kind="invalid_json")] * 3)
    result = generate_knowledge_candidates(bundle, state, client, max_attempts=3, alias_config=ALIASES)
    final = finalize_knowledge_candidates(bundle, result.new_state, alias_config=ALIASES)
    assert final.total_documents == 0  # 失败场景不出文档（不伪造）


def test_finalize_rejects_state_with_illegal_person(tmp_path: Path) -> None:
    bundle = _load_bundle(tmp_path, [_enriched_scene("scene_k1", 1, 10)])
    state = create_knowledge_run(bundle, model_id="m")
    data = json.loads(state.model_dump_json())
    data["scene_states"][0]["status"] = "success"
    data["scene_states"][0]["candidates"] = {
        "scene_id": "scene_k1",
        "facts": [],
        "relations": [
            {
                "subject": "神秘人",
                "relation": "妹妹",
                "target": "妃",
                "title": "t",
                "summary": "s",
                "reality_status": "objective",
                "line_start": 1,
                "line_end": 2,
            }
        ],
        "events": [],
    }
    with pytest.raises(ValueError, match="非法人物名"):
        finalize_knowledge_candidates(bundle, data, alias_config=ALIASES)


# ---------- 人工审核状态与门禁 ----------


def test_create_review_all_draft_with_conflict_notes(tmp_path: Path) -> None:
    scenes = [_enriched_scene("scene_k1", 1, 10), _enriched_scene("scene_k2", 11, 20)]
    responses = [
        _payload_json("scene_k1", facts=[_fact_item(predicate="死因", value="事故")]),
        _payload_json("scene_k2", facts=[_fact_item(predicate="死因", value="自杀", line_start=11, line_end=12)]),
    ]
    bundle, state = _completed_state(tmp_path, scenes, responses)
    final = finalize_knowledge_candidates(bundle, state, alias_config=ALIASES)
    review = create_knowledge_review(bundle, final, reviewer="project_owner_01")
    assert review.review_status == "draft"
    assert review.total_candidates == len(final.fact_documents)
    statuses = {item.review_status for item in review.card_reviews}
    assert statuses == {ReviewStatus.needs_review}
    assert all("冲突组" in item.notes for item in review.card_reviews)


def test_review_allows_approved_card_while_top_level_draft(tmp_path: Path) -> None:
    bundle = _load_bundle(tmp_path, [_enriched_scene("scene_k1", 1, 10)])
    review = create_knowledge_review(
        bundle,
        finalize_knowledge_candidates(bundle, create_knowledge_run(bundle, model_id="m"), alias_config=ALIASES),
        reviewer="r",
    )
    data = json.loads(review.model_dump_json())
    data["total_candidates"] = 1
    data["card_reviews"] = [
        {
            "card_id": "fact_x",
            "document_type": "fact",
            "scene_id": "scene_k1",
            "review_status": "approved",
            "reviewer": "r",
            "notes": "",
        }
    ]
    assert KnowledgeReviewDocument.model_validate(data).card_reviews[0].review_status is ReviewStatus.approved


def test_review_top_level_approved_requires_complete_decisions(tmp_path: Path) -> None:
    bundle = _load_bundle(tmp_path, [_enriched_scene("scene_k1", 1, 10)])
    review = create_knowledge_review(
        bundle,
        finalize_knowledge_candidates(bundle, create_knowledge_run(bundle, model_id="m"), alias_config=ALIASES),
    )
    data = json.loads(review.model_dump_json())
    data["review_status"] = "approved"
    data["reviewer"] = "reviewer"
    data["total_candidates"] = 1
    data["card_reviews"] = [
        {
            "card_id": "fact_x",
            "document_type": "fact",
            "scene_id": "scene_k1",
            "review_status": "draft",
            "reviewer": "",
            "notes": "",
        }
    ]
    with pytest.raises(ValidationError, match="未定稿"):
        KnowledgeReviewDocument.model_validate(data)


def test_review_save_load_roundtrip(tmp_path: Path) -> None:
    bundle = _load_bundle(tmp_path, [_enriched_scene("scene_k1", 1, 10)])
    final = finalize_knowledge_candidates(bundle, create_knowledge_run(bundle, model_id="m"), alias_config=ALIASES)
    review = create_knowledge_review(bundle, final, reviewer="project_owner_01")
    path = tmp_path / "knowledge_review.json"
    save_knowledge_review(path, review)
    assert load_knowledge_review(path) == review


# ---------- 原子保存与产物 ----------


def test_save_documents_jsonl_roundtrip_and_empty(tmp_path: Path) -> None:
    scenes = [_enriched_scene("scene_k1", 1, 10)]
    bundle, state = _completed_state(tmp_path, scenes, [_payload_json("scene_k1", facts=[_fact_item()])])
    final = finalize_knowledge_candidates(bundle, state, alias_config=ALIASES)
    path = tmp_path / "facts.jsonl"
    save_documents_jsonl(path, final.fact_documents)
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 1
    assert lines[0]["id"] == final.fact_documents[0].id
    empty = tmp_path / "empty.jsonl"
    save_documents_jsonl(empty, [])
    assert empty.read_text(encoding="utf-8") == ""
    assert not (tmp_path / "facts.jsonl.tmp").exists()  # 无 tmp 残留


def test_save_run_with_manifest_pair(tmp_path: Path) -> None:
    bundle = _load_bundle(tmp_path, [_enriched_scene("scene_k1", 1, 10)])
    final = finalize_knowledge_candidates(bundle, create_knowledge_run(bundle, model_id="m"), alias_config=ALIASES)
    state = create_knowledge_run(bundle, model_id="m")
    manifest = build_knowledge_run_manifest(
        bundle, state, final, attempted_scene_ids=[], started_at="t0", completed_at="t1"
    )
    state_path = tmp_path / "run.json"
    manifest_path = tmp_path / "manifest.json"
    save_knowledge_run_with_manifest(state_path, manifest_path, state, manifest)
    assert load_knowledge_run(state_path) == state
    loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded_manifest["model_id"] == "m"
    assert loaded_manifest["scene_status_counts"] == {"pending": 1}
    assert not (tmp_path / "run.json.tmp").exists()
    assert not (tmp_path / "manifest.json.tmp").exists()


def test_manifest_rejects_invalid_state(tmp_path: Path) -> None:
    bundle = _load_bundle(tmp_path, [_enriched_scene("scene_k1", 1, 10)])
    other = _load_bundle(tmp_path / "o", [_enriched_scene("scene_k1", 1, 10, text="别的\n" * 10)])
    state = create_knowledge_run(other, model_id="m")
    final = finalize_knowledge_candidates(bundle, create_knowledge_run(bundle, model_id="m"), alias_config=ALIASES)
    with pytest.raises(ValueError):
        build_knowledge_run_manifest(bundle, state, final, attempted_scene_ids=[])


def test_quality_report_contents(tmp_path: Path) -> None:
    scenes = [_enriched_scene("scene_k1", 1, 10), _enriched_scene("scene_k2", 11, 20)]
    responses = [
        _payload_json(
            "scene_k1",
            facts=[_fact_item(predicate="死因", value="事故")],
            relations=[_relation_item()],
            events=[_event_item()],
        ),
        _payload_json("scene_k2", facts=[_fact_item(predicate="死因", value="自杀", line_start=11, line_end=12)]),
    ]
    bundle, state = _completed_state(tmp_path, scenes, responses)
    final = finalize_knowledge_candidates(bundle, state, alias_config=ALIASES)
    report = build_knowledge_quality_report(bundle, state, final)
    assert report["scene_status_counts"] == {"success": 2}
    assert report["card_counts"] == {"fact": 2, "relation": 1, "event": 1, "total": 4}
    assert report["conflict_group_count"] == 1
    assert report["duplicate_group_count"] == 0
    assert report["per_unit"][UNIT_A]["facts"] == 2
    assert report["gates"]["no_approved_cards"] is True
    assert report["gates"]["review_statuses"] == {"needs_review": 2, "draft": 2}
    assert report["gates"]["embedding_generated"] is False
    assert report["model_calls"]["minimum_required_calls"] == 2
    assert report["evidence_quality"]["overbroad_counts"] == {
        "fact": 0,
        "relation": 0,
        "event": 0,
        "total": 0,
    }


def test_run_state_save_rejects_invalid(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="非法"):
        save_knowledge_run(tmp_path / "run.json", {"schema_version": 999})


def test_breakpoint_resume_after_partial(tmp_path: Path) -> None:
    # 两场景：第一场景成功后中断（状态已落盘），续跑只处理剩余场景。
    scenes = [_enriched_scene("scene_k1", 1, 10), _enriched_scene("scene_k2", 11, 20)]
    bundle = _load_bundle(tmp_path, scenes)
    state_path = tmp_path / "run.json"
    state = create_knowledge_run(bundle, model_id="m")
    client1 = ScriptedClient([_payload_json("scene_k1", facts=[_fact_item()])])
    result1 = generate_knowledge_candidates(
        bundle, state, client1, scene_ids=["scene_k1"], state_path=state_path, alias_config=ALIASES
    )
    assert result1.succeeded_scene_ids == ["scene_k1"]
    # 模拟中断后重启：从磁盘加载，默认选择只补 pending（成功场景不再选中）。
    state_loaded = load_knowledge_run(state_path)
    client2 = ScriptedClient(
        [_payload_json("scene_k2", facts=[_fact_item(value="第二场景事实", line_start=11, line_end=12)])]
    )
    result2 = generate_knowledge_candidates(bundle, state_loaded, client2, state_path=state_path, alias_config=ALIASES)
    assert result2.attempted_scene_ids == ["scene_k2"]
    assert result2.skipped_scene_ids == []  # 成功场景默认不选中（结果不被触碰）
    assert len(client2.calls) == 1
    # 第一场景的成功候选原样保留（成功结果保护）
    states = {item.scene_id: item for item in result2.new_state.scene_states}
    assert states["scene_k1"].candidates is not None
    final = finalize_knowledge_candidates(bundle, result2.new_state, alias_config=ALIASES)
    assert len(final.fact_documents) == 2
