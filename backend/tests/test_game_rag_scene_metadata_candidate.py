"""P4B 场景元数据候选生成与断点续审基础设施测试：合成 bundle + 替身模型客户端
覆盖候选运行状态、三摘要绑定、严格 JSON 解析、失败重试、断点恢复、显式重跑审计、
候选合并（最多 needs_review、人工字段保护）、长场景分片归并、原子保存故障注入、
运行 manifest，以及真实未冻结目录的零调用零写入拒绝。

不测试 Pydantic / 标准库自身，只测试项目契约；不放宽断言迁就实现。
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

import knowledge.game_rag.scene_metadata_candidate as candidate_module
import knowledge.game_rag.scene_metadata_review as review_module
from knowledge.game_rag.models import (
    ContentScope,
    RealityStatus,
    ReviewStatus,
    SceneDocument,
    SourceSpan,
    StoryContext,
    TemporalScope,
)
from knowledge.game_rag.scene_metadata_candidate import (
    CANDIDATE_CHUNK_MAX_LINES,
    CANDIDATE_RUN_SCHEMA_VERSION,
    CandidateParseError,
    build_candidate_run_manifest,
    create_candidate_run,
    generate_scene_candidates,
    load_candidate_run,
    merge_candidates_into_review,
    parse_scene_candidate,
    save_candidate_run,
    save_candidate_run_with_manifest,
    select_candidate_scenes,
    validate_candidate_run,
    write_candidate_run_manifest,
)
from knowledge.game_rag.scene_metadata_review import (
    SceneMetadataDecision,
    create_scene_metadata_review,
    load_frozen_scene_bundle,
    save_scene_metadata_review,
    validate_scene_metadata_review,
)

UNIT_A = "vol99_9合成场景A"
UNIT_B = "vol99_9合成场景B"
PATH_A = "gametext/纸上魔法使/synth_a.txt"
PATH_B = "gametext/纸上魔法使/synth_b.txt"
MODEL_ID = "synthetic-model"

REVIEW_DIR = (
    Path(__file__).resolve().parents[1] / "data" / "knowledge" / "tsukiyashiro_kisaki" / "scene_boundary_review"
)

needs_p3_review_files = pytest.mark.skipif(
    not (REVIEW_DIR / "boundary_overrides.json").exists() or not (REVIEW_DIR / "low_candidate_review.json").exists(),
    reason="P3 审核决定文件不存在（data/knowledge 未随测试环境分发）",
)


# ---------- 合成 bundle 与替身客户端 ----------


def _scene(
    scene_id: str,
    unit_id: str,
    source_path: str,
    line_start: int,
    line_end: int,
    *,
    speakers: list[str] | None = None,
) -> SceneDocument:
    span_lines = line_end - line_start + 1
    body = "\n".join(f"{unit_id} 原文第{i}行" for i in range(1, span_lines + 1))
    return SceneDocument(
        id=scene_id,
        title=f"合成 {scene_id} L{line_start}-{line_end}",
        text=body,
        story=StoryContext(
            volume_number=99,
            story_unit_id=unit_id,
            story_title=unit_id,
            content_scope=ContentScope.main_story,
            temporal_scope=None,
        ),
        source=SourceSpan(source_path=source_path, line_start=line_start, line_end=line_end),
        speakers=speakers or [],
        mentioned_characters=[],
        present_characters=[],
        review_status=ReviewStatus.draft,
    )


def _manifest(units: dict[str, int]) -> dict:
    return {
        "schema_version": 1,
        "boundary_review_status": "approved",
        "reviewer": "boundary-auditor",
        "min_dialogue_turns": 6,
        "source_prefix": "gametext/纸上魔法使",
        "units": {
            uid: {
                "story_title": uid,
                "boundaries": [11] if uid == UNIT_A and count == 2 else [],
                "decisions": {},
                "adds": [],
                "scenes": count,
            }
            for uid, count in units.items()
        },
        "total_scenes": sum(units.values()),
        "scene_review_status": "draft",
        "note": "synthetic manifest",
    }


def _default_scenes() -> list[SceneDocument]:
    return [
        _scene("scene_a1", UNIT_A, PATH_A, 1, 10, speakers=["妃", "琉璃"]),
        _scene("scene_a2", UNIT_A, PATH_A, 11, 20, speakers=["妃"]),
        _scene("scene_b1", UNIT_B, PATH_B, 1, 15, speakers=["汀"]),
    ]


def _write_bundle(tmp_path: Path, scenes: list[SceneDocument], manifest: dict) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    scene_path = tmp_path / "scenes.jsonl"
    manifest_path = tmp_path / "boundary_manifest.json"
    scene_path.write_text("".join(s.model_dump_json() + "\n" for s in scenes), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return scene_path, manifest_path


def _load_bundle(tmp_path: Path, scenes: list[SceneDocument] | None = None):
    scenes = scenes if scenes is not None else _default_scenes()
    scene_path, manifest_path = _write_bundle(tmp_path, scenes, _manifest({UNIT_A: 2, UNIT_B: 1}))
    return load_frozen_scene_bundle(scene_path, manifest_path)


def _candidate_json(
    scene_id: str,
    line_start: int,
    *,
    line_end: int | None = None,
    viewpoint: str = "妃第一人称",
    temporal: str = "current",
    reality: str = "objective",
    mentioned: list[str] | None = None,
    present: list[str] | None = None,
) -> str:
    return json.dumps(
        {
            "scene_id": scene_id,
            "viewpoint": viewpoint,
            "temporal_scope": temporal,
            "reality_status": reality,
            "mentioned_characters": mentioned if mentioned is not None else [],
            "present_characters": present if present is not None else ["妃"],
            "evidence": [{"line_start": line_start, "line_end": line_end if line_end is not None else line_start}],
            "reasons": [f"{scene_id} 合成判断理由"],
            "warnings": [],
        },
        ensure_ascii=False,
    )


class RecordingClient:
    """记录全部调用的替身（用于断言零模型调用）。"""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        return "{}"


class ScriptedClient:
    """按调用序消费脚本的替身：str 正常返回，异常实例抛出。"""

    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.script:
            raise AssertionError("模型客户端被意外额外调用")
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class PerSceneClient:
    """按 prompt 中的 scene_id 与 evidence 允许范围返回合法候选 JSON 的替身。"""

    def __init__(self, **overrides) -> None:
        self.prompts: list[str] = []
        self.overrides = overrides

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        scene_id = re.search(r"scene_id: (\S+)", prompt).group(1)
        allowed = re.search(r"必须落在 L(\d+)-L(\d+) 内", prompt)
        line_start = int(allowed.group(1))
        line_end = int(allowed.group(2))
        chunk = re.search(r"第 (\d+)/(\d+) 片", prompt)
        temporal = self.overrides.get("temporal_scope", "current")
        if chunk and "temporal_by_chunk" in self.overrides:
            temporal = self.overrides["temporal_by_chunk"][int(chunk.group(1)) - 1]
        return _candidate_json(
            scene_id,
            line_start,
            line_end=min(line_start + 2, line_end),
            viewpoint=self.overrides.get("viewpoint", "妃第一人称"),
            temporal=temporal,
            reality=self.overrides.get("reality_status", "objective"),
            mentioned=self.overrides.get("mentioned", []),
            present=self.overrides.get("present", ["妃"]),
        )


def _run_state(tmp_path: Path):
    bundle = _load_bundle(tmp_path)
    return bundle, create_candidate_run(bundle, model_id=MODEL_ID)


def _decision(scene: SceneDocument, **overrides) -> SceneMetadataDecision:
    values: dict = {
        "scene_id": scene.id,
        "story_unit_id": scene.story.story_unit_id,
        "source": scene.source,
        "review_status": ReviewStatus.draft,
    }
    values.update(overrides)
    return SceneMetadataDecision(**values)


def _approved_decision(scene: SceneDocument, *, reviewer: str = "human-1") -> SceneMetadataDecision:
    return _decision(
        scene,
        viewpoint="第三人称",
        temporal_scope=TemporalScope.current,
        reality_status=RealityStatus.objective,
        mentioned_characters=[],
        present_characters=["妃"],
        evidence=[
            SourceSpan(
                source_path=scene.source.source_path,
                line_start=scene.source.line_start,
                line_end=scene.source.line_start,
            )
        ],
        reasons=["人工已定稿"],
        review_status=ReviewStatus.approved,
        reviewer=reviewer,
    )


def _review_doc_with_decisions(bundle, decisions: list[SceneMetadataDecision]):
    base = create_scene_metadata_review(bundle, reviewer="meta-auditor")
    return base.model_copy(update={"scene_decisions": list(decisions)})


# ---------- 创建与确定性 ----------


class TestCreateCandidateRun:
    def test_initial_state_deterministic_and_bound(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        first = create_candidate_run(bundle, model_id=MODEL_ID)
        second = create_candidate_run(bundle, model_id=MODEL_ID)
        assert first.model_dump(mode="json") == second.model_dump(mode="json")
        assert first.schema_version == CANDIDATE_RUN_SCHEMA_VERSION
        assert first.total_source_scenes == 3
        assert first.model_id == MODEL_ID
        assert [item.scene_id for item in first.scene_states] == ["scene_a1", "scene_a2", "scene_b1"]
        assert all(item.status.value == "pending" for item in first.scene_states)
        assert all(item.candidate is None and item.attempts == 0 for item in first.scene_states)
        assert first.source_manifest.manifest_sha256 == bundle.manifest_digest
        assert first.source_manifest.scenes_sha256 == bundle.scenes_digest
        assert first.source_manifest.bundle_sha256 == bundle.bundle_digest
        assert first.generation_params["chunk_max_lines"] == CANDIDATE_CHUNK_MAX_LINES

    def test_custom_generation_params_merged(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        state = create_candidate_run(bundle, model_id=MODEL_ID, generation_params={"temperature": 0.1})
        assert state.generation_params["temperature"] == 0.1
        assert state.generation_params["chunk_max_lines"] == CANDIDATE_CHUNK_MAX_LINES

    @pytest.mark.parametrize("value", [0, -1, True, "150"])
    def test_invalid_chunk_max_lines_rejected(self, tmp_path, value):
        bundle = _load_bundle(tmp_path)
        with pytest.raises(ValueError, match="chunk_max_lines"):
            create_candidate_run(bundle, model_id=MODEL_ID, generation_params={"chunk_max_lines": value})

    def test_empty_model_id_rejected(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        with pytest.raises(ValueError, match="model_id 不得为空"):
            create_candidate_run(bundle, model_id="  ")

    def test_tampered_bundle_rejected(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        bundle.scenes[0].text = "篡改后的正文"
        with pytest.raises(ValueError, match="完整性校验失败"):
            create_candidate_run(bundle, model_id=MODEL_ID)

    def test_save_load_roundtrip_and_deterministic_bytes(self, tmp_path):
        bundle, state = _run_state(tmp_path / "frozen")
        path = tmp_path / "candidate_run.json"
        save_candidate_run(path, state)
        loaded = load_candidate_run(path)
        assert loaded.model_dump(mode="json") == state.model_dump(mode="json")
        first_bytes = path.read_bytes()
        save_candidate_run(path, loaded)
        assert path.read_bytes() == first_bytes  # 确定性：重复保存字节一致

    def test_state_contains_no_timestamps(self, tmp_path):
        bundle, state = _run_state(tmp_path / "frozen")
        payload = state.model_dump_json()
        for key in ("started_at", "completed_at", "timestamp"):
            assert key not in payload

    def test_save_rejects_invalid_state(self, tmp_path):
        path = tmp_path / "candidate_run.json"
        with pytest.raises(ValueError, match="拒绝写出"):
            save_candidate_run(path, {"schema_version": 1})
        assert not path.exists()
        assert not list(tmp_path.glob("*.tmp"))


# ---------- 校验：三摘要绑定与篡改拒绝 ----------


class TestValidateCandidateRun:
    def test_valid_state_passes(self, tmp_path):
        bundle, state = _run_state(tmp_path / "frozen")
        assert validate_candidate_run(state, bundle) == []
        assert validate_candidate_run(state.model_dump(mode="json"), bundle) == []

    def test_bundle_digest_mismatch_rejected(self, tmp_path):
        bundle_a, state = _run_state(tmp_path / "bundle_a")
        # 构造内容不同的第二份 bundle：改 note 不影响场景，摘要必然不同
        scenes = _default_scenes()
        manifest = _manifest({UNIT_A: 2, UNIT_B: 1})
        manifest["note"] = "another synthetic manifest"
        scene_path, manifest_path = _write_bundle(tmp_path / "bundle_b", scenes, manifest)
        bundle_b = load_frozen_scene_bundle(scene_path, manifest_path)
        errors = validate_candidate_run(state, bundle_b)
        assert any("manifest_sha256 与 bundle 不一致" in e for e in errors)
        assert any("bundle_sha256 与 bundle 不一致" in e for e in errors)

    def test_scene_content_change_rejected(self, tmp_path):
        bundle_a, state = _run_state(tmp_path / "bundle_a")
        scenes = _default_scenes()
        scenes[0] = scenes[0].model_copy(update={"text": "另一份正文"})
        bundle_b = _load_bundle(tmp_path / "bundle_b", scenes=scenes)
        errors = validate_candidate_run(state, bundle_b)
        assert any("scenes_sha256 与 bundle 不一致" in e for e in errors)

    def test_scene_source_tampered_in_state(self, tmp_path):
        bundle, state = _run_state(tmp_path / "frozen")
        raw = state.model_dump(mode="json")
        raw["scene_states"][0]["source"]["line_end"] = 99
        errors = validate_candidate_run(raw, bundle)
        assert any("source 被篡改" in e for e in errors)

    def test_candidate_evidence_out_of_span_in_state(self, tmp_path):
        bundle = _load_bundle(tmp_path / "frozen")
        client = PerSceneClient()
        result = generate_scene_candidates(bundle, create_candidate_run(bundle, model_id=MODEL_ID), client)
        state = result.new_state
        state.scene_states[0].candidate.evidence[0].line_end = 99  # 场景仅到 L10
        errors = validate_candidate_run(state, bundle)
        assert any("候选 evidence" in e and "超出场景范围" in e for e in errors)

    def test_candidate_evidence_wrong_path_in_state(self, tmp_path):
        bundle = _load_bundle(tmp_path / "frozen")
        result = generate_scene_candidates(bundle, create_candidate_run(bundle, model_id=MODEL_ID), PerSceneClient())
        state = result.new_state
        state.scene_states[0].candidate.evidence[0].source_path = "gametext/纸上魔法使/other.txt"
        errors = validate_candidate_run(state, bundle)
        assert any("与场景 source" in e and "不一致" in e for e in errors)

    def test_unsupported_schema_version_rejected(self, tmp_path):
        bundle, state = _run_state(tmp_path / "frozen")
        raw = state.model_dump(mode="json")
        raw["schema_version"] = CANDIDATE_RUN_SCHEMA_VERSION + 1
        errors = validate_candidate_run(raw, bundle)
        assert any("schema_version 必须为" in e for e in errors)

    def test_scene_set_mismatch_rejected(self, tmp_path):
        bundle, state = _run_state(tmp_path / "frozen")
        raw = state.model_dump(mode="json")
        raw["scene_states"][2]["scene_id"] = "scene_unknown"
        errors = validate_candidate_run(raw, bundle)
        assert any("含未知 scene" in e for e in errors)
        assert any("缺少 scene" in e for e in errors)


# ---------- 场景选择 ----------


class TestSelectCandidateScenes:
    def test_select_pending_in_frozen_order(self, tmp_path):
        bundle, state = _run_state(tmp_path / "frozen")
        assert select_candidate_scenes(state) == ["scene_a1", "scene_a2", "scene_b1"]

    def test_select_retry_failed_toggle(self, tmp_path):
        bundle = _load_bundle(tmp_path / "frozen")
        client = ScriptedClient(
            ["坏 JSON", "坏 JSON", "坏 JSON"]  # scene_a1 耗尽重试
            + [_candidate_json("scene_a2", 11), _candidate_json("scene_b1", 1)]  # 其余场景成功
        )
        result = generate_scene_candidates(bundle, create_candidate_run(bundle, model_id=MODEL_ID), client)
        assert select_candidate_scenes(result.new_state) == ["scene_a1"]  # 默认含失败重试
        assert select_candidate_scenes(result.new_state, retry_failed=False) == []

    def test_select_specified_ids_follow_frozen_order(self, tmp_path):
        bundle, state = _run_state(tmp_path / "frozen")
        assert select_candidate_scenes(state, scene_ids=["scene_b1", "scene_a2"]) == ["scene_a2", "scene_b1"]

    def test_select_unknown_id_rejected(self, tmp_path):
        bundle, state = _run_state(tmp_path / "frozen")
        with pytest.raises(ValueError, match="不在候选运行状态中"):
            select_candidate_scenes(state, scene_ids=["scene_nothing"])


# ---------- prompt 构建 ----------


class TestBuildCandidatePrompt:
    def test_deterministic_and_contains_contract(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        scene = bundle.scenes[0]
        from knowledge.game_rag.scene_metadata_candidate import build_candidate_prompt

        first = build_candidate_prompt(scene)
        second = build_candidate_prompt(scene)
        assert first == second
        assert "scene_id: scene_a1" in first
        assert f"L{scene.source.line_start}-{scene.source.line_end}" in first
        assert "必须落在 L1-L10 内" in first
        for label in ("梦境", "回忆", "书中故事", "宣传元叙事", "魔法重现", "无法判断"):
            assert label in first
        assert "不要 markdown 代码围栏" in first
        assert "speakers（仅供参考，不等于在场人物）" in first

    def test_chunk_prompt_declares_fragment(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        scene = bundle.scenes[0]
        from knowledge.game_rag.scene_metadata_candidate import build_candidate_prompt

        span = SourceSpan(source_path=PATH_A, line_start=3, line_end=7)
        prompt = build_candidate_prompt(scene, span=span, chunk_index=1, total_chunks=2)
        assert "第 1/2 片" in prompt
        assert "必须落在 L3-L7 内" in prompt
        assert "L3: " in prompt and "L7: " in prompt
        assert "L8: " not in prompt

    def test_span_outside_scene_rejected(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        scene = bundle.scenes[0]
        from knowledge.game_rag.scene_metadata_candidate import build_candidate_prompt

        span = SourceSpan(source_path=PATH_A, line_start=5, line_end=50)
        with pytest.raises(ValueError, match="必须落在场景 span"):
            build_candidate_prompt(scene, span=span)

    def test_text_span_line_mismatch_rejected(self, tmp_path):
        scenes = _default_scenes()
        scenes[0] = scenes[0].model_copy(update={"text": "只有一行"})  # span 声称 10 行
        bundle = _load_bundle(tmp_path, scenes=scenes)
        from knowledge.game_rag.scene_metadata_candidate import build_candidate_prompt

        with pytest.raises(ValueError, match="text 行数.*与 span 行数.*不一致"):
            build_candidate_prompt(bundle.scenes[0])


# ---------- 严格 JSON 解析 ----------


class TestParseSceneCandidate:
    def _scene(self, tmp_path):
        return _load_bundle(tmp_path).scenes[0]  # scene_a1 L1-10

    def test_valid_json_parsed_and_normalized(self, tmp_path):
        scene = self._scene(tmp_path)
        raw = _candidate_json(
            "scene_a1",
            2,
            line_end=4,
            mentioned=[" 琉璃 ", "妃", "琉璃"],
            present=["妃", " 汀 "],
        )
        candidate = parse_scene_candidate(raw, scene)
        assert candidate.scene_id == "scene_a1"
        assert candidate.mentioned_characters == ["妃", "琉璃"]  # 去空白、去重、稳定排序
        assert candidate.present_characters == ["妃", "汀"]
        assert candidate.evidence == [
            SourceSpan(source_path=PATH_A, line_start=2, line_end=4)
        ]  # source_path 由解析器补齐
        assert candidate.reasons == ["scene_a1 合成判断理由"]

    def test_unknown_values_allowed(self, tmp_path):
        scene = self._scene(tmp_path)
        raw = _candidate_json("scene_a1", 1, viewpoint="unknown", temporal="unknown", reality="unknown", present=[])
        candidate = parse_scene_candidate(raw, scene)
        assert candidate.viewpoint == "unknown"
        assert candidate.temporal_scope is TemporalScope.unknown
        assert candidate.reality_status is RealityStatus.unknown
        assert candidate.present_characters == []

    def test_markdown_fence_rejected(self, tmp_path):
        scene = self._scene(tmp_path)
        fenced = "```json\n" + _candidate_json("scene_a1", 1) + "\n```"
        with pytest.raises(CandidateParseError) as exc_info:
            parse_scene_candidate(fenced, scene)
        assert exc_info.value.error_kind == "markdown_fence"

    def test_invalid_json_rejected(self, tmp_path):
        scene = self._scene(tmp_path)
        with pytest.raises(CandidateParseError) as exc_info:
            parse_scene_candidate("这不是 JSON", scene)
        assert exc_info.value.error_kind == "invalid_json"

    def test_non_object_json_rejected(self, tmp_path):
        scene = self._scene(tmp_path)
        with pytest.raises(CandidateParseError) as exc_info:
            parse_scene_candidate("[1, 2, 3]", scene)
        assert exc_info.value.error_kind == "invalid_json"

    def test_extra_field_rejected(self, tmp_path):
        scene = self._scene(tmp_path)
        payload = json.loads(_candidate_json("scene_a1", 1))
        payload["unexpected_field"] = "额外字段"
        with pytest.raises(CandidateParseError) as exc_info:
            parse_scene_candidate(json.dumps(payload), scene)
        assert exc_info.value.error_kind == "schema_violation"

    def test_missing_required_field_rejected(self, tmp_path):
        scene = self._scene(tmp_path)
        payload = json.loads(_candidate_json("scene_a1", 1))
        del payload["viewpoint"]
        with pytest.raises(CandidateParseError) as exc_info:
            parse_scene_candidate(json.dumps(payload), scene)
        assert exc_info.value.error_kind == "schema_violation"

    def test_illegal_enum_rejected(self, tmp_path):
        scene = self._scene(tmp_path)
        raw = _candidate_json("scene_a1", 1, temporal="yesterday")
        with pytest.raises(CandidateParseError) as exc_info:
            parse_scene_candidate(raw, scene)
        assert exc_info.value.error_kind == "schema_violation"

    def test_duplicate_json_key_rejected(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        raw = _candidate_json("scene_a1", 1).replace(
            '"scene_id": "scene_a1",', '"scene_id": "scene_a1", "scene_id": "scene_a1",', 1
        )
        with pytest.raises(CandidateParseError) as exc_info:
            parse_scene_candidate(raw, bundle.scenes[0])
        assert exc_info.value.error_kind == "duplicate_json_key"

    def test_illegal_viewpoint_rejected(self, tmp_path):
        scene = self._scene(tmp_path)
        raw = _candidate_json("scene_a1", 1, viewpoint="随便写的自由文本视角")
        with pytest.raises(CandidateParseError) as exc_info:
            parse_scene_candidate(raw, scene)
        assert exc_info.value.error_kind == "schema_violation"

    def test_blank_character_name_rejected(self, tmp_path):
        scene = self._scene(tmp_path)
        raw = _candidate_json("scene_a1", 1, present=["妃", "   "])
        with pytest.raises(CandidateParseError) as exc_info:
            parse_scene_candidate(raw, scene)
        assert exc_info.value.error_kind == "schema_violation"

    def test_blank_warning_rejected_before_merge(self, tmp_path):
        scene = self._scene(tmp_path)
        payload = json.loads(_candidate_json("scene_a1", 1))
        payload["warnings"] = ["   "]
        with pytest.raises(CandidateParseError) as exc_info:
            parse_scene_candidate(json.dumps(payload, ensure_ascii=False), scene)
        assert exc_info.value.error_kind == "schema_violation"

    def test_scene_id_mismatch_rejected(self, tmp_path):
        scene = self._scene(tmp_path)
        raw = _candidate_json("scene_a2", 1)
        with pytest.raises(CandidateParseError) as exc_info:
            parse_scene_candidate(raw, scene)
        assert exc_info.value.error_kind == "scene_id_mismatch"

    def test_evidence_out_of_scene_span_rejected(self, tmp_path):
        scene = self._scene(tmp_path)
        raw = _candidate_json("scene_a1", 8, line_end=15)  # 场景仅到 L10
        with pytest.raises(CandidateParseError) as exc_info:
            parse_scene_candidate(raw, scene)
        assert exc_info.value.error_kind == "evidence_out_of_range"

    def test_evidence_out_of_chunk_span_rejected(self, tmp_path):
        scene = self._scene(tmp_path)
        chunk = SourceSpan(source_path=PATH_A, line_start=4, line_end=6)
        raw = _candidate_json("scene_a1", 2, line_end=3)  # 在场景内但不在分片内
        with pytest.raises(CandidateParseError) as exc_info:
            parse_scene_candidate(raw, scene, span=chunk)
        assert exc_info.value.error_kind == "evidence_out_of_range"

    def test_empty_evidence_rejected(self, tmp_path):
        scene = self._scene(tmp_path)
        payload = json.loads(_candidate_json("scene_a1", 1))
        payload["evidence"] = []
        with pytest.raises(CandidateParseError) as exc_info:
            parse_scene_candidate(json.dumps(payload), scene)
        assert exc_info.value.error_kind == "schema_violation"

    def test_empty_reasons_rejected(self, tmp_path):
        scene = self._scene(tmp_path)
        payload = json.loads(_candidate_json("scene_a1", 1))
        payload["reasons"] = []
        with pytest.raises(CandidateParseError) as exc_info:
            parse_scene_candidate(json.dumps(payload), scene)
        assert exc_info.value.error_kind == "schema_violation"

    def test_non_string_output_rejected(self, tmp_path):
        scene = self._scene(tmp_path)
        with pytest.raises(CandidateParseError) as exc_info:
            parse_scene_candidate(12345, scene)  # type: ignore[arg-type]
        assert exc_info.value.error_kind == "invalid_output"


# ---------- 候选生成、失败重试与断点恢复 ----------


class TestGenerateSceneCandidates:
    def test_success_flow(self, tmp_path):
        bundle = _load_bundle(tmp_path / "frozen")
        client = PerSceneClient()
        result = generate_scene_candidates(bundle, create_candidate_run(bundle, model_id=MODEL_ID), client)
        assert result.attempted_scene_ids == ["scene_a1", "scene_a2", "scene_b1"]
        assert result.succeeded_scene_ids == result.attempted_scene_ids
        assert result.failed_scene_ids == [] and result.skipped_scene_ids == []
        assert len(client.prompts) == 3
        state = result.new_state
        assert all(item.status.value == "success" for item in state.scene_states)
        assert all(item.candidate is not None for item in state.scene_states)
        assert all(item.last_failure is None for item in state.scene_states)
        assert all(item.attempts == 1 for item in state.scene_states)
        assert validate_candidate_run(state, bundle) == []

    def test_deterministic_given_same_client(self, tmp_path):
        bundle = _load_bundle(tmp_path / "frozen")
        first = generate_scene_candidates(bundle, create_candidate_run(bundle, model_id=MODEL_ID), PerSceneClient())
        second = generate_scene_candidates(bundle, create_candidate_run(bundle, model_id=MODEL_ID), PerSceneClient())
        assert first.new_state.model_dump(mode="json") == second.new_state.model_dump(mode="json")

    def test_invalid_max_attempts_rejected(self, tmp_path):
        bundle, state = _run_state(tmp_path / "frozen")
        with pytest.raises(ValueError, match="max_attempts 必须"):
            generate_scene_candidates(bundle, state, PerSceneClient(), max_attempts=0)

    def test_structurally_invalid_state_rejected(self, tmp_path):
        bundle = _load_bundle(tmp_path / "frozen")
        with pytest.raises(ValueError, match="候选运行状态非法"):
            generate_scene_candidates(bundle, {"schema_version": 1}, PerSceneClient())

    def test_bundle_mismatch_rejected_before_model_calls(self, tmp_path):
        bundle_a, state = _run_state(tmp_path / "bundle_a")
        scenes = _default_scenes()
        scenes[2] = scenes[2].model_copy(update={"text": "另一份正文"})
        bundle_b = _load_bundle(tmp_path / "bundle_b", scenes=scenes)
        client = RecordingClient()
        with pytest.raises(ValueError, match="未通过校验"):
            generate_scene_candidates(bundle_b, state, client)
        assert client.calls == []  # 任何模型调用之前拒绝

    def test_tampered_state_rejected_before_model_calls(self, tmp_path):
        bundle, state = _run_state(tmp_path / "frozen")
        state.scene_states[0].source = SourceSpan(source_path=PATH_A, line_start=1, line_end=99)
        client = RecordingClient()
        with pytest.raises(ValueError, match="未通过校验"):
            generate_scene_candidates(bundle, state, client)
        assert client.calls == []

    def test_mutated_model_consistency_rejected_before_model_calls(self, tmp_path):
        bundle, state = _run_state(tmp_path / "frozen")
        state.scene_states[0].status = candidate_module.CandidateGenerationStatus.success
        client = RecordingClient()
        with pytest.raises(ValueError, match="候选运行状态非法"):
            generate_scene_candidates(bundle, state, client)
        assert client.calls == []

    def test_invalid_json_exhausts_attempts(self, tmp_path):
        bundle = _load_bundle(tmp_path / "frozen")
        client = ScriptedClient(["坏 JSON"] * 9)  # 3 场景 × max_attempts=3
        result = generate_scene_candidates(
            bundle, create_candidate_run(bundle, model_id=MODEL_ID), client, max_attempts=3
        )
        assert result.succeeded_scene_ids == []
        assert result.failed_scene_ids == ["scene_a1", "scene_a2", "scene_b1"]
        state = result.new_state
        assert all(item.status.value == "failed" for item in state.scene_states)
        assert all(item.last_failure.error_kind == "invalid_json" for item in state.scene_states)
        assert all(item.attempts == 3 for item in state.scene_states)
        assert all(item.candidate is None for item in state.scene_states)

    def test_timeout_failure_kind(self, tmp_path):
        bundle = _load_bundle(tmp_path / "frozen")
        client = ScriptedClient([TimeoutError("模型超时")] * 3)
        result = generate_scene_candidates(
            bundle, create_candidate_run(bundle, model_id=MODEL_ID), client, scene_ids=["scene_a1"]
        )
        assert result.new_state.scene_states[0].last_failure.error_kind == "timeout"

    def test_model_error_failure_kind(self, tmp_path):
        bundle = _load_bundle(tmp_path / "frozen")
        client = ScriptedClient([RuntimeError("连接被拒绝")] * 3)
        result = generate_scene_candidates(
            bundle, create_candidate_run(bundle, model_id=MODEL_ID), client, scene_ids=["scene_a1"]
        )
        failure = result.new_state.scene_states[0].last_failure
        assert failure.error_kind == "model_error"
        assert "RuntimeError" in failure.detail

    def test_failure_detail_truncated_and_prompt_not_leaked(self, tmp_path):
        bundle = _load_bundle(tmp_path / "frozen")
        leak = "SECRET-SENTINEL-不得进入运行状态" * 50
        client = ScriptedClient([RuntimeError(f"boom {leak}")] * 3)
        result = generate_scene_candidates(
            bundle, create_candidate_run(bundle, model_id=MODEL_ID), client, scene_ids=["scene_a1"]
        )
        detail = result.new_state.scene_states[0].last_failure.detail
        assert "SECRET-SENTINEL" not in detail  # 失败摘要不携带完整错误内容
        assert len(detail) <= 201  # 200 字符 + 截断标记

    def test_schema_failure_detail_does_not_persist_model_value(self, tmp_path):
        bundle = _load_bundle(tmp_path / "frozen")
        leak = "SECRET-SENTINEL-model-output"
        client = ScriptedClient([_candidate_json("scene_a1", 1, viewpoint=leak)] * 3)
        result = generate_scene_candidates(
            bundle, create_candidate_run(bundle, model_id=MODEL_ID), client, scene_ids=["scene_a1"]
        )
        failure = result.new_state.scene_states[0].last_failure
        assert failure.error_kind == "schema_violation"
        assert leak not in failure.detail

    def test_retry_then_success_in_same_run(self, tmp_path):
        bundle = _load_bundle(tmp_path / "frozen")
        client = ScriptedClient(
            ["坏 JSON", _candidate_json("scene_a1", 1), _candidate_json("scene_a2", 11), _candidate_json("scene_b1", 1)]
        )
        result = generate_scene_candidates(
            bundle, create_candidate_run(bundle, model_id=MODEL_ID), client, max_attempts=3
        )
        assert result.succeeded_scene_ids == ["scene_a1", "scene_a2", "scene_b1"]
        assert result.new_state.scene_states[0].attempts == 2  # 失败一次后重试成功
        assert result.new_state.scene_states[0].last_failure is None

    def test_success_scene_skipped_without_rerun_reason(self, tmp_path):
        bundle = _load_bundle(tmp_path / "frozen")
        state = create_candidate_run(bundle, model_id=MODEL_ID)
        first = generate_scene_candidates(bundle, state, PerSceneClient())
        old_candidate = first.new_state.scene_states[0].candidate
        client = PerSceneClient(present=["完全不同的人物"])
        second = generate_scene_candidates(bundle, first.new_state, client, scene_ids=["scene_a1"])
        assert second.attempted_scene_ids == ["scene_a1"]
        assert second.skipped_scene_ids == ["scene_a1"]  # 成功结果不被无意覆盖
        assert second.new_state.scene_states[0].candidate == old_candidate
        assert client.prompts == []  # 未触发任何模型调用

    def test_full_rerun_requires_explicit_selection(self, tmp_path):
        bundle = _load_bundle(tmp_path / "frozen")
        state = create_candidate_run(bundle, model_id=MODEL_ID)
        first = generate_scene_candidates(bundle, state, PerSceneClient())
        # 不指定 scene_ids 时成功场景不在选择范围：rerun 原因指向未选中场景 → 拒绝
        with pytest.raises(ValueError, match="指向未选中的场景"):
            generate_scene_candidates(bundle, first.new_state, PerSceneClient(), rerun_reasons={"scene_a1": "重新生成"})

    def test_explicit_rerun_with_reason_audited(self, tmp_path):
        bundle = _load_bundle(tmp_path / "frozen")
        state = create_candidate_run(bundle, model_id=MODEL_ID)
        first = generate_scene_candidates(bundle, state, PerSceneClient())
        client = PerSceneClient(viewpoint="第三人称", present=["汀"])
        second = generate_scene_candidates(
            bundle,
            first.new_state,
            client,
            scene_ids=["scene_a1"],
            rerun_reasons={"scene_a1": "模型升级后重新生成"},
        )
        assert second.succeeded_scene_ids == ["scene_a1"]
        item = second.new_state.scene_states[0]
        assert item.candidate.viewpoint == "第三人称"  # 候选被显式替换
        assert item.rerun_history[0].rerun_reason == "模型升级后重新生成"
        assert item.rerun_history[0].outcome == "success"
        assert item.rerun_history[0].previous_status.value == "success"

    def test_explicit_rerun_failure_preserves_old_candidate(self, tmp_path):
        bundle = _load_bundle(tmp_path / "frozen")
        state = create_candidate_run(bundle, model_id=MODEL_ID)
        first = generate_scene_candidates(bundle, state, PerSceneClient())
        old_candidate = first.new_state.scene_states[0].candidate
        client = ScriptedClient(["坏 JSON"] * 3)
        second = generate_scene_candidates(
            bundle,
            first.new_state,
            client,
            scene_ids=["scene_a1"],
            rerun_reasons={"scene_a1": "重跑失败场景验证"},
            max_attempts=3,
        )
        assert second.failed_scene_ids == ["scene_a1"]
        item = second.new_state.scene_states[0]
        assert item.status.value == "success"  # 旧成功状态不被损坏
        assert item.candidate == old_candidate  # 旧候选保留
        assert item.rerun_history[0].outcome == "failed"  # 重跑失败被审计
        assert item.last_failure is not None and item.last_failure.error_kind == "invalid_json"

    def test_rerun_reason_for_non_success_rejected(self, tmp_path):
        bundle = _load_bundle(tmp_path / "frozen")
        state = create_candidate_run(bundle, model_id=MODEL_ID)
        with pytest.raises(ValueError, match="rerun 原因只适用于已成功场景"):
            generate_scene_candidates(
                bundle, state, PerSceneClient(), scene_ids=["scene_a1"], rerun_reasons={"scene_a1": "提前重跑"}
            )

    def test_blank_rerun_reason_rejected(self, tmp_path):
        bundle = _load_bundle(tmp_path / "frozen")
        state = create_candidate_run(bundle, model_id=MODEL_ID)
        first = generate_scene_candidates(bundle, state, PerSceneClient())
        with pytest.raises(ValueError, match="rerun 原因不得为空白字符串"):
            generate_scene_candidates(
                bundle, first.new_state, PerSceneClient(), scene_ids=["scene_a1"], rerun_reasons={"scene_a1": "  "}
            )

    def test_state_path_persists_progress_per_scene(self, tmp_path):
        bundle = _load_bundle(tmp_path / "frozen")
        state_path = tmp_path / "run" / "candidate_run.json"
        client = ScriptedClient(["坏 JSON"] * 3 + [_candidate_json("scene_a2", 11), _candidate_json("scene_b1", 1)])
        result = generate_scene_candidates(
            bundle, create_candidate_run(bundle, model_id=MODEL_ID), client, max_attempts=3, state_path=state_path
        )
        assert result.failed_scene_ids == ["scene_a1"]
        persisted = load_candidate_run(state_path)
        assert [item.status.value for item in persisted.scene_states] == ["failed", "success", "success"]

    def test_crash_resume_from_persisted_state(self, tmp_path):
        bundle = _load_bundle(tmp_path / "frozen")
        state_path = tmp_path / "run" / "candidate_run.json"
        # scene_a1 成功、scene_a2 处理中进程被中断（KeyboardInterrupt 不被收敛为失败）
        client = ScriptedClient([_candidate_json("scene_a1", 1), KeyboardInterrupt()])
        with pytest.raises(KeyboardInterrupt):
            generate_scene_candidates(
                bundle, create_candidate_run(bundle, model_id=MODEL_ID), client, state_path=state_path
            )
        persisted = load_candidate_run(state_path)
        assert persisted.scene_states[0].status.value == "success"
        assert persisted.scene_states[1].status.value == "pending"  # 中断场景未落盘损坏
        # 断点续跑：从持久化状态继续，剩余场景完成
        resumed = generate_scene_candidates(bundle, persisted, PerSceneClient(), state_path=state_path)
        assert resumed.succeeded_scene_ids == ["scene_a2", "scene_b1"]
        assert resumed.attempted_scene_ids == ["scene_a2", "scene_b1"]
        final = load_candidate_run(state_path)
        assert all(item.status.value == "success" for item in final.scene_states)

    def test_failed_scene_retried_in_next_run(self, tmp_path):
        bundle = _load_bundle(tmp_path / "frozen")
        state = create_candidate_run(bundle, model_id=MODEL_ID)
        first = generate_scene_candidates(bundle, state, ScriptedClient(["坏 JSON"] * 9), max_attempts=3)
        assert all(item.status.value == "failed" for item in first.new_state.scene_states)
        second = generate_scene_candidates(bundle, first.new_state, PerSceneClient())
        assert second.succeeded_scene_ids == ["scene_a1", "scene_a2", "scene_b1"]
        item = second.new_state.scene_states[0]
        assert item.attempts == 4  # 上一轮 3 次 + 本轮 1 次
        assert item.last_failure is None  # 成功后清空失败摘要

    def test_idempotent_full_run(self, tmp_path):
        bundle = _load_bundle(tmp_path / "frozen")
        state = create_candidate_run(bundle, model_id=MODEL_ID)
        first = generate_scene_candidates(bundle, state, PerSceneClient())
        second = generate_scene_candidates(bundle, first.new_state, PerSceneClient())
        assert second.attempted_scene_ids == []  # 全部成功：无待处理场景
        assert second.new_state.model_dump(mode="json") == first.new_state.model_dump(mode="json")


# ---------- 长场景分片归并 ----------


def _long_scene_bundle(tmp_path: Path):
    scene = _scene("scene_long", UNIT_A, PATH_A, 1, 320, speakers=["妃"])
    scene_path, manifest_path = _write_bundle(tmp_path, [scene], _manifest({UNIT_A: 1}))
    return load_frozen_scene_bundle(scene_path, manifest_path)


class TestLongSceneChunking:
    def test_chunked_generation_merges_back_to_scene_id(self, tmp_path):
        bundle = _long_scene_bundle(tmp_path / "frozen")
        expected_chunks = -(-320 // CANDIDATE_CHUNK_MAX_LINES)  # ceil(320/150) = 3
        client = PerSceneClient(mentioned=["琉璃"], present=["妃", "汀"])
        result = generate_scene_candidates(bundle, create_candidate_run(bundle, model_id=MODEL_ID), client)
        assert len(client.prompts) == expected_chunks
        assert result.succeeded_scene_ids == ["scene_long"]
        state = result.new_state
        assert len(state.scene_states) == 1  # 不重切 P3 场景：仍只有一个场景状态
        assert state.scene_states[0].source == bundle.scenes[0].source
        candidate = state.scene_states[0].candidate
        assert candidate.scene_id == "scene_long"  # evidence 与决定归并回原 scene_id
        assert len(candidate.evidence) == expected_chunks  # 各分片 evidence 并集
        assert candidate.mentioned_characters == ["琉璃"]
        assert candidate.present_characters == ["妃", "汀"]  # 去重 + sorted(set()) 稳定排序
        assert candidate.reasons == ["scene_long 合成判断理由"]  # 理由去重
        assert validate_candidate_run(state, bundle) == []

    def test_custom_chunk_max_lines_controls_actual_prompt_count(self, tmp_path):
        bundle = _long_scene_bundle(tmp_path / "frozen")
        state = create_candidate_run(bundle, model_id=MODEL_ID, generation_params={"chunk_max_lines": 100})
        client = PerSceneClient()
        result = generate_scene_candidates(bundle, state, client, scene_ids=["scene_long"])
        assert result.succeeded_scene_ids == ["scene_long"]
        assert len(client.prompts) == 4

    def test_chunk_disagreement_becomes_unknown_with_warning(self, tmp_path):
        bundle = _long_scene_bundle(tmp_path / "frozen")
        client = PerSceneClient(temporal_by_chunk=["current", "flashback", "flashback"])
        result = generate_scene_candidates(bundle, create_candidate_run(bundle, model_id=MODEL_ID), client)
        candidate = result.new_state.scene_states[0].candidate
        assert candidate.temporal_scope is TemporalScope.unknown  # 分歧不擅自择优
        assert any("分片 temporal_scope 意见不一致" in warning for warning in candidate.warnings)

    def test_chunk_agreement_kept(self, tmp_path):
        bundle = _long_scene_bundle(tmp_path / "frozen")
        client = PerSceneClient(temporal_by_chunk=["flashback", "flashback", "flashback"])
        result = generate_scene_candidates(bundle, create_candidate_run(bundle, model_id=MODEL_ID), client)
        assert result.new_state.scene_states[0].candidate.temporal_scope is TemporalScope.flashback

    def test_single_chunk_failure_fails_scene(self, tmp_path):
        bundle = _long_scene_bundle(tmp_path / "frozen")
        # 第 1 片成功，第 2 片 3 次全部失败 → 整场景失败
        client = ScriptedClient([_candidate_json("scene_long", 1), "坏 JSON", "坏 JSON", "坏 JSON"])
        result = generate_scene_candidates(
            bundle, create_candidate_run(bundle, model_id=MODEL_ID), client, max_attempts=3
        )
        assert result.failed_scene_ids == ["scene_long"]
        item = result.new_state.scene_states[0]
        assert item.status.value == "failed"
        assert item.candidate is None  # 无完整归并不产出部分候选
        assert item.attempts == 4


# ---------- 候选合并进 P4A 审核文档 ----------


class TestMergeCandidatesIntoReview:
    def _bundle_state_with_candidates(self, tmp_path):
        bundle = _load_bundle(tmp_path / "frozen")
        state = create_candidate_run(bundle, model_id=MODEL_ID)
        result = generate_scene_candidates(bundle, state, PerSceneClient())
        return bundle, result.new_state

    def test_merge_fills_empty_fields_and_caps_at_needs_review(self, tmp_path):
        bundle, run_state = self._bundle_state_with_candidates(tmp_path)
        review = create_scene_metadata_review(bundle, reviewer="meta-auditor")
        report = merge_candidates_into_review(bundle, review, run_state)
        assert report.merged_scene_ids == ["scene_a1", "scene_a2", "scene_b1"]
        assert report.skipped_conflict == {}
        assert report.review_doc.review_status == "draft"  # 顶层绝不自动 approved
        for decision in report.review_doc.scene_decisions:
            assert decision.review_status is ReviewStatus.needs_review  # 最多 needs_review
            assert decision.viewpoint == "妃第一人称"
            assert decision.temporal_scope is TemporalScope.current
            assert decision.reality_status is RealityStatus.objective
            assert decision.mentioned_characters == []
            assert decision.present_characters == ["妃"]
            assert decision.evidence and decision.reasons
            assert decision.reviewer == "meta-auditor"  # reviewer 不被候选改写
        assert validate_scene_metadata_review(report.review_doc, bundle) == []

    def test_merge_is_idempotent(self, tmp_path):
        bundle, run_state = self._bundle_state_with_candidates(tmp_path)
        review = create_scene_metadata_review(bundle, reviewer="meta-auditor")
        first = merge_candidates_into_review(bundle, review, run_state)
        second = merge_candidates_into_review(bundle, first.review_doc, run_state)
        assert second.review_doc.model_dump(mode="json") == first.review_doc.model_dump(mode="json")

    def test_human_fields_not_overwritten_by_default(self, tmp_path):
        bundle, run_state = self._bundle_state_with_candidates(tmp_path)
        decisions = [
            _decision(bundle.scenes[0], viewpoint="第三人称"),  # 人工已填，与候选冲突
            _decision(bundle.scenes[1]),
            _decision(bundle.scenes[2]),
        ]
        review = _review_doc_with_decisions(bundle, decisions)
        report = merge_candidates_into_review(bundle, review, run_state)
        assert report.skipped_conflict == {"scene_a1": ["viewpoint"]}
        assert "scene_a1" not in report.merged_scene_ids
        decision = report.review_doc.scene_decisions[0]
        assert decision.viewpoint == "第三人称"  # 人工字段原样保留
        assert decision.review_status is ReviewStatus.draft  # 跳过场景状态不变
        assert decision.temporal_scope is None

    def test_human_confirmed_empty_character_array_is_not_treated_as_unreviewed(self, tmp_path):
        bundle, run_state = self._bundle_state_with_candidates(tmp_path)
        decisions = [
            _decision(bundle.scenes[0], present_characters=[]),
            _decision(bundle.scenes[1]),
            _decision(bundle.scenes[2]),
        ]
        review = _review_doc_with_decisions(bundle, decisions)
        report = merge_candidates_into_review(bundle, review, run_state)
        assert report.skipped_conflict["scene_a1"] == ["present_characters"]
        assert report.review_doc.scene_decisions[0].present_characters == []
        assert report.review_doc.scene_decisions[0].review_status is ReviewStatus.draft

    def test_existing_human_warning_is_preserved_when_candidate_merges(self, tmp_path):
        bundle, run_state = self._bundle_state_with_candidates(tmp_path)
        decisions = [
            _decision(bundle.scenes[0], warnings=["人工备注"]),
            _decision(bundle.scenes[1]),
            _decision(bundle.scenes[2]),
        ]
        review = _review_doc_with_decisions(bundle, decisions)
        report = merge_candidates_into_review(bundle, review, run_state)
        assert "scene_a1" in report.merged_scene_ids
        assert report.review_doc.scene_decisions[0].warnings == ["人工备注"]

    def test_overwrite_policy_replaces_with_audit(self, tmp_path):
        bundle, run_state = self._bundle_state_with_candidates(tmp_path)
        decisions = [
            _decision(bundle.scenes[0], viewpoint="第三人称"),
            _decision(bundle.scenes[1]),
            _decision(bundle.scenes[2]),
        ]
        review = _review_doc_with_decisions(bundle, decisions)
        report = merge_candidates_into_review(bundle, review, run_state, on_conflict="overwrite")
        assert report.overwritten_fields == {"scene_a1": ["viewpoint"]}
        decision = report.review_doc.scene_decisions[0]
        assert decision.viewpoint == "妃第一人称"  # 显式覆盖生效
        assert decision.review_status is ReviewStatus.needs_review  # 仍最多 needs_review
        assert any("P4B 候选显式覆盖人工字段" in warning for warning in decision.warnings)

    def test_approved_and_rejected_decisions_skipped(self, tmp_path):
        bundle, run_state = self._bundle_state_with_candidates(tmp_path)
        decisions = [
            _approved_decision(bundle.scenes[0]),
            _decision(bundle.scenes[1], review_status=ReviewStatus.rejected),
            _decision(bundle.scenes[2]),
        ]
        review = _review_doc_with_decisions(bundle, decisions)
        report = merge_candidates_into_review(bundle, review, run_state)
        assert report.skipped_final_scene_ids == ["scene_a1", "scene_b1".replace("scene_b1", "scene_a2")]
        assert report.merged_scene_ids == ["scene_b1"]
        assert report.review_doc.scene_decisions[0].review_status is ReviewStatus.approved
        assert report.review_doc.scene_decisions[1].review_status is ReviewStatus.rejected

    def test_merge_rejects_approved_top_level_doc(self, tmp_path):
        bundle, run_state = self._bundle_state_with_candidates(tmp_path)
        decisions = [_approved_decision(scene) for scene in bundle.scenes]
        review = _review_doc_with_decisions(bundle, decisions)
        review = review.model_copy(update={"review_status": "approved"})
        with pytest.raises(ValueError, match="不接受候选合并"):
            merge_candidates_into_review(bundle, review, run_state)

    def test_merge_rejects_v1_review_doc_without_migration(self, tmp_path):
        bundle, run_state = self._bundle_state_with_candidates(tmp_path)
        review = create_scene_metadata_review(bundle, reviewer="meta-auditor")
        raw = review.model_dump(mode="json")
        raw["schema_version"] = 1
        with pytest.raises(ValueError, match="schema_version 必须为"):
            merge_candidates_into_review(bundle, raw, run_state)
        # v1 形状（缺 v2 摘要字段）同样被结构校验拒绝，不静默迁移
        v1_shape = review.model_dump(mode="json")
        v1_shape["source_manifest"].pop("scenes_sha256")
        v1_shape["source_manifest"].pop("bundle_sha256")
        with pytest.raises(ValueError, match="未通过校验"):
            merge_candidates_into_review(bundle, v1_shape, run_state)

    def test_merge_rejects_state_bound_to_other_bundle(self, tmp_path):
        bundle, run_state = self._bundle_state_with_candidates(tmp_path / "a")
        scenes = _default_scenes()
        scenes[0] = scenes[0].model_copy(update={"text": "另一份正文"})
        other_bundle = _load_bundle(tmp_path / "b", scenes=scenes)
        review = create_scene_metadata_review(other_bundle, reviewer="meta-auditor")
        with pytest.raises(ValueError, match="候选运行状态未通过校验"):
            merge_candidates_into_review(other_bundle, review, run_state)

    def test_merge_leaves_scenes_without_candidate_untouched(self, tmp_path):
        bundle = _load_bundle(tmp_path / "frozen")
        state = create_candidate_run(bundle, model_id=MODEL_ID)
        result = generate_scene_candidates(bundle, state, PerSceneClient(), scene_ids=["scene_a1", "scene_a2"])
        review = create_scene_metadata_review(bundle, reviewer="meta-auditor")
        report = merge_candidates_into_review(bundle, review, result.new_state)
        assert report.no_candidate_scene_ids == ["scene_b1"]
        decision = report.review_doc.scene_decisions[2]
        assert decision.review_status is ReviewStatus.draft
        assert decision.viewpoint is None and decision.evidence is None

    def test_merge_invalid_policy_rejected(self, tmp_path):
        bundle, run_state = self._bundle_state_with_candidates(tmp_path)
        review = create_scene_metadata_review(bundle, reviewer="meta-auditor")
        with pytest.raises(ValueError, match="on_conflict 必须为"):
            merge_candidates_into_review(bundle, review, run_state, on_conflict="force")

    def test_merged_doc_still_usable_by_p4a_save(self, tmp_path):
        bundle, run_state = self._bundle_state_with_candidates(tmp_path)
        review = create_scene_metadata_review(bundle, reviewer="meta-auditor")
        report = merge_candidates_into_review(bundle, review, run_state)
        path = tmp_path / "merged_review.json"
        save_scene_metadata_review(path, report.review_doc)
        assert path.exists()


# ---------- 运行 manifest ----------


class TestRunManifest:
    def test_manifest_records_model_params_and_counts(self, tmp_path):
        bundle = _load_bundle(tmp_path / "frozen")
        state = create_candidate_run(bundle, model_id=MODEL_ID, generation_params={"temperature": 0.2})
        result = generate_scene_candidates(bundle, state, PerSceneClient())
        manifest = build_candidate_run_manifest(
            result.new_state, result, started_at="2026-08-26T10:00:00+00:00", completed_at="2026-08-26T10:05:00+00:00"
        )
        assert manifest.model_id == MODEL_ID
        assert manifest.generation_params["temperature"] == 0.2
        assert manifest.source_bundle.manifest_sha256 == bundle.manifest_digest
        assert manifest.source_bundle.scenes_sha256 == bundle.scenes_digest
        assert manifest.source_bundle.bundle_sha256 == bundle.bundle_digest
        assert manifest.total_scenes == 3
        assert manifest.scene_status_counts == {"pending": 0, "success": 3, "failed": 0}
        assert manifest.run_counts == {"attempted": 3, "succeeded": 3, "failed": 0, "skipped": 0}
        assert manifest.attempted_scene_ids == ["scene_a1", "scene_a2", "scene_b1"]
        assert manifest.started_at == "2026-08-26T10:00:00+00:00"

    def test_manifest_rejects_result_from_different_state(self, tmp_path):
        bundle = _load_bundle(tmp_path / "frozen")
        initial = create_candidate_run(bundle, model_id=MODEL_ID)
        result = generate_scene_candidates(bundle, initial, PerSceneClient())
        with pytest.raises(ValueError, match="run_result.new_state"):
            build_candidate_run_manifest(initial, result, started_at="t0", completed_at="t1")

    def test_write_manifest_atomic(self, tmp_path):
        bundle = _load_bundle(tmp_path / "frozen")
        result = generate_scene_candidates(bundle, create_candidate_run(bundle, model_id=MODEL_ID), PerSceneClient())
        manifest = build_candidate_run_manifest(result.new_state, result, started_at="t0", completed_at="t1")
        path = tmp_path / "run_manifest.json"
        write_candidate_run_manifest(path, manifest)
        assert json.loads(path.read_text(encoding="utf-8"))["model_id"] == MODEL_ID
        assert not list(tmp_path.glob("*.tmp"))

    def test_save_with_manifest_pair_commit(self, tmp_path):
        bundle = _load_bundle(tmp_path / "frozen")
        result = generate_scene_candidates(bundle, create_candidate_run(bundle, model_id=MODEL_ID), PerSceneClient())
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        state_path = run_dir / "candidate_run.json"
        manifest_path = run_dir / "run_manifest.json"
        manifest = save_candidate_run_with_manifest(
            state_path, manifest_path, result.new_state, result, started_at="t0", completed_at="t1"
        )
        assert state_path.exists() and manifest_path.exists()
        assert load_candidate_run(state_path).model_dump(mode="json") == result.new_state.model_dump(mode="json")
        assert json.loads(manifest_path.read_text(encoding="utf-8"))["run_counts"] == manifest.run_counts
        assert "started_at" not in state_path.read_text(encoding="utf-8")  # 时间戳只进 manifest
        assert not list(run_dir.glob("*.tmp")) and not list(run_dir.glob("*.tmp.old"))

    def test_save_with_manifest_rejects_different_dirs(self, tmp_path):
        bundle = _load_bundle(tmp_path / "frozen")
        result = generate_scene_candidates(bundle, create_candidate_run(bundle, model_id=MODEL_ID), PerSceneClient())
        with pytest.raises(ValueError, match="必须位于同一目录"):
            save_candidate_run_with_manifest(
                tmp_path / "a" / "state.json",
                tmp_path / "b" / "manifest.json",
                result.new_state,
                result,
                started_at="t0",
                completed_at="t1",
            )

    def test_save_with_manifest_rejects_same_path(self, tmp_path):
        bundle = _load_bundle(tmp_path / "frozen")
        result = generate_scene_candidates(bundle, create_candidate_run(bundle, model_id=MODEL_ID), PerSceneClient())
        path = tmp_path / "run.json"
        with pytest.raises(ValueError, match="两个不同路径"):
            save_candidate_run_with_manifest(
                path,
                path,
                result.new_state,
                result,
                started_at="t0",
                completed_at="t1",
            )
        assert not path.exists()


# ---------- 原子保存故障注入 ----------


class TestAtomicSaveFaultInjection:
    def _old_and_new_states(self, tmp_path):
        bundle = _load_bundle(tmp_path / "frozen")
        old = create_candidate_run(bundle, model_id="old-model")
        new = create_candidate_run(bundle, model_id="new-model")
        return old, new

    def test_write_failure_keeps_old_file(self, tmp_path, monkeypatch):
        old, new = self._old_and_new_states(tmp_path)
        path = tmp_path / "candidate_run.json"
        save_candidate_run(path, old)

        real_write_text = Path.write_text

        def failing_write_text(self_path, data, *args, **kwargs):
            if str(self_path).endswith(".tmp"):
                raise OSError("injected: tmp write failure")
            return real_write_text(self_path, data, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", failing_write_text)
        with pytest.raises(OSError, match="injected"):
            save_candidate_run(path, new)
        monkeypatch.undo()

        assert load_candidate_run(path).model_id == "old-model"  # 旧文件不变
        assert not list(tmp_path.glob("*.tmp"))  # tmp 清理
        assert not list(tmp_path.glob("*.tmp.old"))

    def test_replace_failure_keeps_old_file(self, tmp_path, monkeypatch):
        old, new = self._old_and_new_states(tmp_path)
        path = tmp_path / "candidate_run.json"
        save_candidate_run(path, old)

        real_replace = os.replace

        def failing_replace(src, dst, *args, **kwargs):
            if str(dst).endswith("candidate_run.json"):
                raise OSError("injected: replace failure")
            return real_replace(src, dst, *args, **kwargs)

        monkeypatch.setattr(review_module.os, "replace", failing_replace)
        with pytest.raises(OSError, match="injected"):
            save_candidate_run(path, new)
        monkeypatch.undo()

        assert load_candidate_run(path).model_id == "old-model"
        assert not list(tmp_path.glob("*.tmp"))

    def test_first_save_failure_writes_no_formal_file(self, tmp_path, monkeypatch):
        bundle = _load_bundle(tmp_path / "frozen")
        state = create_candidate_run(bundle, model_id=MODEL_ID)
        path = tmp_path / "candidate_run.json"

        real_write_text = Path.write_text

        def failing_write_text(self_path, data, *args, **kwargs):
            if str(self_path).endswith(".tmp"):
                raise OSError("injected: tmp write failure")
            return real_write_text(self_path, data, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", failing_write_text)
        with pytest.raises(OSError, match="injected"):
            save_candidate_run(path, state)
        monkeypatch.undo()

        assert not path.exists()
        assert not list(tmp_path.glob("*.tmp"))

    def test_pair_rollback_on_manifest_failure(self, tmp_path, monkeypatch):
        bundle = _load_bundle(tmp_path / "frozen")
        result = generate_scene_candidates(bundle, create_candidate_run(bundle, model_id=MODEL_ID), PerSceneClient())
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        state_path = run_dir / "candidate_run.json"
        manifest_path = run_dir / "run_manifest.json"
        save_candidate_run(state_path, result.new_state)
        old_content = state_path.read_text(encoding="utf-8")

        real_replace = os.replace

        def failing_replace(src, dst, *args, **kwargs):
            if str(dst).endswith("run_manifest.json"):
                raise OSError("injected: manifest replace failure")
            return real_replace(src, dst, *args, **kwargs)

        monkeypatch.setattr(review_module.os, "replace", failing_replace)
        with pytest.raises(ValueError, match="已回滚"):
            save_candidate_run_with_manifest(
                state_path, manifest_path, result.new_state, result, started_at="t0", completed_at="t1"
            )
        monkeypatch.undo()

        assert state_path.read_text(encoding="utf-8") == old_content  # 状态回滚，无混合版本
        assert not manifest_path.exists()
        assert not list(run_dir.glob("*.tmp")) and not list(run_dir.glob("*.tmp.old"))

    def test_pair_refuses_to_overwrite_recovery_copy(self, tmp_path):
        bundle = _load_bundle(tmp_path / "frozen")
        result = generate_scene_candidates(bundle, create_candidate_run(bundle, model_id=MODEL_ID), PerSceneClient())
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        state_path = run_dir / "candidate_run.json"
        manifest_path = run_dir / "run_manifest.json"
        backup = run_dir / "candidate_run.json.tmp.old"
        backup.write_text("RECOVERY COPY\n", encoding="utf-8")

        with pytest.raises(ValueError, match="未恢复的旧版备份"):
            save_candidate_run_with_manifest(
                state_path, manifest_path, result.new_state, result, started_at="t0", completed_at="t1"
            )
        assert backup.read_text(encoding="utf-8") == "RECOVERY COPY\n"  # 恢复副本不被覆盖
        assert not list(run_dir.glob("*.tmp"))


# ---------- 架构守卫：候选路径不可达 approved/enriched ----------


class TestArchitecturalGuards:
    def test_candidate_module_never_imports_enriched_paths(self):
        assert not hasattr(candidate_module, "apply_approved_scene_metadata")
        assert not hasattr(candidate_module, "write_enriched_scenes")

    def test_public_package_exports_parse_exception(self):
        import knowledge.game_rag as game_rag

        assert game_rag.CandidateParseError is CandidateParseError

    def test_hint_contract_is_structured_and_complete(self):
        from knowledge.game_rag.scene_metadata_candidate import DEFAULT_HINT_CONTRACT

        labels = [hint.label for hint in DEFAULT_HINT_CONTRACT]
        for required in ("梦境", "回忆", "书中故事", "宣传元叙事", "无法判断"):
            assert required in labels
        for hint in DEFAULT_HINT_CONTRACT:
            assert hint.temporal_scope_hints and hint.reality_status_hints


# ---------- 真实冻结目录：未运行模型前零调用、零写入 ----------


class TestRealFrozenDirectory:
    def test_real_bundle_creates_pending_state_in_memory_with_zero_calls_zero_writes(self):
        """正式冻结包可创建 262 条 pending 状态；未运行模型时保持零调用零写入。"""
        before = sorted(p.name for p in REVIEW_DIR.iterdir()) if REVIEW_DIR.exists() else None
        client = RecordingClient()
        bundle = load_frozen_scene_bundle(REVIEW_DIR / "scenes.jsonl", REVIEW_DIR / "boundary_manifest.json")
        state = create_candidate_run(bundle, model_id=MODEL_ID)
        assert client.calls == []
        after = sorted(p.name for p in REVIEW_DIR.iterdir()) if REVIEW_DIR.exists() else None
        assert before == after
        assert len(state.scene_states) == 262
        assert all(item.status is candidate_module.CandidateGenerationStatus.pending for item in state.scene_states)
        assert not (REVIEW_DIR / "candidate_run.json").exists()
        assert not (REVIEW_DIR / "run_manifest.json").exists()

    @needs_p3_review_files
    def test_p3_review_state_is_approved_and_reviewer_unchanged(self):
        """正式冻结授权只改变审核状态，不改变审核人。"""
        overrides = json.loads((REVIEW_DIR / "boundary_overrides.json").read_text(encoding="utf-8"))
        low = json.loads((REVIEW_DIR / "low_candidate_review.json").read_text(encoding="utf-8"))
        assert overrides["boundary_review_status"] == "approved"
        assert low["review_status"] == "approved"
        assert overrides["reviewer"] == low["reviewer"] == "project_owner_01"
