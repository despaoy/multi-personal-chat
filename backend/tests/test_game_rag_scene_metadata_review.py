"""P4A 场景元数据审核基础设施测试：合成 bundle 覆盖输入门禁、审核状态、校验、
原子保存、审核包、approved 应用与 enriched 输出门禁。

不测试 Pydantic / 标准库自身，只测试项目契约。
真实未冻结目录的拒绝路径单独覆盖（P4 不得提前消费 draft 场景）。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from knowledge.game_rag.models import (
    ContentScope,
    RealityStatus,
    ReviewStatus,
    SceneDocument,
    SourceSpan,
    StoryContext,
    TemporalScope,
)
from knowledge.game_rag.scene_metadata_review import (
    SCENE_METADATA_REVIEW_SCHEMA_VERSION,
    SceneMetadataDecision,
    SceneMetadataReviewDocument,
    apply_approved_scene_metadata,
    create_scene_metadata_review,
    generate_scene_metadata_review_pack,
    load_frozen_scene_bundle,
    load_scene_metadata_review,
    save_scene_metadata_review,
    validate_scene_metadata_review,
    write_enriched_scenes,
)

UNIT_A = "vol99_9合成场景A"
UNIT_B = "vol99_9合成场景B"
PATH_A = "gametext/纸上魔法使/synth_a.txt"
PATH_B = "gametext/纸上魔法使/synth_b.txt"

REVIEW_DIR = (
    Path(__file__).resolve().parents[1] / "data" / "knowledge" / "tsukiyashiro_kisaki" / "scene_boundary_review"
)

needs_p3_review_files = pytest.mark.skipif(
    not (REVIEW_DIR / "boundary_overrides.json").exists() or not (REVIEW_DIR / "low_candidate_review.json").exists(),
    reason="P3 审核决定文件不存在（data/knowledge 未随测试环境分发）",
)


# ---------- 合成 bundle 构造 ----------


def _scene(
    scene_id: str,
    unit_id: str,
    source_path: str,
    line_start: int,
    line_end: int,
    *,
    speakers: list[str] | None = None,
    text: str | None = None,
    review_status: ReviewStatus = ReviewStatus.draft,
) -> SceneDocument:
    span_lines = line_end - line_start + 1
    body = text if text is not None else "\n".join(f"{unit_id} 原文第{i}行" for i in range(1, span_lines + 1))
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
        review_status=review_status,
    )


def _manifest(
    units: dict[str, int],
    *,
    total: int | None = None,
    boundary_status: str = "approved",
    scene_status: str = "draft",
    schema_version: int = 1,
) -> dict:
    return {
        "schema_version": schema_version,
        "boundary_review_status": boundary_status,
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
        "total_scenes": total if total is not None else sum(units.values()),
        "scene_review_status": scene_status,
        "note": "synthetic manifest",
    }


def _default_scenes() -> list[SceneDocument]:
    return [
        _scene("scene_a1", UNIT_A, PATH_A, 1, 10, speakers=["妃", "琉璃"]),
        _scene("scene_a2", UNIT_A, PATH_A, 11, 20, speakers=["妃"]),
        _scene("scene_b1", UNIT_B, PATH_B, 1, 15, speakers=["汀"]),
    ]


def _write_bundle(
    tmp_path: Path,
    scenes: list[SceneDocument],
    manifest: dict,
    *,
    scene_text: str | None = None,
) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    scene_path = tmp_path / "scenes.jsonl"
    manifest_path = tmp_path / "boundary_manifest.json"
    payload = scene_text if scene_text is not None else "".join(s.model_dump_json() + "\n" for s in scenes)
    scene_path.write_text(payload, encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return scene_path, manifest_path


def _load_bundle(
    tmp_path: Path,
    scenes: list[SceneDocument] | None = None,
    manifest: dict | None = None,
):
    scenes = scenes if scenes is not None else _default_scenes()
    manifest = manifest if manifest is not None else _manifest({UNIT_A: 2, UNIT_B: 1})
    scene_path, manifest_path = _write_bundle(tmp_path, scenes, manifest)
    return load_frozen_scene_bundle(scene_path, manifest_path)


def _approved_doc(bundle, *, reviewer: str = "meta-auditor") -> SceneMetadataReviewDocument:
    base = create_scene_metadata_review(bundle, reviewer=reviewer)
    decisions = [
        SceneMetadataDecision(
            scene_id=scene.id,
            story_unit_id=scene.story.story_unit_id,
            source=scene.source,
            viewpoint="妃第一人称",
            temporal_scope=TemporalScope.current,
            reality_status=RealityStatus.objective,
            mentioned_characters=["琉璃"],
            present_characters=["妃", "琉璃"],
            evidence=[
                SourceSpan(
                    source_path=scene.source.source_path,
                    line_start=scene.source.line_start,
                    line_end=scene.source.line_start,
                )
            ],
            reasons=["场景开头明确视角与现实层"],
            review_status=ReviewStatus.approved,
            reviewer=reviewer,
        )
        for scene in bundle.scenes
    ]
    return SceneMetadataReviewDocument(
        schema_version=base.schema_version,
        source_manifest=base.source_manifest,
        total_source_scenes=base.total_source_scenes,
        reviewer=reviewer,
        review_status="approved",
        scene_decisions=decisions,
        notes=base.notes,
        created_by=base.created_by,
    )


# ---------- 冻结输入门禁 ----------


class TestLoadFrozenSceneBundle:
    def test_valid_bundle_loads(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        assert bundle.total_scenes == 3
        assert [scene.id for scene in bundle.scenes] == ["scene_a1", "scene_a2", "scene_b1"]
        assert bundle.manifest.boundary_review_status == "approved"
        assert len(bundle.manifest_digest) == 64

    def test_blank_lines_in_scenes_jsonl_tolerated(self, tmp_path):
        scenes = _default_scenes()
        manifest = _manifest({UNIT_A: 2, UNIT_B: 1})
        scene_path, manifest_path = _write_bundle(tmp_path, scenes, manifest)
        scene_path.write_text(
            scenes[0].model_dump_json()
            + "\n\n"
            + scenes[1].model_dump_json()
            + "\n \n"
            + scenes[2].model_dump_json()
            + "\n",
            encoding="utf-8",
        )
        bundle = load_frozen_scene_bundle(scene_path, manifest_path)
        assert bundle.total_scenes == 3

    def test_missing_scenes_rejected(self, tmp_path):
        scenes = _default_scenes()
        manifest = _manifest({UNIT_A: 2, UNIT_B: 1})
        scene_path, manifest_path = _write_bundle(tmp_path, scenes, manifest)
        scene_path.unlink()
        with pytest.raises(ValueError, match="scenes.jsonl 不存在"):
            load_frozen_scene_bundle(scene_path, manifest_path)

    def test_missing_manifest_rejected(self, tmp_path):
        scenes = _default_scenes()
        manifest = _manifest({UNIT_A: 2, UNIT_B: 1})
        scene_path, manifest_path = _write_bundle(tmp_path, scenes, manifest)
        manifest_path.unlink()
        with pytest.raises(ValueError, match="boundary_manifest.json 不存在"):
            load_frozen_scene_bundle(scene_path, manifest_path)

    def test_invalid_manifest_json_rejected(self, tmp_path):
        scenes = _default_scenes()
        manifest = _manifest({UNIT_A: 2, UNIT_B: 1})
        scene_path, manifest_path = _write_bundle(tmp_path, scenes, manifest)
        manifest_path.write_text("{ not valid json", encoding="utf-8")
        with pytest.raises(ValueError, match="不是合法 JSON"):
            load_frozen_scene_bundle(scene_path, manifest_path)

    def test_unsupported_schema_version_rejected(self, tmp_path):
        scenes = _default_scenes()
        manifest = _manifest({UNIT_A: 2, UNIT_B: 1}, schema_version=99)
        scene_path, manifest_path = _write_bundle(tmp_path, scenes, manifest)
        with pytest.raises(ValueError, match="schema_version 不受支持"):
            load_frozen_scene_bundle(scene_path, manifest_path)

    def test_boundary_not_approved_rejected(self, tmp_path):
        scenes = _default_scenes()
        manifest = _manifest({UNIT_A: 2, UNIT_B: 1}, boundary_status="draft")
        scene_path, manifest_path = _write_bundle(tmp_path, scenes, manifest)
        with pytest.raises(ValueError, match="boundary_review_status 必须为 approved"):
            load_frozen_scene_bundle(scene_path, manifest_path)

    def test_scene_review_status_not_draft_rejected(self, tmp_path):
        scenes = _default_scenes()
        manifest = _manifest({UNIT_A: 2, UNIT_B: 1}, scene_status="approved")
        scene_path, manifest_path = _write_bundle(tmp_path, scenes, manifest)
        with pytest.raises(ValueError, match="scene_review_status 必须为 draft"):
            load_frozen_scene_bundle(scene_path, manifest_path)

    def test_total_scenes_mismatch_rejected(self, tmp_path):
        scenes = _default_scenes()
        manifest = _manifest({UNIT_A: 2, UNIT_B: 1}, total=2)
        scene_path, manifest_path = _write_bundle(tmp_path, scenes, manifest)
        with pytest.raises(ValueError, match="与 manifest.total_scenes"):
            load_frozen_scene_bundle(scene_path, manifest_path)

    def test_unit_scene_count_mismatch_rejected(self, tmp_path):
        scenes = _default_scenes()
        # total 保持与实际场景数一致（3），只让单元级计数不一致，命中单元级校验
        manifest = _manifest({UNIT_A: 3, UNIT_B: 0})
        scene_path, manifest_path = _write_bundle(tmp_path, scenes, manifest)
        with pytest.raises(ValueError, match="场景数 2 与 manifest 记录的 3 不一致"):
            load_frozen_scene_bundle(scene_path, manifest_path)

    def test_duplicate_scene_id_rejected(self, tmp_path):
        scenes = _default_scenes()
        scenes[1] = _scene("scene_a1", UNIT_A, PATH_A, 11, 20)
        manifest = _manifest({UNIT_A: 2, UNIT_B: 1})
        scene_path, manifest_path = _write_bundle(tmp_path, scenes, manifest)
        with pytest.raises(ValueError, match="重复 scene id"):
            load_frozen_scene_bundle(scene_path, manifest_path)

    def test_span_overlap_rejected(self, tmp_path):
        scenes = _default_scenes()
        scenes[1] = _scene("scene_a2", UNIT_A, PATH_A, 5, 20)  # 与 L1-10 重叠
        manifest = _manifest({UNIT_A: 2, UNIT_B: 1})
        scene_path, manifest_path = _write_bundle(tmp_path, scenes, manifest)
        with pytest.raises(ValueError, match="无序或重叠"):
            load_frozen_scene_bundle(scene_path, manifest_path)

    def test_dos_eof_marker_rejected(self, tmp_path):
        scenes = _default_scenes()
        scenes[0] = _scene("scene_a1", UNIT_A, PATH_A, 1, 10, text="正常一行\n\x1a\n其余行\n")
        manifest = _manifest({UNIT_A: 2, UNIT_B: 1})
        scene_path, manifest_path = _write_bundle(tmp_path, scenes, manifest)
        with pytest.raises(ValueError, match="\\\\x1a"):
            load_frozen_scene_bundle(scene_path, manifest_path)

    def test_non_draft_input_scene_rejected(self, tmp_path):
        scenes = _default_scenes()
        scenes[0] = _scene("scene_a1", UNIT_A, PATH_A, 1, 10, review_status=ReviewStatus.approved)
        manifest = _manifest({UNIT_A: 2, UNIT_B: 1})
        scene_path, manifest_path = _write_bundle(tmp_path, scenes, manifest)
        with pytest.raises(ValueError, match="review_status 必须为 draft"):
            load_frozen_scene_bundle(scene_path, manifest_path)

    def test_unknown_story_unit_rejected(self, tmp_path):
        scenes = _default_scenes()
        scenes[2] = _scene("scene_b1", "vol99_9未登记单元", PATH_B, 1, 15)
        manifest = _manifest({UNIT_A: 2, UNIT_B: 1})
        scene_path, manifest_path = _write_bundle(tmp_path, scenes, manifest)
        with pytest.raises(ValueError, match="未在 manifest.units 中登记"):
            load_frozen_scene_bundle(scene_path, manifest_path)

    def test_unit_with_multiple_source_paths_rejected(self, tmp_path):
        scenes = _default_scenes()
        scenes[1] = _scene("scene_a2", UNIT_A, PATH_B, 11, 20)  # 同单元换文件
        manifest = _manifest({UNIT_A: 2, UNIT_B: 1})
        scene_path, manifest_path = _write_bundle(tmp_path, scenes, manifest)
        with pytest.raises(ValueError, match="多个 source_path"):
            load_frozen_scene_bundle(scene_path, manifest_path)

    def test_invalid_scene_line_rejected(self, tmp_path):
        manifest = _manifest({UNIT_A: 2, UNIT_B: 1})
        scene_path, manifest_path = _write_bundle(tmp_path, _default_scenes(), manifest)
        scene_path.write_text('{"id": "broken"}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="不是合法 SceneDocument"):
            load_frozen_scene_bundle(scene_path, manifest_path)

    def test_manifest_extra_field_rejected(self, tmp_path):
        manifest = _manifest({UNIT_A: 2, UNIT_B: 1})
        manifest["unknown_field"] = 1
        scene_path, manifest_path = _write_bundle(tmp_path, _default_scenes(), manifest)
        with pytest.raises(ValueError, match="结构校验失败"):
            load_frozen_scene_bundle(scene_path, manifest_path)

    def test_manifest_boundaries_must_match_scene_starts(self, tmp_path):
        manifest = _manifest({UNIT_A: 2, UNIT_B: 1})
        manifest["units"][UNIT_A]["boundaries"] = [12]
        scene_path, manifest_path = _write_bundle(tmp_path, _default_scenes(), manifest)
        with pytest.raises(ValueError, match="与 manifest.boundaries"):
            load_frozen_scene_bundle(scene_path, manifest_path)

    def test_manifest_story_title_must_match_scenes(self, tmp_path):
        manifest = _manifest({UNIT_A: 2, UNIT_B: 1})
        manifest["units"][UNIT_A]["story_title"] = "错误标题"
        scene_path, manifest_path = _write_bundle(tmp_path, _default_scenes(), manifest)
        with pytest.raises(ValueError, match="story_title"):
            load_frozen_scene_bundle(scene_path, manifest_path)

    def test_source_path_must_be_under_manifest_prefix(self, tmp_path):
        manifest = _manifest({UNIT_A: 2, UNIT_B: 1})
        manifest["source_prefix"] = "other/prefix"
        scene_path, manifest_path = _write_bundle(tmp_path, _default_scenes(), manifest)
        with pytest.raises(ValueError, match="source_prefix"):
            load_frozen_scene_bundle(scene_path, manifest_path)


# ---------- 初始审核状态 ----------


class TestCreateSceneMetadataReview:
    def test_covers_all_scenes_in_frozen_order(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        doc = create_scene_metadata_review(bundle, reviewer="meta-auditor")
        assert [d.scene_id for d in doc.scene_decisions] == [s.id for s in bundle.scenes]
        assert doc.total_source_scenes == 3
        assert doc.review_status == "draft"

    def test_pending_fields_all_none(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        doc = create_scene_metadata_review(bundle)
        for decision in doc.scene_decisions:
            assert decision.viewpoint is None
            assert decision.temporal_scope is None
            assert decision.reality_status is None
            assert decision.mentioned_characters is None  # None ≠ 空数组（未审核 ≠ 确认无）
            assert decision.present_characters is None
            assert decision.evidence is None
            assert decision.reasons == []
            assert decision.warnings == []
            assert decision.review_status is ReviewStatus.draft

    def test_speakers_not_auto_filled_into_present(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        doc = create_scene_metadata_review(bundle)
        for decision in doc.scene_decisions:
            assert decision.present_characters is None
            assert decision.mentioned_characters is None

    def test_input_bundle_unchanged(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        before = [scene.model_dump(mode="json") for scene in bundle.scenes]
        create_scene_metadata_review(bundle, reviewer="meta-auditor")
        after = [scene.model_dump(mode="json") for scene in bundle.scenes]
        assert before == after

    def test_deterministic_across_runs(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        first = create_scene_metadata_review(bundle, reviewer="meta-auditor")
        second = create_scene_metadata_review(bundle, reviewer="meta-auditor")
        assert first.model_dump(mode="json") == second.model_dump(mode="json")

    def test_reviewer_inherited_by_records_and_allowed_empty(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        doc = create_scene_metadata_review(bundle, reviewer="meta-auditor")
        assert doc.reviewer == "meta-auditor"
        assert all(d.reviewer == "meta-auditor" for d in doc.scene_decisions)
        empty = create_scene_metadata_review(bundle)
        assert empty.reviewer == ""
        assert all(d.reviewer == "" for d in empty.scene_decisions)

    def test_schema_version_and_manifest_ref(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        doc = create_scene_metadata_review(bundle)
        assert doc.schema_version == SCENE_METADATA_REVIEW_SCHEMA_VERSION
        assert doc.source_manifest.manifest_sha256 == bundle.manifest_digest
        assert doc.source_manifest.scenes_sha256 == bundle.scenes_digest
        assert doc.source_manifest.bundle_sha256 == bundle.bundle_digest
        assert doc.source_manifest.total_scenes == bundle.total_scenes
        assert doc.source_manifest.reviewer == bundle.manifest.reviewer
        assert doc.created_by

    def test_rejects_bundle_scene_mutated_after_load(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        bundle.scenes[0].text = "加载后被原地篡改的正文"
        with pytest.raises(ValueError, match="scenes 在加载后被修改"):
            create_scene_metadata_review(bundle)


# ---------- 审核状态校验 ----------


class TestValidateSceneMetadataReview:
    def test_valid_draft_passes(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        doc = create_scene_metadata_review(bundle, reviewer="meta-auditor")
        assert validate_scene_metadata_review(doc, bundle) == []

    def test_rejects_bundle_manifest_mutated_after_review_creation(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        doc = create_scene_metadata_review(bundle, reviewer="meta-auditor")
        bundle.manifest.note = "加载后被修改"
        errors = validate_scene_metadata_review(doc, bundle)
        assert any("manifest 在加载后被修改" in error for error in errors)

    def test_missing_scene_rejected(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        doc = create_scene_metadata_review(bundle, reviewer="meta-auditor")
        raw = doc.model_dump(mode="json")
        raw["scene_decisions"].pop()
        raw["total_source_scenes"] = 2  # 数量自洽但集合不完整
        errors = validate_scene_metadata_review(raw, bundle)
        assert any("缺少 scene" in e for e in errors)

    def test_unknown_scene_rejected(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        doc = create_scene_metadata_review(bundle, reviewer="meta-auditor")
        raw = doc.model_dump(mode="json")
        raw["scene_decisions"][1]["scene_id"] = "scene_unknown"
        errors = validate_scene_metadata_review(raw, bundle)
        assert any("未知 scene" in e for e in errors)
        assert any("缺少 scene" in e for e in errors)

    def test_duplicate_scene_rejected(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        doc = create_scene_metadata_review(bundle, reviewer="meta-auditor")
        raw = doc.model_dump(mode="json")
        raw["scene_decisions"][1]["scene_id"] = raw["scene_decisions"][0]["scene_id"]
        errors = validate_scene_metadata_review(raw, bundle)
        assert any("重复 scene_id" in e for e in errors)

    def test_source_tampered_rejected(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        doc = create_scene_metadata_review(bundle, reviewer="meta-auditor")
        raw = doc.model_dump(mode="json")
        raw["scene_decisions"][0]["source"]["line_start"] = 2
        errors = validate_scene_metadata_review(raw, bundle)
        assert any("source 被篡改" in e for e in errors)

    def test_story_unit_tampered_rejected(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        doc = create_scene_metadata_review(bundle, reviewer="meta-auditor")
        raw = doc.model_dump(mode="json")
        raw["scene_decisions"][0]["story_unit_id"] = UNIT_B
        errors = validate_scene_metadata_review(raw, bundle)
        assert any("story_unit_id 被篡改" in e for e in errors)

    def test_manifest_digest_mismatch_rejected(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        # digest 钉住的是 manifest 内容：换一份内容不同的 manifest（note 不同），
        # 场景集合本身可以相同
        manifest = _manifest({UNIT_A: 2, UNIT_B: 1})
        manifest["note"] = "synthetic manifest v2"
        scene_path, manifest_path = _write_bundle(tmp_path / "other2", _default_scenes(), manifest)
        different = load_frozen_scene_bundle(scene_path, manifest_path)
        assert different.manifest_digest != bundle.manifest_digest

        doc = create_scene_metadata_review(bundle, reviewer="meta-auditor")
        errors = validate_scene_metadata_review(doc, different)
        assert any("manifest_sha256 与 bundle 不一致" in e for e in errors)

    def test_scenes_digest_mismatch_rejected_even_when_manifest_is_unchanged(self, tmp_path):
        bundle = _load_bundle(tmp_path / "original")
        changed_scenes = _default_scenes()
        changed_scenes[0] = changed_scenes[0].model_copy(update={"text": "被替换但结构仍合法的正文"})
        different = _load_bundle(
            tmp_path / "changed",
            scenes=changed_scenes,
            manifest=_manifest({UNIT_A: 2, UNIT_B: 1}),
        )
        assert different.manifest_digest == bundle.manifest_digest
        assert different.scenes_digest != bundle.scenes_digest
        assert different.bundle_digest != bundle.bundle_digest

        doc = create_scene_metadata_review(bundle, reviewer="meta-auditor")
        errors = validate_scene_metadata_review(doc, different)
        assert any("scenes_sha256 与 bundle 不一致" in error for error in errors)
        assert any("bundle_sha256 与 bundle 不一致" in error for error in errors)

    def test_invalid_temporal_scope_rejected(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        doc = create_scene_metadata_review(bundle, reviewer="meta-auditor")
        raw = doc.model_dump(mode="json")
        raw["scene_decisions"][0]["temporal_scope"] = "bogus"
        errors = validate_scene_metadata_review(raw, bundle)
        assert errors and "结构非法" in errors[0]

    def test_invalid_reality_status_rejected(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        doc = create_scene_metadata_review(bundle, reviewer="meta-auditor")
        raw = doc.model_dump(mode="json")
        raw["scene_decisions"][0]["reality_status"] = "maybe"
        errors = validate_scene_metadata_review(raw, bundle)
        assert errors and "结构非法" in errors[0]

    def test_invalid_viewpoint_rejected(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        doc = create_scene_metadata_review(bundle, reviewer="meta-auditor")
        raw = doc.model_dump(mode="json")
        raw["scene_decisions"][0]["viewpoint"] = "随便写的视角"
        errors = validate_scene_metadata_review(raw, bundle)
        assert errors and "结构非法" in errors[0]

    def test_valid_viewpoint_forms_accepted(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        scene = bundle.scenes[0]
        for value in ("琉璃第一人称", "克丽索贝莉露第一人称", "第三人称", "多视角", "unknown"):
            SceneMetadataDecision(
                scene_id=scene.id,
                story_unit_id=scene.story.story_unit_id,
                source=scene.source,
                viewpoint=value,
            )

    def test_empty_character_name_rejected(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        doc = create_scene_metadata_review(bundle, reviewer="meta-auditor")
        raw = doc.model_dump(mode="json")
        raw["scene_decisions"][0]["mentioned_characters"] = ["妃", "   "]
        errors = validate_scene_metadata_review(raw, bundle)
        assert errors and "结构非法" in errors[0]

    def test_duplicate_characters_normalized(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        scene = bundle.scenes[0]
        decision = SceneMetadataDecision(
            scene_id=scene.id,
            story_unit_id=scene.story.story_unit_id,
            source=scene.source,
            mentioned_characters=["琉璃", " 妃 ", "妃"],
            present_characters=["妃"],
        )
        assert decision.mentioned_characters == ["妃", "琉璃"]

    def test_evidence_out_of_span_rejected(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        scene = bundle.scenes[0]  # L1-10
        with pytest.raises(ValueError, match="超出场景范围"):
            SceneMetadataDecision(
                scene_id=scene.id,
                story_unit_id=scene.story.story_unit_id,
                source=scene.source,
                evidence=[SourceSpan(source_path=scene.source.source_path, line_start=9, line_end=12)],
            )

    def test_evidence_wrong_path_rejected(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        scene = bundle.scenes[0]
        with pytest.raises(ValueError, match="与场景 source"):
            SceneMetadataDecision(
                scene_id=scene.id,
                story_unit_id=scene.story.story_unit_id,
                source=scene.source,
                evidence=[SourceSpan(source_path=PATH_B, line_start=1, line_end=2)],
            )

    def test_approved_with_none_field_rejected(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        scene = bundle.scenes[0]
        raw = {
            "scene_id": scene.id,
            "story_unit_id": scene.story.story_unit_id,
            "source": scene.source.model_dump(mode="json"),
            "viewpoint": None,  # approved 却保留未审核字段
            "temporal_scope": "current",
            "reality_status": "objective",
            "mentioned_characters": [],
            "present_characters": ["妃"],
            "evidence": [
                {"source_path": scene.source.source_path, "line_start": 1, "line_end": 2},
            ],
            "reasons": ["理由"],
            "review_status": "approved",
            "reviewer": "meta-auditor",
        }
        with pytest.raises(ValueError, match="不得保留未审核字段"):
            SceneMetadataDecision.model_validate(raw)

    def test_approved_record_reviewer_empty_rejected(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        scene = bundle.scenes[0]
        raw = {
            "scene_id": scene.id,
            "story_unit_id": scene.story.story_unit_id,
            "source": scene.source.model_dump(mode="json"),
            "viewpoint": "妃第一人称",
            "temporal_scope": "current",
            "reality_status": "objective",
            "mentioned_characters": [],
            "present_characters": ["妃"],
            "evidence": [{"source_path": scene.source.source_path, "line_start": 1, "line_end": 2}],
            "reasons": ["理由"],
            "review_status": "approved",
            "reviewer": "",
        }
        with pytest.raises(ValueError, match="reviewer 不得为空"):
            SceneMetadataDecision.model_validate(raw)

    def test_top_approved_with_draft_scene_rejected(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        doc = _approved_doc(bundle)
        raw = doc.model_dump(mode="json")
        raw["scene_decisions"][0]["review_status"] = "draft"
        raw["scene_decisions"][0]["viewpoint"] = None
        raw["scene_decisions"][0]["temporal_scope"] = None
        raw["scene_decisions"][0]["reality_status"] = None
        raw["scene_decisions"][0]["mentioned_characters"] = None
        raw["scene_decisions"][0]["present_characters"] = None
        raw["scene_decisions"][0]["evidence"] = None
        raw["scene_decisions"][0]["reasons"] = []
        errors = validate_scene_metadata_review(raw, bundle)
        assert any("不得存在 draft 场景" in e for e in errors)

    def test_top_approved_with_empty_reviewer_rejected(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        # 记录级 reviewer 合法，但顶层 reviewer 为空：由校验器拒绝
        doc = _approved_doc(bundle)
        raw = doc.model_dump(mode="json")
        raw["reviewer"] = ""
        errors = validate_scene_metadata_review(raw, bundle)
        assert any("reviewer 不得为空" in e for e in errors)

    def test_approved_doc_passes_validation(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        doc = _approved_doc(bundle)
        assert validate_scene_metadata_review(doc, bundle) == []

    def test_require_complete_flags_draft(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        doc = create_scene_metadata_review(bundle, reviewer="meta-auditor")
        assert not any("require_complete" in e for e in validate_scene_metadata_review(doc, bundle))
        errors = validate_scene_metadata_review(doc, bundle, require_complete=True)
        assert len(errors) == 3
        assert all("仍为 draft" in e for e in errors)

    def test_require_complete_accepts_needs_review(self, tmp_path):
        """needs_review 属于「已明确审核」，通过完整性检查但不构成 approved。"""
        bundle = _load_bundle(tmp_path)
        doc = create_scene_metadata_review(bundle, reviewer="meta-auditor")
        raw = doc.model_dump(mode="json")
        for decision in raw["scene_decisions"]:
            decision["review_status"] = "needs_review"
        errors = validate_scene_metadata_review(raw, bundle, require_complete=True)
        assert errors == []


# ---------- approved 应用流程 ----------


class TestApplyApprovedSceneMetadata:
    def test_draft_doc_rejected(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        doc = create_scene_metadata_review(bundle, reviewer="meta-auditor")
        with pytest.raises(ValueError, match="必须为 approved"):
            apply_approved_scene_metadata(bundle, doc)

    def test_needs_review_record_rejected(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        doc = _approved_doc(bundle)
        raw = doc.model_dump(mode="json")
        raw["scene_decisions"][1]["review_status"] = "needs_review"
        raw["scene_decisions"][1]["viewpoint"] = None
        raw["scene_decisions"][1]["temporal_scope"] = None
        raw["scene_decisions"][1]["reality_status"] = None
        raw["scene_decisions"][1]["mentioned_characters"] = None
        raw["scene_decisions"][1]["present_characters"] = None
        raw["scene_decisions"][1]["evidence"] = None
        raw["scene_decisions"][1]["reasons"] = []
        with pytest.raises(ValueError, match="不得存在 needs_review"):
            apply_approved_scene_metadata(bundle, raw)

    def test_rejected_record_rejected(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        doc = _approved_doc(bundle)
        raw = doc.model_dump(mode="json")
        raw["scene_decisions"][2]["review_status"] = "rejected"
        raw["scene_decisions"][2]["viewpoint"] = None
        raw["scene_decisions"][2]["temporal_scope"] = None
        raw["scene_decisions"][2]["reality_status"] = None
        raw["scene_decisions"][2]["mentioned_characters"] = None
        raw["scene_decisions"][2]["present_characters"] = None
        raw["scene_decisions"][2]["evidence"] = None
        raw["scene_decisions"][2]["reasons"] = []
        with pytest.raises(ValueError, match="不得存在 rejected"):
            apply_approved_scene_metadata(bundle, raw)

    def test_all_approved_success_and_metadata_written(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        doc = _approved_doc(bundle)
        enriched = apply_approved_scene_metadata(bundle, doc)
        assert len(enriched) == 3
        for scene, decision in zip(enriched, doc.scene_decisions):
            assert scene.review_status is ReviewStatus.approved
            assert scene.story.viewpoint == decision.viewpoint
            assert scene.story.temporal_scope is decision.temporal_scope
            assert scene.reality_status is decision.reality_status
            assert scene.mentioned_characters == decision.mentioned_characters
            assert scene.present_characters == decision.present_characters

    def test_output_order_and_conservation(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        doc = _approved_doc(bundle)
        enriched = apply_approved_scene_metadata(bundle, doc)
        assert [s.id for s in enriched] == [s.id for s in bundle.scenes]
        for original, applied in zip(bundle.scenes, enriched):
            assert applied.text == original.text
            assert applied.source == original.source
            assert applied.speakers == original.speakers
            assert applied.story.story_unit_id == original.story.story_unit_id
            assert applied.id == original.id

    def test_input_bundle_unchanged(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        before = [scene.model_dump(mode="json") for scene in bundle.scenes]
        doc = _approved_doc(bundle)
        apply_approved_scene_metadata(bundle, doc)
        after = [scene.model_dump(mode="json") for scene in bundle.scenes]
        assert before == after
        assert all(scene.review_status is ReviewStatus.draft for scene in bundle.scenes)

    def test_deterministic(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        doc = _approved_doc(bundle)
        first = apply_approved_scene_metadata(bundle, doc)
        second = apply_approved_scene_metadata(bundle, doc)
        assert [s.model_dump(mode="json") for s in first] == [s.model_dump(mode="json") for s in second]


# ---------- 原子保存 ----------


class TestAtomicSave:
    def test_roundtrip(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        doc = create_scene_metadata_review(bundle, reviewer="meta-auditor")
        path = tmp_path / "scene_metadata_review.json"
        save_scene_metadata_review(path, doc)
        loaded = load_scene_metadata_review(path)
        assert loaded.model_dump(mode="json") == doc.model_dump(mode="json")
        assert validate_scene_metadata_review(loaded, bundle) == []

    def test_roundtrip_via_dict(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        doc = create_scene_metadata_review(bundle, reviewer="meta-auditor")
        path = tmp_path / "scene_metadata_review.json"
        save_scene_metadata_review(path, doc.model_dump(mode="json"))
        loaded = load_scene_metadata_review(path)
        assert loaded.model_dump(mode="json") == doc.model_dump(mode="json")

    def test_no_tmp_after_success(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        doc = create_scene_metadata_review(bundle)
        path = tmp_path / "scene_metadata_review.json"
        save_scene_metadata_review(path, doc)
        assert path.exists()
        assert not list(tmp_path.glob("*.tmp"))

    def test_save_rejects_invalid_document(self, tmp_path):
        path = tmp_path / "scene_metadata_review.json"
        with pytest.raises(ValueError, match="拒绝写出"):
            save_scene_metadata_review(path, {"schema_version": 1})
        assert not path.exists()
        assert not list(tmp_path.glob("*.tmp"))

    def test_save_revalidates_model_mutated_after_construction(self, tmp_path):
        bundle = _load_bundle(tmp_path / "frozen")
        doc = create_scene_metadata_review(bundle)
        doc.total_source_scenes = 99
        path = tmp_path / "scene_metadata_review.json"
        with pytest.raises(ValueError, match="拒绝写出"):
            save_scene_metadata_review(path, doc)
        assert not path.exists()

    def test_write_failure_keeps_old_file(self, tmp_path, monkeypatch):
        bundle = _load_bundle(tmp_path / "frozen")
        old = create_scene_metadata_review(bundle, reviewer="old")
        new = create_scene_metadata_review(bundle, reviewer="new")
        path = tmp_path / "state.json"
        save_scene_metadata_review(path, old)

        real_write_text = Path.write_text

        def failing_write_text(self_path, data, *args, **kwargs):
            if str(self_path).endswith(".tmp"):
                raise OSError("injected: tmp write failure")
            return real_write_text(self_path, data, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", failing_write_text)
        with pytest.raises(OSError, match="injected"):
            save_scene_metadata_review(path, new)
        monkeypatch.undo()

        assert load_scene_metadata_review(path).reviewer == "old"  # 旧文件不变
        assert not list(tmp_path.glob("*.tmp"))  # 失败后清理未完成 tmp
        assert not list(tmp_path.glob("*.tmp.old"))

    def test_replace_failure_keeps_old_file(self, tmp_path, monkeypatch):
        bundle = _load_bundle(tmp_path / "frozen")
        old = create_scene_metadata_review(bundle, reviewer="old")
        new = create_scene_metadata_review(bundle, reviewer="new")
        path = tmp_path / "state.json"
        save_scene_metadata_review(path, old)

        import knowledge.game_rag.scene_metadata_review as smr

        real_replace = os.replace

        def failing_replace(src, dst, *args, **kwargs):
            if str(dst).endswith("state.json"):
                raise OSError("injected: replace failure")
            return real_replace(src, dst, *args, **kwargs)

        monkeypatch.setattr(smr.os, "replace", failing_replace)
        with pytest.raises(OSError, match="injected"):
            save_scene_metadata_review(path, new)
        monkeypatch.undo()

        assert load_scene_metadata_review(path).reviewer == "old"  # 替换失败旧文件不变
        assert not list(tmp_path.glob("*.tmp"))

    def test_first_save_failure_writes_no_formal_file(self, tmp_path, monkeypatch):
        bundle = _load_bundle(tmp_path / "frozen")
        doc = create_scene_metadata_review(bundle)
        path = tmp_path / "state.json"

        real_write_text = Path.write_text

        def failing_write_text(self_path, data, *args, **kwargs):
            if str(self_path).endswith(".tmp"):
                raise OSError("injected: tmp write failure")
            return real_write_text(self_path, data, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", failing_write_text)
        with pytest.raises(OSError, match="injected"):
            save_scene_metadata_review(path, doc)
        monkeypatch.undo()

        assert not path.exists()  # 失败不产生不完整正式文件
        assert not list(tmp_path.glob("*.tmp"))


# ---------- 人工审核包 ----------


class TestReviewPack:
    def test_deterministic(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        assert generate_scene_metadata_review_pack(bundle) == generate_scene_metadata_review_pack(bundle)

    def test_contains_all_scenes_and_fill_slots(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        pack = generate_scene_metadata_review_pack(bundle)
        for scene in bundle.scenes:
            assert scene.id in pack
            assert f"L{scene.source.line_start}-{scene.source.line_end}" in pack
        for slot in (
            "viewpoint: ______",
            "temporal_scope: ______",
            "reality_status: ______",
            "mentioned_characters: ______",
            "present_characters: ______",
            "evidence: ______",
            "reason: ______",
            "review_status: ______",
        ):
            assert pack.count(slot) == bundle.total_scenes

    def test_short_scene_shows_full_text(self, tmp_path):
        bundle = _load_bundle(tmp_path)
        pack = generate_scene_metadata_review_pack(bundle)
        assert "vol99_9合成场景A 原文第1行" in pack
        assert "vol99_9合成场景A 原文第10行" in pack

    def test_long_scene_excerpted_with_marker(self, tmp_path):
        long_scene = _scene("scene_long", UNIT_A, PATH_A, 1, 60, speakers=["妃"])
        scenes = [long_scene, _scene("scene_b1", UNIT_B, PATH_B, 1, 15)]
        manifest = _manifest({UNIT_A: 1, UNIT_B: 1})
        scene_path, manifest_path = _write_bundle(tmp_path, scenes, manifest)
        bundle = load_frozen_scene_bundle(scene_path, manifest_path)

        pack = generate_scene_metadata_review_pack(bundle)
        assert "原文第60行" in pack  # 首尾各 8 行在摘录内
        assert "原文第1行" in pack
        assert "中间省略 44 行" in pack
        assert "摘录不构成完整人工审核" in pack
        assert "原文第20行" not in pack  # 中间行不展示

    def test_write_to_path_atomic(self, tmp_path):
        bundle = _load_bundle(tmp_path / "frozen")
        out = tmp_path / "pack" / "scene_metadata_review.md"
        content = generate_scene_metadata_review_pack(bundle, out_path=out)
        assert out.read_text(encoding="utf-8") == content
        assert not list((tmp_path / "pack").glob("*.tmp"))


# ---------- enriched 输出门禁 ----------


class TestWriteEnrichedScenes:
    def _approved(self, tmp_path):
        bundle = _load_bundle(tmp_path / "frozen")
        doc = _approved_doc(bundle)
        return bundle, doc

    def test_writes_pair_with_source_identity(self, tmp_path):
        bundle, doc = self._approved(tmp_path)
        out_dir = tmp_path / "enriched"
        manifest = write_enriched_scenes(bundle, doc, out_dir)

        scenes_path = out_dir / "enriched_scenes.jsonl"
        manifest_path = out_dir / "enriched_manifest.json"
        assert scenes_path.exists() and manifest_path.exists()
        records = [json.loads(line) for line in scenes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(records) == len(bundle.scenes)
        assert all(r["review_status"] == "approved" for r in records)
        assert manifest["total_scenes"] == len(bundle.scenes)
        assert manifest["source_boundary_manifest"]["manifest_sha256"] == bundle.manifest_digest
        assert manifest["source_boundary_manifest"]["scenes_sha256"] == bundle.scenes_digest
        assert manifest["source_boundary_manifest"]["bundle_sha256"] == bundle.bundle_digest
        assert manifest["source_boundary_manifest"]["boundary_review_status"] == "approved"
        assert manifest["scene_review_status"] == "approved"
        assert not list(out_dir.glob("*.tmp")) and not list(out_dir.glob("*.tmp.old"))

    def test_rejects_draft_review_document(self, tmp_path):
        bundle = _load_bundle(tmp_path / "frozen")
        out_dir = tmp_path / "enriched"
        doc = create_scene_metadata_review(bundle, reviewer="meta-auditor")
        with pytest.raises(ValueError, match="顶层 review_status 必须为 approved"):
            write_enriched_scenes(bundle, doc, out_dir)
        assert not out_dir.exists() or not list(out_dir.iterdir())  # 拒绝路径零文件写入

    def test_rejects_review_for_different_scene_content(self, tmp_path):
        bundle, doc = self._approved(tmp_path / "original")
        changed_scenes = _default_scenes()
        changed_scenes[0] = changed_scenes[0].model_copy(update={"text": "另一份正文"})
        different = _load_bundle(
            tmp_path / "changed",
            scenes=changed_scenes,
            manifest=_manifest({UNIT_A: 2, UNIT_B: 1}),
        )
        with pytest.raises(ValueError, match="scenes_sha256 与 bundle 不一致"):
            write_enriched_scenes(different, doc, tmp_path / "enriched")
        assert not (tmp_path / "enriched").exists() or not list((tmp_path / "enriched").iterdir())

    def test_does_not_touch_frozen_bundle_files(self, tmp_path):
        frozen_dir = tmp_path / "frozen"
        bundle = _load_bundle(frozen_dir)
        doc = _approved_doc(bundle)
        before = {
            name: (frozen_dir / name).read_text(encoding="utf-8") for name in ("scenes.jsonl", "boundary_manifest.json")
        }
        write_enriched_scenes(bundle, doc, tmp_path / "enriched")
        after = {
            name: (frozen_dir / name).read_text(encoding="utf-8") for name in ("scenes.jsonl", "boundary_manifest.json")
        }
        assert before == after

    def test_secondary_replace_failure_restores_old_pair(self, tmp_path, monkeypatch):
        bundle, doc = self._approved(tmp_path)
        out_dir = tmp_path / "enriched"
        write_enriched_scenes(bundle, doc, out_dir)
        primary = out_dir / "enriched_scenes.jsonl"
        secondary = out_dir / "enriched_manifest.json"
        before = (primary.read_bytes(), secondary.read_bytes())

        import knowledge.game_rag.scene_metadata_review as smr

        real_replace = os.replace

        def failing_replace(src, dst, *args, **kwargs):
            if Path(dst) == secondary:
                raise OSError("injected: secondary replace failure")
            return real_replace(src, dst, *args, **kwargs)

        monkeypatch.setattr(smr.os, "replace", failing_replace)
        with pytest.raises(ValueError, match="已回滚"):
            write_enriched_scenes(bundle, doc, out_dir)
        monkeypatch.undo()

        assert (primary.read_bytes(), secondary.read_bytes()) == before
        assert not list(out_dir.glob("*.tmp")) and not list(out_dir.glob("*.tmp.old"))

    def test_transient_rollback_failure_retries_and_reports_recovery(self, tmp_path, monkeypatch):
        bundle, doc = self._approved(tmp_path)
        out_dir = tmp_path / "enriched"
        write_enriched_scenes(bundle, doc, out_dir)
        primary = out_dir / "enriched_scenes.jsonl"
        secondary = out_dir / "enriched_manifest.json"
        before = (primary.read_bytes(), secondary.read_bytes())

        import knowledge.game_rag.scene_metadata_review as smr

        real_replace = os.replace
        restore_attempts = 0

        def flaky_replace(src, dst, *args, **kwargs):
            nonlocal restore_attempts
            if Path(dst) == secondary:
                raise OSError("injected: secondary replace failure")
            if Path(dst) == primary and str(src).endswith(".tmp.old"):
                restore_attempts += 1
                if restore_attempts == 1:
                    raise OSError("injected: transient rollback failure")
            return real_replace(src, dst, *args, **kwargs)

        monkeypatch.setattr(smr.os, "replace", flaky_replace)
        with pytest.raises(ValueError, match="已回滚"):
            write_enriched_scenes(bundle, doc, out_dir)
        monkeypatch.undo()

        assert restore_attempts == 2
        assert (primary.read_bytes(), secondary.read_bytes()) == before
        assert not list(out_dir.glob("*.tmp")) and not list(out_dir.glob("*.tmp.old"))

    def test_double_rollback_failure_preserves_backup_and_reports_path(self, tmp_path, monkeypatch):
        bundle, doc = self._approved(tmp_path)
        out_dir = tmp_path / "enriched"
        write_enriched_scenes(bundle, doc, out_dir)
        primary = out_dir / "enriched_scenes.jsonl"
        secondary = out_dir / "enriched_manifest.json"
        old_primary = primary.read_bytes()

        import knowledge.game_rag.scene_metadata_review as smr

        real_replace = os.replace

        def failing_replace(src, dst, *args, **kwargs):
            if Path(dst) == secondary:
                raise OSError("injected: secondary replace failure")
            if Path(dst) == primary and str(src).endswith(".tmp.old"):
                raise OSError("injected: rollback failure")
            return real_replace(src, dst, *args, **kwargs)

        monkeypatch.setattr(smr.os, "replace", failing_replace)
        with pytest.raises(ValueError, match=r"回滚失败.*tmp\.old"):
            write_enriched_scenes(bundle, doc, out_dir)
        monkeypatch.undo()

        backup = out_dir / "enriched_scenes.jsonl.tmp.old"
        assert backup.read_bytes() == old_primary
        assert primary.exists() and secondary.exists()  # 故障现场保留，等待人工恢复
        assert not list(out_dir.glob("*.tmp"))

    def test_backup_failure_cleans_both_temporary_files(self, tmp_path, monkeypatch):
        bundle, doc = self._approved(tmp_path)
        out_dir = tmp_path / "enriched"
        write_enriched_scenes(bundle, doc, out_dir)
        primary = out_dir / "enriched_scenes.jsonl"
        secondary = out_dir / "enriched_manifest.json"
        before = (primary.read_bytes(), secondary.read_bytes())

        import knowledge.game_rag.scene_metadata_review as smr

        real_replace = os.replace

        def failing_replace(src, dst, *args, **kwargs):
            if str(dst).endswith(".tmp.old"):
                raise OSError("injected: backup failure")
            return real_replace(src, dst, *args, **kwargs)

        monkeypatch.setattr(smr.os, "replace", failing_replace)
        with pytest.raises(OSError, match="backup failure"):
            write_enriched_scenes(bundle, doc, out_dir)
        monkeypatch.undo()

        assert (primary.read_bytes(), secondary.read_bytes()) == before
        assert not list(out_dir.glob("*.tmp")) and not list(out_dir.glob("*.tmp.old"))

    def test_stale_backup_blocks_write_without_overwriting_recovery_copy(self, tmp_path):
        bundle, doc = self._approved(tmp_path)
        out_dir = tmp_path / "enriched"
        out_dir.mkdir()
        backup = out_dir / "enriched_scenes.jsonl.tmp.old"
        backup.write_text("RECOVERY COPY\n", encoding="utf-8")

        with pytest.raises(ValueError, match="未恢复的旧版备份"):
            write_enriched_scenes(bundle, doc, out_dir)

        assert backup.read_text(encoding="utf-8") == "RECOVERY COPY\n"
        assert not list(out_dir.glob("*.tmp"))


# ---------- 真实冻结目录只读接入 ----------


class TestRealFrozenDirectory:
    def test_real_frozen_bundle_loads_without_creating_p4_files(self):
        """正式冻结后 P4A 可只读加载 262 场景，但不会自动落盘审核状态。"""
        before = sorted(p.name for p in REVIEW_DIR.iterdir()) if REVIEW_DIR.exists() else None
        bundle = load_frozen_scene_bundle(REVIEW_DIR / "scenes.jsonl", REVIEW_DIR / "boundary_manifest.json")
        review = create_scene_metadata_review(bundle, reviewer="")
        after = sorted(p.name for p in REVIEW_DIR.iterdir()) if REVIEW_DIR.exists() else None
        assert before == after
        assert bundle.total_scenes == 262
        assert bundle.manifest.boundary_review_status == "approved"
        assert bundle.manifest.scene_review_status == "draft"
        assert all(decision.review_status is ReviewStatus.draft for decision in review.scene_decisions)
        assert not (REVIEW_DIR / "scene_metadata_review.json").exists()

    @needs_p3_review_files
    def test_p3_review_state_is_approved_and_frozen(self):
        """项目负责人授权后两层 P3 审核均 approved，正式冻结双文件存在。"""
        overrides = json.loads((REVIEW_DIR / "boundary_overrides.json").read_text(encoding="utf-8"))
        low = json.loads((REVIEW_DIR / "low_candidate_review.json").read_text(encoding="utf-8"))
        assert overrides["boundary_review_status"] == "approved"
        assert low["review_status"] == "approved"
        assert overrides["reviewer"] == low["reviewer"] == "project_owner_01"
        assert (REVIEW_DIR / "scenes.jsonl").exists()
        assert (REVIEW_DIR / "boundary_manifest.json").exists()
