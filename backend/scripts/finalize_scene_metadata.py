#!/usr/bin/env python3
"""P4D：262 个冻结场景的人工元数据定稿（批量应用 manual_decisions.json）。

用法（在仓库根目录执行）：
  py -3.10 backend/scripts/finalize_scene_metadata.py --dry-run
      # 试运行：校验决定完备性并预览统计，不写出任何文件
  py -3.10 backend/scripts/finalize_scene_metadata.py
      # 定稿：构建全部 262 条 needs_review 记录并原子保存审核文档

输入（均在 backend/data/knowledge/tsukiyashiro_kisaki/scene_metadata_review/）：
  - manual_decisions.json  人工定稿决定（scene_id → 字段决定，附录性数据资产）
  - character_aliases.json 人物 canonical 别名映射
  - candidate_run.json     P4C 候选运行状态（m/p 缺省时的推导来源与差异统计基准）
  - scene_metadata_review.json  当前 P4A 审核文档（被本脚本定稿替换）

产物：
  - scene_metadata_review.json  定稿后的 P4A 审核文档（全部 needs_review，顶层 draft）

规则：
  - 决定文件必须覆盖全部冻结场景（缺一即拒绝，保证断点续做以决定为单位增量）；
  - m/p（mentioned/present）必须在每条人工决定中显式给出，再经别名归一与
    非人物校验；不得从模型候选隐式继承，避免候选变化导致人工定稿漂移；
  - evidence 使用决定中的具体行区间（少量关键行），不再沿用候选的整段 span；
  - 全部记录 review_status=needs_review、reviewer=project_owner_01，顶层保持 draft；
  - 幂等：同一输入重复运行产物完全一致；不修改冻结包、候选状态与游戏原文；
  - 不调用 apply_approved_scene_metadata / write_enriched_scenes，不产生 approved。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from knowledge.game_rag import (  # noqa: E402
    ReviewStatus,
    SceneMetadataDecision,
    SceneMetadataReviewDocument,
    load_candidate_run,
    load_frozen_scene_bundle,
    load_scene_metadata_review,
    save_scene_metadata_review,
    validate_scene_metadata_review,
)

BASE = REPO_ROOT / "backend" / "data" / "knowledge" / "tsukiyashiro_kisaki"
REVIEW_DIR = BASE / "scene_metadata_review"
BOUNDARY_DIR = BASE / "scene_boundary_review"
SCENES_PATH = BOUNDARY_DIR / "scenes.jsonl"
MANIFEST_PATH = BOUNDARY_DIR / "boundary_manifest.json"
REVIEW_PATH = REVIEW_DIR / "scene_metadata_review.json"
CANDIDATE_PATH = REVIEW_DIR / "candidate_run.json"
DECISIONS_PATH = REVIEW_DIR / "manual_decisions.json"
ALIASES_PATH = REVIEW_DIR / "character_aliases.json"

REVIEWER = "project_owner_01"
CREATED_BY = "backend/scripts/finalize_scene_metadata.py"
EXPECTED_BUNDLE_SHA256 = "b82d753eb93dbc1614b89e98c8498f03884c7d5dd433d223ef853708052c8595"

# 非人物条目：书籍名/家族名/故事角色称谓等，不进入人物字段（见 character_aliases.json notes）
NON_PERSON_TERMS = frozenset(
    {
        "翡翠",
        "红宝石",
        "蓝宝石",
        "紫水晶",
        "磷灰石",
        "芙蓉石",
        "黑珍珠",
        "萤石",
        "白珍珠",
        "绿幽灵水晶",
        "黑曜石",
        "黑玛瑙",
        "青金石",
        "紫翠玉",
        "缟玛瑙",
        "潘多拉",
        "魔法之书",
        "黑榴石的死神花样",
        "勿忘日记",
        "观星少女",
        "幻影水晶的迷乱反射",
        "纸上魔法使",
        "萤色光景",
        "天使",
        "遊行寺家",
        "遊行寺本家",
    }
)


def _fail(message: str) -> None:
    print(f"[P4D] 错误：{message}", file=sys.stderr)
    raise SystemExit(1)


def load_aliases() -> dict[str, str]:
    data = json.loads(ALIASES_PATH.read_text(encoding="utf-8"))
    aliases = data.get("aliases")
    if not isinstance(aliases, dict):
        _fail("character_aliases.json 缺少 aliases 对象")
    return {str(k): str(v) for k, v in aliases.items()}


def normalize_names(names: list[str], aliases: dict[str, str]) -> list[str]:
    """别名归一 + 非人物剔除；顺序稳定（SceneMetadataDecision 构造时再规范化排序）。"""
    out: list[str] = []
    for name in names:
        canonical = aliases.get(name, name)
        if canonical in NON_PERSON_TERMS or name in NON_PERSON_TERMS:
            continue
        if canonical not in out:
            out.append(canonical)
    return out


def build_decision(
    scene: Any,
    decision: dict[str, Any],
    aliases: dict[str, str],
) -> SceneMetadataDecision:
    source = scene.source.model_copy(deep=True)
    vp = decision["vp"]
    for field in ("m", "p"):
        if field not in decision or not isinstance(decision[field], list):
            _fail(f"{scene.id}：人工决定必须显式提供数组字段 {field}")
    mentioned = normalize_names(list(decision["m"]), aliases)
    present = normalize_names(list(decision["p"]), aliases)
    if set(decision["m"]) & NON_PERSON_TERMS:
        _fail(f"{scene.id}：mentioned 含非人物条目")
    if set(decision["p"]) & NON_PERSON_TERMS:
        _fail(f"{scene.id}：present 含非人物条目")

    evidence = [
        {"source_path": scene.source.source_path, "line_start": int(a), "line_end": int(b)} for a, b in decision["ev"]
    ]
    reasons = [part.strip() for part in decision["r"].split("；") if part.strip()]
    warnings = list(decision.get("w") or [])

    return SceneMetadataDecision(
        scene_id=scene.id,
        story_unit_id=scene.story.story_unit_id,
        source=source,
        viewpoint=vp,
        temporal_scope=decision["tp"],
        reality_status=decision["rl"],
        mentioned_characters=mentioned,
        present_characters=present,
        evidence=evidence,
        reasons=reasons,
        warnings=warnings,
        review_status=ReviewStatus.needs_review,
        reviewer=REVIEWER,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="P4D 人工元数据定稿")
    parser.add_argument("--dry-run", action="store_true", help="只校验与统计，不写出文件")
    args = parser.parse_args()

    # 门禁：冻结包三摘要（与 P4C 相同的既定指纹）
    bundle = load_frozen_scene_bundle(SCENES_PATH, MANIFEST_PATH)
    if bundle.bundle_digest != EXPECTED_BUNDLE_SHA256:
        _fail(f"冻结包摘要不一致：{bundle.bundle_digest}")

    for path in (REVIEW_PATH, CANDIDATE_PATH, DECISIONS_PATH, ALIASES_PATH):
        if not path.is_file():
            _fail(f"输入文件不存在：{path}")

    old_review = load_scene_metadata_review(REVIEW_PATH)
    errors = validate_scene_metadata_review(old_review, bundle)
    if errors:
        _fail("现有审核文档未通过 bundle 校验:\n- " + "\n- ".join(errors))
    if old_review.review_status != "draft":
        _fail(f"顶层 review_status 必须为 draft，当前为 {old_review.review_status!r}")

    candidate_run = load_candidate_run(CANDIDATE_PATH)
    run_errors: list[str] = []
    if candidate_run.source_manifest.bundle_sha256 != bundle.bundle_digest:
        run_errors.append("candidate_run 与冻结包摘要不一致")
    if run_errors:
        _fail("\n- ".join(run_errors))
    candidates = {
        state.scene_id: (state.candidate.model_dump() if state.candidate else None)
        for state in candidate_run.scene_states
    }

    aliases = load_aliases()
    decisions_doc = json.loads(DECISIONS_PATH.read_text(encoding="utf-8"))
    decisions = decisions_doc.get("decisions")
    if not isinstance(decisions, dict):
        _fail("manual_decisions.json 缺少 decisions 对象")

    scene_ids = [scene.id for scene in bundle.scenes]
    missing = [sid for sid in scene_ids if sid not in decisions]
    if missing:
        _fail(f"决定未覆盖 {len(missing)} 个场景（断点续做需补齐后重跑）: {missing[:5]}…")
    unknown = [sid for sid in decisions if sid not in set(scene_ids)]
    if unknown:
        _fail(f"决定包含未知场景 id: {unknown[:5]}…")

    # 构建定稿文档（冻结顺序；scene_id/source/story_unit_id 原样复制，不可篡改）
    new_decisions: list[SceneMetadataDecision] = []
    for scene in bundle.scenes:
        new_decisions.append(build_decision(scene, decisions[scene.id], aliases))

    new_review = SceneMetadataReviewDocument(
        schema_version=old_review.schema_version,
        source_manifest=old_review.source_manifest.model_copy(deep=True),
        total_source_scenes=old_review.total_source_scenes,
        reviewer=REVIEWER,
        review_status="draft",
        scene_decisions=new_decisions,
        notes=(
            "P4D 人工定稿：262 个场景全部完成人工元数据定稿（viewpoint/temporal_scope/"
            "reality_status/人物/evidence/reasons/warnings），全部记录为 needs_review，"
            "等待外部复核后再人工置为 approved。人物名已按 character_aliases.json 归一；"
            "3 个候选失败场景（scene_0cfa2e664341ebce/scene_504131e9df789cb9/"
            "scene_cca1256fbf01a2f2）由人工从原文从零定稿。"
        ),
        created_by=CREATED_BY,
    )

    # 完整校验（结构 + bundle 绑定 + 跨字段一致性）
    errors = validate_scene_metadata_review(new_review, bundle)
    if errors:
        _fail("定稿文档未通过 P4A 校验:\n- " + "\n- ".join(errors))
    # 硬性状态检查
    statuses = [d.review_status for d in new_review.scene_decisions]
    if len(new_review.scene_decisions) != len(scene_ids):
        _fail("记录数与冻结场景数不一致")
    if any(s is not ReviewStatus.needs_review for s in statuses):
        _fail("存在非 needs_review 的记录")
    if new_review.review_status != "draft":
        _fail("顶层 review_status 必须为 draft")
    for d in new_review.scene_decisions:
        if d.viewpoint is None or d.temporal_scope is None or d.reality_status is None:
            _fail(f"{d.scene_id}：分类字段为 None")
        if d.mentioned_characters is None or d.present_characters is None or not d.evidence:
            _fail(f"{d.scene_id}：人物/evidence 字段为 None 或空")

    # 差异统计（相对模型候选；失败场景计为人工从零定稿）
    stats = _diff_stats(new_decisions, candidates)
    _print_summary(new_review, stats, candidates)

    if args.dry_run:
        print("[P4D] dry-run：校验通过，未写出任何文件")
        return

    save_scene_metadata_review(REVIEW_PATH, new_review)
    print(f"[P4D] 定稿已原子保存：{REVIEW_PATH}")

    # 复核：重新加载并验证
    reloaded = load_scene_metadata_review(REVIEW_PATH)
    errors = validate_scene_metadata_review(reloaded, bundle)
    if errors:
        _fail("重载后校验失败:\n- " + "\n- ".join(errors))
    if [d.scene_id for d in reloaded.scene_decisions] != scene_ids:
        _fail("重载后场景顺序/集合不一致")
    print(f"[P4D] 复核通过：{len(reloaded.scene_decisions)} 条记录，顺序与冻结包一致")


def _diff_stats(
    new_decisions: list[SceneMetadataDecision],
    candidates: dict[str, dict[str, Any] | None],
) -> dict[str, Any]:
    fields = ("viewpoint", "temporal_scope", "reality_status")
    changed = {f: 0 for f in fields}
    changed["mentioned_characters"] = 0
    changed["present_characters"] = 0
    from_scratch = 0
    for d in new_decisions:
        cand = candidates.get(d.scene_id)
        if cand is None:
            from_scratch += 1
            continue
        for f in fields:
            if getattr(d, f) != cand.get(f):
                changed[f] += 1
        if list(d.mentioned_characters or []) != list(cand.get("mentioned_characters") or []):
            changed["mentioned_characters"] += 1
        if list(d.present_characters or []) != list(cand.get("present_characters") or []):
            changed["present_characters"] += 1
    return {"changed": changed, "from_scratch": from_scratch}


def _print_summary(
    review: SceneMetadataReviewDocument,
    stats: dict[str, Any],
    candidates: dict[str, dict[str, Any] | None],
) -> None:
    import collections

    vp = collections.Counter(d.viewpoint for d in review.scene_decisions)
    tp = collections.Counter(d.temporal_scope.value for d in review.scene_decisions)
    rl = collections.Counter(d.reality_status.value for d in review.scene_decisions)
    warn_scenes = sum(1 for d in review.scene_decisions if d.warnings)
    print(f"[P4D] 记录总数：{len(review.scene_decisions)}（全部 needs_review，顶层 draft）")
    print(f"[P4D] viewpoint 分布：{dict(vp.most_common())}")
    print(f"[P4D] temporal_scope 分布：{dict(tp.most_common())}")
    print(f"[P4D] reality_status 分布：{dict(rl.most_common())}")
    print(f"[P4D] 人工修正（相对候选）: {stats['changed']}；失败场景从零定稿: {stats['from_scratch']}")
    print(f"[P4D] 保留 warnings 的场景数：{warn_scenes}")


if __name__ == "__main__":
    main()
