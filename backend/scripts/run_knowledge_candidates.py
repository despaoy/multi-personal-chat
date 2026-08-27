#!/usr/bin/env python3
"""P5：从 262 个 approved enriched 场景生成事实卡/关系卡/事件卡候选。

用法（在仓库根目录执行）：
  py -3.10 backend/scripts/run_knowledge_candidates.py smoke
      # 代表场景试运行（11 个场景：普通现实/回忆/梦境/书中故事/reconstruction/
      # 多视角/长场景/宣传元叙事/character_claim/猫视角/失败回归）
  py -3.10 backend/scripts/run_knowledge_candidates.py run
      # 全量运行（断点续跑：逐场景原子保存状态，重跑只处理 pending/failed）
  py -3.10 backend/scripts/run_knowledge_candidates.py status
      # 查看当前运行状态与进度
  py -3.10 backend/scripts/run_knowledge_candidates.py finalize
      # 从运行状态确定性产出候选文档、审核状态、质量报告与人工复核材料

产物目录：backend/data/knowledge/tsukiyashiro_kisaki/knowledge_candidate_review/
  - knowledge_candidate_run.json  模型运行状态（确定性，断点续跑载体）
  - knowledge_run_manifest.json   运行 manifest（唯一携带时间戳/调用统计）
  - facts_candidate.jsonl         事实卡候选（draft/needs_review）
  - relations_candidate.jsonl     关系卡候选（draft/needs_review）
  - events_candidate.jsonl        事件卡候选（draft/needs_review）
  - knowledge_review.json         人工审核状态（全部 draft，冲突卡 needs_review）
  - knowledge_quality_report.json 质量报告（去重/冲突/分布/失败）
  - knowledge_review.md           人工复核材料

模型客户端：OllamaKnowledgeClient 是最薄适配层，复用 P4C
run_scene_metadata_candidates.OllamaCandidateClient 的请求契约
（POST {base_url}/api/chat；options: temperature/top_p/num_predict/num_ctx；
模型与地址同 OLLAMA_MODEL/OLLAMA_BASE_URL 环境变量默认值）。不直接 import
OllamaProvider 的原因：该模块导入链会触发 Redis 连接与 .env 写入等应用级
副作用（db.adapter），不适合独立批处理进程。

门禁（本脚本严格遵守）：
  - enriched 双文件必须通过 load_enriched_scene_bundle 门禁（262 approved），
    任何候选生成之前完成；
  - 不修改 P3/P4 的任何审核决定或产物（enriched 输入只读）；
  - 候选文档至多 needs_review，绝不自动 approved；不生成 embedding/向量/索引；
  - 模型不可用时准确记录失败并以非零退出码停止，绝不伪造候选。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from knowledge.game_rag import (  # noqa: E402
    KnowledgeGenerationStatus,
    build_knowledge_quality_report,
    build_knowledge_run_manifest,
    create_knowledge_review,
    create_knowledge_run,
    finalize_knowledge_candidates,
    generate_knowledge_candidates,
    load_alias_config,
    load_enriched_scene_bundle,
    load_knowledge_run,
    save_documents_jsonl,
    save_knowledge_review,
    save_knowledge_run_with_manifest,
)

BASE = REPO_ROOT / "backend" / "data" / "knowledge" / "tsukiyashiro_kisaki"
ENRICHED_DIR = BASE / "scene_metadata_enriched"
ENRICHED_SCENES_PATH = ENRICHED_DIR / "enriched_scenes.jsonl"
ENRICHED_MANIFEST_PATH = ENRICHED_DIR / "enriched_manifest.json"
ALIASES_PATH = BASE / "scene_metadata_review" / "character_aliases.json"

OUT_DIR = BASE / "knowledge_candidate_review"
STATE_PATH = OUT_DIR / "knowledge_candidate_run.json"
MANIFEST_OUT_PATH = OUT_DIR / "knowledge_run_manifest.json"
FACTS_PATH = OUT_DIR / "facts_candidate.jsonl"
RELATIONS_PATH = OUT_DIR / "relations_candidate.jsonl"
EVENTS_PATH = OUT_DIR / "events_candidate.jsonl"
REVIEW_PATH = OUT_DIR / "knowledge_review.json"
QUALITY_REPORT_PATH = OUT_DIR / "knowledge_quality_report.json"
REVIEW_MD_PATH = OUT_DIR / "knowledge_review.md"

# 模型与运行参数（与用户阶段要求一致；num_predict 为本阶段输出规模适配值）。
OLLAMA_MODEL = "qwen2.5:7b"
MODEL_ID = "ollama:qwen2.5:7b"
OLLAMA_BASE_URL = "http://localhost:11434"
TEMPERATURE = 0.2
TOP_P = 0.9
NUM_PREDICT = 3072
NUM_CTX = 16384
REQUEST_TIMEOUT_S = 300
MAX_ATTEMPTS = 3
REVIEWER = "project_owner_01"

# Smoke 代表场景：覆盖普通现实/回忆/梦境/魔法之书正文/reconstruction/多视角/
# 长场景/宣传元叙事/character_claim/猫视角，以及 P4C 失败场景的回归验证。
SMOKE_SCENE_IDS = [
    "scene_6bf3e7cf2262e0e6",  # 普通现实（琉璃入学首日）
    "scene_6d539b1b02ec01e2",  # 回忆（妃不幸其一）
    "scene_5df1b207252c417a",  # 梦境（萤色光辉中的妃）
    "scene_93d01b8f0e6a6ba9",  # 魔法之书正文（紫水晶故事）
    "scene_cb83a63c51df2e79",  # reconstruction（青金石再现世界）
    "scene_e156abf7feeca1dd",  # 多视角（理央独白+琉璃叙述）
    "scene_e5d4aa08d08dfb1e",  # 长场景（695 行，3 分片）
    "scene_b33f3adaf98b715d",  # 宣传元叙事
    "scene_b876c94fe61c7915",  # character_claim（妃日记）
    "scene_cc6a73c7c868182e",  # 猫视角（追加剧本开场）
    "scene_0cfa2e664341ebce",  # P4C 失败场景回归（彼方独白）
]


def _fail(message: str) -> None:
    print(f"[P5] 错误：{message}", file=sys.stderr)
    raise SystemExit(1)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017  # 运行环境为 Python 3.10，无 datetime.UTC


class OllamaKnowledgeClient:
    """最薄 KnowledgeModelClient 适配层：同步调用本机 Ollama /api/chat。

    - 请求契约与 inference.model_manager.OllamaProvider 一致（messages/stream/options），
      额外显式设置 num_ctx 防止 Ollama 静默截断超长 prompt；
    - 超时抛 TimeoutError（P5 收敛为 error_kind=timeout）；HTTP/连接错误抛 RuntimeError
      （P5 收敛为 error_kind=model_error）——异常消息不含 prompt 或密钥；
    - 累计调用统计（次数/token/时延/失败）供运行 manifest 使用，不进入运行状态；
    - 每次调用后打印进度行（ASCII），便于长任务外部观察。
    """

    def __init__(
        self,
        *,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
        num_predict: int = NUM_PREDICT,
        num_ctx: int = NUM_CTX,
        timeout_s: float = REQUEST_TIMEOUT_S,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.num_predict = num_predict
        self.num_ctx = num_ctx
        self.timeout_s = timeout_s
        self.stats: dict[str, float | int] = {
            "calls": 0,
            "failures": 0,
            "prompt_tokens": 0,
            "eval_tokens": 0,
            "total_latency_s": 0.0,
            "max_latency_s": 0.0,
        }

    def __call__(self, prompt: str) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "num_predict": self.num_predict,
                    "num_ctx": self.num_ctx,
                },
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.monotonic()
        self.stats["calls"] = int(self.stats["calls"]) + 1
        call_no = int(self.stats["calls"])
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            self.stats["failures"] = int(self.stats["failures"]) + 1
            raise RuntimeError(f"ollama http status {exc.code}") from exc
        except urllib.error.URLError as exc:
            self.stats["failures"] = int(self.stats["failures"]) + 1
            if isinstance(exc.reason, TimeoutError):
                raise TimeoutError("ollama request timeout") from exc
            raise RuntimeError(f"ollama connection error: {type(exc.reason).__name__}") from exc
        except TimeoutError as exc:
            self.stats["failures"] = int(self.stats["failures"]) + 1
            raise TimeoutError("ollama request timeout") from exc
        latency = time.monotonic() - started
        self.stats["total_latency_s"] = float(self.stats["total_latency_s"]) + latency
        self.stats["max_latency_s"] = max(float(self.stats["max_latency_s"]), latency)
        try:
            data = json.loads(body.decode("utf-8"))
            content = data["message"]["content"]
            self.stats["prompt_tokens"] = int(self.stats["prompt_tokens"]) + int(data.get("prompt_eval_count") or 0)
            self.stats["eval_tokens"] = int(self.stats["eval_tokens"]) + int(data.get("eval_count") or 0)
        except (json.JSONDecodeError, KeyError, TypeError, UnicodeDecodeError) as exc:
            self.stats["failures"] = int(self.stats["failures"]) + 1
            raise RuntimeError("ollama response parse error") from exc
        if not isinstance(content, str):
            self.stats["failures"] = int(self.stats["failures"]) + 1
            raise RuntimeError("ollama response content is not a string")
        print(
            f"[P5] call#{call_no} ok {latency:.1f}s eval={int(data.get('eval_count') or 0)}tok",
            flush=True,
        )
        return content


def _load_inputs():
    bundle = load_enriched_scene_bundle(ENRICHED_SCENES_PATH, ENRICHED_MANIFEST_PATH)
    if len(bundle.scenes) != 262:
        _fail(f"enriched 场景数必须为 262，当前为 {len(bundle.scenes)}")
    alias_config = load_alias_config(ALIASES_PATH)
    return bundle, alias_config


def _load_or_create_state(bundle):
    if STATE_PATH.is_file():
        return load_knowledge_run(STATE_PATH)
    return create_knowledge_run(bundle, model_id=MODEL_ID)


def _print_state_summary(state) -> None:
    counts = {}
    for item in state.scene_states:
        counts[item.status.value] = counts.get(item.status.value, 0) + 1
    total_attempts = sum(item.attempts for item in state.scene_states)
    print(f"[P5] 状态: {counts} 累计调用={total_attempts}")


def phase_status() -> None:
    if not STATE_PATH.is_file():
        print("[P5] 运行状态不存在（先执行 smoke/run）")
        return
    state = load_knowledge_run(STATE_PATH)
    _print_state_summary(state)
    failed = [
        (item.scene_id, item.last_failure.error_kind if item.last_failure else "?")
        for item in state.scene_states
        if item.status is KnowledgeGenerationStatus.failed
    ]
    for scene_id, error_kind in failed[:10]:
        print(f"[P5] failed: {scene_id} ({error_kind})")
    if len(failed) > 10:
        print(f"[P5] …共 {len(failed)} 个失败场景")


def _run_phase(scene_ids: list[str] | None, label: str) -> None:
    bundle, alias_config = _load_inputs()
    state = _load_or_create_state(bundle)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = OllamaKnowledgeClient()
    started_at = _utc_now()
    pending = sum(1 for i in state.scene_states if i.status is KnowledgeGenerationStatus.pending)
    failed = sum(1 for i in state.scene_states if i.status is KnowledgeGenerationStatus.failed)
    print(f"[P5] {label} 开始：pending={pending} failed={failed}（{started_at}）", flush=True)
    result = generate_knowledge_candidates(
        bundle,
        state,
        client,
        scene_ids=scene_ids,
        max_attempts=MAX_ATTEMPTS,
        state_path=STATE_PATH,
        alias_config=alias_config,
    )
    finalization = finalize_knowledge_candidates(bundle, result.new_state, alias_config=alias_config)
    manifest = build_knowledge_run_manifest(
        bundle,
        result.new_state,
        finalization,
        attempted_scene_ids=result.attempted_scene_ids,
        model_call_stats={str(k): v for k, v in client.stats.items()},
        started_at=started_at,
        completed_at=_utc_now(),
    )
    save_knowledge_run_with_manifest(STATE_PATH, MANIFEST_OUT_PATH, result.new_state, manifest)
    _print_state_summary(result.new_state)
    print(
        f"[P5] {label} 完成：attempted={len(result.attempted_scene_ids)} "
        f"succeeded={len(result.succeeded_scene_ids)} failed={len(result.failed_scene_ids)} "
        f"skipped={len(result.skipped_scene_ids)} 本次调用={result.total_attempts}"
    )
    if result.failed_scene_ids:
        print(f"[P5] 失败场景: {result.failed_scene_ids}")
    print(f"[P5] manifest 已提交：{MANIFEST_OUT_PATH}")


def phase_smoke() -> None:
    _run_phase(SMOKE_SCENE_IDS, "smoke")


def phase_run() -> None:
    _run_phase(None, "全量运行")


def _clip(text: str, limit: int = 300) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + "…"


def _build_review_md(
    bundle,
    finalization,
    quality_report: dict[str, Any],
    manifest: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# P5 知识卡候选人工复核材料")
    lines.append("")
    lines.append("## 概览")
    lines.append("")
    counts = quality_report["card_counts"]
    lines.append(
        f"- 模型：{manifest['model_id']}（temperature={TEMPERATURE}, top_p={TOP_P}, num_ctx={NUM_CTX}, timeout={REQUEST_TIMEOUT_S}s, max_attempts={MAX_ATTEMPTS}）"
    )
    lines.append(f"- 场景状态：{quality_report['scene_status_counts']}")
    lines.append(
        f"- 候选数量：fact={counts['fact']} relation={counts['relation']} event={counts['event']}（共 {counts['total']}）"
    )
    lines.append(
        f"- 重复组：{quality_report['duplicate_group_count']}；冲突组：{quality_report['conflict_group_count']}（冲突卡已标记 needs_review）"
    )
    lines.append(
        f"- 失败场景：{len(quality_report['failed_scenes'])}；空产出场景：{len(quality_report['empty_scenes'])}"
    )
    lines.append(f"- 现实层分布：{quality_report['reality_status_distribution']}")
    lines.append("")
    lines.append("> 候选状态均为 draft（冲突组为 needs_review）；本材料不构成 approved。")
    lines.append("> evidence 为系统按行号从场景原文截取的原文；复核时请对照 source span 查看上下文。")
    lines.append("")

    if finalization.conflict_groups:
        lines.append("## 冲突组（needs_review，人工裁决）")
        lines.append("")
        for group in finalization.conflict_groups:
            lines.append(f"- **{group.group_key}**（{group.card_type}）：{group.description}")
            lines.append(f"  - 相关卡片：{'、'.join(group.card_ids)}")
        lines.append("")

    if finalization.duplicate_groups:
        lines.append("## 重复组（跨场景同一事实/关系，保留多证据来源）")
        lines.append("")
        for group in finalization.duplicate_groups[:200]:
            lines.append(f"- {group.payload_key}（{group.card_type}）：{len(group.card_ids)} 张卡片")
        if len(finalization.duplicate_groups) > 200:
            lines.append(f"- …共 {len(finalization.duplicate_groups)} 组（完整清单见 knowledge_quality_report.json）")
        lines.append("")

    status_by_card = {}
    review_note_by_card = {}
    review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
    for item in review["card_reviews"]:
        status_by_card[item["card_id"]] = item["review_status"]
        review_note_by_card[item["card_id"]] = item.get("notes", "")

    current_unit = None
    for doc, doc_type in [
        *[(d, "fact") for d in finalization.fact_documents],
        *[(d, "relation") for d in finalization.relation_documents],
        *[(d, "event") for d in finalization.event_documents],
    ]:
        unit = doc.story.story_unit_id
        if unit != current_unit:
            current_unit = unit
            lines.append(f"## {unit}（{doc.story.story_title}）")
            lines.append("")
        status = status_by_card.get(doc.id, doc.review_status.value)
        if doc_type == "fact":
            content = f"{doc.subject} — {doc.predicate} — {doc.value}"
        elif doc_type == "relation":
            content = f"{doc.subject} — {doc.relation} — {doc.target}"
        else:
            content = doc.title
        lines.append(f"### {doc.id}（{doc_type}｜{status}）")
        lines.append("")
        lines.append(f"- 内容：{content}")
        lines.append(f"- 标题：{doc.title}")
        lines.append(f"- 概括：{doc.summary}")
        if doc_type == "event":
            lines.append(f"- 参与者：{'、'.join(doc.participants) or '（无）'}")
            if doc.causes:
                lines.append(f"- 起因：{'；'.join(doc.causes)}")
            if doc.outcomes:
                lines.append(f"- 结果：{'；'.join(doc.outcomes)}")
        lines.append(
            f"- 出处：{doc.source.source_path} L{doc.source.line_start}-L{doc.source.line_end}"
            f"（现实层 {doc.reality_status.value}，时间层 {doc.story.temporal_scope.value}）"
        )
        lines.append(f"- evidence：{_clip(doc.evidence_text)}")
        note = review_note_by_card.get(doc.id, "")
        if note:
            lines.append(f"- 复核提示：{note}")
        lines.append("- 人工复核：审核意见 ______；review_status（draft/needs_review/approved/rejected）______")
        lines.append("")

    lines.append("## 失败场景（无候选，需人工后续处理）")
    lines.append("")
    if quality_report["failed_scenes"]:
        for item in quality_report["failed_scenes"]:
            lines.append(f"- {item['scene_id']}：{item['error_kind']}（attempts={item['attempts']}）")
    else:
        lines.append("- 无")
    lines.append("")
    return "\n".join(lines) + "\n"


def phase_finalize() -> None:
    bundle, alias_config = _load_inputs()
    if not STATE_PATH.is_file():
        _fail(f"候选运行状态不存在：{STATE_PATH}（先执行 smoke/run）")
    state = load_knowledge_run(STATE_PATH)
    pending = [item.scene_id for item in state.scene_states if item.status is KnowledgeGenerationStatus.pending]
    if pending:
        _fail(f"存在 {len(pending)} 个 pending 场景（先完成全量运行）: {pending[:5]}…")

    finalization = finalize_knowledge_candidates(bundle, state, alias_config=alias_config)
    for doc in [*finalization.fact_documents, *finalization.relation_documents, *finalization.event_documents]:
        if doc.review_status.value == "approved":
            _fail(f"候选路径出现 approved 卡片：{doc.id}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_documents_jsonl(FACTS_PATH, finalization.fact_documents)
    save_documents_jsonl(RELATIONS_PATH, finalization.relation_documents)
    save_documents_jsonl(EVENTS_PATH, finalization.event_documents)
    print(
        f"[P5] 候选文档已保存：fact={len(finalization.fact_documents)} "
        f"relation={len(finalization.relation_documents)} event={len(finalization.event_documents)}"
    )

    review = create_knowledge_review(bundle, finalization, reviewer=REVIEWER)
    save_knowledge_review(REVIEW_PATH, review)
    print(f"[P5] 人工审核状态已保存：{REVIEW_PATH}（{review.total_candidates} 张卡片，顶层 draft）")

    quality_report = build_knowledge_quality_report(bundle, state, finalization)
    QUALITY_REPORT_PATH.write_text(json.dumps(quality_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"[P5] 质量报告已保存：{QUALITY_REPORT_PATH}"
        f"（重复组 {quality_report['duplicate_group_count']}，冲突组 {quality_report['conflict_group_count']}）"
    )

    manifest = build_knowledge_run_manifest(
        bundle,
        state,
        finalization,
        attempted_scene_ids=[],
        model_call_stats={},
        started_at="",
        completed_at="",
    )
    if MANIFEST_OUT_PATH.is_file():
        previous = json.loads(MANIFEST_OUT_PATH.read_text(encoding="utf-8"))
        manifest = build_knowledge_run_manifest(
            bundle,
            state,
            finalization,
            attempted_scene_ids=previous.get("attempted_scene_ids", []),
            model_call_stats=previous.get("model_call_stats", {}),
            started_at=previous.get("started_at", ""),
            completed_at=previous.get("completed_at", ""),
        )
    manifest_payload = json.loads(json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False))
    MANIFEST_OUT_PATH.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    review_md = _build_review_md(bundle, finalization, quality_report, manifest_payload)
    REVIEW_MD_PATH.write_text(review_md, encoding="utf-8", newline="\n")
    print(f"[P5] 人工复核材料已保存：{REVIEW_MD_PATH}（{len(review_md.splitlines())} 行）")


def main() -> None:
    parser = argparse.ArgumentParser(description="P5 知识卡候选生成")
    parser.add_argument("phase", choices=["smoke", "run", "status", "finalize"])
    args = parser.parse_args()
    if args.phase == "smoke":
        phase_smoke()
    elif args.phase == "run":
        phase_run()
    elif args.phase == "status":
        phase_status()
    else:
        phase_finalize()


if __name__ == "__main__":
    main()
