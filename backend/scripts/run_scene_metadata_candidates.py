#!/usr/bin/env python3
"""P4C：基于正式冻结包运行真实 P4B 场景元数据候选生成，产出人工复核材料。

用法（在仓库根目录执行）：
  py -3.10 backend/scripts/run_scene_metadata_candidates.py smoke
      # 代表场景试运行（8 个场景：普通现实/回忆/梦境/书中故事/宣传元叙事/长场景分片）
  py -3.10 backend/scripts/run_scene_metadata_candidates.py run
      # 全量运行（断点续跑：逐场景原子保存状态，重跑只处理 pending/failed）
  py -3.10 backend/scripts/run_scene_metadata_candidates.py status
      # 查看当前运行状态与进度
  py -3.10 backend/scripts/run_scene_metadata_candidates.py finalize [--findings <path>]
      # 整体提交状态+manifest、合并候选进 P4A 审核文档、生成质量报告与人工复核材料

产物目录：backend/data/knowledge/tsukiyashiro_kisaki/scene_metadata_review/
  - candidate_run.json            P4B 候选运行状态（确定性，无时间戳）
  - run_manifest.json             P4B 运行 manifest（唯一携带时间戳的产物）
  - scene_metadata_review.json    P4A 审核文档（候选合并后，最多 needs_review，顶层 draft）
  - candidate_quality_report.json P4C 质量统计报告（含确定性分层抽检样本）
  - candidate_review.md           人工复核材料（按故事单元与场景顺序）
  - model_call_stats.json         模型调用统计（非确定性运行数据，仅本次进程）

模型客户端：OllamaCandidateClient 是最薄适配层，复用 inference.model_manager.OllamaProvider
的请求契约（POST {base_url}/api/chat；options: temperature/top_p/num_predict；模型与地址同
OLLAMA_MODEL/OLLAMA_BASE_URL 环境变量默认值）。不直接 import OllamaProvider 的原因：该模块
导入链会触发 Redis 连接与 .env 写入等应用级副作用（db.adapter），不适合独立批处理进程；
适配层按同一请求契约独立实现，参数与端点在本文件顶部常量中显式记录。

门禁（本脚本严格遵守）：
  - 冻结包三摘要必须与 EXPECTED_BUNDLE_SHA256 一致，任何不一致在模型调用前拒绝；
  - 不修改 scenes.jsonl / boundary_manifest.json / 游戏原文 / P3 边界决定；
  - 候选只能进入 needs_review，绝不自动 approved，不调用 apply_approved_scene_metadata
    或 write_enriched_scenes，不生成事实卡/关系卡/事件卡/embedding；
  - 模型不可用时准确报告阻塞并以非零退出码停止，绝不伪造候选。
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
    CandidateGenerationStatus,
    FrozenSceneBundle,
    create_candidate_run,
    create_scene_metadata_review,
    generate_scene_candidates,
    load_candidate_run,
    load_frozen_scene_bundle,
    load_scene_metadata_review,
    merge_candidates_into_review,
    save_candidate_run_with_manifest,
    save_scene_metadata_review,
    validate_candidate_run,
    validate_scene_metadata_review,
)
from knowledge.game_rag.models import ReviewStatus, SceneDocument  # noqa: E402
from knowledge.game_rag.scene_metadata_candidate import (  # noqa: E402
    CANDIDATE_CHUNK_MAX_LINES,
    SceneCandidateState,
)

# ---------- 常量：输入冻结包与输出目录 ----------

BUNDLE_DIR = BACKEND_DIR / "data" / "knowledge" / "tsukiyashiro_kisaki" / "scene_boundary_review"
SCENES_PATH = BUNDLE_DIR / "scenes.jsonl"
MANIFEST_PATH = BUNDLE_DIR / "boundary_manifest.json"
# 权威 bundle_sha256（P3 正式冻结包，外部复核口径）——运行前后必须一致。
EXPECTED_BUNDLE_SHA256 = "b82d753eb93dbc1614b89e98c8498f03884c7d5dd433d223ef853708052c8595"

OUT_DIR = BACKEND_DIR / "data" / "knowledge" / "tsukiyashiro_kisaki" / "scene_metadata_review"
STATE_PATH = OUT_DIR / "candidate_run.json"
MANIFEST_OUT_PATH = OUT_DIR / "run_manifest.json"
REVIEW_PATH = OUT_DIR / "scene_metadata_review.json"
QUALITY_REPORT_PATH = OUT_DIR / "candidate_quality_report.json"
REVIEW_MD_PATH = OUT_DIR / "candidate_review.md"
CALL_STATS_PATH = OUT_DIR / "model_call_stats.json"

P4A_REVIEWER = "project_owner_01"

# ---------- 常量：模型与生成参数（实际运行记录，写入运行状态与 manifest） ----------

MODEL_ID = "ollama:qwen2.5:7b"
OLLAMA_BASE_URL = "http://localhost:11434"  # 与 OllamaProvider 默认一致
OLLAMA_MODEL = "qwen2.5:7b"  # 本机已安装（Q4_K_M，context_length 32768）
TEMPERATURE = 0.2  # 结构化输出偏低温（非 0：保留少量重试多样性）
TOP_P = 0.9  # 与 OllamaProvider 默认一致
NUM_PREDICT = 1024  # 候选 JSON 约 200-400 token，上限留足
NUM_CTX = 16384  # 最大 prompt 约 6K 字符（约 4-6K token）+ 输出；防 Ollama 截断 prompt 头部
REQUEST_TIMEOUT_S = 300  # 首次加载/长分片的安全余量；超时由 P4B 收敛为可重试失败
MAX_ATTEMPTS = 3  # P4B 默认重试上限

GENERATION_PARAMS: dict[str, int | float | str | bool] = {
    "chunk_max_lines": CANDIDATE_CHUNK_MAX_LINES,
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "num_predict": NUM_PREDICT,
    "num_ctx": NUM_CTX,
    "timeout_s": REQUEST_TIMEOUT_S,
    "max_attempts": MAX_ATTEMPTS,
    "provider": "ollama",
    "base_url": OLLAMA_BASE_URL,
}

# ---------- 常量：smoke 代表场景（覆盖六类叙事现象与长场景分片） ----------

SMOKE_SCENE_IDS = [
    "scene_6d7f74124bb1d663",  # vol01，30 行：普通现实场景（夜子独白）
    "scene_993aa9d89a6c9a42",  # vol02，49 行：普通现实场景（琉璃视角）
    "scene_6d539b1b02ec01e2",  # vol03，85 行：回忆（「月社妃不幸的故事，其一」）
    "scene_1337d70a10d7cbb5",  # vol03，658 行：书中故事（魔法之书《蓝宝石的存在证明》）+ 5 分片
    "scene_ebf7ba4dbfed1ab9",  # vol05，240 行：梦境/回忆混合叙述 + 2 分片
    "scene_e5d4aa08d08dfb1e",  # vol05，695 行：最长场景 + 5 分片
    "scene_4c57301908d12659",  # vol13，79 行：克丽索贝莉露独白场景
    "scene_b33f3adaf98b715d",  # epilogue_meta，37 行：宣传元叙事（体验版宣传对话）
]

# ---------- 常量：确定性分层抽检（预注册算法，避免事后选择偏差） ----------
#
# 样本量 N_TARGET = max(30, ceil(总场景数 * 10%))，当前 262 场景 → 30，取 32 留余量。
# 分层顺序（先到先得，全部按冻结场景顺序，确定性）：
#   stratum_failed    全部 failed 场景（失败必查）
#   stratum_disagree  含「意见不一致」告警的场景，上限 8 个
#   stratum_unknown   viewpoint/temporal_scope/reality_status 任一为 unknown 的场景，上限 8 个
#   stratum_unit_rr   其余场景按故事单元轮转（单元按首次出现顺序，单元内取最早未采样场景）
#   stratum_tail      若仍不足 N_TARGET，按冻结顺序补足
# 抽检只记录问题与建议，绝不擅自批准。

SPOT_CHECK_N_TARGET = 32
SPOT_CHECK_DISAGREE_CAP = 8
SPOT_CHECK_UNKNOWN_CAP = 8

QUALITY_REPORT_SCHEMA_VERSION = 1
RUNNER_ID = "backend/scripts/run_scene_metadata_candidates.py"


# ---------- 模型客户端适配层 ----------


class OllamaCandidateClient:
    """最薄 CandidateModelClient 适配层：同步调用本机 Ollama /api/chat。

    - 请求契约与 inference.model_manager.OllamaProvider 一致（messages/stream/options），
      额外显式设置 num_ctx 防止 Ollama 静默截断超长 prompt；
    - 超时抛 TimeoutError（P4B 收敛为 error_kind=timeout）；HTTP/连接错误抛 RuntimeError
      （P4B 收敛为 error_kind=model_error）——异常消息不含 prompt 或密钥；
    - 累计调用统计（次数/token/时延/失败）供运行报告使用，不进入 P4B 运行状态。
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
        return content


# ---------- 公共工具 ----------


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")  # noqa: UP017  # 运行环境为 Python 3.10，无 datetime.UTC


def _fail(message: str) -> None:
    print(f"[P4C] 阻塞：{message}", file=sys.stderr)
    raise SystemExit(1)


def load_bundle_with_gate() -> FrozenSceneBundle:
    """加载冻结包并验证三摘要；bundle_sha256 必须与权威值一致。"""
    if not SCENES_PATH.is_file() or not MANIFEST_PATH.is_file():
        _fail(f"冻结双文件不存在：{SCENES_PATH} / {MANIFEST_PATH}")
    bundle = load_frozen_scene_bundle(SCENES_PATH, MANIFEST_PATH)
    if bundle.bundle_digest != EXPECTED_BUNDLE_SHA256:
        _fail(
            "冻结包 bundle_sha256 与权威值不一致："
            f"{bundle.bundle_digest} != {EXPECTED_BUNDLE_SHA256}（拒绝任何模型调用）"
        )
    return bundle


def load_or_create_state(bundle: FrozenSceneBundle):
    """加载既有候选运行状态；不存在时创建初始状态（全 pending）。"""
    if STATE_PATH.is_file():
        state = load_candidate_run(STATE_PATH)
        errors = validate_candidate_run(state, bundle)
        if errors:
            _fail("既有候选运行状态未通过校验:\n- " + "\n- ".join(errors))
        if state.model_id != MODEL_ID:
            _fail(f"既有运行状态 model_id={state.model_id!r} 与本次 {MODEL_ID!r} 不一致（拒绝混跑）")
        if state.generation_params != GENERATION_PARAMS:
            _fail(
                "既有运行状态 generation_params 与本次参数不一致（拒绝混跑）：\n"
                f"- 既有: {state.generation_params}\n- 本次: {GENERATION_PARAMS}"
            )
        print(f"[P4C] 续跑既有运行状态：{STATE_PATH}")
        return state, False
    state = create_candidate_run(bundle, model_id=MODEL_ID, generation_params=GENERATION_PARAMS)
    print(f"[P4C] 创建初始运行状态：model_id={MODEL_ID}")
    return state, True


def _write_call_stats(
    client: OllamaCandidateClient,
    phase: str,
    bundle: FrozenSceneBundle,
    state,
) -> None:
    total_attempts = sum(item.attempts for item in state.scene_states)
    minimum_required_calls = _minimum_required_calls(bundle)
    payload = {
        "phase": phase,
        "recorded_at": _utc_now(),
        "note": (
            "client_stats 仅统计本次进程；lifetime_state_stats 从 candidate_run.json 的逐场景 "
            "attempts 确定性推导，覆盖全部断点续跑。二者均不进入 P4A 审核文档。"
        ),
        "client_stats": dict(client.stats),
        "lifetime_state_stats": {
            "total_attempts": total_attempts,
            "minimum_required_calls": minimum_required_calls,
            "excess_retry_calls": max(0, total_attempts - minimum_required_calls),
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CALL_STATS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _print_state_summary(state) -> None:
    counts = {"pending": 0, "success": 0, "failed": 0}
    for item in state.scene_states:
        counts[item.status.value] += 1
    total_attempts = sum(item.attempts for item in state.scene_states)
    print(
        f"[P4C] 状态：total={state.total_source_scenes} "
        f"pending={counts['pending']} success={counts['success']} failed={counts['failed']} "
        f"累计模型调用={total_attempts}"
    )
    failed_ids = [item.scene_id for item in state.scene_states if item.status is CandidateGenerationStatus.failed]
    if failed_ids:
        print(f"[P4C] 失败场景（{len(failed_ids)}）：{', '.join(failed_ids)}")


# ---------- smoke / run ----------


def phase_smoke() -> None:
    bundle = load_bundle_with_gate()
    state, _ = load_or_create_state(bundle)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = OllamaCandidateClient()
    started_at = _utc_now()
    print(f"[P4C] smoke 开始：{len(SMOKE_SCENE_IDS)} 个代表场景（{started_at}）")
    result = generate_scene_candidates(
        bundle,
        state,
        client,
        scene_ids=SMOKE_SCENE_IDS,
        max_attempts=MAX_ATTEMPTS,
        state_path=STATE_PATH,
    )
    manifest = save_candidate_run_with_manifest(
        STATE_PATH,
        MANIFEST_OUT_PATH,
        result.new_state,
        result,
        started_at=started_at,
        completed_at=_utc_now(),
    )
    _write_call_stats(client, "smoke", bundle, result.new_state)
    _print_state_summary(result.new_state)
    print(
        f"[P4C] smoke 完成：attempted={len(result.attempted_scene_ids)} "
        f"succeeded={len(result.succeeded_scene_ids)} failed={len(result.failed_scene_ids)} "
        f"skipped={len(result.skipped_scene_ids)}"
    )
    print(f"[P4C] manifest 已提交：{MANIFEST_OUT_PATH}（attempted={len(manifest.attempted_scene_ids)}）")


def phase_run() -> None:
    bundle = load_bundle_with_gate()
    state, _ = load_or_create_state(bundle)
    client = OllamaCandidateClient()
    started_at = _utc_now()
    pending = sum(1 for item in state.scene_states if item.status is CandidateGenerationStatus.pending)
    failed = sum(1 for item in state.scene_states if item.status is CandidateGenerationStatus.failed)
    if pending + failed == 0:
        print("[P4C] 无待处理场景（全部已成功），仅执行整体提交。")
    print(f"[P4C] 全量运行开始：pending={pending} failed={failed}（{started_at}）")
    result = generate_scene_candidates(
        bundle,
        state,
        client,
        max_attempts=MAX_ATTEMPTS,
        state_path=STATE_PATH,
    )
    manifest = save_candidate_run_with_manifest(
        STATE_PATH,
        MANIFEST_OUT_PATH,
        result.new_state,
        result,
        started_at=started_at,
        completed_at=_utc_now(),
    )
    _write_call_stats(client, "run", bundle, result.new_state)
    _print_state_summary(result.new_state)
    print(
        f"[P4C] 全量运行完成：attempted={len(result.attempted_scene_ids)} "
        f"succeeded={len(result.succeeded_scene_ids)} failed={len(result.failed_scene_ids)} "
        f"skipped={len(result.skipped_scene_ids)}"
    )
    print(f"[P4C] manifest 已提交：{MANIFEST_OUT_PATH}（attempted={len(manifest.attempted_scene_ids)}）")


def phase_status() -> None:
    bundle = load_bundle_with_gate()
    state, _ = load_or_create_state(bundle)
    _print_state_summary(state)


# ---------- 质量统计与抽检 ----------


def _scene_line_count(scene: SceneDocument) -> int:
    return scene.source.line_end - scene.source.line_start + 1


def _scene_chunk_count(scene: SceneDocument) -> int:
    total = _scene_line_count(scene)
    return (total + CANDIDATE_CHUNK_MAX_LINES - 1) // CANDIDATE_CHUNK_MAX_LINES


def _minimum_required_calls(bundle: FrozenSceneBundle) -> int:
    return sum(_scene_chunk_count(scene) for scene in bundle.scenes)


def _categorize_warning(warning: str) -> str:
    if "意见不一致" in warning:
        return "chunk_disagreement"
    if "分片" in warning:
        return "chunk_other"
    return "model_warning"


def select_spot_check_sample(bundle: FrozenSceneBundle, state) -> list[str]:
    """预注册的确定性分层抽检（算法见文件头部注释；只依赖冻结顺序与候选内容）。"""
    by_id = {scene.id: scene for scene in bundle.scenes}
    sampled: list[str] = []
    sampled_set: set[str] = set()

    def add(scene_id: str) -> bool:
        if scene_id in sampled_set:
            return False
        sampled.append(scene_id)
        sampled_set.add(scene_id)
        return True

    # stratum_failed：全部 failed 场景
    for item in state.scene_states:
        if item.status is CandidateGenerationStatus.failed:
            add(item.scene_id)

    # stratum_disagree：含「意见不一致」告警（上限）
    disagree_count = 0
    for item in state.scene_states:
        if disagree_count >= SPOT_CHECK_DISAGREE_CAP:
            break
        if (
            item.candidate is not None
            and any("意见不一致" in w for w in item.candidate.warnings)
            and add(item.scene_id)
        ):
            disagree_count += 1

    # stratum_unknown：任一分类字段为 unknown（上限）
    unknown_count = 0
    for item in state.scene_states:
        if unknown_count >= SPOT_CHECK_UNKNOWN_CAP:
            break
        candidate = item.candidate
        if (
            candidate is not None
            and (
                candidate.viewpoint == "unknown"
                or candidate.temporal_scope.value == "unknown"
                or candidate.reality_status.value == "unknown"
            )
            and add(item.scene_id)
        ):
            unknown_count += 1

    # stratum_unit_rr：按故事单元轮转补足
    unit_order: list[str] = []
    unit_scenes: dict[str, list[str]] = {}
    for scene in bundle.scenes:
        uid = scene.story.story_unit_id
        if uid not in unit_scenes:
            unit_order.append(uid)
            unit_scenes[uid] = []
        unit_scenes[uid].append(scene.id)
    unit_cursor = {uid: 0 for uid in unit_order}
    while len(sampled) < SPOT_CHECK_N_TARGET:
        progressed = False
        for uid in unit_order:
            if len(sampled) >= SPOT_CHECK_N_TARGET:
                break
            scenes = unit_scenes[uid]
            cursor = unit_cursor[uid]
            while cursor < len(scenes):
                scene_id = scenes[cursor]
                cursor += 1
                if add(scene_id):
                    progressed = True
                    break
            unit_cursor[uid] = cursor
        if not progressed:
            break

    # stratum_tail：仍不足则按冻结顺序补足
    if len(sampled) < SPOT_CHECK_N_TARGET:
        for scene in bundle.scenes:
            if len(sampled) >= SPOT_CHECK_N_TARGET:
                break
            add(scene.id)
    # 校验：样本必须真实存在
    unknown = [scene_id for scene_id in sampled if scene_id not in by_id]
    if unknown:  # pragma: no cover - 内部不变量
        raise RuntimeError(f"抽检样本含未知 scene id: {unknown}")
    return sampled


def build_quality_report(
    bundle: FrozenSceneBundle,
    state,
    spot_check_sample: list[str],
    findings: dict[str, Any] | None,
) -> dict[str, Any]:
    """从运行状态确定性推导质量统计（不含时间戳；抽检发现由调用方提供）。"""
    candidates = [item for item in state.scene_states if item.candidate is not None]
    status_counts = {"pending": 0, "success": 0, "failed": 0}
    for item in state.scene_states:
        status_counts[item.status.value] += 1

    viewpoint_dist: dict[str, int] = {}
    temporal_dist: dict[str, int] = {}
    reality_dist: dict[str, int] = {}
    for item in candidates:
        viewpoint_dist[item.candidate.viewpoint] = viewpoint_dist.get(item.candidate.viewpoint, 0) + 1
        temporal_dist[item.candidate.temporal_scope.value] = (
            temporal_dist.get(item.candidate.temporal_scope.value, 0) + 1
        )
        reality_dist[item.candidate.reality_status.value] = reality_dist.get(item.candidate.reality_status.value, 0) + 1

    unknown_counts = {
        "viewpoint_unknown": sum(1 for item in candidates if item.candidate.viewpoint == "unknown"),
        "temporal_scope_unknown": sum(1 for item in candidates if item.candidate.temporal_scope.value == "unknown"),
        "reality_status_unknown": sum(1 for item in candidates if item.candidate.reality_status.value == "unknown"),
    }
    candidate_total = len(candidates)
    empty_mentioned = sum(1 for item in candidates if item.candidate.mentioned_characters == [])
    empty_present = sum(1 for item in candidates if item.candidate.present_characters == [])

    # evidence 统计与越界复查（P4B 校验已保证，这里独立复核作为报告数据）
    total_spans = 0
    out_of_range = 0
    scenes_with_evidence = 0
    by_id = {scene.id: scene for scene in bundle.scenes}
    for item in candidates:
        spans = item.candidate.evidence
        if spans:
            scenes_with_evidence += 1
        total_spans += len(spans)
        scene = by_id[item.scene_id]
        for span in spans:
            if (
                span.source_path != scene.source.source_path
                or span.line_start < scene.source.line_start
                or span.line_end > scene.source.line_end
            ):
                out_of_range += 1

    disagree = {
        "viewpoint": 0,
        "temporal_scope": 0,
        "reality_status": 0,
    }
    disagreement_scenes: list[str] = []
    warning_total = 0
    warning_categories: dict[str, int] = {}
    scenes_with_warnings = 0
    for item in candidates:
        has_warning = bool(item.candidate.warnings)
        if has_warning:
            scenes_with_warnings += 1
        for warning in item.candidate.warnings:
            warning_total += 1
            category = _categorize_warning(warning)
            warning_categories[category] = warning_categories.get(category, 0) + 1
        for field_name in disagree:
            if any(f"分片 {field_name} 意见不一致" in w for w in item.candidate.warnings):
                disagree[field_name] += 1
                if item.scene_id not in disagreement_scenes:
                    disagreement_scenes.append(item.scene_id)

    unit_stats: dict[str, dict[str, int | float]] = {}
    for scene in bundle.scenes:
        uid = scene.story.story_unit_id
        stats = unit_stats.setdefault(uid, {"success": 0, "total": 0, "failed": 0, "rate": 0.0})
        stats["total"] = int(stats["total"]) + 1
    state_by_id = {item.scene_id: item for item in state.scene_states}
    for uid, stats in unit_stats.items():
        for scene in bundle.scenes:
            if scene.story.story_unit_id != uid:
                continue
            item = state_by_id[scene.id]
            if item.status is CandidateGenerationStatus.success:
                stats["success"] = int(stats["success"]) + 1
            elif item.status is CandidateGenerationStatus.failed:
                stats["failed"] = int(stats["failed"]) + 1
        stats["rate"] = round(int(stats["success"]) / int(stats["total"]), 4)

    failed_scenes = [
        {
            "scene_id": item.scene_id,
            "story_unit_id": item.story_unit_id,
            "error_kind": item.last_failure.error_kind if item.last_failure else "unknown",
            "attempts": item.attempts,
        }
        for item in state.scene_states
        if item.status is CandidateGenerationStatus.failed
    ]

    longest = sorted(bundle.scenes, key=_scene_line_count, reverse=True)[:10]
    longest_scenes = [
        {
            "scene_id": scene.id,
            "story_unit_id": scene.story.story_unit_id,
            "lines": _scene_line_count(scene),
            "chunks": _scene_chunk_count(scene),
            "status": state_by_id[scene.id].status.value,
        }
        for scene in longest
    ]

    total_attempts = sum(item.attempts for item in state.scene_states)
    min_calls = _minimum_required_calls(bundle)

    report: dict[str, Any] = {
        "schema_version": QUALITY_REPORT_SCHEMA_VERSION,
        "generator": RUNNER_ID,
        "model_id": state.model_id,
        "generation_params": dict(state.generation_params),
        "source_bundle": {
            "manifest_sha256": state.source_manifest.manifest_sha256,
            "scenes_sha256": state.source_manifest.scenes_sha256,
            "bundle_sha256": state.source_manifest.bundle_sha256,
            "boundary_review_status": state.source_manifest.boundary_review_status,
            "reviewer": state.source_manifest.reviewer,
            "total_scenes": state.source_manifest.total_scenes,
        },
        "status_counts": status_counts,
        "model_calls": {
            "total_attempts": total_attempts,
            "minimum_required_calls": min_calls,
            "excess_retry_calls": max(0, total_attempts - min_calls),
            "note": "total_attempts 为状态中逐场景累计模型调用（含分片与重试）；"
            "minimum_required_calls 为全部分片各一次调用的理论下限；差值为重试/失败损耗上界。",
        },
        "viewpoint_distribution": dict(sorted(viewpoint_dist.items())),
        "temporal_scope_distribution": dict(sorted(temporal_dist.items())),
        "reality_status_distribution": dict(sorted(reality_dist.items())),
        "unknown_usage": {
            **unknown_counts,
            "candidate_total": candidate_total,
            "viewpoint_unknown_ratio": round(unknown_counts["viewpoint_unknown"] / candidate_total, 4)
            if candidate_total
            else 0.0,
            "temporal_scope_unknown_ratio": round(unknown_counts["temporal_scope_unknown"] / candidate_total, 4)
            if candidate_total
            else 0.0,
            "reality_status_unknown_ratio": round(unknown_counts["reality_status_unknown"] / candidate_total, 4)
            if candidate_total
            else 0.0,
        },
        "empty_character_arrays": {
            "mentioned_empty": empty_mentioned,
            "present_empty": empty_present,
            "candidate_total": candidate_total,
            "mentioned_empty_ratio": round(empty_mentioned / candidate_total, 4) if candidate_total else 0.0,
            "present_empty_ratio": round(empty_present / candidate_total, 4) if candidate_total else 0.0,
        },
        "evidence": {
            "total_spans": total_spans,
            "avg_per_candidate": round(total_spans / candidate_total, 2) if candidate_total else 0.0,
            "out_of_range_spans": out_of_range,
            "scenes_with_evidence": scenes_with_evidence,
            "scene_coverage_ratio": round(scenes_with_evidence / candidate_total, 4) if candidate_total else 0.0,
        },
        "chunk_disagreements": {
            **disagree,
            "scenes_with_any_disagreement": disagreement_scenes,
        },
        "warnings_summary": {
            "total": warning_total,
            "by_category": dict(sorted(warning_categories.items())),
            "scenes_with_warnings": scenes_with_warnings,
        },
        "unit_success_rates": unit_stats,
        "failed_scenes": failed_scenes,
        "longest_scenes": longest_scenes,
        "spot_check": {
            "n_target": SPOT_CHECK_N_TARGET,
            "sample_size": len(spot_check_sample),
            "algorithm": (
                "预注册确定性分层：全部 failed 场景；含「意见不一致」告警场景（上限 "
                f"{SPOT_CHECK_DISAGREE_CAP}）；任一分类字段为 unknown 的场景（上限 {SPOT_CHECK_UNKNOWN_CAP}）；"
                "其余按故事单元轮转补足至目标样本量；仍不足按冻结顺序补足。全部按冻结场景顺序。"
            ),
            "sampled_scene_ids": spot_check_sample,
            "findings": (findings or {}).get("findings", []),
            "recommendations": (findings or {}).get("recommendations", []),
            "note": "抽检只记录问题与建议，不擅自批准；发现内容不改变任何 review_status。",
        },
        "gates": {
            "candidates_are_needs_review_only": True,
            "top_level_review_status": "draft",
            "approved_artifacts_generated": False,
        },
    }
    return report


# ---------- 人工复核材料（Markdown） ----------

REVIEW_MD_FULL_TEXT_MAX_LINES = 40
REVIEW_MD_EXCERPT_EDGE_LINES = 8
REVIEW_MD_EVIDENCE_MAX_LINES = 12
REVIEW_MD_EVIDENCE_EDGE_LINES = 4


def _scene_excerpt_block(scene: SceneDocument) -> str:
    """无候选场景的原文展示（短场景完整、长场景首尾摘录），供人工直接审核定位。"""
    text_lines = scene.text.split("\n")
    offset = scene.source.line_start
    body: list[str] = []
    if len(text_lines) <= REVIEW_MD_FULL_TEXT_MAX_LINES:
        body = [f"L{offset + index}: {line}" for index, line in enumerate(text_lines)]
        note = "（完整场景原文；仍以 source span 对应的冻结原文为准）"
    else:
        omitted = len(text_lines) - 2 * REVIEW_MD_EXCERPT_EDGE_LINES
        head = [f"L{offset + index}: {line}" for index, line in enumerate(text_lines[:REVIEW_MD_EXCERPT_EDGE_LINES])]
        tail = [
            f"L{offset + index}: {line}"
            for index, line in enumerate(
                text_lines[-REVIEW_MD_EXCERPT_EDGE_LINES:], start=len(text_lines) - REVIEW_MD_EXCERPT_EDGE_LINES
            )
        ]
        body = head + [f"……（中间省略 {omitted} 行；摘录不代替完整场景审核）……"] + tail
        note = "（仅首尾摘录；长场景审核必须对照完整 source span 原文）"
    return "\n".join(["", f"**原文**{note}：", "```text", *body, "```"])


def _evidence_excerpt(scene: SceneDocument, line_start: int, line_end: int) -> list[str]:
    """evidence 对应原文（超长 span 只展示首尾摘录并显式声明）。"""
    text_lines = scene.text.split("\n")
    offset = scene.source.line_start
    span_lines = [
        f"L{offset + index}: {line}"
        for index, line in enumerate(text_lines)
        if line_start <= offset + index <= line_end
    ]
    if len(span_lines) <= REVIEW_MD_EVIDENCE_MAX_LINES:
        return span_lines
    head = span_lines[:REVIEW_MD_EVIDENCE_EDGE_LINES]
    tail = span_lines[-REVIEW_MD_EVIDENCE_EDGE_LINES:]
    omitted = len(span_lines) - 2 * REVIEW_MD_EVIDENCE_EDGE_LINES
    return head + [f"……（中间省略 {omitted} 行；摘录不代替完整场景审核）……"] + tail


def _render_review_md_scene(scene: SceneDocument, item: SceneCandidateState, model_id: str) -> list[str]:
    lines: list[str] = []
    line_count = _scene_line_count(scene)
    chunk_count = _scene_chunk_count(scene)
    lines.append(
        f"### {scene.id}｜{scene.source.source_path} L{scene.source.line_start}-{scene.source.line_end}"
        f"｜{line_count} 行｜分片 {chunk_count}"
    )
    lines.append(f"- story_unit: {scene.story.story_unit_id}｜{scene.story.story_title}")
    lines.append(f"- speakers（仅供参考，不等于在场人物）: {'、'.join(scene.speakers) or '（无）'}")
    lines.append("")

    if item.candidate is not None:
        candidate = item.candidate
        lines.append(f"**模型候选**（model_id={model_id}；仅供人工参考，不是决定；最多 needs_review）：")
        lines.append(f"- viewpoint: {candidate.viewpoint}")
        lines.append(f"- temporal_scope: {candidate.temporal_scope.value}")
        lines.append(f"- reality_status: {candidate.reality_status.value}")
        lines.append(f"- mentioned_characters: {', '.join(candidate.mentioned_characters) or '[]'}")
        lines.append(f"- present_characters: {', '.join(candidate.present_characters) or '[]'}")
        evidence_desc = ", ".join(f"L{span.line_start}-L{span.line_end}" for span in candidate.evidence)
        lines.append(f"- evidence（{len(candidate.evidence)} 条）: {evidence_desc}")
        lines.append("")
        lines.append("**evidence 对应原文**（摘录；摘录不代替完整场景审核，完整原文见 source span）：")
        lines.append("```text")
        for span in candidate.evidence:
            lines.extend(_evidence_excerpt(scene, span.line_start, span.line_end))
        lines.append("```")
        lines.append("")
        if line_count > REVIEW_MD_FULL_TEXT_MAX_LINES:
            text_lines = scene.text.split("\n")
            omitted = len(text_lines) - 2 * REVIEW_MD_EXCERPT_EDGE_LINES
            lines.append("**场景首尾摘录**（仅辅助定位；长场景审核必须对照完整 source span 原文）：")
            lines.append("```text")
            lines.extend(text_lines[:REVIEW_MD_EXCERPT_EDGE_LINES])
            lines.append(f"……（中间省略 {omitted} 行）……")
            lines.extend(text_lines[-REVIEW_MD_EXCERPT_EDGE_LINES:])
            lines.append("```")
            lines.append("")
        lines.append("**reasons**（模型判断理由）：")
        for reason in candidate.reasons:
            lines.append(f"- {reason}")
        if candidate.warnings:
            lines.append("")
            lines.append("**warnings**（含长场景分片分歧，需人工重点复核）：")
            for warning in candidate.warnings:
                lines.append(f"- {warning}")
        else:
            lines.append("")
            lines.append("**warnings**: （无）")
    elif item.status is CandidateGenerationStatus.failed:
        failure = item.last_failure
        lines.append(
            f"**候选生成失败**（error_kind={failure.error_kind if failure else 'unknown'}，"
            f"attempts={item.attempts}）：需人工直接审核，无候选参考。"
        )
        lines.append(_scene_excerpt_block(scene))
    else:
        lines.append("**候选尚未生成**（pending）：需先补跑候选或人工直接审核。")
        lines.append(_scene_excerpt_block(scene))

    lines.append("")
    lines.append("**人工填写区**（候选仅供参考；最终以人工判断为准）：")
    lines.append("- viewpoint: ______（<人物名>第一人称 / 第三人称 / 多视角 / unknown）")
    lines.append("- temporal_scope: ______（current / flashback / reconstruction / hypothetical / unknown）")
    lines.append(
        "- reality_status: ______（objective / character_claim / inferred / fictional / conflicted / unknown）"
    )
    lines.append("- mentioned_characters: ______（已审核且确认无则 []）")
    lines.append("- present_characters: ______（当前叙事层实际在场；已审核且确认无则 []）")
    lines.append(f"- evidence: ______（场景内行号范围，如 L{scene.source.line_start}-L{scene.source.line_start}）")
    lines.append("- reason: ______")
    lines.append("- 最终 review_status: ______（draft / needs_review / approved / rejected）")
    lines.append("")
    return lines


def build_review_md(bundle: FrozenSceneBundle, state, merge_report, quality_report: dict[str, Any]) -> str:
    lines: list[str] = [
        "# P4C 场景元数据候选人工复核材料",
        "",
        f"- 源冻结包：bundle_sha256={bundle.bundle_digest[:16]}…，boundary_review_status="
        f"{bundle.manifest.boundary_review_status}，reviewer={bundle.manifest.reviewer}，"
        f"total_scenes={bundle.total_scenes}",
        f"- 候选模型：model_id={state.model_id}，generation_params={json.dumps(state.generation_params, ensure_ascii=False)}",
        f"- 候选状态：success={quality_report['status_counts']['success']} "
        f"failed={quality_report['status_counts']['failed']} pending={quality_report['status_counts']['pending']}",
        f"- 合并策略：on_conflict={merge_report.on_conflict}；合并 {len(merge_report.merged_scene_ids)} 个场景，"
        f"冲突跳过 {len(merge_report.skipped_conflict)} 个，无候选 {len(merge_report.no_candidate_scene_ids)} 个",
        "- 语义约定：候选是模型输出，仅供人工参考；合并后记录最多 needs_review，绝不自动 approved；"
        "None=尚未审核，unknown=已审核但无法判断，空人物数组=已审核且确认无人",
        "- 原文展示：evidence 与长场景只展示摘录，摘录不代替完整场景审核；长场景审核必须对照完整 source span 原文",
        "- speakers 仅为参考：书中故事/回忆/梦境/转述中的台词不属于当前叙事层在场人物",
        "",
        "---",
        "",
    ]
    unit_scenes: dict[str, list[SceneDocument]] = {}
    for scene in bundle.scenes:
        unit_scenes.setdefault(scene.story.story_unit_id, []).append(scene)
    state_by_id = {item.scene_id: item for item in state.scene_states}
    for uid, scenes in unit_scenes.items():
        lines.append(f"## {uid}（{scenes[0].story.story_title}｜{len(scenes)} 个场景）")
        lines.append("")
        for scene in scenes:
            lines.extend(_render_review_md_scene(scene, state_by_id[scene.id], state.model_id))
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


# ---------- finalize ----------


def _prepare_review_for_finalize(bundle: FrozenSceneBundle, state, review_path: Path):
    """加载既有审核进度后合并候选；首次运行才创建空白审核文档。"""
    if review_path.is_file():
        review = load_scene_metadata_review(review_path)
        errors = validate_scene_metadata_review(review, bundle)
        if errors:
            raise ValueError("既有 P4A 审核文档未通过校验:\n- " + "\n- ".join(errors))
        if review.review_status == "approved":
            raise ValueError("既有 P4A 审核文档已整体 approved，候选阶段不得再次 finalize")
    else:
        review = create_scene_metadata_review(bundle, reviewer=P4A_REVIEWER)

    approved_before = {
        decision.scene_id for decision in review.scene_decisions if decision.review_status is ReviewStatus.approved
    }
    merge_report = merge_candidates_into_review(bundle, review, state, on_conflict="skip")
    approved_after = {
        decision.scene_id
        for decision in merge_report.review_doc.scene_decisions
        if decision.review_status is ReviewStatus.approved
    }
    if approved_after != approved_before:
        raise ValueError("候选合并改变了既有 approved 场景集合，拒绝写出")
    return merge_report


def phase_finalize(findings_path: Path | None) -> None:
    bundle = load_bundle_with_gate()
    if not STATE_PATH.is_file():
        _fail(f"候选运行状态不存在：{STATE_PATH}（先执行 smoke/run）")
    state = load_candidate_run(STATE_PATH)
    errors = validate_candidate_run(state, bundle)
    if errors:
        _fail("候选运行状态未通过校验:\n- " + "\n- ".join(errors))

    findings: dict[str, Any] | None = None
    if findings_path is not None:
        if not findings_path.is_file():
            _fail(f"抽检发现文件不存在：{findings_path}")
        findings = json.loads(findings_path.read_text(encoding="utf-8"))

    # 1. 首次创建或加载既有 P4A 审核文档，再以 skip 策略合并候选。
    #    续跑 finalize 必须保留所有既有人工字段与审核状态。
    try:
        merge_report = _prepare_review_for_finalize(bundle, state, REVIEW_PATH)
    except ValueError as exc:
        _fail(str(exc))
    new_review = merge_report.review_doc
    if new_review.review_status != "draft":
        _fail(f"合并后顶层 review_status 必须为 draft，当前为 {new_review.review_status!r}")
    save_scene_metadata_review(REVIEW_PATH, new_review)
    residual = validate_scene_metadata_review(load_scene_metadata_review(REVIEW_PATH), bundle)
    if residual:
        _fail("合并后的审核文档未通过 P4A 校验:\n- " + "\n- ".join(residual))
    print(
        f"[P4C] P4A 审核文档已保存：{REVIEW_PATH}（merged={len(merge_report.merged_scene_ids)}，"
        f"conflict_skipped={len(merge_report.skipped_conflict)}，"
        f"no_candidate={len(merge_report.no_candidate_scene_ids)}）"
    )

    # 2. 质量报告（含确定性分层抽检样本）
    sample = select_spot_check_sample(bundle, state)
    quality_report = build_quality_report(bundle, state, sample, findings)
    QUALITY_REPORT_PATH.write_text(json.dumps(quality_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[P4C] 质量报告已保存：{QUALITY_REPORT_PATH}（抽检样本 {len(sample)} 个场景）")

    # 3. 人工复核材料
    review_md = build_review_md(bundle, state, merge_report, quality_report)
    REVIEW_MD_PATH.write_text(review_md, encoding="utf-8", newline="\n")
    print(f"[P4C] 人工复核材料已保存：{REVIEW_MD_PATH}（{len(review_md.splitlines())} 行）")

    # 4. 配对校验：manifest 与状态必须一致（若 manifest 存在）
    if MANIFEST_OUT_PATH.is_file():
        manifest = json.loads(MANIFEST_OUT_PATH.read_text(encoding="utf-8"))
        state_json = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        checks = {
            "model_id": manifest.get("model_id") == state_json.get("model_id"),
            "bundle_sha256": manifest.get("source_bundle", {}).get("bundle_sha256")
            == state_json.get("source_manifest", {}).get("bundle_sha256"),
            "total_scenes": manifest.get("total_scenes") == state_json.get("total_source_scenes"),
            "scene_status_counts_sum": sum(manifest.get("scene_status_counts", {}).values())
            == len(state_json.get("scene_states", [])),
        }
        bad = [name for name, ok in checks.items() if not ok]
        if bad:
            _fail(f"candidate_run 与 run_manifest 配对校验失败：{bad}")
        print("[P4C] candidate_run 与 run_manifest 配对校验通过")


def main() -> None:
    parser = argparse.ArgumentParser(description="P4C 场景元数据候选真实运行")
    parser.add_argument("phase", choices=["smoke", "run", "status", "finalize"])
    parser.add_argument("--findings", type=Path, default=None, help="抽检发现 JSON 文件（finalize 可选）")
    args = parser.parse_args()
    if args.phase == "smoke":
        phase_smoke()
    elif args.phase == "run":
        phase_run()
    elif args.phase == "status":
        phase_status()
    else:
        phase_finalize(args.findings)


if __name__ == "__main__":
    main()
